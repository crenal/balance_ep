#include "dispatch.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stddef.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ void grid_barrier_balance(int *counter, int *sense, int blocks) {
  __shared__ int local_sense;
  if (threadIdx.x == 0) {
    __threadfence();
    int s = sense[0];
    local_sense = s;
    int ticket = atomicAdd(counter, 1);
    if (ticket == blocks - 1) {
      counter[0] = 0;
      __threadfence();
      atomicExch(sense, s ^ 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = local_sense;
    while (atomicAdd(sense, 0) == s) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ int node_block_index_balance(int local_node, int remote_node) {
  return remote_node < local_node ? remote_node : remote_node - 1;
}

__global__ void preprocess_balance_kernel(const bool *routing_map, int *counts, int *offsets,
                                          int *dst_index, int *local_counts, int num_tokens,
                                          int expert_num, int blocks, int *barrier_counter,
                                          int *barrier_sense) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int global_tokens = num_tokens * npes;
  int experts_per_rank = (expert_num + npes - 1) / npes;
  for (int g = tid; g < global_tokens; g += stride) {
    int src = g / num_tokens;
    size_t base = (size_t)g * (size_t)expert_num;
    for (int j = 0; j < npes; ++j) {
      bool has = false;
      int start = j * experts_per_rank;
      int end = start + experts_per_rank;
      if (start < expert_num) {
        if (end > expert_num) end = expert_num;
        for (int e = start; e < end; ++e) {
          if (routing_map[base + (size_t)e]) {
            has = true;
            break;
          }
        }
      }
      if (has) atomicAdd(&counts[src * npes + j], 1);
    }
  }

  grid_barrier_balance(barrier_counter, barrier_sense, blocks);

  for (int j = tid; j < npes; j += stride) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  grid_barrier_balance(barrier_counter, barrier_sense, blocks);

  for (int g = tid; g < global_tokens; g += stride) {
    int src = g / num_tokens;
    size_t base = (size_t)g * (size_t)expert_num;
    for (int j = 0; j < npes; ++j) {
      bool has = false;
      int start = j * experts_per_rank;
      int end = start + experts_per_rank;
      if (start < expert_num) {
        if (end > expert_num) end = expert_num;
        for (int e = start; e < end; ++e) {
          if (routing_map[base + (size_t)e]) {
            has = true;
            break;
          }
        }
      }
      if (has) {
        int local = atomicAdd(&local_counts[src * npes + j], 1);
        dst_index[(size_t)g * (size_t)npes + (size_t)j] = offsets[src * npes + j] + local;
      } else {
        dst_index[(size_t)g * (size_t)npes + (size_t)j] = -1;
      }
    }
  }
}

__global__ void node_counts_kernel(const int *dst_index, int *node_counts, int num_tokens,
                                   int node_npes, int nnodes) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int global_tokens = num_tokens * npes;
  for (int g = tid; g < global_tokens; g += stride) {
    int src_rank = g / num_tokens;
    for (int dst_node = 0; dst_node < nnodes; ++dst_node) {
      bool has = false;
      int base_rank = dst_node * node_npes;
      for (int lr = 0; lr < node_npes; ++lr) {
        int dst_rank = base_rank + lr;
        if (dst_rank >= npes) break;
        int idx = dst_index[(size_t)g * (size_t)npes + (size_t)dst_rank];
        if (idx >= 0) {
          has = true;
          break;
        }
      }
      if (has) atomicAdd(&node_counts[src_rank * nnodes + dst_node], 1);
    }
  }
}

static __device__ __forceinline__ void dispatch_stage1_balance(
    const void *input_tokens, void *output_tokens, const int *dst_index, int num_tokens,
    int hidden_size, int bytes_per_elem, int node_npes, int nnodes, const void *mid_buf,
    uint64_t *mid_flags, int chunk_tokens_local, int num_chunks, int npes, int mype, int node_id,
    int local_rank, const int *src_local_of_forward, const int *src_forward_local_of_dst_forward,
    size_t token_bytes, cg::thread_block_tile<128> tile) {
  int warp_id = tile.thread_rank() >> 5;
  int num_warps = tile.size() >> 5;
  int node_base = node_id * node_npes;
  for (int chunk_id = blockIdx.x; chunk_id < num_chunks; chunk_id += gridDim.x) {
    int t_begin = chunk_id * chunk_tokens_local;
    int t_end = t_begin + chunk_tokens_local;
    if (t_end > num_tokens) t_end = num_tokens;
    for (int t = t_begin + warp_id; t < t_end; t += num_warps) {
      size_t src_offset = (size_t)t * token_bytes;
      size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
      for (int lr = 0; lr < node_npes; ++lr) {
        int dst_rank = node_base + lr;
        int idx = dst_index[g * (size_t)npes + (size_t)dst_rank];
        if (idx < 0) continue;
        size_t dst_offset = (size_t)idx * token_bytes;
        nvshmemx_putmem_warp((char *)output_tokens + dst_offset,
                             (const char *)input_tokens + src_offset, token_bytes, dst_rank);
      }
    }
    if (nnodes > 1 && mid_buf && mid_flags) {
      for (int k = 1; k < nnodes; ++k) {
        int remote_node = node_id - k;
        if (remote_node < 0) remote_node += nnodes;
        int node_idx = node_block_index_balance(node_id, remote_node);
        uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_chunks + chunk_id;
        if (tile.thread_rank() == 0) {
          nvshmem_signal_wait_until(flag_ptr, NVSHMEM_CMP_EQ, 1ull);
        }
        tile.sync();
        int src_forward_local =
            src_forward_local_of_dst_forward[(remote_node * nnodes + node_id) * node_npes +
                                             local_rank];
        if (src_forward_local >= 0 && src_forward_local < node_npes) {
          int src_local = src_local_of_forward[remote_node * node_npes + src_forward_local];
          int src_rank = remote_node * node_npes + src_local;
          size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
          for (int t = t_begin + warp_id; t < t_end; t += num_warps) {
            size_t g_remote = (size_t)src_rank * (size_t)num_tokens + (size_t)t;
            const char *mid_ptr = (const char *)mid_buf + mid_base + (size_t)t * token_bytes;
            for (int lr = 0; lr < node_npes; ++lr) {
              int dst_rank = node_base + lr;
              int idx = dst_index[g_remote * (size_t)npes + (size_t)dst_rank];
              if (idx < 0) continue;
              size_t dst_offset = (size_t)idx * token_bytes;
              nvshmemx_putmem_warp((char *)output_tokens + dst_offset, mid_ptr, token_bytes,
                                   dst_rank);
            }
          }
        }
        tile.sync();
        if (tile.thread_rank() == 0) {
          flag_ptr[0] = 0;
        }
      }
    }
  }
}

static __device__ __forceinline__ void dispatch_stage2_balance(
    const void *local_buf, const int *dst_index, int num_tokens, int hidden_size, int bytes_per_elem,
    int node_npes, int nnodes, const void *mid_buf, uint64_t *mid_flags, int chunk_tokens_local,
    int num_chunks, int npes, int node_id, int local_rank, const int *src_local_of_forward,
    const int *dst_forward_local, size_t token_bytes, cg::thread_block_tile<128> tile) {
  (void)hidden_size;
  (void)bytes_per_elem;
  if (nnodes <= 1 || !mid_buf || !mid_flags) return;
  int src_local = src_local_of_forward[node_id * node_npes + local_rank];
  int src_rank = node_id * node_npes + src_local;
  int warp_id = tile.thread_rank() >> 5;
  int num_warps = tile.size() >> 5;
  for (int chunk_id = blockIdx.x; chunk_id < num_chunks; chunk_id += gridDim.x) {
    int t_begin = chunk_id * chunk_tokens_local;
    int t_end = t_begin + chunk_tokens_local;
    if (t_end > num_tokens) t_end = num_tokens;
    for (int t = t_begin + warp_id; t < t_end; t += num_warps) {
      size_t src_offset = (size_t)t * token_bytes;
      size_t g = (size_t)src_rank * (size_t)num_tokens + (size_t)t;
      for (int k = 1; k < nnodes; ++k) {
        int dst_node = node_id + k;
        if (dst_node >= nnodes) dst_node -= nnodes;
        bool need_send = false;
        int base_rank = dst_node * node_npes;
        for (int lr = 0; lr < node_npes; ++lr) {
          int dst_rank = base_rank + lr;
          int idx = dst_index[g * (size_t)npes + (size_t)dst_rank];
          if (idx >= 0) {
            need_send = true;
            break;
          }
        }
        if (!need_send) continue;
        int dst_fwd_local =
            dst_forward_local[(node_id * nnodes + dst_node) * node_npes + local_rank];
        int dst_pe = dst_node * node_npes + dst_fwd_local;
        int node_idx = node_block_index_balance(dst_node, node_id);
        size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
        size_t mid_offset = mid_base + (size_t)t * token_bytes;
        nvshmemx_putmem_warp((char *)mid_buf + mid_offset, (const char *)local_buf + src_offset,
                             token_bytes, dst_pe);
      }
    }
    tile.sync();
    if (tile.thread_rank() == 0) {
      for (int k = 1; k < nnodes; ++k) {
        int dst_node = node_id + k;
        if (dst_node >= nnodes) dst_node -= nnodes;
        int dst_fwd_local =
            dst_forward_local[(node_id * nnodes + dst_node) * node_npes + local_rank];
        int dst_pe = dst_node * node_npes + dst_fwd_local;
        int node_idx = node_block_index_balance(dst_node, node_id);
        uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_chunks + chunk_id;
        nvshmemx_signal_op(flag_ptr, 1ull, NVSHMEM_SIGNAL_SET, dst_pe);
      }
    }
  }
}

__global__ void dispatch_balance_kernel(const void *input_tokens, void *output_tokens,
                                        const int *dst_index, const void *local_buf, int num_tokens,
                                        int hidden_size, int bytes_per_elem, int node_npes, int nnodes,
                                        const void *mid_buf, uint64_t *mid_flags, int chunk_tokens,
                                        const int *src_local_of_forward,
                                        const int *dst_forward_local,
                                        const int *src_forward_local_of_dst_forward) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int node_id = mype / node_npes;
  int local_rank = mype - node_id * node_npes;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  int chunk_tokens_local = chunk_tokens > 0 ? chunk_tokens : num_tokens;
  int num_chunks = (num_tokens + chunk_tokens_local - 1) / chunk_tokens_local;
  if (num_chunks <= 0) return;
  auto tile = cg::tiled_partition<128>(cg::this_thread_block());
  int tile_id = tile.meta_group_rank();
  int tile_count = tile.meta_group_size();
  if (nnodes <= 1) {
    if (tile_id == 0) {
      dispatch_stage1_balance(input_tokens, output_tokens, dst_index, num_tokens, hidden_size,
                              bytes_per_elem, node_npes, nnodes, mid_buf, mid_flags,
                              chunk_tokens_local, num_chunks, npes, mype, node_id, local_rank,
                              src_local_of_forward, src_forward_local_of_dst_forward, token_bytes,
                              tile);
    }
    return;
  }
  bool stage1 = tile_id < (tile_count / 2);
  if (stage1) {
    dispatch_stage1_balance(input_tokens, output_tokens, dst_index, num_tokens, hidden_size,
                            bytes_per_elem, node_npes, nnodes, mid_buf, mid_flags,
                            chunk_tokens_local, num_chunks, npes, mype, node_id, local_rank,
                            src_local_of_forward, src_forward_local_of_dst_forward, token_bytes,
                            tile);
  } else {
    dispatch_stage2_balance(local_buf, dst_index, num_tokens, hidden_size, bytes_per_elem, node_npes,
                            nnodes, mid_buf, mid_flags, chunk_tokens_local, num_chunks, npes, node_id,
                            local_rank, src_local_of_forward, dst_forward_local, token_bytes, tile);
  }
}

__global__ void gather_balance_kernel(const void *input_tokens, void *local_buf,
                                      const int *dst_index, int num_tokens, int hidden_size,
                                      int bytes_per_elem, int node_npes, int nnodes,
                                      const int *src_forward_local) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int node_id = mype / node_npes;
  int src_local = mype - node_id * node_npes;
  int fwd_local = src_forward_local[node_id * node_npes + src_local];
  int fwd_rank = node_id * node_npes + fwd_local;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;

  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int warp_id = (threadIdx.x >> 5);
  int num_warps = (blockDim.x >> 5);
  for (int t = warp_id + blockIdx.x * num_warps; t < num_tokens; t += gridDim.x * num_warps) {
    size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
    bool need = false;
    for (int k = 1; k < nnodes; ++k) {
      int dst_node = node_id + k;
      if (dst_node >= nnodes) dst_node -= nnodes;
      int base_rank = dst_node * node_npes;
      for (int lr = 0; lr < node_npes; ++lr) {
        int dst_rank = base_rank + lr;
        if (dst_rank >= npes) break;
        int idx = dst_index[g * (size_t)npes + (size_t)dst_rank];
        if (idx >= 0) {
          need = true;
          break;
        }
      }
      if (need) break;
    }
    if (!need) continue;
    size_t offset = (size_t)t * token_bytes;
    if (fwd_rank == mype) {
      const uint4 *src = (const uint4 *)((const char *)input_tokens + offset);
      uint4 *dst = (uint4 *)((char *)local_buf + offset);
      size_t n = token_bytes / sizeof(uint4);
      for (size_t i = (size_t)warp.thread_rank(); i < n; i += (size_t)warp.size()) {
        dst[i] = src[i];
      }
    } else {
      nvshmemx_putmem_warp((char *)local_buf + offset, (const char *)input_tokens + offset,
                           token_bytes, fwd_rank);
    }
  }
}

static void compute_balance_maps_host(int npes, int node_npes, int nnodes, const int *counts_h,
                                      const int *node_counts_h, int *src_forward_local_h,
                                      int *src_local_of_forward_h, int *dst_forward_local_h,
                                      int *src_forward_local_of_dst_forward_h) {
  const double inv_intra = 1.0 / 200.0;
  const double inv_inter = 1.0 / 50.0;

  int nodes = nnodes > 0 ? nnodes : 1;
  int rpn = node_npes > 0 ? node_npes : npes;

  for (int n = 0; n < nodes; ++n) {
    for (int i = 0; i < rpn; ++i) {
      src_forward_local_h[n * rpn + i] = i;
      src_local_of_forward_h[n * rpn + i] = i;
    }
  }
  for (int a = 0; a < nodes; ++a) {
    for (int b = 0; b < nodes; ++b) {
      for (int f = 0; f < rpn; ++f) {
        dst_forward_local_h[(a * nodes + b) * rpn + f] = f;
        src_forward_local_of_dst_forward_h[(a * nodes + b) * rpn + f] = f;
      }
    }
  }

  int *intra_send = (int *)malloc((size_t)npes * sizeof(int));
  int *intra_recv = (int *)malloc((size_t)npes * sizeof(int));
  int *inter_send = (int *)malloc((size_t)npes * sizeof(int));
  int *inter_recv = (int *)malloc((size_t)npes * sizeof(int));
  int *dist_total = (int *)malloc((size_t)npes * (size_t)nodes * sizeof(int));
  if (!intra_send || !intra_recv || !inter_send || !inter_recv || !dist_total) {
    if (intra_send) free(intra_send);
    if (intra_recv) free(intra_recv);
    if (inter_send) free(inter_send);
    if (inter_recv) free(inter_recv);
    if (dist_total) free(dist_total);
    return;
  }

  for (int r = 0; r < npes; ++r) {
    intra_send[r] = 0;
    intra_recv[r] = 0;
    inter_send[r] = 0;
    inter_recv[r] = 0;
  }
  for (int s = 0; s < npes; ++s) {
    int snode = s / rpn;
    for (int b = 0; b < nodes; ++b) {
      int sum = 0;
      int base_rank = b * rpn;
      for (int lr = 0; lr < rpn; ++lr) {
        int d = base_rank + lr;
        if (d >= npes) break;
        sum += counts_h[s * npes + d];
        if (b == snode && d != s) {
          intra_send[s] += counts_h[s * npes + d];
        }
      }
      dist_total[s * nodes + b] = sum;
    }
  }
  for (int d = 0; d < npes; ++d) {
    int sum = 0;
    for (int s = 0; s < npes; ++s) {
      if (s == d) continue;
      sum += counts_h[s * npes + d];
    }
    intra_recv[d] = sum;
  }

  for (int n = 0; n < nodes; ++n) {
    int used_fwd[64];
    int weights[64];
    int order[64];
    int m = rpn;
    if (m > 64) m = 64;
    for (int i = 0; i < m; ++i) {
      used_fwd[i] = 0;
      int src_rank = n * rpn + i;
      int off = 0;
      for (int b = 0; b < nodes; ++b) {
        if (b == n) continue;
        off += node_counts_h[src_rank * nodes + b];
      }
      weights[i] = off;
      order[i] = i;
    }
    for (int i = 0; i < m; ++i) {
      int best = i;
      for (int j = i + 1; j < m; ++j) {
        if (weights[order[j]] > weights[order[best]]) best = j;
      }
      int tmp = order[i];
      order[i] = order[best];
      order[best] = tmp;
    }
    for (int pi = 0; pi < m; ++pi) {
      int s = order[pi];
      int best_f = -1;
      double best_score = 0.0;
      for (int f = 0; f < m; ++f) {
        if (used_fwd[f]) continue;
        int src_rank = n * rpn + s;
        int f_rank = n * rpn + f;
        int w = weights[s];
        int add_intra = (f == s) ? 0 : w;
        double a = ((double)(inter_send[f_rank] + w)) * inv_inter;
        double b = ((double)(intra_recv[f_rank] + add_intra)) * inv_intra;
        double c = ((double)(intra_send[src_rank] + add_intra)) * inv_intra;
        double score = a;
        if (b > score) score = b;
        if (c > score) score = c;
        if (best_f < 0 || score < best_score) {
          best_f = f;
          best_score = score;
        }
      }
      if (best_f < 0) best_f = s;
      used_fwd[best_f] = 1;
      src_forward_local_h[n * rpn + s] = best_f;
      src_local_of_forward_h[n * rpn + best_f] = s;
      int src_rank = n * rpn + s;
      int f_rank = n * rpn + best_f;
      inter_send[f_rank] += weights[s];
      if (best_f != s) {
        intra_send[src_rank] += weights[s];
        intra_recv[f_rank] += weights[s];
      }
    }
  }

  for (int a = 0; a < nodes; ++a) {
    for (int b = 0; b < nodes; ++b) {
      if (a == b) continue;
      int used_dst[64];
      int order[64];
      int m = rpn;
      if (m > 64) m = 64;
      for (int i = 0; i < m; ++i) {
        used_dst[i] = 0;
        order[i] = i;
      }
      for (int i = 0; i < m; ++i) {
        int best = i;
        for (int j = i + 1; j < m; ++j) {
          int f1 = order[j];
          int f0 = order[best];
          int s1 = src_local_of_forward_h[a * rpn + f1];
          int s0 = src_local_of_forward_h[a * rpn + f0];
          int r1 = a * rpn + s1;
          int r0 = a * rpn + s0;
          int w1 = node_counts_h[r1 * nodes + b];
          int w0 = node_counts_h[r0 * nodes + b];
          int d1 = dist_total[r1 * nodes + b];
          int d0 = dist_total[r0 * nodes + b];
          int k1 = w1 * 4 + d1;
          int k0 = w0 * 4 + d0;
          if (k1 > k0) best = j;
        }
        int tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
      }
      for (int pi = 0; pi < m; ++pi) {
        int f = order[pi];
        int s = src_local_of_forward_h[a * rpn + f];
        int src_rank = a * rpn + s;
        int w = node_counts_h[src_rank * nodes + b];
        int dist = dist_total[src_rank * nodes + b];
        int best_j = -1;
        double best_score = 0.0;
        for (int j = 0; j < m; ++j) {
          if (used_dst[j]) continue;
          int dst_rank = b * rpn + j;
          int self_cnt = counts_h[src_rank * npes + dst_rank];
          int add_intra = dist - self_cnt;
          if (add_intra < 0) add_intra = 0;
          double a1 = ((double)(inter_recv[dst_rank] + w)) * inv_inter;
          double b1 = ((double)(intra_send[dst_rank] + add_intra)) * inv_intra;
          double score = a1 > b1 ? a1 : b1;
          if (best_j < 0 || score < best_score) {
            best_j = j;
            best_score = score;
          }
        }
        if (best_j < 0) best_j = f;
        used_dst[best_j] = 1;
        dst_forward_local_h[(a * nodes + b) * rpn + f] = best_j;
        src_forward_local_of_dst_forward_h[(a * nodes + b) * rpn + best_j] = f;
        int dst_rank = b * rpn + best_j;
        int self_cnt = counts_h[src_rank * npes + dst_rank];
        int add_intra = dist - self_cnt;
        if (add_intra < 0) add_intra = 0;
        inter_recv[dst_rank] += w;
        intra_send[dst_rank] += add_intra;
      }
    }
  }

  free(intra_send);
  free(intra_recv);
  free(inter_send);
  free(inter_recv);
  free(dist_total);
}

int pre_process_balance(const bool *routing_map, int *dst_index, const DispatchConfig *cfg) {
  if (!cfg) return 1;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int expert_num = cfg->expert_num;
  int node_npes = cfg->node_npes;
  int nnodes = cfg->nnodes;

  int *counts = cfg->counts;
  int *offsets = cfg->offsets;
  int *local_counts = cfg->local_counts;
  int *barrier_counter = cfg->barrier_counter;
  int *barrier_sense = cfg->barrier_sense;
  int *node_counts = cfg->node_counts;
  int *src_forward_local = cfg->src_forward_local;
  int *src_local_of_forward = cfg->src_local_of_forward;
  int *dst_forward_local = cfg->dst_forward_local;
  int *src_forward_local_of_dst_forward = cfg->src_forward_local_of_dst_forward;

  if (!routing_map || !dst_index || !counts || !offsets || !local_counts || !barrier_counter ||
      !barrier_sense || !node_counts || !src_forward_local || !src_local_of_forward ||
      !dst_forward_local || !src_forward_local_of_dst_forward) {
    return 1;
  }

  cudaMemset(counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(local_counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(barrier_counter, 0, sizeof(int));
  cudaMemset(barrier_sense, 0, sizeof(int));
  cudaMemset(node_counts, 0, (size_t)npes * (size_t)nnodes * sizeof(int));

  int threads = 256;
  int global_tokens = num_tokens * npes;
  int blocks = (global_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;

  preprocess_balance_kernel<<<blocks, threads>>>(routing_map, counts, offsets, dst_index, local_counts,
                                                 num_tokens, expert_num, blocks, barrier_counter,
                                                 barrier_sense);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;

  int blocks2 = (global_tokens + threads - 1) / threads;
  if (blocks2 > 4096) blocks2 = 4096;
  node_counts_kernel<<<blocks2, threads>>>(dst_index, node_counts, num_tokens, node_npes, nnodes);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;

  int *counts_h = (int *)malloc((size_t)npes * (size_t)npes * sizeof(int));
  int *node_counts_h = (int *)malloc((size_t)npes * (size_t)nnodes * sizeof(int));
  int *src_forward_local_h = (int *)malloc((size_t)nnodes * (size_t)node_npes * sizeof(int));
  int *src_local_of_forward_h = (int *)malloc((size_t)nnodes * (size_t)node_npes * sizeof(int));
  int *dst_forward_local_h =
      (int *)malloc((size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int));
  int *src_forward_local_of_dst_forward_h =
      (int *)malloc((size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int));
  if (!counts_h || !node_counts_h || !src_forward_local_h || !src_local_of_forward_h ||
      !dst_forward_local_h || !src_forward_local_of_dst_forward_h) {
    if (counts_h) free(counts_h);
    if (node_counts_h) free(node_counts_h);
    if (src_forward_local_h) free(src_forward_local_h);
    if (src_local_of_forward_h) free(src_local_of_forward_h);
    if (dst_forward_local_h) free(dst_forward_local_h);
    if (src_forward_local_of_dst_forward_h) free(src_forward_local_of_dst_forward_h);
    return 1;
  }

  cudaMemcpy(counts_h, counts, (size_t)npes * (size_t)npes * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(node_counts_h, node_counts, (size_t)npes * (size_t)nnodes * sizeof(int),
             cudaMemcpyDeviceToHost);

  compute_balance_maps_host(npes, node_npes, nnodes, counts_h, node_counts_h, src_forward_local_h,
                            src_local_of_forward_h, dst_forward_local_h,
                            src_forward_local_of_dst_forward_h);

  cudaMemcpy(src_forward_local, src_forward_local_h,
             (size_t)nnodes * (size_t)node_npes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(src_local_of_forward, src_local_of_forward_h,
             (size_t)nnodes * (size_t)node_npes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dst_forward_local, dst_forward_local_h,
             (size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(src_forward_local_of_dst_forward, src_forward_local_of_dst_forward_h,
             (size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int),
             cudaMemcpyHostToDevice);

  free(counts_h);
  free(node_counts_h);
  free(src_forward_local_h);
  free(src_local_of_forward_h);
  free(dst_forward_local_h);
  free(src_forward_local_of_dst_forward_h);
  return 0;
}

int dispatch_tokens_balance(const void *input_tokens, void *output_tokens, const int *dst_index,
                            const DispatchConfig *cfg) {
  if (!cfg) return 1;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;
  int node_npes = cfg->node_npes;
  int nnodes = cfg->nnodes;
  int chunk_tokens = cfg->chunk_tokens;
  if (chunk_tokens <= 0) chunk_tokens = num_tokens;

  const void *mid_buf = cfg->mid_buf;
  uint64_t *mid_flags = cfg->mid_flags;
  void *local_buf = cfg->local_buf;
  const int *src_forward_local = cfg->src_forward_local;
  const int *src_local_of_forward = cfg->src_local_of_forward;
  const int *dst_forward_local = cfg->dst_forward_local;
  const int *src_forward_local_of_dst_forward = cfg->src_forward_local_of_dst_forward;
  if (!input_tokens || !output_tokens || !dst_index || !local_buf || !src_forward_local ||
      !src_local_of_forward || !dst_forward_local || !src_forward_local_of_dst_forward) {
    return 1;
  }

  int threads = 256;
  int blocks_gather = (num_tokens + (threads / 32) - 1) / (threads / 32);
  if (blocks_gather > 4096) blocks_gather = 4096;
  gather_balance_kernel<<<blocks_gather, threads>>>(input_tokens, local_buf, dst_index, num_tokens,
                                                    hidden_size, bytes_per_elem, node_npes, nnodes,
                                                    src_forward_local);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  nvshmem_quiet();
  nvshmem_barrier_all();

  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  dispatch_balance_kernel<<<blocks, threads>>>(input_tokens, output_tokens, dst_index, local_buf,
                                               num_tokens, hidden_size, bytes_per_elem, node_npes,
                                               nnodes, mid_buf, mid_flags, chunk_tokens,
                                               src_local_of_forward, dst_forward_local,
                                               src_forward_local_of_dst_forward);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  nvshmem_quiet();
  return 0;
}
