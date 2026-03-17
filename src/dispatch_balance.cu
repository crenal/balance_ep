#include "dispatch.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stddef.h>

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
                                        const int *src_forward_local,
                                        const int *src_local_of_forward,
                                        const int *dst_forward_local,
                                        const int *src_forward_local_of_dst_forward,
                                        int *barrier_counter, int *barrier_sense) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int node_id = mype / node_npes;
  int local_rank = mype - node_id * node_npes;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  int chunk_tokens_local = chunk_tokens > 0 ? chunk_tokens : num_tokens;
  int num_chunks = (num_tokens + chunk_tokens_local - 1) / chunk_tokens_local;
  if (num_chunks <= 0) return;

  if (nnodes > 1) {
    int src_local = local_rank;
    int fwd_local = src_forward_local[node_id * node_npes + src_local];
    int fwd_rank = node_id * node_npes + fwd_local;

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

    grid_barrier_balance(barrier_counter, barrier_sense, gridDim.x);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      nvshmem_quiet();
      nvshmem_barrier_all();
    }
    grid_barrier_balance(barrier_counter, barrier_sense, gridDim.x);
  }

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

__device__ __forceinline__ float max2f(float a, float b) { return a > b ? a : b; }

__global__ void compute_src_forward_kernel(const int *counts, const int *node_counts,
                                           int *src_forward_local, int *src_local_of_forward,
                                           int node_npes, int nnodes) {
  if (threadIdx.x != 0) return;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int rpn = node_npes;
  if (rpn < 1 || rpn > 4) return;
  int node = (int)blockIdx.x;
  if (node < 0 || node >= nnodes) return;
  if (node * rpn >= npes) return;

  int weights[4] = {0, 0, 0, 0};
  int intra_send_base[4] = {0, 0, 0, 0};
  int intra_recv_base[4] = {0, 0, 0, 0};

  int base_rank = node * rpn;
  for (int s = 0; s < rpn; ++s) {
    int src_rank = base_rank + s;
    int off = 0;
    for (int b = 0; b < nnodes; ++b) {
      if (b == node) continue;
      off += node_counts[src_rank * nnodes + b];
    }
    weights[s] = off;

    int send_sum = 0;
    for (int lr = 0; lr < rpn; ++lr) {
      if (lr == s) continue;
      int d = base_rank + lr;
      if (d < npes) send_sum += counts[src_rank * npes + d];
    }
    intra_send_base[s] = send_sum;
  }

  for (int f = 0; f < rpn; ++f) {
    int rank = base_rank + f;
    int recv_sum = 0;
    for (int s = 0; s < npes; ++s) {
      if (s == rank) continue;
      recv_sum += counts[s * npes + rank];
    }
    intra_recv_base[f] = recv_sum;
  }

  float inv_intra = 1.0f / 200.0f;
  float inv_inter = 1.0f / 50.0f;

  int best_perm[4] = {0, 1, 2, 3};
  float best_score = -1.0f;

  int perm[4];
  int used[4];
  for (int i = 0; i < 4; ++i) used[i] = 0;

  for (int a = 0; a < rpn; ++a) {
    perm[0] = a;
    used[a] = 1;
    if (rpn == 1) {
      float inter_send_f[4] = {0, 0, 0, 0};
      float intra_recv_f[4] = {0, 0, 0, 0};
      float intra_send_s[4] = {0, 0, 0, 0};
      for (int i = 0; i < rpn; ++i) {
        inter_send_f[i] = 0.0f;
        intra_recv_f[i] = (float)intra_recv_base[i];
        intra_send_s[i] = (float)intra_send_base[i];
      }
      for (int s = 0; s < rpn; ++s) {
        int f = perm[s];
        int w = weights[s];
        inter_send_f[f] += (float)w;
        if (f != s) {
          intra_recv_f[f] += (float)w;
          intra_send_s[s] += (float)w;
        }
      }
      float score = 0.0f;
      for (int f = 0; f < rpn; ++f) {
        score = max2f(score, inter_send_f[f] * inv_inter);
        score = max2f(score, intra_recv_f[f] * inv_intra);
      }
      for (int s = 0; s < rpn; ++s) {
        score = max2f(score, intra_send_s[s] * inv_intra);
      }
      if (best_score < 0.0f || score < best_score) {
        best_score = score;
        for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
      }
      used[a] = 0;
      continue;
    }
    for (int b = 0; b < rpn; ++b) {
      if (used[b]) continue;
      perm[1] = b;
      used[b] = 1;
      if (rpn == 2) {
        float inter_send_f[4] = {0, 0, 0, 0};
        float intra_recv_f[4] = {0, 0, 0, 0};
        float intra_send_s[4] = {0, 0, 0, 0};
        for (int i = 0; i < rpn; ++i) {
          inter_send_f[i] = 0.0f;
          intra_recv_f[i] = (float)intra_recv_base[i];
          intra_send_s[i] = (float)intra_send_base[i];
        }
        for (int s = 0; s < rpn; ++s) {
          int f = perm[s];
          int w = weights[s];
          inter_send_f[f] += (float)w;
          if (f != s) {
            intra_recv_f[f] += (float)w;
            intra_send_s[s] += (float)w;
          }
        }
        float score = 0.0f;
        for (int f = 0; f < rpn; ++f) {
          score = max2f(score, inter_send_f[f] * inv_inter);
          score = max2f(score, intra_recv_f[f] * inv_intra);
        }
        for (int s = 0; s < rpn; ++s) {
          score = max2f(score, intra_send_s[s] * inv_intra);
        }
        if (best_score < 0.0f || score < best_score) {
          best_score = score;
          for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
        }
        used[b] = 0;
        continue;
      }
      for (int c = 0; c < rpn; ++c) {
        if (used[c]) continue;
        perm[2] = c;
        used[c] = 1;
        if (rpn == 3) {
          float inter_send_f[4] = {0, 0, 0, 0};
          float intra_recv_f[4] = {0, 0, 0, 0};
          float intra_send_s[4] = {0, 0, 0, 0};
          for (int i = 0; i < rpn; ++i) {
            inter_send_f[i] = 0.0f;
            intra_recv_f[i] = (float)intra_recv_base[i];
            intra_send_s[i] = (float)intra_send_base[i];
          }
          for (int s = 0; s < rpn; ++s) {
            int f = perm[s];
            int w = weights[s];
            inter_send_f[f] += (float)w;
            if (f != s) {
              intra_recv_f[f] += (float)w;
              intra_send_s[s] += (float)w;
            }
          }
          float score = 0.0f;
          for (int f = 0; f < rpn; ++f) {
            score = max2f(score, inter_send_f[f] * inv_inter);
            score = max2f(score, intra_recv_f[f] * inv_intra);
          }
          for (int s = 0; s < rpn; ++s) {
            score = max2f(score, intra_send_s[s] * inv_intra);
          }
          if (best_score < 0.0f || score < best_score) {
            best_score = score;
            for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
          }
          used[c] = 0;
          continue;
        }
        for (int d = 0; d < rpn; ++d) {
          if (used[d]) continue;
          perm[3] = d;

          float inter_send_f[4] = {0, 0, 0, 0};
          float intra_recv_f[4] = {0, 0, 0, 0};
          float intra_send_s[4] = {0, 0, 0, 0};
          for (int i = 0; i < rpn; ++i) {
            inter_send_f[i] = 0.0f;
            intra_recv_f[i] = (float)intra_recv_base[i];
            intra_send_s[i] = (float)intra_send_base[i];
          }
          for (int s = 0; s < rpn; ++s) {
            int f = perm[s];
            int w = weights[s];
            inter_send_f[f] += (float)w;
            if (f != s) {
              intra_recv_f[f] += (float)w;
              intra_send_s[s] += (float)w;
            }
          }
          float score = 0.0f;
          for (int f = 0; f < rpn; ++f) {
            score = max2f(score, inter_send_f[f] * inv_inter);
            score = max2f(score, intra_recv_f[f] * inv_intra);
          }
          for (int s = 0; s < rpn; ++s) {
            score = max2f(score, intra_send_s[s] * inv_intra);
          }
          if (best_score < 0.0f || score < best_score) {
            best_score = score;
            for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
          }
        }
        used[c] = 0;
      }
      used[b] = 0;
    }
    used[a] = 0;
  }

  for (int i = 0; i < rpn; ++i) {
    src_forward_local[base_rank + i] = best_perm[i];
  }
  for (int i = 0; i < rpn; ++i) {
    int f = best_perm[i];
    src_local_of_forward[base_rank + f] = i;
  }
}

__global__ void compute_dst_forward_kernel(const int *counts, const int *node_counts,
                                           const int *src_local_of_forward, int *dst_forward_local,
                                           int *src_forward_local_of_dst_forward, int node_npes,
                                           int nnodes) {
  if (threadIdx.x != 0) return;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int rpn = node_npes;
  if (rpn < 1 || rpn > 4) return;
  int b = (int)blockIdx.x;
  if (b < 0 || b >= nnodes) return;
  if (b * rpn >= npes) return;

  float inv_intra = 1.0f / 200.0f;
  float inv_inter = 1.0f / 50.0f;

  int base_rank = b * rpn;
  float intra_send[4] = {0, 0, 0, 0};
  float inter_recv[4] = {0, 0, 0, 0};
  for (int j = 0; j < rpn; ++j) {
    int dst_rank = base_rank + j;
    int send_sum = 0;
    for (int lr = 0; lr < rpn; ++lr) {
      if (lr == j) continue;
      int d = base_rank + lr;
      if (d < npes) send_sum += counts[dst_rank * npes + d];
    }
    intra_send[j] = (float)send_sum;
    inter_recv[j] = 0.0f;
  }

  for (int f = 0; f < rpn; ++f) {
    dst_forward_local[(b * nnodes + b) * rpn + f] = f;
    src_forward_local_of_dst_forward[(b * nnodes + b) * rpn + f] = f;
  }

  for (int a = 0; a < nnodes; ++a) {
    if (a == b) continue;
    if (a * rpn >= npes) continue;

    int w_f[4] = {0, 0, 0, 0};
    int src_rank_f[4] = {0, 0, 0, 0};
    int dist_f[4] = {0, 0, 0, 0};
    for (int f = 0; f < rpn; ++f) {
      int s = src_local_of_forward[a * rpn + f];
      int src_rank = a * rpn + s;
      src_rank_f[f] = src_rank;
      w_f[f] = node_counts[src_rank * nnodes + b];
      int dist = 0;
      for (int lr = 0; lr < rpn; ++lr) {
        int d = base_rank + lr;
        if (d < npes) dist += counts[src_rank * npes + d];
      }
      dist_f[f] = dist;
    }

    int best_perm[4] = {0, 1, 2, 3};
    float best_score = -1.0f;

    int perm[4];
    int used[4];
    for (int i = 0; i < 4; ++i) used[i] = 0;

    for (int p0 = 0; p0 < rpn; ++p0) {
      perm[0] = p0;
      used[p0] = 1;
      if (rpn == 1) {
        float tmp_inter[4] = {inter_recv[0], 0, 0, 0};
        float tmp_intra[4] = {intra_send[0], 0, 0, 0};
        int j = perm[0];
        int src_rank = src_rank_f[0];
        int dst_rank = base_rank + j;
        int self_cnt = counts[src_rank * npes + dst_rank];
        int add_intra = dist_f[0] - self_cnt;
        if (add_intra < 0) add_intra = 0;
        tmp_inter[j] += (float)w_f[0];
        tmp_intra[j] += (float)add_intra;
        float score = max2f(tmp_inter[0] * inv_inter, tmp_intra[0] * inv_intra);
        if (best_score < 0.0f || score < best_score) {
          best_score = score;
          for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
        }
        used[p0] = 0;
        continue;
      }
      for (int p1 = 0; p1 < rpn; ++p1) {
        if (used[p1]) continue;
        perm[1] = p1;
        used[p1] = 1;
        if (rpn == 2) {
          float tmp_inter[4] = {inter_recv[0], inter_recv[1], 0, 0};
          float tmp_intra[4] = {intra_send[0], intra_send[1], 0, 0};
          for (int f = 0; f < rpn; ++f) {
            int j = perm[f];
            int src_rank = src_rank_f[f];
            int dst_rank = base_rank + j;
            int self_cnt = counts[src_rank * npes + dst_rank];
            int add_intra = dist_f[f] - self_cnt;
            if (add_intra < 0) add_intra = 0;
            tmp_inter[j] += (float)w_f[f];
            tmp_intra[j] += (float)add_intra;
          }
          float score = 0.0f;
          for (int j = 0; j < rpn; ++j) {
            score = max2f(score, max2f(tmp_inter[j] * inv_inter, tmp_intra[j] * inv_intra));
          }
          if (best_score < 0.0f || score < best_score) {
            best_score = score;
            for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
          }
          used[p1] = 0;
          continue;
        }
        for (int p2 = 0; p2 < rpn; ++p2) {
          if (used[p2]) continue;
          perm[2] = p2;
          used[p2] = 1;
          if (rpn == 3) {
            float tmp_inter[4] = {inter_recv[0], inter_recv[1], inter_recv[2], 0};
            float tmp_intra[4] = {intra_send[0], intra_send[1], intra_send[2], 0};
            for (int f = 0; f < rpn; ++f) {
              int j = perm[f];
              int src_rank = src_rank_f[f];
              int dst_rank = base_rank + j;
              int self_cnt = counts[src_rank * npes + dst_rank];
              int add_intra = dist_f[f] - self_cnt;
              if (add_intra < 0) add_intra = 0;
              tmp_inter[j] += (float)w_f[f];
              tmp_intra[j] += (float)add_intra;
            }
            float score = 0.0f;
            for (int j = 0; j < rpn; ++j) {
              score = max2f(score, max2f(tmp_inter[j] * inv_inter, tmp_intra[j] * inv_intra));
            }
            if (best_score < 0.0f || score < best_score) {
              best_score = score;
              for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
            }
            used[p2] = 0;
            continue;
          }
          for (int p3 = 0; p3 < rpn; ++p3) {
            if (used[p3]) continue;
            perm[3] = p3;
            float tmp_inter[4] = {inter_recv[0], inter_recv[1], inter_recv[2], inter_recv[3]};
            float tmp_intra[4] = {intra_send[0], intra_send[1], intra_send[2], intra_send[3]};
            for (int f = 0; f < rpn; ++f) {
              int j = perm[f];
              int src_rank = src_rank_f[f];
              int dst_rank = base_rank + j;
              int self_cnt = counts[src_rank * npes + dst_rank];
              int add_intra = dist_f[f] - self_cnt;
              if (add_intra < 0) add_intra = 0;
              tmp_inter[j] += (float)w_f[f];
              tmp_intra[j] += (float)add_intra;
            }
            float score = 0.0f;
            for (int j = 0; j < rpn; ++j) {
              score = max2f(score, max2f(tmp_inter[j] * inv_inter, tmp_intra[j] * inv_intra));
            }
            if (best_score < 0.0f || score < best_score) {
              best_score = score;
              for (int i = 0; i < 4; ++i) best_perm[i] = perm[i];
            }
          }
          used[p2] = 0;
        }
        used[p1] = 0;
      }
      used[p0] = 0;
    }

    for (int f = 0; f < rpn; ++f) {
      int j = best_perm[f];
      int src_rank = src_rank_f[f];
      int dst_rank = base_rank + j;
      int self_cnt = counts[src_rank * npes + dst_rank];
      int add_intra = dist_f[f] - self_cnt;
      if (add_intra < 0) add_intra = 0;
      dst_forward_local[(a * nnodes + b) * rpn + f] = j;
      src_forward_local_of_dst_forward[(a * nnodes + b) * rpn + j] = f;
      inter_recv[j] += (float)w_f[f];
      intra_send[j] += (float)add_intra;
    }
  }
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

  if (node_npes < 1 || node_npes > 4) return 1;

  compute_src_forward_kernel<<<nnodes, 1>>>(counts, node_counts, src_forward_local,
                                            src_local_of_forward, node_npes, nnodes);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 1;

  compute_dst_forward_kernel<<<nnodes, 1>>>(counts, node_counts, src_local_of_forward,
                                            dst_forward_local, src_forward_local_of_dst_forward,
                                            node_npes, nnodes);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 1;

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
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
  int *barrier_counter = cfg->barrier_counter;
  int *barrier_sense = cfg->barrier_sense;
  const int *src_forward_local = cfg->src_forward_local;
  const int *src_local_of_forward = cfg->src_local_of_forward;
  const int *dst_forward_local = cfg->dst_forward_local;
  const int *src_forward_local_of_dst_forward = cfg->src_forward_local_of_dst_forward;
  if (!input_tokens || !output_tokens || !dst_index || !local_buf || !src_forward_local ||
      !src_local_of_forward || !dst_forward_local || !src_forward_local_of_dst_forward ||
      !barrier_counter || !barrier_sense) {
    return 1;
  }

  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;

  cudaMemset(barrier_counter, 0, sizeof(int));
  cudaMemset(barrier_sense, 0, sizeof(int));

  dispatch_balance_kernel<<<blocks, threads>>>(
      input_tokens, output_tokens, dst_index, local_buf, num_tokens, hidden_size, bytes_per_elem,
      node_npes, nnodes, mid_buf, mid_flags, chunk_tokens, src_forward_local, src_local_of_forward,
      dst_forward_local, src_forward_local_of_dst_forward, barrier_counter, barrier_sense);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  nvshmem_quiet();
  return 0;
}
