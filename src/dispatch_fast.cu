#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>

__device__ __forceinline__ void grid_barrier_fast(int *counter, int *sense, int blocks) {
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

__global__ void preprocess_fast_kernel(const bool *routing_map, int *counts, int *offsets,
                                       int *dst_index, int *local_counts, int *round_num,
                                       int num_tokens, int expert_num, int blocks,
                                       int *barrier_counter, int *barrier_sense) {
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

  grid_barrier_fast(barrier_counter, barrier_sense, blocks);

  for (int j = tid; j < npes; j += stride) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  grid_barrier_fast(barrier_counter, barrier_sense, blocks);

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
        dst_index[(size_t)g * (size_t)npes + (size_t)j] =
            offsets[src * npes + j] + local;
      } else {
        dst_index[(size_t)g * (size_t)npes + (size_t)j] = -1;
      }
    }
  }

  grid_barrier_fast(barrier_counter, barrier_sense, blocks);

  if (round_num && blockIdx.x == 0 && threadIdx.x == 0) {
    extern __shared__ int shared[];
    int *row_used = shared;
    int *col_used = shared + npes;
    int mype = nvshmem_my_pe();
    for (int i = 0; i < npes * npes; ++i) {
      local_counts[i] = 0;
    }
    for (int r = 0; r < npes; ++r) {
      round_num[r] = -1;
    }
    for (int r = 0; r < npes; ++r) {
      for (int i = 0; i < npes; ++i) {
        row_used[i] = 0;
        col_used[i] = 0;
      }
      for (int pick = 0; pick < npes; ++pick) {
        int best_i = -1;
        int best_j = -1;
        int best_v = -1;
        for (int i = 0; i < npes; ++i) {
          if (row_used[i]) continue;
          for (int j = 0; j < npes; ++j) {
            if (col_used[j]) continue;
            int idx = i * npes + j;
            if (local_counts[idx]) continue;
            int v = counts[idx];
            if (best_i < 0 || v > best_v) {
              best_i = i;
              best_j = j;
              best_v = v;
            }
          }
        }
        if (best_i < 0) break;
        row_used[best_i] = 1;
        col_used[best_j] = 1;
        local_counts[best_i * npes + best_j] = 1;
        if (best_i == mype) {
          round_num[r] = best_j;
        }
      }
    }
  }
}

int pre_process_fast(const bool *routing_map, int *dst_index, int *round_num,
                     const DispatchConfig *cfg) {
  if (!cfg) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int expert_num = cfg->expert_num;

  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;

  int *counts = cfg->counts;
  int *offsets = cfg->offsets;
  int *local_counts = cfg->local_counts;
  int *barrier_counter = cfg->barrier_counter;
  int *barrier_sense = cfg->barrier_sense;
  if (!counts || !offsets || !local_counts || !barrier_counter || !barrier_sense) {
    return 1;
  }

  cudaMemset(counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(local_counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(barrier_counter, 0, sizeof(int));
  cudaMemset(barrier_sense, 0, sizeof(int));

  int threads = 256;
  int global_tokens = num_tokens * npes;
  int blocks = (global_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;

  size_t shared_bytes = (size_t)npes * 2 * sizeof(int);
  preprocess_fast_kernel<<<blocks, threads, shared_bytes>>>(
      routing_map, counts, offsets, dst_index, local_counts, round_num, num_tokens, expert_num,
      blocks, barrier_counter, barrier_sense);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  return 0;
}

__global__ void dispatch_fast_kernel(const void *input_tokens, void *output_tokens,
                                     const int *dst_index, const int *round_num, int num_tokens,
                                     int hidden_size, int bytes_per_elem, int chunk_tokens) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int chunk_id = blockIdx.x;
  int num_chunks = (num_tokens + chunk_tokens - 1) / chunk_tokens;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  for (int cid = chunk_id; cid < num_chunks; cid += gridDim.x) {
    int t_begin = cid * chunk_tokens;
    int t_end = t_begin + chunk_tokens;
    if (t_end > num_tokens) t_end = num_tokens;
    for (int r = 0; r < npes; ++r) {
      int dst = round_num ? round_num[r] : r;
      if (dst < 0 || dst >= npes) continue;
      for (int t = t_begin + threadIdx.x; t < t_end; t += blockDim.x) {
        size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
        int idx = dst_index[g * (size_t)npes + (size_t)dst];
        if (idx < 0) continue;
        size_t src_offset = (size_t)t * token_bytes;
        size_t dst_offset = (size_t)idx * token_bytes;
        nvshmem_putmem((char *)output_tokens + dst_offset,
                       (const char *)input_tokens + src_offset, token_bytes, dst);
      }
      __syncthreads();
    }
  }
}

int dispatch_tokens_fast(const void *input_tokens, void *output_tokens, const int *dst_index,
                         const int *round_num, const DispatchConfig *cfg) {
  if (!cfg) return 1;
  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;
  int chunk_tokens = cfg->chunk_tokens;
  if (chunk_tokens <= 0) chunk_tokens = num_tokens;

  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  dispatch_fast_kernel<<<blocks, threads>>>(input_tokens, output_tokens, dst_index, round_num,
                                            num_tokens, hidden_size, bytes_per_elem, chunk_tokens);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  nvshmem_quiet();
  return 0;
}
