#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <stddef.h>

// 全网格栅栏：基于计数器与翻转位，保证所有 block 完成前一阶段后再进入下一阶段
__device__ __forceinline__ void grid_barrier(int *counter, int *sense, int blocks) {
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

// 单次 kernel 分三阶段完成计数、前缀与索引构建
// routing_map 维度为 [global_token][expert]，global_token = src_rank * num_tokens + local_token
__global__ void preprocess_kernel(const bool *routing_map, int *counts, int *offsets,
                                  int *intranode_index, int *local_counts, int num_tokens,
                                  int expert_num, int blocks, int *barrier_counter,
                                  int *barrier_sense) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 阶段 1：统计每个源 rank 的 token 在目标 rank 上的命中数量
  int global_tokens = num_tokens * npes;
  for (int g = tid; g < global_tokens; g += stride) {
    int src = g / num_tokens;
    size_t base = (size_t)g * (size_t)expert_num;
    int experts_per_rank = (expert_num + npes - 1) / npes;
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

  grid_barrier(barrier_counter, barrier_sense, blocks);

  // 阶段 2：对每个目标 rank 做源 rank 维度的前缀和，得到连续布局起点
  for (int j = tid; j < npes; j += stride) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  grid_barrier(barrier_counter, barrier_sense, blocks);

  // 阶段 3：为本 rank 的每个本地 token 计算其写入目标 rank 的最终位置
  int mype = nvshmem_my_pe();
  int experts_per_rank = (expert_num + npes - 1) / npes;
  for (int t = tid; t < num_tokens; t += stride) {
    size_t global_t = (size_t)mype * (size_t)num_tokens + (size_t)t;
    for (int j = 0; j < npes; ++j) {
      bool has = false;
      int start = j * experts_per_rank;
      int end = start + experts_per_rank;
      if (start < expert_num) {
        if (end > expert_num) end = expert_num;
        for (int e = start; e < end; ++e) {
          if (routing_map[global_t * (size_t)expert_num + (size_t)e]) {
            has = true;
            break;
          }
        }
      }
      if (has) {
        int local = atomicAdd(&local_counts[j], 1);
        // intranode_index[t][j] 为目标 rank j 的连续写入位置
        intranode_index[t * npes + j] = offsets[mype * npes + j] + local;
      } else {
        intranode_index[t * npes + j] = -1;
      }
    }
  }
}

// 按 intranode_index 将本地 token 发送到目标 rank，输出缓冲为连续紧凑布局
__global__ void dispatch_kernel(const void *input_tokens, void *output_tokens,
                                const int *intranode_index, int num_tokens, int hidden_size,
                                int bytes_per_elem) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int t = tid; t < num_tokens; t += stride) {
    // token_bytes 是每个 token 在连续缓冲区中的占用字节数
    size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
    // 本地输入 token 在 input_tokens 中的起始偏移
    size_t src_offset = (size_t)t * token_bytes;
    for (int j = 0; j < npes; ++j) {
      int idx = intranode_index[t * npes + j];
      if (idx < 0) continue;
      // output buffer 为紧凑连续布局，idx 为全局位置
      size_t dst_offset = (size_t)idx * token_bytes;
      // 在设备端发起 NVSHMEM put，将本地 token 写入目标 rank 的输出缓冲区
      nvshmem_putmem((char *)output_tokens + dst_offset, (const char *)input_tokens + src_offset,
                     token_bytes, j);
    }
  }
}

int pre_process(const bool *routing_map, int *intranode_index, const DispatchConfig *cfg) {
  if (!cfg) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int expert_num = cfg->expert_num;

  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;

  // 预处理阶段使用的临时缓冲区由调用方在 cfg 中提供
  int *counts = cfg->counts;
  int *offsets = cfg->offsets;
  int *local_counts = cfg->local_counts;
  int *barrier_counter = cfg->barrier_counter;
  int *barrier_sense = cfg->barrier_sense;
  if (!counts || !offsets || !local_counts || !barrier_counter || !barrier_sense) {
    return 1;
  }
  // 清零计数与栅栏状态，保证每次 pre_process 结果可复用
  cudaMemset(counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(local_counts, 0, (size_t)npes * sizeof(int));
  cudaMemset(barrier_counter, 0, sizeof(int));
  cudaMemset(barrier_sense, 0, sizeof(int));

  int threads = 256;
  int global_tokens = num_tokens * npes;
  int blocks = (global_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  preprocess_kernel<<<blocks, threads>>>(routing_map, counts, offsets, intranode_index,
                                         local_counts, num_tokens, expert_num, blocks,
                                         barrier_counter, barrier_sense);
  return 0;
}

int dispatch_tokens(const void *input_tokens, void *output_tokens, const int *intranode_index,
                    const DispatchConfig *cfg) {
  if (!cfg) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;

  // 使用 intranode_index 决定每个 token 需要发送到哪些 rank
  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  dispatch_kernel<<<blocks, threads>>>(input_tokens, output_tokens, intranode_index, num_tokens,
                                       hidden_size, bytes_per_elem);
  // 确保所有 put 在本 rank 完成提交
  nvshmem_quiet();
  return 0;
}
