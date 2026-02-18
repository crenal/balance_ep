#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stddef.h>

// 全网格栅栏：基于计数器与翻转位，保证所有 block 完成前一阶段后再进入下一阶段
// 该栅栏仅在单个 kernel 内使用，用于分阶段的并行前缀计算
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

// 计算当前节点在 mid_buf 中的块索引
// 每个 rank 有 (nnodes-1) 个块，按节点编号顺序压缩存放
__device__ __forceinline__ int node_block_index(int local_node, int remote_node) {
  return remote_node < local_node ? remote_node : remote_node - 1;
}

// 单次 kernel 分三阶段完成计数、前缀与索引构建
// routing_map 维度为 [global_token][expert]，global_token = src_rank * num_tokens + local_token
__global__ void preprocess_kernel(const bool *routing_map, int *counts, int *offsets,
                                  IntranodeIndex *intranode_index, int *local_counts,
                                  int num_tokens, int expert_num, int node_npes, int blocks,
                                  int *barrier_counter, int *barrier_sense) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 阶段 1：统计每个源 rank 的 token 在目标 rank 上的命中数量
  // counts[src][dst] = src 上有多少 token 需要发送到 dst
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
  // offsets[src][dst] 表示 dst 端为 src 预留的起始偏移
  for (int j = tid; j < npes; j += stride) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  grid_barrier(barrier_counter, barrier_sense, blocks);

  // 阶段 3：为每个 global_token 计算其发往各目标 rank 的写入位置与路由类型
  // intranode_index 以 global_token 为索引，保证跨节点转发时可被中继 rank 访问
  int experts_per_rank = (expert_num + npes - 1) / npes;
  for (int g = tid; g < global_tokens; g += stride) {
    int src = g / num_tokens;
    int src_node = src / node_npes;
    int src_local = src - src_node * node_npes;
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
      IntranodeIndex entry;
      entry.target_rank = j;
      if (has) {
        int local = atomicAdd(&local_counts[src * npes + j], 1);
        entry.index = offsets[src * npes + j] + local;
        int dst_node = j / node_npes;
        int dst_local = j - dst_node * node_npes;
        // route 由节点与 local_rank 关系决定
        if (src_node == dst_node) {
          entry.route = 0;
        } else if (src_local == dst_local) {
          entry.route = 1;
        } else {
          entry.route = 2;
        }
      } else {
        entry.index = -1;
        entry.route = -1;
      }
      intranode_index[(size_t)g * (size_t)npes + (size_t)j] = entry;
    }
  }
}

// 按 intranode_index 将本地 token 发送到目标 rank，输出缓冲为连续紧凑布局
// route=0/1：直接写目标 rank 输出缓冲
// route=2：写目标节点同号 local_rank 的 mid_buf，并置位 flag
__global__ void dispatch_kernel(const void *input_tokens, void *output_tokens,
                                const IntranodeIndex *intranode_index, int num_tokens,
                                int hidden_size, int bytes_per_elem, int node_npes, int nnodes,
                                const void *mid_buf, uint64_t *mid_flags) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;

  int mype = nvshmem_my_pe();
  int src_node = mype / node_npes;
  int src_local = mype - src_node * node_npes;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int t = tid; t < num_tokens; t += stride) {
    // 每个 token 在连续缓冲中的字节长度
    size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
    // 本地 token 在 input_tokens 中的偏移
    size_t src_offset = (size_t)t * token_bytes;
    // global_token 索引
    size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
    for (int j = 0; j < npes; ++j) {
      const IntranodeIndex entry = intranode_index[g * (size_t)npes + (size_t)j];
      int idx = entry.index;
      if (idx < 0) continue;
      // 目标输出在 dst 端的紧凑位置
      size_t dst_offset = (size_t)idx * token_bytes;
      if (entry.route == 0 || entry.route == 1) {
        // 同节点或跨节点同号 local_rank，直接写目标输出缓冲
        nvshmem_putmem((char *)output_tokens + dst_offset,
                       (const char *)input_tokens + src_offset, token_bytes, j);
      } else {
        int dst_node = j / node_npes;
        int dst_local = j - dst_node * node_npes;
        if (dst_local != src_local && nnodes > 1 && mid_buf && mid_flags) {
          // 跨节点异号 local_rank：写入目标节点同号 local_rank 的中转缓冲
          int node_idx = node_block_index(dst_node, src_node);
          size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
          size_t mid_offset = mid_base + (size_t)t * token_bytes;
          uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_tokens + t;
          nvshmem_putmem((char *)mid_buf + mid_offset, (const char *)input_tokens + src_offset,
                         token_bytes, dst_node * node_npes + src_local);
          nvshmem_fence();
          // 置位 flag 表示该 token 已到达中转缓冲
          nvshmemx_signal_op(flag_ptr, 1ull, NVSHMEM_SIGNAL_SET,
                             dst_node * node_npes + src_local);
        }
      }
    }
  }
}

// 在目标节点同号 local_rank 上执行的中继 kernel
// 读取 mid_buf 中收到的 token 并转发到本节点内其它 local_rank 的输出缓冲
__global__ void relay_kernel(void *output_tokens, const IntranodeIndex *intranode_index,
                             int num_tokens, int hidden_size, int bytes_per_elem, int node_npes,
                             int nnodes, const void *mid_buf, uint64_t *mid_flags) {
  int npes = nvshmem_n_pes();
  if (npes <= 0 || nnodes <= 1 || !mid_buf || !mid_flags) return;

  int mype = nvshmem_my_pe();
  int node_id = mype / node_npes;
  int local_rank = mype - node_id * node_npes;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int t = tid; t < num_tokens; t += stride) {
    for (int remote_node = 0; remote_node < nnodes; ++remote_node) {
      if (remote_node == node_id) continue;
      int node_idx = node_block_index(node_id, remote_node);
      // 该节点上同号 local_rank 作为转发源的 rank
      int src_rank = remote_node * node_npes + local_rank;
      size_t g = (size_t)src_rank * (size_t)num_tokens + (size_t)t;
      bool need_forward = false;
      // 判断本地节点是否存在需要转发的目标
      for (int lr = 0; lr < node_npes; ++lr) {
        if (lr == local_rank) continue;
        int dst_rank = node_id * node_npes + lr;
        const IntranodeIndex entry =
            intranode_index[g * (size_t)npes + (size_t)dst_rank];
        if (entry.index >= 0 && entry.route == 2) {
          need_forward = true;
          break;
        }
      }
      if (!need_forward) continue;
      uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_tokens + t;
      // 等待该 token 的中转数据到达
      nvshmem_signal_wait_until(flag_ptr, NVSHMEM_CMP_EQ, 1ull);
      size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
      const char *mid_ptr = (const char *)mid_buf + mid_base + (size_t)t * token_bytes;
      // 向本节点内其它 local_rank 转发
      for (int lr = 0; lr < node_npes; ++lr) {
        if (lr == local_rank) continue;
        int dst_rank = node_id * node_npes + lr;
        const IntranodeIndex entry =
            intranode_index[g * (size_t)npes + (size_t)dst_rank];
        if (entry.index < 0 || entry.route != 2) continue;
        size_t dst_offset = (size_t)entry.index * token_bytes;
        nvshmem_putmem((char *)output_tokens + dst_offset, mid_ptr, token_bytes, dst_rank);
      }
      // 重置 flag，便于下一次复用
      flag_ptr[0] = 0;
    }
  }
}
// host 侧入口：生成 intranode_index
int pre_process(const bool *routing_map, IntranodeIndex *intranode_index, const DispatchConfig *cfg) {
  if (!cfg) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int expert_num = cfg->expert_num;
  int node_npes = cfg->node_npes;

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
  cudaMemset(local_counts, 0, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMemset(barrier_counter, 0, sizeof(int));
  cudaMemset(barrier_sense, 0, sizeof(int));

  int threads = 256;
  int global_tokens = num_tokens * npes;
  int blocks = (global_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  preprocess_kernel<<<blocks, threads>>>(routing_map, counts, offsets, intranode_index,
                                         local_counts, num_tokens, expert_num, node_npes, blocks,
                                         barrier_counter, barrier_sense);
  return 0;
}

// host 侧入口：执行 dispatch 与可选的跨节点转发
int dispatch_tokens(const void *input_tokens, void *output_tokens,
                    const IntranodeIndex *intranode_index, const DispatchConfig *cfg) {
  if (!cfg) return 1;

  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;
  int node_npes = cfg->node_npes;
  int nnodes = cfg->nnodes;
  const void *mid_buf = cfg->mid_buf;
  uint64_t *mid_flags = cfg->mid_flags;

  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  dispatch_kernel<<<blocks, threads>>>(input_tokens, output_tokens, intranode_index, num_tokens,
                                       hidden_size, bytes_per_elem, node_npes, nnodes, mid_buf,
                                       mid_flags);
  // 确保本 rank 发起的 put 已提交
  nvshmem_quiet();
  // 跨节点转发：等待所有 rank 完成直发/写中转，再由中继 rank 处理转发
  if (nnodes > 1 && mid_buf && mid_flags) {
    nvshmem_barrier_all();
    relay_kernel<<<blocks, threads>>>(output_tokens, intranode_index, num_tokens, hidden_size,
                                      bytes_per_elem, node_npes, nnodes, mid_buf, mid_flags);
    nvshmem_quiet();
  }
  return 0;
}
