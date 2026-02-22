#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stddef.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

static __device__ __forceinline__ void dispatch_stage1(
    const void *input_tokens, void *output_tokens, const IntranodeIndex *intranode_index,
    int num_tokens, int hidden_size, int bytes_per_elem, int node_npes, int nnodes,
    const void *mid_buf, uint64_t *mid_flags, int chunk_tokens_local, int num_chunks, int npes,
    int mype, int src_node, int src_local, size_t token_bytes,
    cg::thread_block_tile<128> tile) {
  for (int chunk_id = blockIdx.x; chunk_id < num_chunks; chunk_id += gridDim.x) {
    int t_begin = chunk_id * chunk_tokens_local;
    int t_end = t_begin + chunk_tokens_local;
    if (t_end > num_tokens) t_end = num_tokens;
    for (int t = t_begin + tile.thread_rank(); t < t_end; t += tile.size()) {
      size_t src_offset = (size_t)t * token_bytes;
      size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
      for (int j = 0; j < npes; ++j) {
        const IntranodeIndex entry = intranode_index[g * (size_t)npes + (size_t)j];
        int idx = entry.index;
        if (idx < 0) continue;
        if (entry.route == 0 || entry.route == 1) {
          size_t dst_offset = (size_t)idx * token_bytes;
          nvshmem_putmem((char *)output_tokens + dst_offset,
                         (const char *)input_tokens + src_offset, token_bytes, j);
        }
      }
      if (nnodes > 1 && mid_buf && mid_flags) {
        for (int dst_node = 0; dst_node < nnodes; ++dst_node) {
          if (dst_node == src_node) continue;
          bool need_send = false;
          for (int lr = 0; lr < node_npes; ++lr) {
            int dst_rank = dst_node * node_npes + lr;
            const IntranodeIndex entry =
                intranode_index[g * (size_t)npes + (size_t)dst_rank];
            if (entry.index >= 0 && entry.route == 2) {
              need_send = true;
              break;
            }
          }
          if (!need_send) continue;
          int node_idx = node_block_index(dst_node, src_node);
          size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
          size_t mid_offset = mid_base + (size_t)t * token_bytes;
          nvshmem_putmem((char *)mid_buf + mid_offset,
                         (const char *)input_tokens + src_offset, token_bytes,
                         dst_node * node_npes + src_local);
        }
      }
    }
    tile.sync();
    nvshmem_quiet();
    if (nnodes > 1 && mid_buf && mid_flags && tile.thread_rank() == 0) {
      for (int dst_node = 0; dst_node < nnodes; ++dst_node) {
        if (dst_node == src_node) continue;
        int node_idx = node_block_index(dst_node, src_node);
        uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_chunks + chunk_id;
        nvshmemx_signal_op(flag_ptr, 1ull, NVSHMEM_SIGNAL_SET,
                           dst_node * node_npes + src_local);
      }
    }
  }
}

static __device__ __forceinline__ void dispatch_stage2(
    void *output_tokens, const IntranodeIndex *intranode_index, int num_tokens, int hidden_size,
    int bytes_per_elem, int node_npes, int nnodes, const void *mid_buf, uint64_t *mid_flags,
    int chunk_tokens_local, int num_chunks, int npes, int node_id, int local_rank,
    size_t token_bytes, cg::thread_block_tile<128> tile) {
  if (nnodes <= 1 || !mid_buf || !mid_flags) return;
  for (int chunk_id = blockIdx.x; chunk_id < num_chunks; chunk_id += gridDim.x) {
    int t_begin = chunk_id * chunk_tokens_local;
    int t_end = t_begin + chunk_tokens_local;
    if (t_end > num_tokens) t_end = num_tokens;
    for (int remote_node = 0; remote_node < nnodes; ++remote_node) {
      if (remote_node == node_id) continue;
      int node_idx = node_block_index(node_id, remote_node);
      uint64_t *flag_ptr = mid_flags + (size_t)node_idx * (size_t)num_chunks + chunk_id;
      nvshmem_signal_wait_until(flag_ptr, NVSHMEM_CMP_EQ, 1ull);
      for (int t = t_begin + tile.thread_rank(); t < t_end; t += tile.size()) {
        int src_rank = remote_node * node_npes + local_rank;
        size_t g = (size_t)src_rank * (size_t)num_tokens + (size_t)t;
        size_t mid_base = (size_t)node_idx * (size_t)num_tokens * token_bytes;
        const char *mid_ptr = (const char *)mid_buf + mid_base + (size_t)t * token_bytes;
        for (int lr = 0; lr < node_npes; ++lr) {
          if (lr == local_rank) continue;
          int dst_rank = node_id * node_npes + lr;
          const IntranodeIndex entry =
              intranode_index[g * (size_t)npes + (size_t)dst_rank];
          if (entry.index < 0 || entry.route != 2) continue;
          size_t dst_offset = (size_t)entry.index * token_bytes;
          nvshmem_putmem((char *)output_tokens + dst_offset, mid_ptr, token_bytes, dst_rank);
        }
      }
      tile.sync();
      nvshmem_quiet();
      if (tile.thread_rank() == 0) {
        flag_ptr[0] = 0;
      }
    }
  }
}

// 单 kernel 双阶段流水化：block 内一半线程做直发/写中转，另一半线程做中转转发
// chunk 以 token 数为粒度，并按 chunk_id % gridDim.x 映射到 block
__global__ void dispatch_kernel(const void *input_tokens, void *output_tokens,
                                const IntranodeIndex *intranode_index, int num_tokens,
                                int hidden_size, int bytes_per_elem, int node_npes, int nnodes,
                                const void *mid_buf, uint64_t *mid_flags, int chunk_tokens) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;

  int mype = nvshmem_my_pe();
  int src_node = mype / node_npes;
  int src_local = mype - src_node * node_npes;
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  int chunk_tokens_local = chunk_tokens > 0 ? chunk_tokens : num_tokens;
  int num_chunks = (num_tokens + chunk_tokens_local - 1) / chunk_tokens_local;
  if (num_chunks <= 0) return;
  auto tile = cg::tiled_partition<128>(cg::this_thread_block());
  int tile_id = tile.meta_group_rank();
  int tile_count = tile.meta_group_size();
  bool stage1 = tile_id < (tile_count / 2);
  if (stage1) {
    dispatch_stage1(input_tokens, output_tokens, intranode_index, num_tokens, hidden_size,
                    bytes_per_elem, node_npes, nnodes, mid_buf, mid_flags, chunk_tokens_local,
                    num_chunks, npes, mype, src_node, src_local, token_bytes, tile);
  }
  else {
    int node_id = mype / node_npes;
    int local_rank = mype - node_id * node_npes;
    dispatch_stage2(output_tokens, intranode_index, num_tokens, hidden_size, bytes_per_elem,
                    node_npes, nnodes, mid_buf, mid_flags, chunk_tokens_local, num_chunks, npes,
                    node_id, local_rank, token_bytes, tile);
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
  // chunk_tokens 控制每个 chunk 的 token 数
  int chunk_tokens = cfg->chunk_tokens;

  int threads = 256;
  int blocks = (num_tokens + threads - 1) / threads;
  if (cfg->blocks_per_kernel > 0) blocks = cfg->blocks_per_kernel;
  dispatch_kernel<<<blocks, threads>>>(input_tokens, output_tokens, intranode_index, num_tokens,
                                       hidden_size, bytes_per_elem, node_npes, nnodes, mid_buf,
                                       mid_flags, chunk_tokens);
  // 确保本 rank 发起的 put 已提交
  nvshmem_quiet();
  return 0;
}
