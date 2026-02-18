#pragma once

#include <stdint.h>

struct IntranodeIndex {
  // 在目标 rank 输出缓冲中的紧凑写入位置，<0 表示该目标不接收
  int index;
  // 目标 rank 编号
  int target_rank;
  // 路由类型：0 同节点直写，1 跨节点同号 local_rank 直写，2 跨节点异号 local_rank 转发
  int route;
};

struct DispatchConfig {
  // 每个 rank 的本地 token 数
  int num_tokens_per_rank;
  // expert 总数，按 rank 轮转映射到目标 rank
  int expert_num;
  // token 隐层维度
  int hidden_size;
  // 单元素字节数
  int bytes_per_elem;
  // kernel 启动的 block 数；<=0 表示按数据量自适应
  int blocks_per_kernel;
  // 每个节点上的 rank 数（local_rank 维度）
  int node_npes;
  // 节点总数
  int nnodes;
  // 跨节点转发用的对称缓冲区（每 rank 维护 nnodes-1 块）
  void *mid_buf;
  // mid_buf 的 flag 数组（每块对应 num_tokens_per_rank 个 flag）
  uint64_t *mid_flags;
  // 预处理阶段的计数矩阵 counts[src][dst]
  int *counts;
  // 预处理阶段的前缀矩阵 offsets[src][dst]
  int *offsets;
  // 预处理阶段的局部计数 local_counts[src][dst]
  int *local_counts;
  // 设备侧网格同步计数器
  int *barrier_counter;
  // 设备侧网格同步翻转位
  int *barrier_sense;
};

// 根据 routing_map 生成全局 intranode_index
int pre_process(const bool *routing_map, IntranodeIndex *intranode_index, const DispatchConfig *cfg);
// 按 intranode_index 执行跨 rank dispatch，必要时使用 mid_buf 转发
int dispatch_tokens(const void *input_tokens, void *output_tokens, const IntranodeIndex *intranode_index,
                    const DispatchConfig *cfg);
