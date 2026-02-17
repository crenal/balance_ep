#pragma once

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
  // 预处理阶段的临时缓冲区（由调用方分配/释放）
  int *counts;
  int *offsets;
  int *local_counts;
  int *barrier_counter;
  int *barrier_sense;
};

int pre_process(const bool *routing_map, int *intranode_index, const DispatchConfig *cfg);
int dispatch_tokens(const void *input_tokens, void *output_tokens, const int *intranode_index,
                    const DispatchConfig *cfg);
