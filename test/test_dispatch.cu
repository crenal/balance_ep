#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 简单线性同余随机数，用于构造稳定可复现的 routing_map
static unsigned int next_rand(unsigned int *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

struct TestBuffers {
  // NVSHMEM 对称内存输入输出
  float *input_tokens;
  float *output_tokens;
  // 设备侧 routing_map 与 intranode_index
  bool *routing_map;
  IntranodeIndex *intranode_index;
  // 跨节点转发的中间缓冲及其 flag
  void *mid_buf;
  uint64_t *mid_flags;
  size_t mid_buf_bytes;
  size_t mid_flags_bytes;
  // 主机侧缓冲区，用于初始化与校验
  float *input_h;
  float *output_h;
  bool *map_h;
};

static int allocate_buffers(TestBuffers *buf, size_t input_bytes, size_t output_bytes,
                            size_t map_bytes, size_t intranode_elems,
                            int num_tokens_per_rank, int npes, int nnodes, int num_chunks) {
  // input/output 需要对称内存，保证跨 rank 可访问
  buf->input_tokens = (float *)nvshmem_malloc(input_bytes);
  buf->output_tokens = (float *)nvshmem_malloc(output_bytes);
  // routing_map 与 intranode_index 使用普通设备内存
  buf->routing_map = nullptr;
  cudaMalloc((void **)&buf->routing_map, map_bytes);
  buf->intranode_index = nullptr;
  cudaMalloc((void **)&buf->intranode_index, intranode_elems * sizeof(IntranodeIndex));
  // 每个 rank 分配 (nnodes-1) 块中转缓冲与 flag
  buf->mid_buf_bytes = (nnodes > 1) ? (size_t)(nnodes - 1) * input_bytes : 0;
  buf->mid_flags_bytes =
      (nnodes > 1) ? (size_t)(nnodes - 1) * (size_t)num_chunks * sizeof(uint64_t) : 0;
  buf->mid_buf = buf->mid_buf_bytes > 0 ? nvshmem_malloc(buf->mid_buf_bytes) : nullptr;
  buf->mid_flags =
      buf->mid_flags_bytes > 0 ? (uint64_t *)nvshmem_malloc(buf->mid_flags_bytes) : nullptr;
  // 主机侧临时缓冲
  buf->input_h = (float *)malloc(input_bytes);
  buf->output_h = (float *)malloc(output_bytes);
  buf->map_h = (bool *)malloc(map_bytes);
  if (!buf->input_tokens || !buf->output_tokens || !buf->routing_map || !buf->intranode_index ||
      (buf->mid_buf_bytes > 0 && !buf->mid_buf) || (buf->mid_flags_bytes > 0 && !buf->mid_flags) ||
      !buf->input_h || !buf->output_h || !buf->map_h) {
    return 1;
  }
  return 0;
}

static void free_buffers(TestBuffers *buf) {
  // 主机侧缓冲
  if (buf->input_h) free(buf->input_h);
  if (buf->output_h) free(buf->output_h);
  if (buf->map_h) free(buf->map_h);
  // NVSHMEM 对称内存
  if (buf->input_tokens) nvshmem_free(buf->input_tokens);
  if (buf->output_tokens) nvshmem_free(buf->output_tokens);
  if (buf->mid_buf) nvshmem_free(buf->mid_buf);
  if (buf->mid_flags) nvshmem_free(buf->mid_flags);
  // 普通设备内存
  if (buf->routing_map) cudaFree(buf->routing_map);
  if (buf->intranode_index) cudaFree(buf->intranode_index);
}

static void generate_map(bool *map_h, size_t global_tokens, int expert_num, int topk) {
  // 每行选择 topk 个 expert 置 1，保证所有 rank 的 map 一致
  unsigned int state = 1234u;
  int kmax = topk > expert_num ? expert_num : topk;
  for (size_t gt = 0; gt < global_tokens; ++gt) {
    size_t row = gt * (size_t)expert_num;
    for (int e = 0; e < expert_num; ++e) {
      map_h[row + (size_t)e] = false;
    }
    int filled = 0;
    while (filled < kmax) {
      int idx = (int)(next_rand(&state) % (unsigned int)expert_num);
      if (!map_h[row + (size_t)idx]) {
        map_h[row + (size_t)idx] = true;
        filled++;
      }
    }
  }
}

static void init_inputs(TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int mype,
                        int npes, int expert_num, int topk, size_t input_bytes,
                        size_t output_bytes, size_t map_bytes) {
  // 构造可验证的输入 token：值包含 src rank 与 token 索引信息
  for (int t = 0; t < num_tokens_per_rank; ++t) {
    for (int h = 0; h < hidden_size; ++h) {
      buf->input_h[t * hidden_size + h] = (float)(mype * 100000 + t * 1000 + h);
    }
  }
  // 复制到设备侧 input，并清零输出与中转缓冲
  cudaMemcpy(buf->input_tokens, buf->input_h, input_bytes, cudaMemcpyHostToDevice);
  cudaMemset(buf->output_tokens, 0, output_bytes);
  if (buf->mid_buf && buf->mid_buf_bytes > 0) {
    cudaMemset(buf->mid_buf, 0, buf->mid_buf_bytes);
  }
  if (buf->mid_flags && buf->mid_flags_bytes > 0) {
    cudaMemset(buf->mid_flags, 0, buf->mid_flags_bytes);
  }

  // 生成全局 routing_map 并拷贝到设备侧
  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  generate_map(buf->map_h, global_tokens, expert_num, topk);
  cudaMemcpy(buf->routing_map, buf->map_h, map_bytes, cudaMemcpyHostToDevice);
}

static int check_output(const TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int npes,
                        int expert_num, int mype) {
  // 将设备侧输出拷回主机进行一致性验证
  cudaMemcpy(buf->output_h, buf->output_tokens,
             (size_t)npes * (size_t)num_tokens_per_rank * (size_t)hidden_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  int total_rows = npes * num_tokens_per_rank;
  int *counts = (int *)malloc((size_t)npes * (size_t)npes * sizeof(int));
  int *offsets = (int *)malloc((size_t)npes * (size_t)npes * sizeof(int));
  if (!counts || !offsets) {
    if (counts) free(counts);
    if (offsets) free(offsets);
    return 1;
  }
  for (int i = 0; i < npes * npes; ++i) counts[i] = 0;

  // 在主机侧重新统计 counts 与 offsets，作为校验基准
  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  int experts_per_rank = (expert_num + npes - 1) / npes;
  for (size_t gt = 0; gt < global_tokens; ++gt) {
    int src = (int)(gt / (size_t)num_tokens_per_rank);
    size_t base = gt * (size_t)expert_num;
    for (int j = 0; j < npes; ++j) {
      bool has = false;
      int start = j * experts_per_rank;
      int end = start + experts_per_rank;
      if (start < expert_num) {
        if (end > expert_num) end = expert_num;
        for (int e = start; e < end; ++e) {
          if (buf->map_h[base + (size_t)e]) {
            has = true;
            break;
          }
        }
      }
      if (has) counts[src * npes + j] += 1;
    }
  }
  for (int j = 0; j < npes; ++j) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  // 验证每个源 rank 在本 rank 的紧凑块是否满足路由关系
  int total_count = 0;
  for (int src = 0; src < npes; ++src) total_count += counts[src * npes + mype];

  unsigned char *received = (unsigned char *)malloc((size_t)num_tokens_per_rank);
  if (!received) {
    free(counts);
    free(offsets);
    return 1;
  }

  for (int src = 0; src < npes; ++src) {
    int block_start = offsets[src * npes + mype];
    int block_len = counts[src * npes + mype];
    for (int i = 0; i < num_tokens_per_rank; ++i) received[i] = 0;
    for (int row = 0; row < block_len; ++row) {
      int pos = block_start + row;
      size_t base = (size_t)pos * (size_t)hidden_size;
      float v0 = buf->output_h[base];
      int t = (int)((v0 - (float)(src * 100000)) / 1000.0f + 0.5f);
      if (t < 0 || t >= num_tokens_per_rank) {
        free(counts);
        free(offsets);
        free(received);
        return 1;
      }
      if (received[t]) {
        free(counts);
        free(offsets);
        free(received);
        return 1;
      }
      received[t] = 1;
      // 校验 token 内容是否与预期一致
      for (int h = 0; h < hidden_size; ++h) {
        float expected_val = (float)(src * 100000 + t * 1000 + h);
        float got = buf->output_h[base + h];
        if (got != expected_val) {
          free(counts);
          free(offsets);
          free(received);
          return 1;
        }
      }
    }
    for (int t = 0; t < num_tokens_per_rank; ++t) {
      size_t global_t = (size_t)src * (size_t)num_tokens_per_rank + (size_t)t;
      size_t base_map = global_t * (size_t)expert_num;
      bool has = false;
      int start = mype * experts_per_rank;
      int end = start + experts_per_rank;
      if (start < expert_num) {
        if (end > expert_num) end = expert_num;
        for (int e = start; e < end; ++e) {
          if (buf->map_h[base_map + (size_t)e]) {
            has = true;
            break;
          }
        }
      }
      if ((has && !received[t]) || (!has && received[t])) {
        free(counts);
        free(offsets);
        free(received);
        return 1;
      }
    }
  }

  // 验证尾部未使用区域保持为 0
  for (int row = total_count; row < total_rows; ++row) {
    size_t base = (size_t)row * (size_t)hidden_size;
    for (int h = 0; h < hidden_size; ++h) {
      if (buf->output_h[base + h] != 0.0f) {
        free(counts);
        free(offsets);
        free(received);
        return 1;
      }
    }
  }

  free(received);

  free(counts);
  free(offsets);
  return 0;
}

// 校验所有中转 flag 已被清零，避免遗留状态
static int check_flags_cleared(const TestBuffers *buf) {
  if (!buf->mid_flags || buf->mid_flags_bytes == 0) return 0;
  uint64_t *flags_h = (uint64_t *)malloc(buf->mid_flags_bytes);
  if (!flags_h) return 1;
  cudaMemcpy(flags_h, buf->mid_flags, buf->mid_flags_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  size_t count = buf->mid_flags_bytes / sizeof(uint64_t);
  for (size_t i = 0; i < count; ++i) {
    if (flags_h[i] != 0ull) {
      free(flags_h);
      return 1;
    }
  }
  free(flags_h);
  return 0;
}

static int parse_positive_int(const char *text, int *out) {
  // 解析正整数参数，失败返回 0
  if (!text || !out) return 0;
  errno = 0;
  char *end = nullptr;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return 0;
  if (value <= 0 || value > INT_MAX) return 0;
  *out = (int)value;
  return 1;
}

static int read_env_int(const char *name, int *out) {
  // 从环境变量读取正整数，不存在则保持默认值
  const char *value = getenv(name);
  if (!value || value[0] == '\0') return 0;
  return parse_positive_int(value, out);
}

int main(int argc, char **argv) {
  // 解析命令行参数：可选覆盖 nnodes 与 ranks_per_node
  int override_nnodes = 0;
  int override_rpn = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--nnodes") == 0 && i + 1 < argc) {
      if (!parse_positive_int(argv[i + 1], &override_nnodes)) return 1;
      ++i;
      continue;
    }
    if (strcmp(argv[i], "--ranks_per_node") == 0 && i + 1 < argc) {
      if (!parse_positive_int(argv[i + 1], &override_rpn)) return 1;
      ++i;
      continue;
    }
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: %s [--nnodes N] [--ranks_per_node R]\n", argv[0]);
      return 0;
    }
    return 1;
  }

  // 初始化 NVSHMEM
  nvshmem_init();

  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
  int actual_node_npes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int actual_local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  int node_npes = actual_node_npes;
  int nnodes = (npes + node_npes - 1) / node_npes;
  // 根据用户覆盖配置推导节点拓扑
  if (override_nnodes > 0 && override_rpn > 0) {
    nnodes = override_nnodes;
    node_npes = override_rpn;
  } else if (override_nnodes > 0) {
    nnodes = override_nnodes;
    node_npes = (npes + nnodes - 1) / nnodes;
  } else if (override_rpn > 0) {
    node_npes = override_rpn;
    nnodes = (npes + node_npes - 1) / node_npes;
  }
  if (nnodes * node_npes != npes) {
    if (mype == 0) {
      printf("Invalid nnodes/ranks_per_node for npes=%d\n", npes);
    }
    nvshmem_finalize();
    return 1;
  }
  // 使用实际 local_rank 选择 GPU
  cudaSetDevice(actual_local_rank);

  // 支持通过环境变量覆盖测试规模参数
  int num_tokens_per_rank = 8192;
  int expert_num = 128;
  int hidden_size = 8192;
  int topk = 4;
  int blocks_per_kernel = 8;
  // chunk_tokens 控制每个 chunk 的 token 数，影响 flag 数量与流水化程度
  int chunk_tokens = 256;
  int bench_iters = 0;
  int bench_warmup = 5;
  int bench_only = 0;
  read_env_int("NUM_TOKENS_PER_RANK", &num_tokens_per_rank);
  read_env_int("EXPERT_NUM", &expert_num);
  read_env_int("HIDDEN_SIZE", &hidden_size);
  read_env_int("TOPK", &topk);
  read_env_int("BLOCKS_PER_KERNEL", &blocks_per_kernel);
  read_env_int("CHUNK_TOKENS", &chunk_tokens);
  read_env_int("BENCH_ITERS", &bench_iters);
  read_env_int("BENCH_WARMUP", &bench_warmup);
  read_env_int("BENCH_ONLY", &bench_only);
  if (topk > expert_num) topk = expert_num;
  const int bytes_per_elem = (int)sizeof(float);
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  if (chunk_tokens <= 0) chunk_tokens = num_tokens_per_rank;
  int num_chunks = (num_tokens_per_rank + chunk_tokens - 1) / chunk_tokens;
  size_t input_bytes = (size_t)num_tokens_per_rank * token_bytes;
  size_t output_bytes = (size_t)npes * (size_t)num_tokens_per_rank * token_bytes;
  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  size_t map_elems = global_tokens * (size_t)expert_num;
  size_t map_bytes = map_elems * sizeof(bool);

  // 分配并初始化所有缓冲
  TestBuffers buf = {};
  int status = allocate_buffers(&buf, input_bytes, output_bytes, map_bytes,
                                (size_t)global_tokens * (size_t)npes, num_tokens_per_rank, npes,
                                nnodes, num_chunks);
  if (status != 0) {
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  init_inputs(&buf, num_tokens_per_rank, hidden_size, mype, npes, expert_num, topk, input_bytes,
              output_bytes, map_bytes);

  // 配置 dispatch 参数与临时缓冲
  DispatchConfig cfg;
  cfg.num_tokens_per_rank = num_tokens_per_rank;
  cfg.expert_num = expert_num;
  cfg.hidden_size = hidden_size;
  cfg.bytes_per_elem = bytes_per_elem;
  cfg.blocks_per_kernel = blocks_per_kernel;
  cfg.chunk_tokens = chunk_tokens;
  cfg.node_npes = node_npes;
  cfg.nnodes = nnodes;
  cfg.mid_buf = buf.mid_buf;
  cfg.mid_flags = buf.mid_flags;
  cudaMalloc((void **)&cfg.counts, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.offsets, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.local_counts, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.barrier_counter, sizeof(int));
  cudaMalloc((void **)&cfg.barrier_sense, sizeof(int));
  if (!cfg.counts || !cfg.offsets || !cfg.local_counts || !cfg.barrier_counter ||
      !cfg.barrier_sense) {
    if (cfg.counts) cudaFree(cfg.counts);
    if (cfg.offsets) cudaFree(cfg.offsets);
    if (cfg.local_counts) cudaFree(cfg.local_counts);
    if (cfg.barrier_counter) cudaFree(cfg.barrier_counter);
    if (cfg.barrier_sense) cudaFree(cfg.barrier_sense);
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  // 生成 intranode_index 并执行 dispatch
  status = pre_process(buf.routing_map, buf.intranode_index, &cfg);
  if (status == 0) {
    status = dispatch_tokens(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
  }
  nvshmem_barrier_all();

  int errors = 0;
  if (bench_iters > 0) {
    size_t local_bytes =
        (size_t)num_tokens_per_rank * (size_t)topk * (size_t)token_bytes;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < bench_warmup; ++i) {
      dispatch_tokens(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
      nvshmem_barrier_all();
    }
    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i) {
      dispatch_tokens(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
      nvshmem_barrier_all();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double sec = ms / 1000.0;
    double bw_gb = (double)local_bytes * (double)bench_iters / sec / 1e9;
    printf("PE %d: avg_bw %.3f GB/s, iters %d\n", mype, bw_gb, bench_iters);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  if (!bench_only) {
    errors = (status != 0)
                 ? 1
                 : check_output(&buf, num_tokens_per_rank, hidden_size, npes, expert_num, mype);
    if (errors == 0) {
      errors = check_flags_cleared(&buf);
    }
    if (errors != 0) {
      printf("PE %d: dispatch_tokens failed\n", mype);
    }
    else{
      printf("PE %d: dispatch_tokens passed\n", mype);
    }
  }
  cudaFree(cfg.counts);
  cudaFree(cfg.offsets);
  cudaFree(cfg.local_counts);
  cudaFree(cfg.barrier_counter);
  cudaFree(cfg.barrier_sense);
  free_buffers(&buf);
  nvshmem_finalize();
  return errors;
}
