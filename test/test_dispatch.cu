#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned int next_rand(unsigned int *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

struct TestBuffers {
  // NVSHMEM 对称内存：每个 rank 的 input/output
  float *input_tokens;
  float *output_tokens;
  // 普通 GPU 内存：全局 routing_map 与本地 intranode_index
  bool *routing_map;
  int *intranode_index;
  // 主机侧缓冲区：用于初始化和校验
  float *input_h;
  float *output_h;
  bool *map_h;
};

static int allocate_buffers(TestBuffers *buf, size_t input_bytes, size_t output_bytes,
                            size_t map_bytes, int num_tokens_per_rank, int npes) {
  // input/output 使用 nvshmem_malloc 保证对称内存
  buf->input_tokens = (float *)nvshmem_malloc(input_bytes);
  buf->output_tokens = (float *)nvshmem_malloc(output_bytes);
  // routing_map 与 intranode_index 使用普通 GPU 内存
  buf->routing_map = nullptr;
  cudaMalloc((void **)&buf->routing_map, map_bytes);
  buf->intranode_index = nullptr;
  cudaMalloc((void **)&buf->intranode_index,
             (size_t)num_tokens_per_rank * (size_t)npes * sizeof(int));
  // 主机侧临时缓冲区
  buf->input_h = (float *)malloc(input_bytes);
  buf->output_h = (float *)malloc(output_bytes);
  buf->map_h = (bool *)malloc(map_bytes);
  if (!buf->input_tokens || !buf->output_tokens || !buf->routing_map || !buf->intranode_index ||
      !buf->input_h || !buf->output_h || !buf->map_h) {
    return 1;
  }
  return 0;
}

static void free_buffers(TestBuffers *buf) {
  // 释放主机缓冲区
  if (buf->input_h) free(buf->input_h);
  if (buf->output_h) free(buf->output_h);
  if (buf->map_h) free(buf->map_h);
  // 释放 NVSHMEM 对称内存
  if (buf->input_tokens) nvshmem_free(buf->input_tokens);
  if (buf->output_tokens) nvshmem_free(buf->output_tokens);
  // 释放普通 GPU 内存
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
  // 构造可验证的输入 token：值包含 src rank 和 token 索引信息
  for (int t = 0; t < num_tokens_per_rank; ++t) {
    for (int h = 0; h < hidden_size; ++h) {
      buf->input_h[t * hidden_size + h] = (float)(mype * 100000 + t * 1000 + h);
    }
  }
  // 复制到设备侧 input，并将输出清零
  cudaMemcpy(buf->input_tokens, buf->input_h, input_bytes, cudaMemcpyHostToDevice);
  cudaMemset(buf->output_tokens, 0, output_bytes);

  // 生成全局 routing_map：map[global_token][expert]
  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  generate_map(buf->map_h, global_tokens, expert_num, topk);
  // 将 routing_map 复制到设备侧
  cudaMemcpy(buf->routing_map, buf->map_h, map_bytes, cudaMemcpyHostToDevice);
}

static int check_output(const TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int npes,
                        int expert_num, int mype) {
  // 输出布局为按源 rank 分块的连续紧凑布局
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

  // counts[src][dst] 统计 src 的 token 在 dst 上被接收的数量
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
  // offsets[src][dst] 为 dst 端接收的 src 块起始位置
  for (int j = 0; j < npes; ++j) {
    int prefix = 0;
    for (int src = 0; src < npes; ++src) {
      offsets[src * npes + j] = prefix;
      prefix += counts[src * npes + j];
    }
  }

  int total_count = 0;
  for (int src = 0; src < npes; ++src) total_count += counts[src * npes + mype];

  unsigned char *received = (unsigned char *)malloc((size_t)num_tokens_per_rank);
  if (!received) {
    free(counts);
    free(offsets);
    return 1;
  }

  // 块内按 token 内容解析出 t，验证块内无重复且与 routing_map 一致
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

  // 超出总接收量的尾部区域应保持为 0
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

int main() {
  nvshmem_init();

  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  cudaSetDevice(mype_node);

  const int num_tokens_per_rank = 8192;
  const int expert_num = 128;
  const int hidden_size = 8192;
  const int topk = 4;
  const int bytes_per_elem = (int)sizeof(float);
  size_t token_bytes = (size_t)hidden_size * (size_t)bytes_per_elem;
  size_t input_bytes = (size_t)num_tokens_per_rank * token_bytes;
  size_t output_bytes = (size_t)npes * (size_t)num_tokens_per_rank * token_bytes;
  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  size_t map_elems = global_tokens * (size_t)expert_num;
  size_t map_bytes = map_elems * sizeof(bool);

  TestBuffers buf = {};
  int status = allocate_buffers(&buf, input_bytes, output_bytes, map_bytes, num_tokens_per_rank,
                                npes);
  if (status != 0) {
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  // 初始化输入 token 与全局 routing_map
  init_inputs(&buf, num_tokens_per_rank, hidden_size, mype, npes, expert_num, topk, input_bytes,
              output_bytes, map_bytes);

  DispatchConfig cfg;
  cfg.num_tokens_per_rank = num_tokens_per_rank;
  cfg.expert_num = expert_num;
  cfg.hidden_size = hidden_size;
  cfg.bytes_per_elem = bytes_per_elem;
  cfg.blocks_per_kernel = 8;
  // 预处理临时缓冲由调用方提供
  cudaMalloc((void **)&cfg.counts, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.offsets, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.local_counts, (size_t)npes * sizeof(int));
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

  // 由 routing_map 生成 intranode_index
  status = pre_process(buf.routing_map, buf.intranode_index, &cfg);
  if (status == 0) {
    // 按 intranode_index 执行 dispatch
    status = dispatch_tokens(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
  }
  nvshmem_barrier_all();

  // 校验输出是否符合 routing_map 语义
  int errors = (status != 0) ? 1
                             : check_output(&buf, num_tokens_per_rank, hidden_size, npes,
                                            expert_num, mype);
  if (errors != 0) {
    printf("PE %d: dispatch_tokens failed\n", mype);
  }
  else{
    printf("PE %d: dispatch_tokens passed\n", mype);
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
