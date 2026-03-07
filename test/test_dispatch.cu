#include "dispatch.h"

#include "test_dispatch_buffers.h"
#include "test_dispatch_check.h"
#include "test_dispatch_env.h"
#include "test_dispatch_inputs.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  nvshmem_init();
  printf("pe %d init nvshmem\n", nvshmem_my_pe());
  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
  int override_nnodes = 0;
  int override_rpn = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--nnodes") == 0 && i + 1 < argc) {
      override_nnodes = atoi(argv[i + 1]);
      ++i;
      continue;
    }
    if (strcmp(argv[i], "--ranks_per_node") == 0 && i + 1 < argc) {
      override_rpn = atoi(argv[i + 1]);
      ++i;
      continue;
    }
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      if (mype == 0) printf("Usage: %s [--nnodes N] [--ranks_per_node R]\n", argv[0]);
      nvshmem_finalize();
      return 0;
    }
  }

  int node_npes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("pe %d local_rank %d\n", nvshmem_my_pe(), local_rank);
  int nnodes = node_npes ? (npes + node_npes - 1) / node_npes : 1;
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
    if (mype == 0) printf("Invalid nnodes/ranks_per_node for npes=%d\n", npes);
    nvshmem_finalize();
    return 1;
  }
  cudaSetDevice(local_rank);
  int num_tokens_per_rank = 8192;
  int expert_num = 128;
  int hidden_size = 8192;
  int topk = 4;
  int blocks_per_kernel = 8;
  int chunk_tokens = 256;
  int bench_iters = 0;
  int bench_warmup = 5;
  int bench_only = 0;
  float alpha = 0.0f;
  read_env_int("NUM_TOKENS_PER_RANK", &num_tokens_per_rank);
  read_env_int("EXPERT_NUM", &expert_num);
  read_env_int("HIDDEN_SIZE", &hidden_size);
  read_env_int("TOPK", &topk); 
  read_env_int("BLOCKS_PER_KERNEL", &blocks_per_kernel);
  read_env_int("CHUNK_TOKENS", &chunk_tokens);
  read_env_int("BENCH_ITERS", &bench_iters);
  read_env_int("BENCH_WARMUP", &bench_warmup);
  read_env_int("BENCH_ONLY", &bench_only);
  if (!read_env_float("ALPHA", &alpha)) {
    read_env_float("ZIPF_ALPHA", &alpha);
  }
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
  size_t mid_buf_bytes = (nnodes > 1) ? (size_t)(nnodes - 1) * input_bytes : 0;
  size_t mid_flags_bytes =
      (nnodes > 1) ? (size_t)(nnodes - 1) * (size_t)num_chunks * sizeof(uint64_t) : 0;

  TestBuffers buf = {};
  printf("pe %d allocate_buffers\n", nvshmem_my_pe());
  int status = allocate_buffers(&buf, input_bytes, output_bytes, map_bytes,
                                (size_t)global_tokens * (size_t)npes, mid_buf_bytes, mid_flags_bytes);
  if (status != 0) {
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  init_inputs(&buf, num_tokens_per_rank, hidden_size, mype, npes, nnodes, node_npes, expert_num,
              topk, alpha, input_bytes, output_bytes, map_bytes);

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
 
