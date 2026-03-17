#include "dispatch.h"

#include "test_dispatch_buffers.h"
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
  }

  int node_npes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
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
    nvshmem_finalize();
    return 1;
  }
  cudaSetDevice(local_rank);

  int num_tokens_per_rank = 1024;
  int expert_num = 128;
  int hidden_size = 1024;
  int topk = 4;
  int blocks_per_kernel = 8;
  int chunk_tokens = 256;
  int dispatch_mode = 0;
  float alpha = 0.0f;
  read_env_int("NUM_TOKENS_PER_RANK", &num_tokens_per_rank);
  read_env_int("EXPERT_NUM", &expert_num);
  read_env_int("HIDDEN_SIZE", &hidden_size);
  read_env_int("TOPK", &topk);
  read_env_int("BLOCKS_PER_KERNEL", &blocks_per_kernel);
  read_env_int("CHUNK_TOKENS", &chunk_tokens);
  read_env_int("DISPATCH_MODE", &dispatch_mode);
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
  size_t map_bytes = global_tokens * (size_t)expert_num * sizeof(bool);
  size_t mid_buf_bytes = (nnodes > 1) ? (size_t)(nnodes - 1) * input_bytes : 0;
  size_t mid_flags_bytes =
      (nnodes > 1) ? (size_t)(nnodes - 1) * (size_t)num_chunks * sizeof(uint64_t) : 0;

  TestBuffers buf = {};
  int status =
      allocate_buffers(&buf, input_bytes, output_bytes, map_bytes, global_tokens * (size_t)npes,
                       mid_buf_bytes, mid_flags_bytes);
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
  cfg.local_buf = buf.local_buf;
  cfg.mid_buf = buf.mid_buf;
  cfg.mid_flags = buf.mid_flags;

  cudaMalloc((void **)&cfg.counts, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.offsets, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.local_counts, (size_t)npes * (size_t)npes * sizeof(int));
  cudaMalloc((void **)&cfg.barrier_counter, sizeof(int));
  cudaMalloc((void **)&cfg.barrier_sense, sizeof(int));
  cfg.node_counts = nullptr;
  cfg.src_forward_local = nullptr;
  cfg.src_local_of_forward = nullptr;
  cfg.dst_forward_local = nullptr;
  cfg.src_forward_local_of_dst_forward = nullptr;
  if (dispatch_mode == 3) {
    cudaMalloc((void **)&cfg.node_counts, (size_t)npes * (size_t)nnodes * sizeof(int));
    cudaMalloc((void **)&cfg.src_forward_local, (size_t)nnodes * (size_t)node_npes * sizeof(int));
    cudaMalloc((void **)&cfg.src_local_of_forward, (size_t)nnodes * (size_t)node_npes * sizeof(int));
    cudaMalloc((void **)&cfg.dst_forward_local,
               (size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int));
    cudaMalloc((void **)&cfg.src_forward_local_of_dst_forward,
               (size_t)nnodes * (size_t)nnodes * (size_t)node_npes * sizeof(int));
  }
  if (!cfg.counts || !cfg.offsets || !cfg.local_counts || !cfg.barrier_counter || !cfg.barrier_sense ||
      (dispatch_mode == 3 &&
       (!cfg.node_counts || !cfg.src_forward_local || !cfg.src_local_of_forward || !cfg.dst_forward_local ||
        !cfg.src_forward_local_of_dst_forward))) {
    if (cfg.counts) cudaFree(cfg.counts);
    if (cfg.offsets) cudaFree(cfg.offsets);
    if (cfg.local_counts) cudaFree(cfg.local_counts);
    if (cfg.barrier_counter) cudaFree(cfg.barrier_counter);
    if (cfg.barrier_sense) cudaFree(cfg.barrier_sense);
    if (cfg.node_counts) cudaFree(cfg.node_counts);
    if (cfg.src_forward_local) cudaFree(cfg.src_forward_local);
    if (cfg.src_local_of_forward) cudaFree(cfg.src_local_of_forward);
    if (cfg.dst_forward_local) cudaFree(cfg.dst_forward_local);
    if (cfg.src_forward_local_of_dst_forward) cudaFree(cfg.src_forward_local_of_dst_forward);
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  int *round_num = nullptr;
  if (dispatch_mode == 1 || dispatch_mode == 2) {
    cudaMalloc((void **)&round_num, (size_t)npes * sizeof(int));
    if (!round_num) {
      cudaFree(cfg.counts);
      cudaFree(cfg.offsets);
      cudaFree(cfg.local_counts);
      cudaFree(cfg.barrier_counter);
      cudaFree(cfg.barrier_sense);
      if (cfg.node_counts) cudaFree(cfg.node_counts);
      if (cfg.src_forward_local) cudaFree(cfg.src_forward_local);
      if (cfg.src_local_of_forward) cudaFree(cfg.src_local_of_forward);
      if (cfg.dst_forward_local) cudaFree(cfg.dst_forward_local);
      if (cfg.src_forward_local_of_dst_forward) cudaFree(cfg.src_forward_local_of_dst_forward);
      free_buffers(&buf);
      nvshmem_finalize();
      return 1;
    }
  }

  if (dispatch_mode == 3) {
    status = pre_process_balance(buf.routing_map, buf.intranode_index, &cfg);
  } else if (dispatch_mode != 0) {
    status = pre_process_fast(buf.routing_map, buf.intranode_index, round_num, &cfg);
  } else {
    status = pre_process(buf.routing_map, buf.intranode_index, &cfg);
  }
  if (status == 0) {
    if (dispatch_mode == 3) {
      status = dispatch_tokens_balance(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
    } else if (dispatch_mode == 2) {
      status = dispatch_tokens_fast(buf.input_tokens, buf.output_tokens, buf.intranode_index, round_num,
                                    &cfg);
    } else {
      status = dispatch_tokens(buf.input_tokens, buf.output_tokens, buf.intranode_index, &cfg);
    }
  }
  nvshmem_barrier_all();
  if (status != 0) {
    if (round_num) cudaFree(round_num);
    cudaFree(cfg.counts);
    cudaFree(cfg.offsets);
    cudaFree(cfg.local_counts);
    cudaFree(cfg.barrier_counter);
    cudaFree(cfg.barrier_sense);
    if (cfg.node_counts) cudaFree(cfg.node_counts);
    if (cfg.src_forward_local) cudaFree(cfg.src_forward_local);
    if (cfg.src_local_of_forward) cudaFree(cfg.src_local_of_forward);
    if (cfg.dst_forward_local) cudaFree(cfg.dst_forward_local);
    if (cfg.src_forward_local_of_dst_forward) cudaFree(cfg.src_forward_local_of_dst_forward);
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }

  float *combined = nullptr;
  cudaMalloc((void **)&combined, input_bytes);
  if (!combined) {
    if (round_num) cudaFree(round_num);
    cudaFree(cfg.counts);
    cudaFree(cfg.offsets);
    cudaFree(cfg.local_counts);
    cudaFree(cfg.barrier_counter);
    cudaFree(cfg.barrier_sense);
    if (cfg.node_counts) cudaFree(cfg.node_counts);
    if (cfg.src_forward_local) cudaFree(cfg.src_forward_local);
    if (cfg.src_local_of_forward) cudaFree(cfg.src_local_of_forward);
    if (cfg.dst_forward_local) cudaFree(cfg.dst_forward_local);
    if (cfg.src_forward_local_of_dst_forward) cudaFree(cfg.src_forward_local_of_dst_forward);
    free_buffers(&buf);
    nvshmem_finalize();
    return 1;
  }
  cudaMemset(combined, 0, input_bytes);

  if (dispatch_mode == 3) {
    status = combine_tokens_balance(buf.output_tokens, combined, buf.intranode_index, &cfg);
  } else if (dispatch_mode != 0) {
    status = combine_tokens_fast(buf.output_tokens, combined, buf.intranode_index, round_num, &cfg);
  } else {
    status = combine_tokens(buf.output_tokens, combined, buf.intranode_index, &cfg);
  }
  nvshmem_barrier_all();

  if (mype == 0) {
    printf("combine status %d\n", status);
  }

  cudaFree(combined);
  if (round_num) cudaFree(round_num);
  cudaFree(cfg.counts);
  cudaFree(cfg.offsets);
  cudaFree(cfg.local_counts);
  cudaFree(cfg.barrier_counter);
  cudaFree(cfg.barrier_sense);
  if (cfg.node_counts) cudaFree(cfg.node_counts);
  if (cfg.src_forward_local) cudaFree(cfg.src_forward_local);
  if (cfg.src_local_of_forward) cudaFree(cfg.src_local_of_forward);
  if (cfg.dst_forward_local) cudaFree(cfg.dst_forward_local);
  if (cfg.src_forward_local_of_dst_forward) cudaFree(cfg.src_forward_local_of_dst_forward);
  free_buffers(&buf);
  nvshmem_finalize();
  return 0;
}

