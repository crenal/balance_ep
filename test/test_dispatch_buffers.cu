#include "test_dispatch_buffers.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int allocate_buffers(TestBuffers *buf, size_t input_bytes, size_t output_bytes, size_t map_bytes,
                     size_t intranode_elems, size_t mid_buf_bytes, size_t mid_flags_bytes) {
  printf("pe %d allocate_buffers\n", nvshmem_my_pe());
  buf->input_tokens = (float *)nvshmem_malloc(input_bytes);
  printf("pe %d input_tokens %p\n", nvshmem_my_pe(), buf->input_tokens);
  buf->output_tokens = (float *)nvshmem_malloc(output_bytes);
  buf->routing_map = nullptr;
  cudaMalloc((void **)&buf->routing_map, map_bytes);
  buf->intranode_index = nullptr;
  cudaMalloc((void **)&buf->intranode_index, intranode_elems * sizeof(int));
  buf->mid_buf_bytes = mid_buf_bytes;
  buf->mid_flags_bytes = mid_flags_bytes;
  printf("start allocate mid_buf %p, mid_flags %p\n", buf->mid_buf, buf->mid_flags);
  buf->mid_buf = buf->mid_buf_bytes ? nvshmem_malloc(buf->mid_buf_bytes) : nullptr;
  printf("pe %d mid_buf %p\n", nvshmem_my_pe(), buf->mid_buf);
  buf->mid_flags = buf->mid_flags_bytes ? (uint64_t *)nvshmem_malloc(buf->mid_flags_bytes) : nullptr;
  printf("pe %d mid_flags %p\n", nvshmem_my_pe(), buf->mid_flags);
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

void free_buffers(TestBuffers *buf) {
  if (buf->input_h) free(buf->input_h);
  if (buf->output_h) free(buf->output_h);
  if (buf->map_h) free(buf->map_h);
  if (buf->input_tokens) nvshmem_free(buf->input_tokens);
  if (buf->output_tokens) nvshmem_free(buf->output_tokens);
  if (buf->mid_buf) nvshmem_free(buf->mid_buf);
  if (buf->mid_flags) nvshmem_free(buf->mid_flags);
  if (buf->routing_map) cudaFree(buf->routing_map);
  if (buf->intranode_index) cudaFree(buf->intranode_index);
}
