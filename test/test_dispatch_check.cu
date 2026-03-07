#include "test_dispatch_check.h"

#include <cuda_runtime.h>
#include <stdlib.h>

int check_output(const TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int npes,
                 int expert_num, int mype) {
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
      for (int h = 0; h < hidden_size; ++h) {
        float expected_val = (float)(src * 100000 + t * 1000 + h);
        if (buf->output_h[base + h] != expected_val) {
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
