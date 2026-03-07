#include "test_dispatch_inputs.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static unsigned int next_rand(unsigned int *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static inline unsigned int mix_u32(unsigned int x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static inline float u32_to_unit_float(unsigned int x) {
  return (float)(x & 0x00ffffffu) * (1.0f / 16777216.0f);
}

static void build_zipf_cdf(float *cdf, int n, float s) {
  if (!cdf || n <= 0) return;
  if (s <= 0.0f) {
    for (int i = 0; i < n; ++i) cdf[i] = (float)(i + 1) / (float)n;
    return;
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += pow((double)(i + 1), -(double)s);
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += pow((double)(i + 1), -(double)s) / sum;
    cdf[i] = (float)acc;
  }
  cdf[n - 1] = 1.0f;
}

static int sample_cdf(const float *cdf, int n, float u) {
  if (!cdf || n <= 0) return 0;
  if (u <= 0.0f) return 0;
  if (u >= 1.0f) return n - 1;
  int i = 0;
  for (; i < n; ++i) {
    if (u <= cdf[i]) break;
  }
  if (i >= n) i = n - 1;
  return i;
}

static void generate_map(bool *map_h, size_t global_tokens, int num_tokens_per_rank, int npes,
                         int nnodes, int node_npes, int expert_num, int topk, float alpha) {
  if (alpha < 0.0f) alpha = 0.0f;
  if (alpha > 1.0f) alpha = 1.0f;
  if (nnodes <= 0) nnodes = 1;
  if (node_npes <= 0) node_npes = npes;
  int experts_per_rank = (expert_num + npes - 1) / npes;

  float zipf_s = 0.0f;
  const float zipf_s_max = 12.0f;
  if (alpha > 0.0f && alpha < 1.0f) zipf_s = alpha * zipf_s_max;

  float *node_cdf = nullptr;
  float *rank_cdf = nullptr;
  if (zipf_s > 0.0f) {
    node_cdf = (float *)malloc((size_t)nnodes * sizeof(float));
    rank_cdf = (float *)malloc((size_t)node_npes * sizeof(float));
    if (node_cdf) build_zipf_cdf(node_cdf, nnodes, zipf_s);
    if (rank_cdf) build_zipf_cdf(rank_cdf, node_npes, zipf_s);
  }

  for (size_t gt = 0; gt < global_tokens; ++gt) {
    size_t row = gt * (size_t)expert_num;
    for (int e = 0; e < expert_num; ++e) {
      map_h[row + (size_t)e] = false;
    }

    int src = (int)(gt / (size_t)num_tokens_per_rank);
    (void)src;

    int dst_node = 0;
    int localrank = 0;
    if (alpha <= 0.0f) {
      float u0 = u32_to_unit_float(mix_u32((unsigned int)gt ^ 0x243f6a88u));
      float u1 = u32_to_unit_float(mix_u32((unsigned int)gt ^ 0x85a308d3u));
      dst_node = (int)(u0 * (float)nnodes);
      if (dst_node >= nnodes) dst_node = nnodes - 1;
      localrank = (int)(u1 * (float)node_npes);
      if (localrank >= node_npes) localrank = node_npes - 1;
    } else if (alpha >= 1.0f) {
      dst_node = 0;
      localrank = 0;
    } else {
      float u0 = u32_to_unit_float(mix_u32((unsigned int)gt ^ 0x243f6a88u));
      float u1 = u32_to_unit_float(mix_u32((unsigned int)gt ^ 0x85a308d3u));
      if (node_cdf) {
        dst_node = sample_cdf(node_cdf, nnodes, u0);
      } else {
        dst_node = 0;
      }
      if (rank_cdf) {
        localrank = sample_cdf(rank_cdf, node_npes, u1);
      } else {
        localrank = 0;
      }
    }

    int dst = dst_node * node_npes + localrank;
    if (dst < 0 || dst >= npes) continue;

    int kmax = topk;
    if (kmax < 1) kmax = 1;
    int start = dst * experts_per_rank;
    if (start >= expert_num) continue;
    int end = start + experts_per_rank;
    if (end > expert_num) end = expert_num;
    int experts_here = end - start;
    if (experts_here <= 0) continue;
    if (kmax > experts_here) kmax = experts_here;

    for (int s = 0; s < kmax; ++s) {
      int e = start + s;
      map_h[row + (size_t)e] = true;
    }
  }
  if (node_cdf) free(node_cdf);
  if (rank_cdf) free(rank_cdf);
}

void init_inputs(TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int mype, int npes,
                 int nnodes, int node_npes, int expert_num, int topk, float alpha,
                 size_t input_bytes, size_t output_bytes, size_t map_bytes) {
  for (int t = 0; t < num_tokens_per_rank; ++t) {
    for (int h = 0; h < hidden_size; ++h) {
      buf->input_h[t * hidden_size + h] = (float)(mype * 100000 + t * 1000 + h);
    }
  }
  cudaMemcpy(buf->input_tokens, buf->input_h, input_bytes, cudaMemcpyHostToDevice);
  cudaMemset(buf->output_tokens, 0, output_bytes);
  if (buf->mid_buf && buf->mid_buf_bytes > 0) {
    cudaMemset(buf->mid_buf, 0, buf->mid_buf_bytes);
  }
  if (buf->mid_flags && buf->mid_flags_bytes > 0) {
    cudaMemset(buf->mid_flags, 0, buf->mid_flags_bytes);
  }

  size_t global_tokens = (size_t)num_tokens_per_rank * (size_t)npes;
  generate_map(buf->map_h, global_tokens, num_tokens_per_rank, npes, nnodes, node_npes, expert_num,
               topk, alpha);
  cudaMemcpy(buf->routing_map, buf->map_h, map_bytes, cudaMemcpyHostToDevice);
}
