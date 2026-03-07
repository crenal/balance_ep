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

static inline int clamp_kmax(int topk, int limit) {
  int kmax = topk < 1 ? 1 : topk;
  if (kmax > limit) kmax = limit;
  return kmax;
}

static inline void mark_rank_experts(bool *map_h, size_t row, int expert_num, int experts_per_rank,
                                     int dst, int kmax) {
  int start = dst * experts_per_rank;
  if (start >= expert_num) return;
  int end = start + experts_per_rank;
  if (end > expert_num) end = expert_num;
  int experts_here = end - start;
  if (experts_here <= 0) return;
  int kmax_node = clamp_kmax(kmax, experts_here);
  for (int s = 0; s < kmax_node; ++s) {
    int e = start + s;
    map_h[row + (size_t)e] = true;
  }
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
  zipf_s = alpha * zipf_s_max;

  float *rank_cdf = nullptr;
  rank_cdf = (float *)malloc((size_t)node_npes * sizeof(float));
  if (rank_cdf) build_zipf_cdf(rank_cdf, node_npes, zipf_s);

  unsigned int *rank_used = (unsigned int *)malloc((size_t)npes * sizeof(unsigned int));
  unsigned int *node_used = (unsigned int *)malloc((size_t)nnodes * sizeof(unsigned int));
  unsigned int stamp = 1;
  if (rank_used) {
    for (int i = 0; i < npes; ++i) rank_used[i] = 0;
  }
  if (node_used) {
    for (int i = 0; i < nnodes; ++i) node_used[i] = 0;
  }

  for (size_t gt = 0; gt < global_tokens; ++gt) {
    size_t row = gt * (size_t)expert_num;
    for (int e = 0; e < expert_num; ++e) {
      map_h[row + (size_t)e] = false;
    }

    (void)num_tokens_per_rank;
    int k_ranks = clamp_kmax(topk, npes);
    int kmax = 1;

    if (!rank_used || !node_used || !rank_cdf) {
      for (int i = 0; i < k_ranks; ++i) {
        mark_rank_experts(map_h, row, expert_num, experts_per_rank, i, kmax);
      }
      continue;
    }

    if (stamp == 0) {
      for (int i = 0; i < npes; ++i) rank_used[i] = 0;
      for (int i = 0; i < nnodes; ++i) node_used[i] = 0;
      stamp = 1;
    }
    unsigned int token_stamp = stamp++;

    int selected = 0;
    int prefer_unique_nodes = (k_ranks <= nnodes) ? 1 : 0;
    int max_attempts = npes * 16;

    for (int pick = 0; pick < k_ranks; ++pick) {
      int dst_node = 0;
      if (prefer_unique_nodes) {
        int attempts = 0;
        while (attempts++ < nnodes * 8) {
          unsigned int seed_node =
              (unsigned int)gt ^ 0x243f6a88u ^ (unsigned int)pick * 0x9e3779b9u ^
              (unsigned int)attempts * 0x7f4a7c15u;
          int cand = (int)(mix_u32(seed_node) % (unsigned int)nnodes);
          if (node_used[cand] != token_stamp) {
            dst_node = cand;
            node_used[cand] = token_stamp;
            break;
          }
        }
      } else {
        unsigned int seed_node = (unsigned int)gt ^ 0x243f6a88u ^ (unsigned int)pick * 0x9e3779b9u;
        dst_node = (int)(mix_u32(seed_node) % (unsigned int)nnodes);
      }

      int attempts = 0;
      while (attempts++ < max_attempts) {
        unsigned int seed =
            (unsigned int)gt ^ (unsigned int)dst_node * 0x9e3779b9u ^ (unsigned int)pick * 0x85a308d3u ^
            (unsigned int)attempts * 0x632be59bu;
        float u = u32_to_unit_float(mix_u32(seed));
        int localrank = sample_cdf(rank_cdf, node_npes, u);
        int dst = dst_node * node_npes + localrank;
        if (dst < 0 || dst >= npes) continue;
        if (rank_used[dst] == token_stamp) continue;
        rank_used[dst] = token_stamp;
        mark_rank_experts(map_h, row, expert_num, experts_per_rank, dst, kmax);
        selected++;
        break;
      }
    }
  }
  if (rank_used) free(rank_used);
  if (node_used) free(node_used);
  if (rank_cdf) free(rank_cdf);
}

static void generate_map_random(bool *map_h, size_t global_tokens, int num_tokens_per_rank, int npes,
                                int nnodes, int node_npes, int expert_num, int topk, float alpha) {
  (void)num_tokens_per_rank;
  (void)npes;
  (void)nnodes;
  (void)node_npes;
  (void)alpha;

  int kmax = topk;
  if (kmax < 1) kmax = 1;
  if (kmax > expert_num) kmax = expert_num;

  for (size_t gt = 0; gt < global_tokens; ++gt) {
    size_t row = gt * (size_t)expert_num;
    for (int e = 0; e < expert_num; ++e) {
      map_h[row + (size_t)e] = false;
    }

    unsigned int state = mix_u32((unsigned int)gt ^ 0xa5a5a5a5u) + 1234u;
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
  const char *random_map_env = getenv("RANDOM_MAP");
  int use_random_map = (random_map_env && random_map_env[0] != '\0' && atoi(random_map_env) != 0);
  if (use_random_map) {
    generate_map_random(buf->map_h, global_tokens, num_tokens_per_rank, npes, nnodes, node_npes,
                        expert_num, topk, alpha);
  } else {
    generate_map(buf->map_h, global_tokens, num_tokens_per_rank, npes, nnodes, node_npes, expert_num,
                 topk, alpha);
  }
  cudaMemcpy(buf->routing_map, buf->map_h, map_bytes, cudaMemcpyHostToDevice);
}
