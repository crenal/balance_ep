#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <stddef.h>

__global__ void combine_fast_kernel_f32(const float *input_tokens, float *output_tokens,
                                        const int *dst_index, const int *round_num, int num_tokens,
                                        int hidden_size) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int t = (int)blockIdx.x;
  if (t >= num_tokens) return;
  size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
  for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
    float acc = 0.0f;
    for (int r = 0; r < npes; ++r) {
      int dst = round_num ? round_num[r] : r;
      if (dst < 0 || dst >= npes) continue;
      int idx = dst_index[g * (size_t)npes + (size_t)dst];
      if (idx < 0) continue;
      const float *remote = input_tokens + (size_t)idx * (size_t)hidden_size + (size_t)h;
      acc += nvshmem_float_g(remote, dst);
    }
    output_tokens[(size_t)t * (size_t)hidden_size + (size_t)h] = acc;
  }
}

int combine_tokens_fast(const void *input_tokens, void *output_tokens, const int *dst_index,
                        const int *round_num, const DispatchConfig *cfg) {
  if (!cfg) return 1;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;
  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;
  if (bytes_per_elem != (int)sizeof(float)) return 1;
  if (!input_tokens || !output_tokens || !dst_index) return 1;
  int threads = 256;
  int blocks = num_tokens > 0 ? num_tokens : 1;
  combine_fast_kernel_f32<<<blocks, threads>>>((const float *)input_tokens, (float *)output_tokens,
                                               dst_index, round_num, num_tokens, hidden_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  return 0;
}

