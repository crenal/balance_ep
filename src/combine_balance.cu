#include "dispatch.h"

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <stddef.h>

__global__ void combine_balance_kernel_f32(const float *input_tokens, float *output_tokens,
                                           const int *dst_index, int num_tokens, int hidden_size,
                                           int node_npes, int nnodes,
                                           const int *src_forward_local,
                                           const int *dst_forward_local) {
  int npes = nvshmem_n_pes();
  if (npes <= 0) return;
  int mype = nvshmem_my_pe();
  int t = (int)blockIdx.x;
  if (t >= num_tokens) return;
  int node_id = node_npes ? (mype / node_npes) : 0;
  int local_rank = node_npes ? (mype - node_id * node_npes) : 0;
  size_t g = (size_t)mype * (size_t)num_tokens + (size_t)t;
  for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
    float acc = 0.0f;
    if (node_npes > 0) {
      int src_fwd = src_forward_local ? src_forward_local[node_id * node_npes + local_rank]
                                      : local_rank;
      if (src_fwd < 0 || src_fwd >= node_npes) src_fwd = local_rank;

      int base = node_id * node_npes;
      for (int lr = 0; lr < node_npes; ++lr) {
        int dst = base + lr;
        if (dst < 0 || dst >= npes) continue;
        int idx = dst_index[g * (size_t)npes + (size_t)dst];
        if (idx < 0) continue;
        const float *remote = input_tokens + (size_t)idx * (size_t)hidden_size + (size_t)h;
        acc += nvshmem_float_g(remote, dst);
      }

      for (int k = 1; k < nnodes; ++k) {
        int dst_node = node_id + k;
        if (dst_node >= nnodes) dst_node -= nnodes;
        int dst_fwd_local = src_fwd;
        if (dst_forward_local) {
          dst_fwd_local =
              dst_forward_local[(node_id * nnodes + dst_node) * node_npes + src_fwd];
        }
        if (dst_fwd_local < 0 || dst_fwd_local >= node_npes) dst_fwd_local = 0;
        int dst = dst_node * node_npes + dst_fwd_local;
        if (dst < 0 || dst >= npes) continue;
        int idx = dst_index[g * (size_t)npes + (size_t)dst];
        if (idx < 0) continue;
        const float *remote = input_tokens + (size_t)idx * (size_t)hidden_size + (size_t)h;
        acc += nvshmem_float_g(remote, dst);
      }
    } else {
      for (int dst = 0; dst < npes; ++dst) {
        int idx = dst_index[g * (size_t)npes + (size_t)dst];
        if (idx < 0) continue;
        const float *remote = input_tokens + (size_t)idx * (size_t)hidden_size + (size_t)h;
        acc += nvshmem_float_g(remote, dst);
      }
    }
    output_tokens[(size_t)t * (size_t)hidden_size + (size_t)h] = acc;
  }
}

int combine_tokens_balance(const void *input_tokens, void *output_tokens, const int *dst_index,
                           const DispatchConfig *cfg) {
  if (!cfg) return 1;
  int npes = nvshmem_n_pes();
  if (npes <= 0) return 1;
  int num_tokens = cfg->num_tokens_per_rank;
  int hidden_size = cfg->hidden_size;
  int bytes_per_elem = cfg->bytes_per_elem;
  if (bytes_per_elem != (int)sizeof(float)) return 1;
  if (!input_tokens || !output_tokens || !dst_index) return 1;
  if (!cfg->src_forward_local || !cfg->dst_forward_local) return 1;
  int threads = 256;
  int blocks = num_tokens > 0 ? num_tokens : 1;
  combine_balance_kernel_f32<<<blocks, threads>>>((const float *)input_tokens,
                                                  (float *)output_tokens, dst_index, num_tokens,
                                                  hidden_size, cfg->node_npes, cfg->nnodes,
                                                  cfg->src_forward_local, cfg->dst_forward_local);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 1;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 1;
  return 0;
}
