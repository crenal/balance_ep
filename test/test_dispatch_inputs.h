#pragma once

#include <stddef.h>

#include "test_dispatch_types.h"

void init_inputs(TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int mype, int npes,
                 int nnodes, int node_npes, int expert_num, int topk, float alpha,
                 size_t input_bytes, size_t output_bytes, size_t map_bytes);
