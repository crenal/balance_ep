#pragma once

#include <stddef.h>
#include <stdint.h>

struct TestBuffers {
  float *input_tokens;
  float *output_tokens;
  bool *routing_map;
  int *intranode_index;
  void *mid_buf;
  uint64_t *mid_flags;
  size_t mid_buf_bytes;
  size_t mid_flags_bytes;
  float *input_h;
  float *output_h;
  bool *map_h;
};
