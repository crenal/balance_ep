#pragma once

#include <stddef.h>

#include "test_dispatch_types.h"

int allocate_buffers(TestBuffers *buf, size_t input_bytes, size_t output_bytes, size_t map_bytes,
                     size_t intranode_elems, size_t mid_buf_bytes, size_t mid_flags_bytes);

void free_buffers(TestBuffers *buf);
