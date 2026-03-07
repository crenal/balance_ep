#pragma once

#include "test_dispatch_types.h"

int check_output(const TestBuffers *buf, int num_tokens_per_rank, int hidden_size, int npes,
                 int expert_num, int mype);
