#pragma once

#include "common.cuh"

void ggml_cuda_op_moe_branch_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
