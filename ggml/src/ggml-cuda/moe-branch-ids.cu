#include "moe-branch-ids.cuh"

static __global__ void moe_branch_ids(
        const int32_t * __restrict__ ids, int32_t * __restrict__ dst,
        const int ne0, const int ne1, const int s1, const int d1,
        const int lo, const int hi, const int mask_base) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0*ne1) {
        return;
    }
    const int i1 = i / ne0;
    const int i0 = i - i1*ne0;
    const int id = ids[i1*s1 + i0];
    dst[i1*d1 + i0] = id >= lo && id < hi ? id - lo : mask_base + i0;
}

void ggml_cuda_op_moe_branch_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids = dst->src[0];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(dst->nb[0] == sizeof(int32_t));

    const int lo        = ggml_get_op_params_i32(dst, 0);
    const int hi        = ggml_get_op_params_i32(dst, 1);
    const int mask_base = ggml_get_op_params_i32(dst, 2);

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int s1  = ids->nb[1]/sizeof(int32_t);
    const int d1  = dst->nb[1]/sizeof(int32_t);

    const int n = ne0*ne1;
    moe_branch_ids<<<(n + 255)/256, 256, 0, ctx.stream()>>>(
        (const int32_t *) ids->data, (int32_t *) dst->data, ne0, ne1, s1, d1, lo, hi, mask_base);
}
