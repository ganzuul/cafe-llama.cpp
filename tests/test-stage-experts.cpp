// test-stage-experts.cpp
//
// Validates GGML_OP_STAGE_EXPERTS: the union of routed experts must be
// gathered into a compact, type-preserving staging tensor whose planes match
// the source expert weights bit-for-bit, and the sorted id union must remap
// correctly so a quantized mul_mat_id can consume the staged copy.
//
// Strategy: build a small expert tensor with real quantized type, run the op,
// then verify
//   (1) n_union == number of distinct in-range ids,
//   (2) ids_sorted is sorted, deduped, -1 padded,
//   (3) each staged plane is byte-identical to the corresponding source plane,
//   (4) remapping the original ids through ids_sorted yields the same experts.
//
// Runs on the CPU backend (the op is host-implemented by design).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);          \
            std::fprintf(stderr, __VA_ARGS__);                                 \
            std::fprintf(stderr, "\n");                                        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// Build the sorted distinct union exactly as the op is specified to, so we can
// cross-check the implementation rather than trusting one source of truth.
static std::vector<int32_t> reference_union(const std::vector<int32_t> & ids, int32_t n_expert) {
    std::vector<int32_t> v;
    for (int32_t id : ids) {
        if (id >= 0 && id < n_expert) {
            v.push_back(id);
        }
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

int main() {
    // Small but non-trivial: 4 experts, top-3, 3 tokens.
    const int64_t n_embd   = 32;
    const int64_t n_ff     = 8;
    const int64_t n_expert = 4;
    const int64_t n_used   = 3;
    const int64_t n_tokens = 3;

    const size_t ctx_size = 16 * 1024 * 1024;
    std::vector<uint8_t> buf(ctx_size);
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr, "ggml_init failed");

    // Expert weights, Q4_0 so we exercise a real quantized type. ne0 must be a
    // multiple of the block size (32) for Q4_0.
    struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff, n_expert);
    CHECK(w != nullptr, "failed to create expert tensor");

    // Deterministic pseudo-random fill so every expert plane differs.
    // With no_alloc the backend owns storage, so quantize into a scratch
    // buffer and upload after allocation.
    std::vector<float> w_f32(ggml_nelements(w));
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & v : w_f32) {
            v = dist(rng);
        }
    }

    // Ids with a deliberate duplicate, an out-of-range entry, and an
    // unsorted ordering, to exercise dedupe / masking / sorting.
    const std::vector<int32_t> ids_host = {
        3, 0, 3,      // token 0: dup + unsorted
        1, 2, 99,     // token 1: out-of-range masked
        2, 1, 0,      // token 2
    };

    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * ids_sorted = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);

    struct ggml_tensor * staged = ggml_stage_experts(ctx, w, ids, ids_sorted);
    CHECK(staged != nullptr, "ggml_stage_experts returned null");
    CHECK(staged->type == GGML_TYPE_Q4_0, "staged tensor must preserve source type, got %d", (int) staged->type);
    CHECK(staged->ne[2] == n_used * n_tokens, "staged expert dim should be max union, got %lld", (long long) staged->ne[2]);

    // The remap op must observe ids_sorted, which STAGE_EXPERTS fills as a side
    // effect. ggml tracks dependencies through src[], and ids_sorted is a leaf,
    // so the edge has to be stated explicitly or the remap may be scheduled
    // (and executed) before the union exists. This mirrors the requirement in
    // the real graph wiring.
    struct ggml_tensor * remapped = ggml_moe_remap_ids(ctx, ids, ids_sorted, -1);
    CHECK(remapped != nullptr, "ggml_moe_remap_ids returned null");
    CHECK(remapped->ne[0] == ids->ne[0] && remapped->ne[1] == ids->ne[1],
          "remapped shape must match ids");
    // Ordering edge: ids_sorted is a leaf filled by STAGE_EXPERTS as a side
    // effect, and ggml tracks dependencies only through src[]. Without this the
    // remap can execute before the union exists. src[2] is unused by the op.
    remapped->src[2] = staged;

    // Execute on the CPU backend.
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(backend != nullptr, "failed to init CPU backend");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, staged);
    ggml_build_forward_expand(gf, remapped);

    ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(in_buf != nullptr, "failed to allocate tensors");

    // Upload host data now that the allocator owns the storages.
    std::memcpy(ids->data, ids_host.data(), ids_host.size() * sizeof(int32_t));
    {
        std::vector<uint8_t> qbytes((size_t) ggml_row_size(GGML_TYPE_Q4_0, n_embd) * n_ff * n_expert);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w_f32.data(), qbytes.data(), 0, n_embd * n_ff, n_expert, nullptr);
        std::memcpy(w->data, qbytes.data(), qbytes.size());
    }


    ggml_backend_graph_compute(backend, gf);

    // (1) n_union
    const std::vector<int32_t> expect = reference_union(ids_host, (int32_t) n_expert);
    const int32_t n_union = ggml_stage_experts_n_union(staged);
    CHECK(n_union == (int32_t) expect.size(),
          "n_union mismatch: got %d, expected %zu", n_union, expect.size());
    std::printf("n_union = %d (expected %zu)\n", n_union, expect.size());

    // (2) ids_sorted is sorted, deduped, -1 padded
    {
        const int32_t * s = (const int32_t *) ids_sorted->data;
        const int64_t total = ids_sorted->ne[0] * ids_sorted->ne[1];
        for (int64_t i = 0; i < (int64_t) expect.size(); ++i) {
            CHECK(s[i] == expect[i], "ids_sorted[%lld] = %d, expected %d",
                  (long long) i, s[i], expect[i]);
        }
        for (int64_t i = (int64_t) expect.size(); i < total; ++i) {
            CHECK(s[i] == -1, "ids_sorted[%lld] = %d, expected -1 padding", (long long) i, s[i]);
        }
        std::printf("ids_sorted = [");
        for (int64_t i = 0; i < total; ++i) {
            std::printf("%d%s", s[i], i + 1 == total ? "" : ", ");
        }
        std::printf("]\n");
    }

    // (3) each staged plane byte-identical to the source expert plane
    {
        const size_t plane_bytes = (size_t) w->ne[1] * w->nb[1];
        for (int32_t k = 0; k < n_union; ++k) {
            const int32_t expert_id = expect[k];
            const char * src = (const char *) w->data + (size_t) expert_id * w->nb[2];
            const char * dstp = (const char *) staged->data + (size_t) k * staged->nb[2];
            CHECK(std::memcmp(src, dstp, plane_bytes) == 0,
                  "staged plane %d != source expert %d (%zu bytes)", k, expert_id, plane_bytes);
        }
        std::printf("all %d staged planes byte-identical to source experts\n", n_union);
    }

    // (4) remapping original ids through ids_sorted recovers the same experts
    {
        const int32_t * s = (const int32_t *) ids_sorted->data;
        for (size_t t = 0; t < (size_t) n_tokens; ++t) {
            for (size_t e = 0; e < (size_t) n_used; ++e) {
                const int32_t orig = ids_host[t * n_used + e];
                if (orig < 0 || orig >= (int32_t) n_expert) {
                    continue; // masked
                }
                // Find orig in the union; that index is the compact expert slot.
                int64_t slot = -1;
                for (int64_t k = 0; k < n_union; ++k) {
                    if (s[k] == orig) { slot = k; break; }
                }
                CHECK(slot >= 0, "id %d not found in union", orig);
                // The staged plane at `slot` must equal source expert `orig`.
                const size_t plane_bytes = (size_t) w->ne[1] * w->nb[1];
                const char * src = (const char *) w->data + (size_t) orig * w->nb[2];
                const char * dstp = (const char *) staged->data + (size_t) slot * staged->nb[2];
                CHECK(std::memcmp(src, dstp, plane_bytes) == 0,
                      "remap mismatch token %zu expert slot %zu (orig %d)", t, e, orig);
            }
        }
        std::printf("id remap recovers the correct expert for every routed slot\n");
    }

    // (5) GGML_OP_MOE_REMAP_IDS produces the same slots as the reference union
    {
        const int32_t * r = (const int32_t *) remapped->data;
        for (size_t t = 0; t < (size_t) n_tokens; ++t) {
            for (size_t e = 0; e < (size_t) n_used; ++e) {
                const int32_t orig = ids_host[t * n_used + e];
                const int64_t idx = t * n_used + e;
                int32_t expect_slot = -1;
                if (orig >= 0 && orig < (int32_t) n_expert) {
                    for (int64_t k = 0; k < n_union; ++k) {
                        if (expect[k] == orig) { expect_slot = (int32_t) k; break; }
                    }
                }
                CHECK(r[idx] == expect_slot,
                      "remap[%zu] = %d, expected %d (orig %d)",
                      (size_t) idx, r[idx], expect_slot, orig);
            }
        }
        std::printf("GGML_OP_MOE_REMAP_IDS slots match the reference union\n");
    }

    // Dedup effectiveness, for the record.
    {
        const int64_t n_ids = n_used * n_tokens;
        std::printf("dedup: %lld ids -> %d unique (%.0f%%)\n",
                    (long long) n_ids, n_union, 100.0 * n_union / n_ids);
    }

    ggml_backend_buffer_free(in_buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("PASS\n");
    return 0;
}
