# MVP On-Ramp: Path to Final Implementation

> **Date:** 2026-09-10
> **Current state:** `mvp-cpu-moe-expert-gather` branch — CPU gather + graph wiring
> **Goal:** 4 incremental steps to the final implementation

---

## Step 0: MVP (✅ Complete)

**Branch:** `mvp-cpu-moe-expert-gather`
**Commit:** 42e9b9cfb

**What's done:**
- New `GGML_OP_MOE_EXPERT_GATHER` op
- CPU forward implementation
- Graph wiring (gather → reshape → repeat → matmul)

**What's deferred:** per-layer rolling, hot/cold routing, LoRA, CUDA graphs, performance hooks

---

## Step 1: Tensor Placement Overrides

**Branch:** `mvp-tensor-placement` (proposed)
**Risk:** Low
**Effort:** ~30 lines, 2 files

**What it does:**
- Ensure expert tensors are placed on CUDA host (pinned) memory
- CPU can read them for gather; GPU can DMA from them
- Uses existing `llm_ffn_exps_host_override()` + `TENSOR_READ_LAZY`

**Files to modify:**
- `common/common.h` — add staging buffer override pattern
- `src/llama-model-loader.cpp` — ensure expert tensors stay on CUDA_Host under mmap

**Why it matters:** Without pinned host memory, the CPU can't efficiently read expert weights for the gather operation. The fork's default placement puts experts on GPU device memory, which requires PCIe sync to read.

**Testability:** Can verify by checking tensor buffer types after model load.

---

## Step 2: Performance Validation Hooks

**Branch:** `mvp-performance-hooks` (proposed)
**Risk:** Low
**Effort:** ~40 lines, 2 files

**What it does:**
- Add timing instrumentation to measure CPU gather time
- Log per-layer timing breakdown (gather, matmul, total)
- Compare with fork's `ggml_cuda_mul_mat_id` timing

**Files to modify:**
- `ggml/src/ggml-cpu/ops.cpp` — timing hook in `ggml_compute_forward_moe_expert_gather`
- `ggml/src/ggml-cuda/ggml-cuda.cu` — timing hook in `ggml_cuda_mul_mat_id_staged`

**Why it matters:** Need to verify the MVP is actually faster than the fork's implementation. Without timing data, we can't measure progress.

**Testability:** Can verify by running with `GGML_DEBUG=INFO` and checking timing logs.

---

## Step 3: Staging Buffer Manager

**Branch:** `mvp-staging-buffer` (proposed)
**Risk:** Medium
**Effort:** ~150 lines, 2 files

**What it does:**
- `ggml_moe_staging_manager` struct: tracks active experts, manages LRU eviction, allocates pinned memory
- Per-layer rolling: reuses experts across layers, evicts least-recent-used
- Integrates with existing `ggml_cuda_expert_lru_cache`

**Files to modify:**
- `ggml/src/ggml-cuda/common.cuh` — add `ggml_moe_staging_manager` struct
- `ggml/src/ggml-cuda/ggml-cuda.cu` — integrate manager into CUDA context

**Why it matters:** Without per-layer rolling, every layer copies all experts from RAM, defeating the purpose of caching. This is the core optimization.

**Testability:** Can verify by checking staging buffer usage patterns in logs.

---

## Step 4: Hot/Cold Expert Routing + CUDA Stream Overlap

**Branch:** `mvp-hot-cold-routing` (proposed)
**Risk:** High
**Effort:** ~200 lines, 3 files

**What it does:**
- Hot experts: copy from GPU cache to staging (fast)
- Cold experts: read from RAM to staging (slower)
- CUDA stream overlap: gather runs asynchronously, matmul waits on event

**Files to modify:**
- `ggml/src/ggml-cpu/ops.cpp` — hot/cold routing in gather
- `ggml/src/ggml-cuda/ggml-cuda.cu` — stream overlap, event-based sync
- `ggml/src/ggml-cuda/common.cuh` — extend LRU cache for staging integration

**Why it matters:** This is the final optimization that makes the design competitive with the fork's implementation. Without it, we're just moving work from GPU to CPU without the caching benefit.

**Testability:** Can verify by comparing decode speed with and without hot/cold routing.

---

## Summary

| Step | Feature | Risk | Effort | Depends On |
|------|---------|------|--------|------------|
| 0 | MVP gather + wiring | ✅ Done | — | — |
| 1 | Tensor placement | Low | ~30 lines | Step 0 |
| 2 | Performance hooks | Low | ~40 lines | Step 0 |
| 3 | Staging buffer | Medium | ~150 lines | Step 1 |
| 4 | Hot/cold + stream overlap | High | ~200 lines | Steps 1-3 |

**Total additional effort:** ~420 lines across 8 files.

**Parallelization:** Steps 1 and 2 can be done in parallel. Steps 3 and 4 must be sequential (each depends on the previous).
