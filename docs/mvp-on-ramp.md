# MVP On-Ramp: Path to Final Implementation

> **Date:** 2026-09-10
> **Current state:** All 4 steps implemented ✅
> **Goal:** 4 incremental steps to the final implementation

---

## Step 0: MVP (✅ Complete)

**Branch:** `mvp-cpu-moe-expert-gather`
**Commit:** 42e9b9cfb

**What's done:**
- New `GGML_OP_MOE_EXPERT_GATHER` op
- CPU forward implementation
- Graph wiring (gather → reshape → repeat → matmul)

**Files modified:**
- `ggml/include/ggml.h` — new op enum + API declaration
- `ggml/src/ggml.c` — tensor constructor + op symbol + GGML_OP_COUNT update
- `ggml/src/ggml-cpu/ops.h` — forward declaration
- `ggml/src/ggml-cpu/ops.cpp` — CPU forward implementation
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU dispatch (3 locations)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — CUDA compatibility check
- `src/llama-graph.cpp` — graph wiring

---

## Step 1: Tensor Placement Overrides (✅ Complete)

**Branch:** `mvp-tensor-placement`
**Commit:** 91e9abd1b

**What's done:**
- Added `llm_ffn_exps_staging_override()` — ensures expert tensors are placed on CUDA host (pinned) memory
- Added `llm_add_n_staging_moe_overrides()` — per-layer staging overrides
- Uses `common_host_buffer_type()` which returns CUDA_Host when available, falling back to CPU buffer type

**Files modified:**
- `common/common.h` — staging buffer override functions

**Lesson learned:** `common_host_buffer_type()` already returns CUDA_Host when CUDA is available, so no CUDA-specific header includes needed.

---

## Step 2: Performance Validation Hooks (✅ Complete)

**Branch:** `mvp-performance-hooks`
**Commit:** 9c4d9f382

**What's done:**
- Added timing instrumentation to `ggml_compute_forward_moe_expert_gather` — logs per-layer gather time in microseconds
- Added timing instrumentation to `ggml_cuda_mul_mat_id` — logs per-layer CUDA time + cumulative total
- Timing is logged via `GGML_LOG_DEBUG` when `GGML_DEBUG >= 20`

**Files modified:**
- `ggml/src/ggml-cpu/ops.cpp` — timing in `ggml_compute_forward_moe_expert_gather`
- `ggml/src/ggml-cuda/ggml-cuda.cu` — timing in `ggml_cuda_mul_mat_id`

**Output format:**
```
moe_expert_gather: ne01=16384 ne1=8 ne2=1024 time=12345 us (12.35 ms)
mul_mat_id: ne0=16384 ne1=8192 ne2=1024 cuda_us=45678 total_cuda_us=45678
```

---

## Step 3: Staging Buffer Manager (✅ Complete)

**Branch:** `mvp-hot-cold-routing`
**Commit:** bbcd98d59

**What's done:**
- Added `ggml_moe_staging_manager` struct — manages pinned memory staging buffers for hot/cold expert routing
- Hot experts: use cached device pointer from staging buffer (fast)
- Cold experts: read from CPU host and copy to staging buffer (slower)
- LRU eviction when staging buffer is full
- Integrates with existing `ggml_cuda_expert_lru_cache`

**Files modified:**
- `ggml/src/ggml-cuda/common.cuh` — add `ggml_moe_staging_manager` struct
- `ggml/src/ggml-cuda/ggml-cuda.cu` — use staging manager in expert loop

**Lesson learned:** Linear scan for LRU eviction is sufficient for small N experts. The staging buffer size is configurable via `max_mb` parameter (default 1024 MB).

---

## Step 4: Hot/Cold Expert Routing + CUDA Stream Overlap (✅ Complete)

**Branch:** `mvp-hot-cold-routing`
**Commit:** bbcd98d59

**What's done:**
- Modified `ggml_cuda_mul_mat_id` expert loop to use staging buffer:
  - **Hot experts**: use cached device pointer (fast, stays on GPU)
  - **Cold experts**: read from CPU host, copy to staging buffer (slower, PCIe)
- CUDA stream overlap: gather runs asynchronously while GPU computes

**Files modified:**
- `ggml/src/ggml-cuda/common.cuh` — staging buffer manager
- `ggml/src/ggml-cuda/ggml-cuda.cu` — hot/cold routing in expert loop

**Lesson learned:** The staging buffer manager handles both hot and cold expert routing in a single pass. The `get_staging()` function returns a pointer to the staging entry, which can be used for both CPU and GPU access.

---

## Lessons Learned

### 1. `common_host_buffer_type()` is the right choice for staging
**Lesson:** Instead of using `ggml_backend_cuda_host_buffer_type()` directly (which requires CUDA header includes), use `common_host_buffer_type()` which returns CUDA_Host when available and falls back to CPU buffer type.

### 2. Linear scan for LRU is sufficient for small N experts
**Lesson:** The staging buffer manager uses a linear scan for LRU eviction, which is O(N) but sufficient for small N experts (typically 8-64). For larger N, consider using a hash map with LRU list.

### 3. Staging buffer size should be configurable
**Lesson:** The staging buffer size is configurable via `max_mb` parameter (default 1024 MB). This allows tuning for different GPU sizes and expert counts.

### 4. Performance hooks should be low-cost
**Lesson:** Timing instrumentation using `ggml_time_us()` is low-cost and can be enabled/disabled via `GGML_DEBUG` level. This avoids conditional compilation and keeps the code clean.

### 5. Graph wiring requires careful shape handling
**Lesson:** The gather output shape `[ne0*ne1, n_exp, n_tok, 1]` requires reshaping to `[ne0, ne1, n_exp*n_tok, 1]` and repeating `cur` to `[ne1, n_exp, n_tok, 1]` for the matmul to work correctly. This was the most challenging part of the MVP.

---

## Summary

| Step | Feature | Risk | Effort | Depends On | Status |
|------|---------|------|--------|------------|--------|
| 0 | MVP gather + wiring | — | ~170 lines | — | ✅ Done |
| 1 | Tensor placement | Low | ~12 lines | Step 0 | ✅ Done |
| 2 | Performance hooks | Low | ~20 lines | Step 0 | ✅ Done |
| 3 | Staging buffer | Medium | ~128 lines | Step 1 | ✅ Done |
| 4 | Hot/cold + stream overlap | High | ~24 lines | Steps 1-3 | ✅ Done |

**Total:** ~230 lines across 8 files.

**Next steps:**
1. Run inference tests to verify correctness
2. Benchmark decode speed vs fork's implementation
3. Tune staging buffer size based on benchmarks
4. Consider adding per-layer rolling optimization (deferred)
