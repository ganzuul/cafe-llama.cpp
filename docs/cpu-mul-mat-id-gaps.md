# Gap Analysis: Storyboard vs Design Document

> **Date:** 2026-09-10
> **Purpose:** Identify gaps between the storyboard code blocks and the design document, flag friction points, and point to relevant documentation

---

## Gap 1: Per-Layer Rolling Staging Buffer

**Status:** ❌ MISSING from storyboard

**Design document says (Section 6.3):**
> The staging buffer must be reused across layers within a forward pass:
> - Layer 0: populate staging with experts 3, 7, 12
> - Layer 1: evict unused, add experts 5, 9
> - No PCIe transfer between layers if experts are already cached

**Storyboard shows:** A static staging buffer with no per-layer management.

**Friction point:** Without per-layer rolling, every layer copies all selected experts from RAM to the staging buffer, defeating the purpose of caching. The staging buffer needs:
1. A **set of active experts** per layer (computed from router output)
2. **Eviction logic** to free space for new experts
3. **Reuse detection** to skip experts already in the buffer

**Where to find docs:**
- `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_expert_lru_cache` (line 1414) — the existing LRU cache structure we can extend
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_pool_alloc` (line ~2020) — pool allocator for temporary GPU buffers
- Prior art: PR #24524's `CachedWeightProvider` — LRU eviction with usage tracking

**Storyboard fix needed:** Add a `moe_staging_manager` struct that:
- Tracks which experts are in the staging buffer
- Reuses experts across layers when possible
- Evicts least-recent-used experts when buffer is full
- Integrates with the existing `ggml_cuda_expert_lru_cache`

---

## Gap 2: Hot vs Cold Expert Routing

**Status:** ⚠️ PARTIALLY COVERED — missing implementation details

**Design document says (Section 6.2):**
> Expert weights must be accessible from both CPU and GPU:
> - **Hot experts:** Cached on GPU via `expert_cache.get_or_alloc()`
> - **Cold experts:** On CPU RAM (via `--cpu-moe` or `TENSOR_READ_LAZY`)
> - **Staging buffer:** CPU-accessible, populated by CPU, read by GPU

**Storyboard shows:** A single path for all experts, no distinction between hot and cold.

**Friction point:** The storyboard's `ggml_moe_expert_gather()` reads all expert weights from the same source. But hot experts should come from the GPU cache (fast, no CPU involvement), while cold experts need CPU RAM reads. The routing logic needs to:
1. Check if an expert is in the GPU cache (hot)
2. If hot, copy from cache to staging (GPU-to-GPU, fast)
3. If cold, read from RAM to staging (CPU-to-staging, slower)

**Where to find docs:**
- `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_expert_lru_cache::get_or_alloc()` (line 1464)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_mul_mat_id()` fallback path (line 1994+) — shows how host vs device weights are handled
- `src/llama-graph.cpp` — `split_exps` logic (line ~2130) — shows how hot/cold split is already done

**Storyboard fix needed:** In `ggml_compute_forward_moe_expert_gather()`, add:
```cpp
// For hot experts (in GPU cache): use expert_cache.get_or_alloc()
// For cold experts (on RAM): read directly from src0->data
// Route to appropriate source based on tensor buffer type
```

---

## Gap 3: CUDA Stream Overlap (Async Execution)

**Status:** ❌ MISSING from storyboard

**Design document says (Section 6.1):**
> Our design avoids these sync points by:
> - Keeping `mul_mat_id` on CPU (async from GPU perspective)
> - Using a pre-populated staging buffer on GPU (no PCIe transfer per layer)

**Storyboard shows:** CPU gather and GPU matmul as sequential operations, no stream overlap.

**Friction point:** The entire performance benefit depends on CPU gather overlapping with GPU's previous layer computation. If CPU and GPU are sequential, we add latency without benefit. The storyboard needs to show:
1. Which CUDA stream the CPU gather uses
2. How the CPU gather is launched asynchronously
3. How the GPU matmul knows when gather is complete

**Where to find docs:**
- `ggml/src/ggml-cuda/common.cuh` — `ggml_backend_cuda_context::stream()` (line ~1619) — how streams are managed
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_context::concurrent_stream_context` (line ~1617) — concurrent stream management
- Prior art: PR #25294's double-buffering — shows overlapping CPU I/O with GPU compute

**Storyboard fix needed:** Add stream management to the gather operation:
```cpp
// CPU gather should be launched on a specific CUDA stream
// with an event that signals completion to the matmul stream
cudaStream_t gather_stream = ctx->streams[device][stream_id];
cudaEvent_t gather_done;
cudaEventCreate(&gather_done);
cudaLaunchGatherKernel(gather_stream, ...);
cudaEventRecord(gather_done, gather_stream);
// Matmul waits on gather_done before starting
```

---

## Gap 4: Tensor Shape Handling for Different MoE Paths

**Status:** ⚠️ PARTIALLY COVERED — assumes single shape

**Design document says (Section 2.2):**
> **Tensor shapes:**
> - `w` (expert weights): `[n_embd, n_ff, n_expert]` for up/gate; `[n_ff, n_embd, n_expert]` for gate_up; `[n_embd, n_ff, n_expert]` for down
> - `cur` (input): `[n_embd, n_expert_used, n_tokens]`
> - `ids` (expert indices): `[n_expert_used, n_tokens]` (i32)
> - Output: `[n_ff, n_expert_used, n_tokens]` or `[n_embd, n_expert_used, n_tokens]`

**Storyboard shows:** `ggml_moe_expert_gather()` assumes `as->ne[0] * as->ne[1]` as the flattened dimension, but doesn't account for the different layouts.

**Friction point:** The fork has 4 different MoE paths (gate_up, up, gate, down), each with different tensor layouts. The gather op needs to handle:
1. `gate_up`: weights are `[n_ff*2, n_embd, n_expert]` — merged gate+up
2. `up`: weights are `[n_ff, n_embd, n_expert]`
3. `gate`: weights are `[n_ff, n_embd, n_expert]`
4. `down`: weights are `[n_embd, n_ff, n_expert]`

**Where to find docs:**
- `src/llama-graph.cpp` — `build_moe_ffn()` lines 2158-2294 — shows how each path calls `mm_id_exps`
- `ggml/src/ggml.c` — `ggml_mul_mat_id()` docblock (line 3362) — shows tensor shape conventions

**Storyboard fix needed:** Add shape handling to `ggml_moe_expert_gather()`:
```cpp
// Handle different tensor layouts:
// - gate_up: ne0 = n_ff*2, ne1 = n_embd
// - up/gate: ne0 = n_ff, ne1 = n_embd
// - down: ne0 = n_embd, ne1 = n_ff
// Output shape depends on input layout
```

---

## Gap 5: LoRA Adapter Integration

**Status:** ⚠️ PARTIALLY COVERED — placeholder only

**Design document says (Section 2.2):**
> `build_lora_mm_id` also handles LoRA adapters with nested `ggml_mul_mat_id` chains

**Storyboard shows:** LoRA handling as a comment placeholder in `build_lora_mm_id()`.

**Friction point:** The fork's LoRA support builds nested `ggml_mul_mat_id` chains for LoRA weights. Our staged approach needs to handle LoRA adapters with the new gather+matmul sequence. The LoRA weights also have expert tensors that need gathering.

**Where to find docs:**
- `src/llama-graph.cpp` — `build_lora_mm_id()` lines 1565-1578 — LoRA adapter handling
- `src/llama-graph.cpp` — `loras` member — how LoRA adapters are stored

**Storyboard fix needed:** Extend `build_lora_mm_id()` to handle LoRA with the new gather op:
```cpp
// For each LoRA adapter:
// 1. Gather LoRA expert weights (same as main weights)
// 2. Run LoRA matmul on gathered weights
// 3. Add to result
```

---

## Gap 6: CUDA Graph Integration

**Status:** ⚠️ PARTIALLY COVERED — missing details

**Design document says (Section 6.4):**
> The fork disables CUDA graphs for sync-requiring MUL_MAT_ID nodes (line 2586). Our design must either:
> - Be compatible with CUDA graphs (no sync points)
> - Or accept that CUDA graphs are disabled for MoE layers

**Storyboard shows:** Removal of sync-based graph disabling, but no actual CUDA graph capture/replay code.

**Friction point:** CUDA graphs require all operations to be deterministic and non-branching. The CPU gather op is non-deterministic from the GPU's perspective (it runs on a different thread, on different hardware). The storyboard needs to address:
1. Can CUDA graphs capture the gather+matmul sequence?
2. If not, how do we minimize the performance impact of disabling graphs?
3. Can we use CUDA streams to overlap gather with graph replay?

**Where to find docs:**
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_graph_update_required()` (line ~2608) — how graphs are managed
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_graph_evaluate_and_capture()` (line ~4035) — graph capture/replay
- Prior art: PR #18958 — explains why MUL_MAT_ID disables CUDA graphs

**Storyboard fix needed:** Add CUDA graph handling to the gather+matmul sequence:
```cpp
// Option 1: Capture gather+matmul as a single graph
// Option 2: Disable graphs for MoE layers (simpler, but less optimal)
// Option 3: Use CUDA streams to overlap gather with graph replay
```

---

## Gap 7: Error Handling and Validation

**Status:** ❌ MISSING from storyboard

**Design document says (Section 6.2):**
> Expert weights must be accessible from both CPU and GPU

**Storyboard shows:** No error handling for invalid expert IDs, missing tensors, or buffer overflow.

**Friction point:** The gather op needs to handle:
1. Invalid expert IDs (router outputs IDs outside valid range)
2. Missing expert tensors (model loading failed)
3. Buffer overflow (more experts than staging buffer capacity)
4. GPU OOM (staging buffer allocation fails)

**Where to find docs:**
- `ggml/src/ggml-cpu/ops.cpp` — existing error handling patterns (GGML_ASSERT, GGML_ABORT)
- `ggml/src/ggml-cuda/common.cuh` — CUDA error handling (CUDA_CHECK macro)

**Storyboard fix needed:** Add error handling throughout:
```cpp
// Validate expert IDs are in range
// Check staging buffer capacity before writing
// Handle GPU OOM gracefully (fallback to non-staged path)
// Log warnings for invalid expert IDs
```

---

## Gap 8: Performance Validation

**Status:** ❌ MISSING from storyboard

**Design document says (Section 6.1):**
> Our design avoids these sync points by keeping `mul_mat_id` on CPU

**Storyboard shows:** No performance validation or benchmarking code.

**Friction point:** The entire design hinges on the assumption that CPU gather is faster than GPU-side gathering + sync. This needs validation:
1. How long does CPU gather take vs GPU gather?
2. Is CPU gather hidden by GPU previous layer computation?
3. What's the throughput impact of the staging buffer?

**Where to find docs:**
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_mul_mat_id_needs_sync()` (line 1875) — sync point detection
- Prior art: PR #24524 benchmark results — +25% on GLM-5.1 754B
- Prior art: PR #25294 benchmark results — 4.7 tok/s on M1 MacBook

**Storyboard fix needed:** Add benchmarking hooks:
```cpp
// Measure CPU gather time
// Measure GPU matmul time
// Compare with fork's ggml_cuda_mul_mat_id timing
// Log per-layer timing breakdown
```

---

## Summary of Gaps

| Gap | Severity | Storyboard Status | Fix Complexity |
|-----|----------|-------------------|----------------|
| 1. Per-layer rolling staging | **HIGH** | Missing | Medium |
| 2. Hot vs cold expert routing | **HIGH** | Partial | Medium |
| 3. CUDA stream overlap | **HIGH** | Missing | High |
| 4. Tensor shape handling | **MEDIUM** | Partial | Low |
| 5. LoRA adapter integration | **MEDIUM** | Placeholder | Medium |
| 6. CUDA graph integration | **MEDIUM** | Partial | High |
| 7. Error handling | **LOW** | Missing | Low |
| 8. Performance validation | **MEDIUM** | Missing | Medium |

**Total:** 8 gaps identified, 3 HIGH severity, 4 MEDIUM, 1 LOW.

---

## Recommended Next Steps

1. **Fix HIGH gaps first** (1, 2, 3) — these are the core of the design
2. **Address MEDIUM gaps** (4, 5, 6, 8) — important for correctness and performance
3. **Handle LOW gaps** (7) — standard error handling, can be done later

**Priority order:**
1. Per-layer rolling staging (Gap 1) — without this, the design doesn't work
2. Hot vs cold routing (Gap 2) — without this, we don't leverage the GPU cache
3. CUDA stream overlap (Gap 3) — without this, we add latency instead of reducing it
4. Tensor shape handling (Gap 4) — without this, only one MoE path works
5. LoRA integration (Gap 5) — without this, LoRA models don't work
6. CUDA graph integration (Gap 6) — without this, we lose CUDA graph optimization
7. Error handling (Gap 7) — without this, crashes on edge cases
8. Performance validation (Gap 8) — without this, we can't verify the design works
