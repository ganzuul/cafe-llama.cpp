# CPU `mul_mat_id` + GPU Expert Cache: Entry Point Analysis

> **Date:** 2026-09-10
> **Fork:** `workstreams/llama.cpp-mtp-slot-state` (commit 8621747a2)
> **Goal:** Implement CPU-side `mul_mat_id` for expert weight lookup, GPU expert cache for hot experts, per-layer rolling staging

---

## 1. Architecture Overview

The fork's current design moves `mul_mat_id` entirely to the GPU (via `ggml_cuda_mul_mat_id`), which the prior-art report identifies as causing ~3× decode regression due to per-layer sync points on cache misses. Our design inverts this: keep `mul_mat_id` on the CPU for index lookup and weight gathering, then dispatch to the GPU for the actual matmul with cached expert weights.

---

## 2. Entry Point Map

### 2.1 Graph Construction: `src/llama-graph.cpp`

**File:** `src/llama-graph.cpp`

**Function:** `llm_graph_context::build_moe_ffn()` — line ~1952

This is the **primary hook**. It constructs the MoE FFN computation graph for a layer:

```
build_moe_ffn(
    const llama_layer & layer,
    ggml_tensor       * cur,
    int64_t             n_embd_head,
    int64_t             n_head,
    int64_t             n_head_kv,
    int                 il)
```

**Key sub-routines:**

| Line | Function/Block | Purpose |
|------|---------------|---------|
| ~2130 | `split_exps` check | Determines if hot/cold split is active |
| ~2134-2140 | `ids_hot` / `ids_cold` creation | Uses `ggml_moe_branch_ids()` to split router IDs into hot/cold branches |
| ~2143-2149 | `mm_id_exps` lambda | The **critical hook** — calls `build_lora_mm_id()` for hot/cold branches |
| ~2158 | `gate_up` call | First MoE gate: `mm_id_exps(gate_up_exps, up_exps_cold, cur, up_exps_s)` |
| ~2177 | `up` call | Second MoE up: `mm_id_exps(up_exps, up_exps_cold, cur, up_exps_s)` |
| ~2190 | `gate` call | Third MoE gate: `mm_id_exps(gate_exps, gate_exps_cold, cur, gate_exps_s)` |
| ~2294 | `down` call | Final MoE down: `mm_id_exps(down_exps, down_exps_cold, cur, down_exps_s)` |

**Current `mm_id_exps` lambda (line 2143):**
```cpp
auto mm_id_exps = [&](ggml_tensor * w_hot, ggml_tensor * w_cold, ggml_tensor * x, ggml_tensor * s) {
    if (!split_exps) {
        return build_lora_mm_id(w_hot, x, selected_experts, s);
    }
    ggml_tensor * o_hot  = build_lora_mm_id(w_hot,  x, ids_hot,  s, n_hot);
    ggml_tensor * o_cold = build_lora_mm_id(w_cold, x, ids_cold, s, n_cold);
    return ggml_add(ctx0, o_hot, o_cold);
};
```

**For each of the 4 MoE paths (gate_up, up, gate, down),** `build_lora_mm_id` is called with the hot expert weights and the corresponding hot/cold IDs.

---

### 2.2 `build_lora_mm_id`: The Node Creator

**File:** `src/llama-graph.cpp`
**Line:** ~1542

```cpp
ggml_tensor * llm_graph_context::build_lora_mm_id(
    ggml_tensor * w,    // expert weights [cols, rows, n_expert]
    ggml_tensor * cur,  // input [cols, n_expert_used, n_tokens]
    ggml_tensor * ids,  // expert indices [n_expert_used, n_tokens]
    ggml_tensor * w_s,  // expert scaling factors (optional)
    int32_t       mask_from) const
```

**What it does:**
1. Creates `ggml_mul_mat_id(ctx0, w, cur, ids)` — the core indirect matmul node
2. Optionally sets mask via `ggml_mul_mat_id_set_mask_from(res, mask_from)`
3. Applies expert scaling if `w_s` is provided
4. Loops through LoRA adapters, building nested `ggml_mul_mat_id` chains

**Entry point for CPU/GPU split:** This is where we would replace the single `ggml_mul_mat_id` with our new CPU-side weight gathering + GPU-side batched matmul sequence.

**Tensor shapes:**
- `w` (expert weights): `[n_embd, n_ff, n_expert]` for up/gate; `[n_ff, n_embd, n_expert]` for gate_up; `[n_embd, n_ff, n_expert]` for down
- `cur` (input): `[n_embd, n_expert_used, n_tokens]`
- `ids` (expert indices): `[n_expert_used, n_tokens]` (i32)
- Output: `[n_ff, n_expert_used, n_tokens]` or `[n_embd, n_expert_used, n_tokens]`

---

### 2.3 CPU Backend Dispatch: `ggml/src/ggml-cpu/ops.cpp`

**File:** `ggml/src/ggml-cpu/ops.cpp`

| Line | Function | Purpose |
|------|---------|---------|
| 12027 | `ggml_compute_forward_moe_branch_ids()` | CPU implementation of `GGML_OP_MOE_BRANCH_IDS` — maps global expert IDs to local branch indices |
| 4386 | `forward_mul_mat_id()` (in repack.cpp) | CPU implementation of `GGML_OP_MUL_MAT_ID` — groups rows by expert, reads from RAM |

**CPU `forward_mul_mat_id` (in `ggml/src/ggml-cpu/repack.cpp`, line 4386):**
- Groups rows by source0 matrix (expert)
- Handles `mask_from` to skip masked slots
- Reads expert weights from `src0->data` (CPU RAM)
- Outputs to `dst->data`

**This is the entry point for the CPU-side `mul_mat_id` weight lookup.** We would modify this function to:
1. Look up expert IDs in the GPU expert cache
2. Fetch cached expert weights (if hit) from the staging buffer
3. Fall back to RAM (if miss)

---

### 2.4 CUDA Backend Dispatch: `ggml/src/ggml-cuda/ggml-cuda.cu`

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

| Line | Function | Purpose |
|------|---------|---------|
| 1875 | `ggml_cuda_mul_mat_id_needs_sync()` | Determines if the MUL_MAT_ID node requires stream synchronization |
| 1908 | `ggml_cuda_mul_mat_id()` | **Primary CUDA entry point** for `GGML_OP_MUL_MAT_ID` |
| 2122 | `ggml_cuda_op_moe_branch_ids()` | CUDA dispatch for `GGML_OP_MOE_BRANCH_IDS` |
| 2586 | CUDA graph check | Disables CUDA graphs for sync-requiring MUL_MAT_ID nodes |

**`ggml_cuda_mul_mat_id()` (line 1908) — current flow:**
```
1. Check if src0 (expert weights) is on host
2. Try fast paths:
   - MMVQ (quantized, small batch) → ggml_cuda_mul_mat_vec_q
   - AMD fast path → ggml_cuda_mul_mat_vec_f
   - MQ (quantized, larger) → ggml_cuda_mul_mat_q
   - MF (f32, larger) → ggml_cuda_mul_mat_f
3. Fallback path (always syncs):
   - Copy ids to host, synchronize
   - Build ids_to_sorted / ids_from_sorted on host
   - Launch get_rows_cuda to gather src1 rows by expert
   - For each expert:
     - Slice src0, check expert_cache.get_or_alloc()
     - Copy from host if not cached
     - Launch ggml_cuda_mul_mat for this expert
   - Launch get_rows_cuda to scatter results
```

**Key insight:** The fallback path (lines 1952+) is the **synchronization bottleneck**. It synchronizes the stream for:
- Copying ids from GPU to host
- Building sorted indices
- Per-expert cache lookups (if src0 is on host)
- Per-expert matmul dispatches

**Entry point for GPU expert cache integration:** The `ctx.expert_cache.get_or_alloc()` call at line ~2030 is where the LRU cache is consulted. We would enhance this to also handle CPU-side weight fetching.

---

### 2.5 GPU Expert Cache: `ggml/src/ggml-cuda/common.cuh`

**File:** `ggml/src/ggml-cuda/common.cuh`

| Line | Struct/Function | Purpose |
|------|----------------|---------|
| 1414 | `struct ggml_cuda_expert_lru_cache` | LRU cache for expert weights on GPU |
| 1445 | `init()` | Initializes cache with size from `GGML_CUDA_MOE_CACHE_MB` env var |
| 1464 | `get_or_alloc()` | **Cache entry point** — returns cached dev_ptr or allocates new |
| 1507 | `clear()` | Frees all cached entries |

**`get_or_alloc()` (line 1464):**
```cpp
void * get_or_alloc(const void * host_ptr, size_t size, cudaStream_t stream, bool & out_allocated) {
    // 1. Increment step counter
    // 2. Check if host_ptr is already cached → return dev_ptr
    // 3. If cache full, evict LRU entry
    // 4. cudaMalloc for new entry
    // 5. cudaMemcpyAsync host → device
    // 6. Store in entries map
    // 7. Return dev_ptr
}
```

**This is the entry point for GPU expert cache hits.** The cache is keyed by `host_ptr` (the host-side expert weight address).

**Per-context storage:** `ggml_backend_cuda_context` (line 1609) has `ggml_cuda_expert_lru_cache expert_cache` as a member.

---

### 2.6 Tensor Placement: `common/common.h` + `src/llama-model-loader.cpp`

**File:** `common/common.h`

| Line | Definition | Purpose |
|------|-----------|---------|
| 1128 | `LLM_FFN_EXPS_REGEX` | Pattern `\.ffn_(up|down|gate|gate_up)_(ch|)exps` for expert tensor matching |
| 1130 | `llm_ffn_exps_block_regex(int idx)` | Per-layer regex `blk\.N\.ffn_.*exps` |
| 1134 | `llm_ffn_exps_cpu_override()` | Override expert tensors to CPU buffer type |
| 1142 | `llm_ffn_exps_host_override()` | Override expert tensors to CUDA host buffer type |
| 1143 | `llm_add_n_host_moe_overrides()` | Add N per-layer host overrides |

**File:** `src/llama-model-loader.cpp`

| Line | Purpose |
|------|---------|
| 1184-1195 | Tensor buffer type override application |
| 1221 | CUDA_Host override handling with mmap |
| 1295 | `TENSOR_READ_LAZY` for on-demand tensor loading |

**Entry point for tensor placement:** The `llm_model_tensor_buft_override` structure at `common/common.h:349` controls which tensors go to which backend. We would add a new override pattern for expert staging tensors.

---

## 3. Data Flow Summary

### Current Fork Flow (GPU-side `mul_mat_id`):
```
llama-graph.cpp::build_moe_ffn()
  → mm_id_exps()
    → build_lora_mm_id()
      → ggml_mul_mat_id() [CPU node]
        → CUDA dispatch: ggml_cuda_mul_mat_id()
          → ids: GPU → host sync
          → ids_to_sorted: build on host
          → get_rows_cuda: gather src1 rows
          → for each expert:
              → expert_cache.get_or_alloc() [LRU hit/miss]
              → ggml_cuda_mul_mat() [per-expert matmul]
          → get_rows_cuda: scatter results
```

### Proposed Flow (CPU-side `mul_mat_id` + GPU expert cache):
```
llama-graph.cpp::build_moe_ffn()
  → mm_id_exps() [MODIFIED]
    → [NEW] CPU-side expert weight gathering:
        → ggml_moe_branch_ids() [hot/cold split]
        → CPU reads expert indices from GPU cache
        → CPU gathers expert weights from staging buffer
    → [NEW] GPU-side batched matmul:
        → ggml_mul_mat() [standard matmul, not indirect]
        → Uses cached expert weights from staging buffer
```

---

## 4. Entry Points Summary

| # | File | Line | Function/Block | Role |
|---|------|------|----------------|------|
| 1 | `src/llama-graph.cpp` | ~1952 | `build_moe_ffn()` | **Primary graph construction hook** |
| 2 | `src/llama-graph.cpp` | ~2143 | `mm_id_exps` lambda | **Hot/cold branch dispatch** |
| 3 | `src/llama-graph.cpp` | ~1542 | `build_lora_mm_id()` | **Node creation** (replace with CPU+GPU sequence) |
| 4 | `ggml/src/ggml-cpu/ops.cpp` | 12027 | `ggml_compute_forward_moe_branch_ids()` | CPU dispatch for ID mapping |
| 5 | `ggml/src/ggml-cpu/repack.cpp` | 4386 | `forward_mul_mat_id()` | **CPU weight lookup entry** |
| 6 | `ggml/src/ggml-cuda/ggml-cuda.cu` | 1908 | `ggml_cuda_mul_mat_id()` | **Current CUDA dispatch** (modify or replace) |
| 7 | `ggml/src/ggml-cuda/ggml-cuda.cu` | 1875 | `ggml_cuda_mul_mat_id_needs_sync()` | Sync point detection |
| 8 | `ggml/src/ggml-cuda/common.cuh` | 1464 | `get_or_alloc()` | **GPU cache hit entry** |
| 9 | `common/common.h` | 1128-1144 | `LLM_FFN_EXPS_REGEX` + overrides | **Tensor placement config** |
| 10 | `ggml/include/ggml.h` | 1454 | `ggml_mul_mat_id()` + `ggml_moe_branch_ids()` | **API entry points** |

---

## 5. Modification Plan

### Phase 1: CPU-side `mul_mat_id` weight gathering

**Files to modify:**
1. `ggml/include/ggml.h` — Add new op or extend `ggml_mul_mat_id` params
2. `ggml/src/ggml.c` — Implement new op
3. `ggml/src/ggml-cpu/ops.cpp` — Implement CPU forward
4. `ggml/src/ggml-cpu/repack.cpp` — Modify `forward_mul_mat_id` for CPU weight lookup

**Key changes:**
- CPU reads expert indices from a GPU-resident tensor (via `ggml_backend_get_tensor`)
- CPU looks up expert weights in a shared staging buffer (CPU-accessible)
- CPU outputs gathered expert weights as a new tensor

### Phase 2: GPU expert cache integration

**Files to modify:**
1. `ggml/src/ggml-cuda/common.cuh` — Extend `ggml_cuda_expert_lru_cache` for CPU-to-GPU staging
2. `ggml/src/ggml-cuda/ggml-cuda.cu` — Modify `ggml_cuda_mul_mat_id` to use staged weights
3. `ggml/src/ggml-cuda/mmid.cu` — Modify kernels for staged input

**Key changes:**
- GPU reads expert weights from staging buffer instead of RAM
- GPU cache hit/miss logic updated for staging buffer
- Per-layer rolling: staging buffer reused across layers

### Phase 3: Graph construction wiring

**Files to modify:**
1. `src/llama-graph.cpp` — `build_moe_ffn()`, `mm_id_exps`, `build_lora_mm_id`
2. `common/common.h` — New tensor placement overrides
3. `src/llama-model-loader.cpp` — Tensor loading for staging buffer

**Key changes:**
- Replace `build_lora_mm_id` call with CPU gather + GPU matmul sequence
- Wire staging buffer tensor placement
- Handle per-layer rolling of staging buffer

---

## 6. Constraints & Considerations

### 6.1 Sync Point Avoidance
The fork's `ggml_cuda_mul_mat_id` fallback path synchronizes the stream at multiple points (lines 1994, 2000, 2003). Our design must avoid these sync points by:
- Keeping `mul_mat_id` on CPU (async from GPU perspective)
- Using a pre-populated staging buffer on GPU (no PCIe transfer per layer)

### 6.2 Tensor Placement
Expert weights must be accessible from both CPU and GPU:
- **Hot experts:** Cached on GPU via `expert_cache.get_or_alloc()`
- **Cold experts:** On CPU RAM (via `--cpu-moe` or `TENSOR_READ_LAZY`)
- **Staging buffer:** CPU-accessible, populated by CPU, read by GPU

### 6.3 Per-Layer Rolling
The staging buffer must be reused across layers within a forward pass:
- Layer 0: populate staging with experts 3, 7, 12
- Layer 1: evict unused, add experts 5, 9
- No PCIe transfer between layers if experts are already cached

### 6.4 CUDA Graph Compatibility
The fork disables CUDA graphs for sync-requiring MUL_MAT_ID nodes (line 2586). Our design must either:
- Be compatible with CUDA graphs (no sync points)
- Or accept that CUDA graphs are disabled for MoE layers

---

## 7. Reference: Prior Art Pitfalls

From `ftq-recipe/docs/prior-art-expert-offload.md`:

> **Critical prior-art pitfall** (from PR #20757 discussion): Moving `mul_mat_id` to the GPU and copying weights on cache miss adds ~3× decode regression because every miss becomes a synchronous PCIe transfer.

Our design avoids this by:
1. Keeping `mul_mat_id` on the CPU (index lookup is fast, no PCIe)
2. Pre-populating the GPU staging buffer (no per-layer PCIe)
3. GPU reads from staging buffer (fast local memory, no PCIe)
