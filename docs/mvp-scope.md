# MVP Scope: CPU-Side Expert Gather + GPU Standard Matmul

> **Date:** 2026-09-10
> **Goal:** Prove CPU gather + GPU standard matmul works and is faster than fork's sync-heavy ggml_cuda_mul_mat_id

---

## MVP Definition

**In scope (must work):**
1. New op `GGML_OP_MOE_EXPERT_GATHER` — CPU gathers expert weights into a compact tensor
2. CPU dispatch — reads expert indices, copies expert weights from RAM to output tensor
3. Modified CUDA dispatch — standard matmul on gathered weights (no row gathering)
4. Graph wiring — replace `build_lora_mm_id` call with gather+matmul sequence for MoE path

**Out of scope (defer to later):**
- Per-layer rolling staging buffer (Gap 1) — MVP uses a simple inline buffer
- Hot vs cold expert routing (Gap 2) — MVP reads all experts from RAM
- CUDA stream overlap (Gap 3) — MVP is synchronous but avoids sync points
- Tensor shape handling for all MoE paths (Gap 4) — MVP handles `up` path only (gate_up can use same code)
- LoRA adapter integration (Gap 5) — MVP works on base model only
- CUDA graph integration (Gap 6) — MVP disables graphs for MoE layers
- Error handling (Gap 7) — basic validation only
- Performance validation hooks (Gap 8) — manual benchmarking only

---

## MVP Architecture

```
Before (fork's current):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → ggml_mul_mat_id() [CPU node]
        → CUDA dispatch: ggml_cuda_mul_mat_id()
          → [SYNC] Copy ids to host
          → [SYNC] Build ids_to_sorted
          → [GPU] get_rows_cuda: gather src1 rows
          → [PER-EXPERT] cache check + matmul
          → [SYNC] Scatter results

After (MVP):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → ggml_moe_expert_gather() [CPU node]  ← NEW
        → CPU reads expert indices from ids tensor
        → CPU copies expert weights from RAM to output tensor
      → ggml_mul_mat() [GPU node]  ← standard matmul
        → GPU reads gathered weights (no gathering needed)
        → No per-layer sync for index lookup
```

---

## Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `ggml/include/ggml.h` | Add `ggml_moe_expert_gather()` API | ~15 |
| `ggml/src/ggml.c` | Implement tensor constructor | ~10 |
| `ggml/src/ggml-cpu/ops.cpp` | Implement CPU forward | ~30 |
| `ggml/src/ggml-cpu/ops.cpp` | Register dispatch | ~5 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Add CUDA dispatch for gathered weights | ~40 |
| `src/llama-graph.cpp` | Replace `build_lora_mm_id` with gather+matmul | ~10 |

**Total new code: ~100 lines**
**Total modified code: ~50 lines**

---

## MVP Implementation Order

1. **Step 1:** Add `ggml_moe_expert_gather()` API + tensor constructor
2. **Step 2:** Implement CPU forward (read ids, copy expert weights)
3. **Step 3:** Implement CUDA dispatch (standard matmul on gathered tensor)
4. **Step 4:** Wire into `build_lora_mm_id()` in `llama-graph.cpp`
5. **Step 5:** Test with a MoE model (e.g., Qwen3-MoE)

---

## MVP Success Criteria

1. **Correctness:** Model produces same output as fork's `ggml_cuda_mul_mat_id()`
2. **Performance:** >= 25% faster decode on a MoE model (target: match or beat PR #24524's +25%)
3. **No regressions:** Non-MoE models still work, LoRA still works

---

## Deferred to Post-MVP

| Feature | Why deferred | Effort |
|---------|--------------|--------|
| Per-layer rolling staging | Requires buffer management, eviction logic | Medium |
| Hot/cold expert routing | Requires GPU cache integration | Medium |
| CUDA stream overlap | Requires event-based sync | High |
| All MoE path layouts | MVP handles up/gate; gate_up needs separate handling | Low |
| LoRA adapters | MVP on base model; LoRA adds complexity | Medium |
| CUDA graph support | MVP disables graphs for MoE layers | High |
| Comprehensive error handling | Basic validation is enough for MVP | Low |
| Performance hooks | Manual benchmarking is enough for MVP | Medium |

---

## Key Decision Points

1. **Staging buffer:** MVP uses inline CPU memory (no CUDA host pinned buffer). Simple, fast for testing.
2. **Expert source:** MVP reads all experts from RAM. Hot/cold routing is deferred.
3. **MoE paths:** MVP handles `up` path. `gate_up` uses same code (different tensor shape).
4. **LoRA:** MVP works on base model. LoRA adapters are not supported in MVP.
5. **CUDA graphs:** MVP disables graphs for MoE layers. Re-enable in post-MVP.
6. **Sync points:** MVP eliminates sync points (core value prop) but doesn't overlap CPU/GPU work.

---

## Risks

1. **CPU gather bottleneck:** If CPU gather is slower than GPU gather + sync, MVP may not show improvement.
   - Mitigation: Measure CPU gather time vs fork's ggml_cuda_mul_mat_id time
2. **Tensor layout mismatch:** The fork's MoE paths have different tensor layouts.
   - Mitigation: MVP handles `up` path only; `gate_up` uses same code with different shape
3. **Memory bandwidth:** CPU gather requires reading expert weights from RAM.
   - Mitigation: For small models (<10B), RAM bandwidth is not the bottleneck

---

## Summary

The MVP is a focused proof-of-concept: CPU gathers expert weights, GPU runs standard matmul. No bells and whistles. If this works and is faster, we iterate on the deferred features.
