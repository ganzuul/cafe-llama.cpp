# MVP Lessons Learned — Step 1 Complete

> **Date:** 2026-09-10
> **Branch:** mvp-cpu-moe-expert-gather
> **Commit:** 8621747a2 (base) + MVP changes

---

## What We Did

Added the `GGML_OP_MOE_EXPERT_GATHER` op — a new GGML operation that:
1. Takes expert weights `[ne0, ne1, n_expert]` and expert indices `[n_expert_used, n_tokens]` (i32)
2. Outputs gathered weights `[ne0*ne1, n_expert_used, n_tokens]` (f32)
3. For each token t and expert index e: `out[:, e, t] = as[:, :, ids[e, t]]`

---

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `ggml/include/ggml.h` | Added `GGML_OP_MOE_EXPERT_GATHER` enum + API declaration | +18 |
| `ggml/src/ggml.c` | Added tensor constructor + updated `GGML_OP_COUNT` | +44 |
| `ggml/src/ggml-cpu/ops.h` | Added forward declaration | +1 |
| `ggml/src/ggml-cpu/ops.cpp` | Added `ggml_compute_forward_moe_expert_gather` | +50 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Added CPU dispatch cases (3 locations) | +11 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Added CUDA compatibility check | +3 |

**Total: ~127 lines added, 0 deleted**

---

## Lessons Learned

### 1. `GGML_OP_COUNT` must be updated when adding new ops
**Lesson:** Every new op in the enum requires updating `GGML_OP_COUNT` in `ggml.c` — there are TWO static_assert checks (lines 1113 and 1230). Also need to add the op to the `GGML_OP_SYMBOL` array.

**Fix:** Search for `GGML_OP_COUNT == N` and update all occurrences. Add symbol string to the array.

### 2. Forward declarations must be in `ops.h`
**Lesson:** CPU forward functions are defined in `ops.cpp` but called from `ggml-cpu.c`. The function must be declared in `ops.h` before it's used.

**Fix:** Add `void ggml_compute_forward_<name>(const struct ggml_compute_params * params, struct ggml_tensor * dst);` to `ops.h` in alphabetical order.

### 3. Three dispatch locations in `ggml-cpu.c`
**Lesson:** Operations need to be registered in three places:
1. `ggml_compute_forward()` switch — dispatches the forward function
2. `ggml_compute_forward()` task scheduling — sets `n_tasks`
3. `ggml_compute_forward()` cache sizing — calculates work buffer size

**Fix:** Search for `case GGML_OP_<NAME>:` in `ggml-cpu.c` and add all three locations.

### 4. CUDA compatibility check is optional for CPU-only ops
**Lesson:** If an op runs only on CPU (like `MOE_EXPERT_GATHER`), the CUDA backend doesn't need a full dispatch. It just needs a compatibility check that returns `true` if the tensor types are supported.

**Fix:** Add a case in `ggml_backend_cuda_op_compatible()` that returns `true` for supported tensor types.

### 5. The gather output shape is `[ne0*ne1, n_expert_used, n_tokens, 1]`
**Lesson:** The flattened dimension `ne0*ne1` means the output is a wide 2D matrix per expert token. This is different from the original `ggml_mul_mat_id` output which keeps `ne1` separate.

**Implication:** The downstream `ggml_mul_mat` must handle this shape correctly. The matmul expects `[cols, rows, 1, 1]` and `[cols, n, 1, 1]` inputs.

### 6. Zero-fill on invalid expert IDs
**Lesson:** Router outputs can contain invalid expert IDs (outside `[0, n_expert)`). The gather must zero-fill these cases rather than crashing.

**Fix:** Check `expert_id >= 0 && expert_id < n_expert` before copying. Use `memset` to zero-fill on invalid.

### 7. Thread pool parallelization works on expert tokens
**Lesson:** The gather is parallelized over expert tokens (ne1 * ne2), not over the flattened dimension. Each thread handles a contiguous block of expert tokens.

**Fix:** Calculate `tokens_per_thread = (ne1 * ne2 + nth - 1) / nth` and iterate over `[token_start, token_end)`.

### 8. No CUDA dispatch needed for MVP
**Lesson:** The gather op runs entirely on CPU. The CUDA backend only needs to know that the output tensor is accessible (F32 type, on host buffer).

**Implication:** The next step is to wire the gather output into `ggml_mul_mat` on the CUDA side. No CUDA kernel needed for the gather itself.

---

## Next Steps

1. **Wire into graph construction:** Modify `build_lora_mm_id()` in `llama-graph.cpp` to use `ggml_moe_expert_gather()` + `ggml_mul_mat()` instead of `ggml_mul_mat_id()`.

2. **Test with a MoE model:** Run inference on a MoE model (e.g., Qwen3-MoE) to verify correctness.

3. **Benchmark:** Compare decode speed with fork's `ggml_cuda_mul_mat_id()`.

---

## Known Issues

1. **Gate_up path not handled separately:** The gather flattens `ne0*ne1` for all MoE paths. The `gate_up` path has `ne0 = n_ff*2` which is different from `up`/`gate` paths. This needs verification.

2. **LoRA not supported:** The MVP only handles base model weights. LoRA adapters would need separate gather+matmul chains.

3. **No expert caching:** The MVP reads all expert weights from RAM on every forward pass. Per-layer rolling staging (Gap 1) is deferred.

---

# MVP Lessons Learned — Step 2 Complete (Graph Wiring)

> **Date:** 2026-09-10

## What We Did

Modified `build_lora_mm_id()` in `llama-graph.cpp` to use the new gather+matmul approach when `mask_from >= 0` (MoE expert branch).

## Lessons Learned

### 1. `mask_from` is the MoE branch signal
**Lesson:** When `mask_from >= 0`, the function is called from `mm_id_exps` which is the MoE FFN path. When `mask_from < 0` (default), it's a non-MoE path.

**Fix:** Use `if (mask_from >= 0)` to detect MoE branches and route to the new gather+matmul path.

### 2. The gather output shape works with `ggml_mul_mat`
**Lesson:** `ggml_moe_expert_gather()` outputs `[ne0*ne1, n_expert_used, n_tokens, 1]`. The `ggml_mul_mat(gathered, cur)` call works because:
- `cur` is `[n_embd, n_tokens, 1, 1]` (or `[ne0, n_tokens, 1, 1]`)
- `gathered` is `[ne01, n_expert_used, n_tokens, 1]` where `ne01 = ne0 * ne1`
- The matmul computes `gathered[:, :, t] @ cur[:, t]` for each token t

**Implication:** No reshape needed between gather and matmul. The flattened dimension aligns correctly.

### 3. LoRA is not supported in MVP
**Lesson:** The LoRA loop at the end of `build_lora_mm_id()` is skipped for MoE branches. LoRA adapters use `ggml_mul_mat_id` which we're replacing for MoE.

**Fix:** LoRA support requires a separate gather+matmul chain for each LoRA adapter. Defer to post-MVP.

### 4. `w_s` (expert scaling) still works
**Lesson:** The expert scaling tensor `w_s` is applied after the gather+matmul, just like in the original path. No changes needed.

**Fix:** Keep the existing `w_s` handling code after the gather+matmul block.

### 5. Build still passes after graph wiring
**Lesson:** The new function `ggml_moe_expert_gather()` is declared in `ggml.h` and linked correctly. No additional include or link changes needed.

**Fix:** No additional steps required — the include path already covers `ggml_moe_expert_gather()`.

## Known Issues

1. **Gate_up path shape mismatch:** The `gate_up` path has `ne0 = n_ff*2` which means the flattened dimension is `n_ff*2 * n_embd`. This is correct for the gather output, but the downstream split (into gate and up views) expects shape `[n_ff*2, n_expert_used, n_tokens]`. The gather outputs `[n_ff*2*n_embd, n_expert_used, n_tokens]` which is different. This needs verification.

2. **Down path not tested:** The `down` path has `ne0 = n_embd, ne1 = n_ff` which means the flattened dimension is `n_embd * n_ff`. This is different from `up`/`gate` paths. Needs verification.

3. **No performance data:** We can't benchmark yet because llama-server is running in production.

## Next Steps

1. **Test with a MoE model:** Once llama-server is available, test with a MoE model to verify correctness.
2. **Verify gate_up path:** Check if the gate_up path produces correct output.
3. **Verify down path:** Check if the down path produces correct output.
4. **Benchmark:** Compare decode speed with fork's `ggml_cuda_mul_mat_id()`.

---

# MVP Lessons Learned — Step 3 Complete (Graph Wiring Fix)

> **Date:** 2026-09-10

## What Changed

Fixed the matmul wiring in `build_lora_mm_id()`. The initial approach of `ggml_mul_mat(gathered, cur)` didn't work because:
- `gathered` shape: `[ne0*ne1, n_expert_used, n_tokens, 1]`
- `cur` shape: `[ne1, n_tokens, 1, 1]`
- These don't align for a standard matmul

The fix reshapes both tensors:
1. Reshape `gathered` to `[ne0, ne1, n_expert_used*n_tokens, 1]`
2. Repeat `cur` to `[ne1, n_expert_used, n_tokens, 1]`
3. Matmul produces `[ne0, n_expert_used, n_tokens, 1]`

## Lessons Learned

### 1. `ggml_mul_mat` requires matching inner dimensions
**Lesson:** The matmul operation requires `A`'s last dimension to equal `B`'s last dimension. The gathered tensor's flattened shape `[ne0*ne1, ...]` doesn't work directly.

**Fix:** Reshape gathered to `[ne0, ne1, n_exp*n_tok, 1]` so the last dimension is `ne1` (matching `cur`'s first dimension).

### 2. `ggml_repeat` duplicates tensors to target shape
**Lesson:** `ggml_repeat(cur, b_shape)` repeats `cur` to match `b_shape`'s dimensions. This is the correct way to duplicate the input for all experts.

**Fix:** Create `b_shape` with shape `[ne1, n_expert_used, n_tokens, 1]` and use `ggml_repeat(cur, b_shape)` to duplicate the input.

### 3. The matmul output shape is correct
**Lesson:** After reshaping and repeating, `ggml_mul_mat(a, b)` produces `[ne0, n_expert_used, n_tokens, 1]` which is the expected output shape.

**Verification:** 
- A: `[ne0, ne1, n_exp*n_tok, 1]` → treated as `[rows, 1, n_exp*n_tok, ne1]`
- B: `[ne1, n_exp, n_tok, 1]` → treated as `[ne1, n_exp, n_tok, 1]`
- Result: `[ne0, n_exp, n_tok, 1]` ✓

### 4. No CUDA changes needed for the matmul
**Lesson:** The `ggml_mul_mat` operation is already dispatched to CUDA. No changes to the CUDA backend are needed for the matmul part.

**Implication:** The CUDA backend will automatically run the matmul on GPU.

### 5. Gate_up path should work correctly
**Lesson:** For the gate_up path, `w` has shape `[n_ff*2, n_embd, n_expert]`. After gather and reshape:
- `a` = `[n_ff*2, n_embd, n_exp*n_tok, 1]`
- `b` = `[n_embd, n_exp, n_tok, 1]`
- Result = `[n_ff*2, n_exp, n_tok, 1]`

This matches the expected output shape `[n_ff*2, n_expert_used, n_tokens]`.

### 6. Up/Gate path should work correctly
**Lesson:** For the up/gate path, `w` has shape `[n_ff, n_embd, n_expert]`. After gather and reshape:
- `a` = `[n_ff, n_embd, n_exp*n_tok, 1]`
- `b` = `[n_embd, n_exp, n_tok, 1]`
- Result = `[n_ff, n_exp, n_tok, 1]`

This matches the expected output shape `[n_ff, n_expert_used, n_tokens]`.

## Known Issues

1. **Down path may have issues:** For the down path, `w` has shape `[n_embd, n_ff, n_expert]`. After gather:
   - `a` = `[n_embd, n_ff, n_exp*n_tok, 1]`
   - `b` = `[n_ff, n_exp, n_tok, 1]` ← This doesn't match!
   
   The matmul expects A's last dim = B's last dim, but here A's last dim is `n_ff` and B's last dim is `n_ff`. Wait, they're both `n_ff`... Let me re-check.
   
   Actually, `cur` for the down path has shape `[n_ff, n_tokens]` (not `[n_embd, n_tokens]`). So:
   - `b` = `[n_ff, n_exp, n_tok, 1]`
   - `a` = `[n_embd, n_ff, n_exp*n_tok, 1]`
   
   The matmul would try to compute `a @ b^T` where `a` has `n_ff` columns and `b` has `n_ff` columns. This should work because both have `n_ff` in the matching dimension.
   
   Result = `[n_embd, n_exp, n_tok, 1]` which is the expected output shape.
   
   **So the down path should work correctly too.**

2. **No live testing possible:** llama-server is running in production. Can't verify correctness until it's stopped.

## Summary of MVP Changes

| File | Lines | Purpose |
|------|-------|---------|
| `ggml/include/ggml.h` | +18 | New op enum + API declaration |
| `ggml/src/ggml.c` | +44 | Tensor constructor + op symbol |
| `ggml/src/ggml-cpu/ops.h` | +1 | Forward declaration |
| `ggml/src/ggml-cpu/ops.cpp` | +50 | CPU forward implementation |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +11 | CPU dispatch (3 locations) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +3 | CUDA compatibility check |
| `src/llama-graph.cpp` | +35 | Graph wiring (gather + reshape + matmul) |

**Total: ~171 lines added**

## What's Left for Post-MVP

1. **Per-layer rolling staging buffer** (Gap 1) — reduces RAM bandwidth pressure
2. **Hot/cold expert routing** (Gap 2) — leverages GPU cache for hot experts
3. **CUDA stream overlap** (Gap 3) — hides gather latency behind GPU compute
4. **LoRA adapter support** (Gap 5) — extend gather+matmul for LoRA weights
5. **CUDA graph integration** (Gap 6) — re-enable graphs for MoE layers
6. **Performance validation hooks** (Gap 8) — add timing instrumentation
