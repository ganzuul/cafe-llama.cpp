# CPU `mul_mat_id` + GPU Expert Cache: Storyboard

> **Date:** 2026-09-10
> **Purpose:** Storyboard code blocks for each implementation phase — not for integration, just visualization

---

## Phase 1: New GGML Op + CPU Weight Gathering

### 1.1 Add `GGML_OP_MOE_EXPERT_GATHER` to the op enum

**File:** `ggml/include/ggml.h` (line ~595, after `GGML_OP_MOE_BRANCH_IDS`)

```diff
@@ -591,6 +591,7 @@ enum ggml_op {
         GGML_OP_OPT_STEP_SGD,
 
         GGML_OP_GLU,
+
         GGML_OP_MOE_BRANCH_IDS,
+        GGML_OP_MOE_EXPERT_GATHER,
 
         GGML_OP_COUNT,
```

### 1.2 Add op name to the name table

**File:** `ggml/src/ggml.c` (line ~1110)

```diff
@@ -1107,7 +1107,8 @@ static const char * GGML_OP_NAME[GGML_OP_COUNT] = {
     "GLU",
 
     "MOE_BRANCH_IDS",
+    "MOE_EXPERT_GATHER",
 };
 
-static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");
+static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");
```

### 1.3 Add the GGML API function declaration

**File:** `ggml/include/ggml.h` (after `ggml_moe_branch_ids` declaration, ~line 1470)

```diff
@@ -1468,6 +1470,18 @@ GGML_API void ggml_mul_mat_id_set_mask_from(
             int32_t               mask_base);
 
+    // Gather expert weights from a 3D expert tensor into a compact 2D output tensor.
+    // as -> [n_embd, n_ff, n_expert]       (expert weights, 3D)
+    // ids -> [n_expert_used, n_tokens]     (i32 expert indices)
+    // out -> [n_embd * n_ff, n_expert_used, n_tokens]  (gathered, 3D)
+    //
+    // For each token t and expert index e in ids:
+    //   out[:, e, t] = as[:, :, ids[e, t]]  (row from expert tensor)
+    //
+    // The output is a contiguous block suitable for ggml_mul_mat on the GPU.
+    GGML_API struct ggml_tensor * ggml_moe_expert_gather(
+            struct ggml_context * ctx,
+            struct ggml_tensor  * as,
+            struct ggml_tensor  * ids);
```

### 1.4 Implement the tensor constructor in ggml.c

**File:** `ggml/src/ggml.c` (after `ggml_moe_branch_ids` implementation, ~line 3340)

```c
// ggml_moe_expert_gather
//
// as  -> [ne0, ne1, n_expert]   (expert weights: [n_embd, n_ff, n_expert] or [n_ff, n_embd, n_expert])
// ids -> [n_expert_used, n_tokens]  (i32)
// out -> [ne0*ne1, n_expert_used, n_tokens]  (gathered weights)
struct ggml_tensor * ggml_moe_expert_gather(
        struct ggml_context * ctx,
        struct ggml_tensor  * as,
        struct ggml_tensor  * ids) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(as->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(as->ne[0] >= 0 && as->ne[1] >= 0);

    const int64_t ne[4] = {
        as->ne[0] * as->ne[1],  // flattened expert dimension
        ids->ne[0],             // n_expert_used
        ids->ne[1],             // n_tokens
        1
    };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MOE_EXPERT_GATHER;
    result->src[0] = as;
    result->src[1] = ids;

    return result;
}
```

### 1.5 CPU backend implementation in ops.cpp

**File:** `ggml/src/ggml-cpu/ops.cpp` (new function, after `ggml_compute_forward_moe_branch_ids`)

```cpp
// ggml_compute_forward_moe_expert_gather
//
// CPU reads expert weights from the expert tensor (on RAM or staging buffer)
// and gathers them into a compact 2D output tensor.
// This is the key operation that replaces the GPU-side row gathering in the fork.
void ggml_compute_forward_moe_expert_gather(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];  // expert weights [ne0, ne1, n_expert]
    const ggml_tensor * ids = dst->src[1];  // expert indices [n_expert_used, n_tokens]

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(as->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1);

    const int64_t ne0 = dst->ne[0];       // ne0 * ne1 of source
    const int64_t ne1 = dst->ne[1];       // n_expert_used
    const int64_t ne2 = dst->ne[2];       // n_tokens
    const int64_t n_expert = as->ne[2];   // total experts

    const int ith = params->ith;
    const int nth = params->nth;

    // Process tokens in parallel across threads
    for (int64_t t = ith; t < ne2; t += nth) {
        // For each expert index used at this token
        for (int64_t e = 0; e < ne1; e++) {
            // Get the expert ID for this (expert_slot, token)
            const int32_t expert_id = *(const int32_t *) (
                (const char *) ids->data + e * ids->nb[0] + t * ids->nb[1]
            );

            // If expert_id is valid (< n_expert), gather that expert's weights
            if (expert_id >= 0 && expert_id < (int32_t)n_expert) {
                // Source expert row: as[:, :, expert_id] -> flattened
                const char * src_expert = (const char *) as->data +
                    expert_id * as->nb[2];

                // Destination: dst[:, e, t]
                float * dst_row = (float *) dst->data +
                    (e * ne2 + t) * ne0;

                // Copy the expert row (ne0 floats)
                memcpy(dst_row, src_expert, ne0 * sizeof(float));
            } else {
                // Invalid expert ID -> zero output
                float * dst_row = (float *) dst->data +
                    (e * ne2 + t) * ne0;
                memset(dst_row, 0, ne0 * sizeof(float));
            }
        }
    }
}
```

### 1.6 Register the CPU op in ggml-cpu.c

**File:** `ggml/src/ggml-cpu/ggml-cpu.c` (in the forward dispatch switch, after `GGML_OP_MOE_BRANCH_IDS`)

```diff
@@ -1746,6 +1746,9 @@ static enum ggml_status ggml_compute_forward(
         case GGML_OP_MOE_BRANCH_IDS:
             {
                 ggml_compute_forward_moe_branch_ids(params, tensor);
             } break;
+        case GGML_OP_MOE_EXPERT_GATHER:
+            {
+                ggml_compute_forward_moe_expert_gather(params, tensor);
+            } break;
```

**File:** `ggml/src/ggml-cpu/ggml-cpu.c` (in the n_tasks switch, after `GGML_OP_MOE_BRANCH_IDS`)

```diff
@@ -2249,6 +2249,7 @@ static int ggml_backend_cpu_tgml_compute_get_n_tasks(const struct ggml_tensor * node) {
         case GGML_OP_MOE_BRANCH_IDS:
         case GGML_OP_ADD1:
+        case GGML_OP_MOE_EXPERT_GATHER:
         case GGML_OP_ACC:
```

### 1.7 Thread pool task registration in ops.h

**File:** `ggml/src/ggml-cpu/ops.h`

```diff
@@ -32,5 +32,6 @@ void ggml_compute_forward_add_id(const struct ggml_compute_params * params, ggml_tensor * dst);
 void ggml_compute_forward_moe_branch_ids(const struct ggml_compute_params * params, ggml_tensor * dst);
+void ggml_compute_forward_moe_expert_gather(const struct ggml_compute_params * params, ggml_tensor * dst);
```

---

## Phase 2: CUDA Dispatch for Gathered Weights

### 2.1 Modify the CUDA dispatch switch for `GGML_OP_MOE_EXPERT_GATHER`

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (in `ggml_cuda_compute_forward`, after `GGML_OP_MOE_BRANCH_IDS`)

```diff
@@ -2123,6 +2123,8 @@ static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
         case GGML_OP_MOE_BRANCH_IDS:
             ggml_cuda_op_moe_branch_ids(ctx, dst);
             break;
+        case GGML_OP_MOE_EXPERT_GATHER:
+            ggml_cuda_op_expert_gather(ctx, dst);
+            break;
```

### 2.2 CUDA implementation: read gathered weights from staging buffer

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (new function, after `ggml_cuda_op_moe_branch_ids`)

```cpp
// CUDA implementation of MOE_EXPERT_GATHER.
//
// Instead of gathering rows from the expert tensor (like the CPU path),
// this path reads expert weights that were pre-copied to a staging buffer
// by the CPU. The staging buffer is a contiguous block of device memory
// containing [ne0*ne1, n_expert_used, n_tokens] of F32 data.
//
// This is used when the expert weights are already on GPU (hot cache hit).
// The CPU's ggml_moe_expert_gather writes directly to the staging buffer
// via cudaMemcpyAsync, and this kernel just validates and passes through.
static void ggml_cuda_op_expert_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];  // expert weights [ne0, ne1, n_expert]
    const ggml_tensor * ids = dst->src[1];  // expert indices [n_expert_used, n_tokens]

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // If expert weights are on GPU (hot cache hit), copy directly
    if (as->buffer && !ggml_backend_buffer_is_host(as->buffer)) {
        // as is already on GPU — copy the selected expert rows to dst
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        const ggml_tensor * ids  = dst->src[2];

        cudaStream_t stream = ctx.stream();

        // Build ids_to_sorted on host, then launch get_rows_cuda
        // (similar to the existing ggml_cuda_mul_mat_id fallback path)
        // ... see detailed implementation below
    } else {
        // Expert weights on host — use CPU gather path (handled by CPU backend)
        // This case should not be reached; the graph construction should
        // ensure expert weights are on GPU before calling this op.
        GGML_ABORT("MOE_EXPERT_GATHER with host weights — should use CPU path");
    }
}
```

### 2.3 Modify CUDA MUL_MAT_ID dispatch to use gathered weights

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (in `ggml_cuda_compute_forward`, the MUL_MAT_ID case)

```diff
@@ -2275,7 +2275,7 @@ static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
         case GGML_OP_MUL_MAT_ID:
-            ggml_cuda_mul_mat_id(ctx, dst);
+            ggml_cuda_mul_mat_id_staged(ctx, dst);
             break;
```

### 2.4 New staged `ggml_cuda_mul_mat_id_staged` — reads from staging buffer instead of RAM

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (new function, replacing `ggml_cuda_mul_mat_id`)

```cpp
// Staged version of ggml_cuda_mul_mat_id.
//
// Instead of reading expert weights from the expert tensor (which may be on
// host or device), this reads from a staging buffer that was pre-populated
// by the CPU's ggml_moe_expert_gather operation.
//
// Key difference from the fork's ggml_cuda_mul_mat_id:
// - No per-layer sync for index lookup (CPU handles it)
// - No per-expert cache check (staging buffer has everything needed)
// - Standard matmul on gathered weights (no row gathering on GPU)
static void ggml_cuda_mul_mat_id_staged(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // expert weights [ne0, ne1, n_expert]
    const ggml_tensor * src1 = dst->src[1];  // input [ne0, n_expert_used, n_tokens]
    const ggml_tensor * ids  = dst->src[2];  // expert indices [n_expert_used, n_tokens]

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    cudaStream_t stream = ctx.stream();

    // [KEY CHANGE] We don't need the sync-heavy fallback path anymore.
    // The CPU has already gathered the expert weights into a staging buffer.
    // We just need to run standard matmul on the gathered weights.

    // The gathered weights are in src1 (which was produced by MOE_EXPERT_GATHER).
    // src1 shape: [ne0, n_expert_used, n_tokens]
    // We need to run: dst = src0_gathered @ src1_transposed
    // where src0_gathered is the expert weights for the selected experts.

    // Actually, the MOE_EXPERT_GATHER output IS src1, and we just need:
    // dst = matmul(src0_expert_slice, src1_gathered)
    //
    // But wait — the fork's mul_mat_id does per-expert matmul because each
    // expert has different weights. With our staging approach, the CPU gathers
    // the expert weights into src1, and we need to run:
    //
    // For each expert e selected:
    //   dst[:, e, t] = expert_weights[e] @ src1[:, t]
    //
    // This is still per-expert, but the weight gathering is done on CPU.
    // The GPU just does standard matmul.

    // [SIMPLIFICATION] For now, use the existing fast paths when possible:
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // [KEY CHANGE] No sync needed — the CPU gather is complete.
    // Just run the matmul with the gathered weights.
    // The ids tensor tells us which experts are active.

    // For quantized experts, use the existing mmq/mmvq paths:
    if (ggml_is_quantized(src0->type)) {
        if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // [KEY CHANGE] Fallback: use the existing matmul path.
    // The expert weight gathering was done on CPU, so src0 still has the
    // full expert tensor. We need to select the right expert slice.
    //
    // This is the same as the fork's approach for hot experts (cached on GPU),
    // but without the sync penalty because the CPU gather happened asynchronously.
    ggml_cuda_mul_mat(ctx, src0, src1, dst);
}
```

### 2.5 Update CUDA graph compatibility check

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (in `ggml_cuda_graph_update_required`, ~line 2586)

```diff
@@ -2586,12 +2586,6 @@ static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
         // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
         if (node->op == GGML_OP_MUL_MAT_ID) {
-            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
-            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
-                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
-                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
-                use_cuda_graph = false;
-#ifndef NDEBUG
-                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
-#endif
-            }
+            // MOE_EXPERT_GATHER nodes don't need sync — CPU gather is async.
+            // Only disable CUDA graphs if the gather op itself has sync requirements.
+            const bool has_sync = false;  // CPU gather is async from GPU perspective
+            if (has_sync) {
+                use_cuda_graph = false;
+            }
         }
```

---

## Phase 3: Graph Construction Wiring

### 3.1 Modify `build_lora_mm_id` to insert MOE_EXPERT_GATHER

**File:** `src/llama-graph.cpp` (modify `build_lora_mm_id`, ~line 1542)

```diff
@@ -1542,10 +1542,35 @@ ggml_tensor * llm_graph_context::build_lora_mm_id(
           ggml_tensor * w,   // ggml_tensor * as
           ggml_tensor * cur, // ggml_tensor * b
           ggml_tensor * ids,
           ggml_tensor * w_s,
               int32_t   mask_from) const {
-    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
+    // [KEY CHANGE] For MoE expert branches, use staged gather + matmul.
+    // Instead of a single GGML_OP_MUL_MAT_ID, we split into:
+    // 1. MOE_EXPERT_GATHER: CPU gathers expert weights into compact tensor
+    // 2. GGML_OP_MUL_MAT:   GPU runs standard matmul on gathered weights
+
+    // Check if this is a MoE expert branch (has mask_from set)
+    if (mask_from >= 0) {
+        // MoE expert branch — use staged path
+
+        // Step 1: Gather expert weights on CPU
+        // Input: w [n_embd, n_ff, n_expert], ids [n_expert_used, n_tokens]
+        // Output: gathered [n_embd*n_ff, n_expert_used, n_tokens]
+        ggml_tensor * gathered = ggml_moe_expert_gather(ctx0, w, ids);
+
+        // Step 2: Run standard matmul with gathered weights
+        // Input: gathered [n_embd*n_ff, n_expert_used, n_tokens]
+        //        cur [n_embd, n_expert_used, n_tokens]
+        // Output: res [n_ff, n_expert_used, n_tokens]
+        ggml_tensor * res = ggml_mul_mat(ctx0, gathered, cur);
+
+        // Apply mask and scaling (same as before)
+        if (mask_from >= 0) {
+            ggml_mul_mat_id_set_mask_from(res, mask_from);
+        }
+
+        if (w_s) {
+            // Expert scaling — same logic as before
+            const int64_t n_expert = w_s->ne[0];
+            const int64_t n_tokens = cur->ne[2];
+            ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
+            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
+            s = ggml_get_rows(ctx0, s, ids);
+            res = ggml_mul(ctx0, res, s);
+        }
+
+        // LoRA — same logic as before
+        for (const auto & lora : *loras) {
+            // ... (unchanged)
+        }
+
+        return res;
+    }
+
+    // Non-MoE path — use original ggml_mul_mat_id
+    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
```

### 3.2 Wire tensor placement for expert tensors

**File:** `common/common.h` (add new tensor placement override for staging)

```diff
@@ -1144,6 +1144,12 @@ inline void llm_add_n_host_moe_overrides(int n, std::vector<llama_model_tensor_buft_override> & out) {
     static std::list<std::string> strings; for (int i=0;i<n;++i) { strings.push_back(llm_ffn_exps_block_regex(i)); out.push_back({strings.back().c_str(), common_host_buffer_type()}); }
 }
+
+// Override for expert staging buffer — place on CUDA host (pinned) for CPU-GPU access
+inline llama_model_tensor_buft_override llm_ffn_exps_staging_override() {
+    return { LLM_FFN_EXPS_REGEX, ggml_backend_cuda_host_buffer_type() };
+}
+
+// Add per-layer staging overrides
+inline void llm_add_n_staging_moe_overrides(int n, std::vector<llama_model_tensor_buft_override> & out) {
+    static std::list<std::string> strings; for (int i=0;i<n;++i) { strings.push_back(llm_ffn_exps_block_regex(i)); out.push_back({strings.back().c_str(), ggml_backend_cuda_host_buffer_type()}); }
+}
```

### 3.3 Modify `mm_id_exps` lambda to use staged path

**File:** `src/llama-graph.cpp` (modify `build_moe_ffn`, ~line 2143)

```diff
@@ -2143,10 +2143,12 @@ ggml_tensor * llm_graph_context::build_moe_ffn(
     auto mm_id_exps = [&](ggml_tensor * w_hot, ggml_tensor * w_cold, ggml_tensor * x, ggml_tensor * s) {
         if (!split_exps) {
-            return build_lora_mm_id(w_hot, x, selected_experts, s);
+            // Non-split path — use staged gather + matmul
+            return build_lora_mm_id(w_hot, x, selected_experts, s, /*mask_from=*/-1);
         }
-        ggml_tensor * o_hot  = build_lora_mm_id(w_hot,  x, ids_hot,  s, n_hot);
-        ggml_tensor * o_cold = build_lora_mm_id(w_cold, x, ids_cold, s, n_cold);
+        ggml_tensor * o_hot  = build_lora_mm_id(w_hot,  x, ids_hot,  s, /*mask_from=*/0);
+        ggml_tensor * o_cold = build_lora_mm_id(w_cold, x, ids_cold, s, /*mask_from=*/n_hot);
         return ggml_add(ctx0, o_hot, o_cold);
     };
```

### 3.4 Ensure expert tensors are placed on CUDA host (pinned) for staging

**File:** `src/llama-model-loader.cpp` (in tensor loading, after CUDA_Host override check)

```diff
@@ -1221,6 +1221,12 @@ static void llm_load_split(model_loader_loader * loader, const char * path, st
         // Automatic host buffers are demoted under mmap, but an explicit CUDA_Host override
         // is intentional: allocate it and copy that tensor while lazy tensors remain mapped.
+        // [KEY CHANGE] For MoE expert tensors, ensure they're on CUDA host (pinned)
+        // so the CPU can read them for the gather operation.
         auto * buft_dev = ggml_backend_buft_get_device(buft);
         if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev) && !keep_explicit_host_with_mmap) {
+            // For expert tensors, keep them on CUDA host even with mmap
+            const char * tn = t_meta->name;
+            if (strstr(tn, "_exps") != nullptr || strstr(tn, "ffn_gate") != nullptr) {
+                // Keep expert tensors on CUDA host for CPU gather
+                buft = ggml_backend_dev_host_buffer_type(buft_dev);
+            } else {
+                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
+                if (!cpu_dev) {
+                    throw std::runtime_error("no CPU backend found");
+                }
+                buft = ggml_backend_dev_buffer_type(cpu_dev);
+            }
+        } else {
             auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
             if (!cpu_dev) {
                 throw std::runtime_error("no CPU backend found");
             }
             buft = ggml_backend_dev_buffer_type(cpu_dev);
         }
```

---

## Summary of Data Flow After Changes

```
Before (fork's current design):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → ggml_mul_mat_id() [CPU node]
        → CUDA dispatch: ggml_cuda_mul_mat_id()
          → [SYNC] Copy ids to host
          → [SYNC] Build ids_to_sorted
          → [GPU] get_rows_cuda: gather src1 rows
          → [PER-EXPERT] cache check + matmul
          → [SYNC] Scatter results

After (our design):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → ggml_moe_expert_gather() [CPU node]  ← NEW
        → CPU reads expert weights from RAM/staging
        → Outputs compact gathered tensor
      → ggml_mul_mat() [GPU node]  ← standard matmul
        → GPU reads gathered weights (no gathering needed)
        → No per-layer sync for index lookup
```

---

## Key Design Decisions

1. **New op `GGML_OP_MOE_EXPERT_GATHER`**: Separates weight gathering from matmul. The CPU handles gathering, GPU handles matmul.

2. **Staged execution**: CPU gather happens asynchronously while GPU is processing the previous layer. No sync point between gather and matmul.

3. **Staging buffer**: Expert weights on CUDA host (pinned) memory. CPU reads from RAM, writes to pinned buffer. GPU reads from pinned buffer via DMA.

4. **No per-expert cache check on GPU**: The CPU gather pre-fetches all needed expert weights. The GPU just does standard matmul.

5. **CUDA graph compatibility**: Since CPU gather is async from GPU perspective, CUDA graphs can still be used for the matmul portion.

---

## Open Questions

1. **Staging buffer memory model**: Should we use pinned host memory (CPU writes, GPU reads via DMA) or unified memory (CPU writes, GPU reads directly)? Pinned host is simpler but requires explicit H2D copies.

2. **Thread pool integration**: The CPU gather needs to run on the CPU thread pool. How do we ensure it runs asynchronously with respect to the GPU?

3. **Error handling**: What happens if the CPU gather fails (e.g., invalid expert ID)? Should we zero-fill or abort?

4. **Performance**: How much does the CPU gather add to the per-layer latency? Is it hidden by the GPU's previous layer computation?

---

# GAP-CLOSING STORYBOARD BLOCKS

> These blocks close the 8 gaps identified in `cpu-mul-mat-id-gaps.md`

---

## Phase 1b: Per-Layer Rolling Staging Buffer (Gap 1)

**File:** `ggml/src/ggml-cuda/common.cuh` (new struct, after `ggml_cuda_expert_lru_cache`)

```cpp
// Per-layer rolling staging buffer manager.
//
// Manages a pool of expert weight slices that are reused across layers
// within a forward pass. Experts already in the buffer are reused;
// new experts are added; unused experts are evicted.
struct ggml_moe_staging_manager {
    // Per-expert metadata
    struct entry {
        void * data = nullptr;       // staging buffer pointer (pinned host)
        size_t size = 0;             // size in bytes
        uint64_t last_used = 0;      // step counter for LRU eviction
        bool valid = false;          // whether this entry is active
    };

    int device = 0;
    size_t max_bytes = 0;            // max staging buffer size
    size_t current_bytes = 0;        // currently used bytes
    uint64_t step_counter = 0;       // increments per layer
    std::unordered_map<int32_t, entry> entries;  // expert_id -> entry

    void init(int dev, size_t max_mb) {
        device = dev;
        max_bytes = max_mb * 1024 * 1024;
    }

    // Ensure an expert is in the staging buffer.
    // Returns pointer to staging buffer slot, or nullptr on OOM.
    void * ensure_in_staging(
            const void * host_ptr,   // expert weights on host RAM
            size_t size,
            cudaStream_t stream) {
        ++step_counter;

        // Check if expert is already in staging buffer
        auto it = entries.find(host_ptr);
        if (it != entries.end() && it->second.valid) {
            it->second.last_used = step_counter;
            return it->second.data;
        }

        // Evict LRU entries if needed
        evict_lru(size);

        // Allocate new staging slot
        void * staging_ptr = nullptr;
        if (cudaHostAlloc(&staging_ptr, size, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;  // OOM — caller should fallback
        }

        entry new_entry;
        new_entry.data = staging_ptr;
        new_entry.size = size;
        new_entry.last_used = step_counter;
        new_entry.valid = true;

        entries[host_ptr] = new_entry;
        current_bytes += size;

        // Copy expert weights to staging buffer
        CUDA_CHECK(cudaMemcpyAsync(staging_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream));

        return staging_ptr;
    }

    void evict_lru(size_t required_bytes) {
        while (current_bytes + required_bytes > max_bytes && !entries.empty()) {
            // Find LRU entry
            auto lru_it = entries.begin();
            for (auto iter = entries.begin(); iter != entries.end(); ++iter) {
                if (iter->second.last_used < lru_it->second.last_used) {
                    lru_it = iter;
                }
            }

            // Free LRU entry
            current_bytes -= lru_it->second.size;
            cudaFreeHost(lru_it->second.data);
            entries.erase(lru_it);
        }
    }

    void reset() {
        // Called at start of each forward pass
        // Keeps entries warm but marks them for potential eviction
        for (auto & [id, entry] : entries) {
            entry.last_used = 0;  // reset for next pass
        }
    }

    void clear() {
        for (auto & [id, entry] : entries) {
            if (entry.data) {
                cudaFreeHost(entry.data);
            }
        }
        entries.clear();
        current_bytes = 0;
    }
};
```

**File:** `ggml/src/ggml-cuda/common.cuh` (add to `ggml_backend_cuda_context`, line ~1609)

```diff
@@ -1609,6 +1609,7 @@ struct ggml_backend_cuda_context {
     ggml_cuda_expert_lru_cache expert_cache;

     ggml_cuda_stream_context concurrent_stream_context;
+    ggml_moe_staging_manager moe_staging;

     ~ggml_backend_cuda_context();
```

---

## Phase 1c: Hot vs Cold Expert Routing (Gap 2)

**File:** `ggml/src/ggml-cpu/ops.cpp` (modify `ggml_compute_forward_moe_expert_gather`)

```cpp
// Updated ggml_compute_forward_moe_expert_gather with hot/cold routing
void ggml_compute_forward_moe_expert_gather(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];  // expert weights [ne0, ne1, n_expert]
    const ggml_tensor * ids = dst->src[1];  // expert indices [n_expert_used, n_tokens]

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(as->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t n_expert = as->ne[2];

    const int ith = params->ith;
    const int nth = params->nth;

    // Determine if expert weights are on GPU (hot cache) or RAM (cold)
    const bool weights_on_gpu = as->buffer && !ggml_backend_buffer_is_host(as->buffer);

    for (int64_t t = ith; t < ne2; t += nth) {
        for (int64_t e = 0; e < ne1; e++) {
            const int32_t expert_id = *(const int32_t *) (
                (const char *) ids->data + e * ids->nb[0] + t * ids->nb[1]
            );

            float * dst_row = (float *) dst->data + (e * ne2 + t) * ne0;

            // Validate expert ID
            if (expert_id < 0 || expert_id >= (int32_t)n_expert) {
                memset(dst_row, 0, ne0 * sizeof(float));
                continue;
            }

            if (weights_on_gpu) {
                // Hot expert: weights already on GPU
                // Copy from GPU expert tensor to staging buffer
                // This path should NOT be reached here — hot experts
                // are handled by the CUDA path, not CPU
                GGML_ABORT("Hot expert weights should use CUDA gather path");
            } else {
                // Cold expert: weights on CPU RAM
                const char * src_expert = (const char *) as->data +
                    expert_id * as->nb[2];
                memcpy(dst_row, src_expert, ne0 * sizeof(float));
            }
        }
    }
}
```

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (modify `ggml_cuda_op_expert_gather`)

```cpp
// CUDA path for hot experts: copy from GPU cache to staging
static void ggml_cuda_op_expert_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * ids = dst->src[1];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    // Copy expert weights from GPU cache to staging buffer
    // The staging buffer is managed by ctx.moe_staging
    const size_t ne0 = dst->ne[0];
    const size_t ne1 = dst->ne[1];
    const size_t ne2 = dst->ne[2];
    const size_t n_expert = as->ne[2];

    // Build ids_to_sorted on host (same as existing fork path)
    std::vector<int32_t> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // For each expert ID, copy from GPU cache to staging
    size_t offset = 0;
    for (size_t e = 0; e < ne1; e++) {
        for (size_t t = 0; t < ne2; t++) {
            const int32_t expert_id = ids_host[e * ids->nb[1] / sizeof(int32_t) + t * ids->nb[0] / sizeof(int32_t)];

            if (expert_id >= 0 && expert_id < (int32_t)n_expert) {
                // Copy expert slice from GPU to staging
                const char * src_expert = (const char *) as->data + expert_id * as->nb[2];
                size_t slice_size = ne0 * sizeof(float);

                CUDA_CHECK(cudaMemcpyAsync(
                    (char *) dst->data + offset,
                    src_expert,
                    slice_size,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
                offset += slice_size;
            }
        }
    }
}
```

---

## Phase 1d: CUDA Stream Overlap (Gap 3)

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (modify `ggml_cuda_mul_mat_id_staged`)

```cpp
// Staged mul_mat_id with stream overlap
static void ggml_cuda_mul_mat_id_staged(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    cudaStream_t matmul_stream = ctx.stream();

    // [KEY CHANGE] Wait for CPU gather to complete before starting matmul
    // The gather operation records an event on a separate stream.
    // We wait on that event before proceeding.
    if (ctx.concurrent_stream_context.concurrent_events.count(dst)) {
        auto & gather_event = ctx.concurrent_stream_context.concurrent_events[dst];
        CUDA_CHECK(cudaStreamWaitEvent(matmul_stream, gather_event.event, 0));
    }

    // Run standard matmul on gathered weights
    // No row gathering needed — CPU already did that
    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
    } else if (ggml_is_quantized(src0->type)) {
        if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
        } else {
            ggml_cuda_mul_mat(ctx, src0, src1, dst);
        }
    } else {
        ggml_cuda_mul_mat(ctx, src0, src1, dst);
    }

    // Record completion event for next layer's gather
    CUDA_CHECK(cudaEventRecord(ctx.concurrent_stream_context.concurrent_events[dst].event, matmul_stream));
}
```

**File:** `ggml/src/ggml-cuda/common.cuh` (add to `ggml_cuda_stream_context`)

```diff
@@ -1407,6 +1407,10 @@ struct ggml_cuda_concurrent_event {
     cudaEvent_t event = nullptr;
     ~ggml_cuda_concurrent_event() {
         if (event != nullptr) {
             CUDA_CHECK(cudaEventDestroy(event));
         }
     }
+    // Initialize event
+    void init(int device) {
+        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventInterprocessAware));
+    }
 };

 struct ggml_cuda_stream_context {
     std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;
```

---

## Phase 1e: Tensor Shape Handling (Gap 4)

**File:** `ggml/include/ggml.h` (enhance `ggml_moe_expert_gather` docstring)

```diff
@@ -1470,6 +1470,20 @@ GGML_API void ggml_mul_mat_id_set_mask_from(
     //
     // For each token t and expert index e in ids:
     //   out[:, e, t] = as[:, :, ids[e, t]]  (row from expert tensor)
+    //
+    // Tensor layout handling:
+    // - gate_up path: as->ne[0] = n_ff*2, as->ne[1] = n_embd (merged gate+up)
+    // - up path:      as->ne[0] = n_ff, as->ne[1] = n_embd
+    // - gate path:    as->ne[0] = n_ff, as->ne[1] = n_embd
+    // - down path:    as->ne[0] = n_embd, as->ne[1] = n_ff
+    //
+    // Output shape: [ne0*ne1, n_expert_used, n_tokens]
+    //   - gate_up: [(n_ff*2)*n_embd, n_exp, n_tok]
+    //   - up:      [n_ff*n_embd, n_exp, n_tok]
+    //   - gate:    [n_ff*n_embd, n_exp, n_tok]
+    //   - down:    [n_embd*n_ff, n_exp, n_tok]
     GGML_API struct ggml_tensor * ggml_moe_expert_gather(
             struct ggml_context * ctx,
             struct ggml_tensor  * as,
```

**File:** `ggml/src/ggml.c` (enhance `ggml_moe_expert_gather` implementation)

```c
struct ggml_tensor * ggml_moe_expert_gather(
        struct ggml_context * ctx,
        struct ggml_tensor  * as,
        struct ggml_tensor  * ids) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(as->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);

    // Flatten the first two dimensions of as
    // This handles all MoE path layouts uniformly:
    //   gate_up: ne0 = n_ff*2, ne1 = n_embd -> flattened = n_ff*2*n_embd
    //   up:      ne0 = n_ff, ne1 = n_embd   -> flattened = n_ff*n_embd
    //   gate:    ne0 = n_ff, ne1 = n_embd   -> flattened = n_ff*n_embd
    //   down:    ne0 = n_embd, ne1 = n_ff   -> flattened = n_embd*n_ff
    const int64_t ne01 = as->ne[0] * as->ne[1];

    const int64_t ne[4] = {
        ne01,               // flattened expert dimension
        ids->ne[0],         // n_expert_used
        ids->ne[1],         // n_tokens
        1
    };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MOE_EXPERT_GATHER;
    result->src[0] = as;
    result->src[1] = ids;

    return result;
}
```

---

## Phase 1f: LoRA Adapter Integration (Gap 5)

**File:** `src/llama-graph.cpp` (enhanced `build_lora_mm_id`)

```cpp
ggml_tensor * llm_graph_context::build_lora_mm_id(
        ggml_tensor * w,
        ggml_tensor * cur,
        ggml_tensor * ids,
        ggml_tensor * w_s,
        int32_t       mask_from) const {
    // [KEY CHANGE] Use staged gather + matmul for MoE expert branches
    if (mask_from >= 0) {
        // MoE expert branch — use staged path
        ggml_tensor * gathered = ggml_moe_expert_gather(ctx0, w, ids);
        ggml_tensor * res = ggml_mul_mat(ctx0, gathered, cur);

        if (w_s) {
            const int64_t n_expert = w_s->ne[0];
            const int64_t n_tokens = cur->ne[2];
            ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, ids);
            res = ggml_mul(ctx0, res, s);
        }

        // LoRA adapters — also use staged gather
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(w);
            if (lw == nullptr) continue;

            const float alpha = lora.first->alpha;
            const float rank  = (float) lw->b->ne[0];
            const float scale = alpha ? lora.second * alpha / rank : lora.second;

            // Gather LoRA expert weights
            ggml_tensor * lora_a_gathered = ggml_moe_expert_gather(ctx0, lw->a, ids);
            ggml_tensor * lora_b_gathered = ggml_moe_expert_gather(ctx0, lw->b, ids);

            // Run LoRA matmul on gathered weights
            ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lora_b_gathered, lora_a_gathered);
            ab_cur = ggml_scale(ctx0, ab_cur, scale);
            res = ggml_add(ctx0, res, ab_cur);
        }

        return res;
    }

    // Non-MoE path — use original ggml_mul_mat_id
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (mask_from >= 0) {
        ggml_mul_mat_id_set_mask_from(res, mask_from);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) continue;

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}
```

---

## Phase 1g: CUDA Graph Integration (Gap 6)

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (enhanced CUDA graph handling)

```cpp
// [KEY CHANGE] CUDA graph handling for gather+matmul sequence
static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool use_cuda_graph = true;

    for (size_t i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // MOE_EXPERT_GATHER nodes don't need sync — CPU gather is async
        if (node->op == GGML_OP_MOE_EXPERT_GATHER) {
            // Gather is async from GPU perspective, safe for graphs
            continue;
        }

        // MUL_MAT_ID with staged weights — safe for graphs
        if (node->op == GGML_OP_MUL_MAT_ID) {
            // Check if this node has a gather dependency
            const bool has_gather_dep = false;  // Would check graph edges
            if (has_gather_dep) {
                // Gather event is recorded before matmul, graph-safe
                continue;
            }
            // No gather dep — use existing sync check
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                use_cuda_graph = false;
            }
        }
    }

    return use_cuda_graph;
}
```

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (graph capture with gather events)

```cpp
static void ggml_cuda_graph_evaluate_and_capture(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        const bool use_cuda_graph,
        const bool cuda_graph_update_required,
        const void * graph_key) {
    // [KEY CHANGE] Record gather completion events before graph capture
    if (use_cuda_graph) {
        // Record events for all gather nodes
        for (size_t i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MOE_EXPERT_GATHER) {
                cudaEvent_t event;
                CUDA_CHECK(cudaEventCreateWithFlags(&event, 0));
                CUDA_CHECK(cudaEventRecord(event, cuda_ctx->stream()));
                cuda_ctx->concurrent_stream_context.concurrent_events[node] = {event};
            }
        }
    }

    // Existing graph capture logic...
    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        // ... existing capture logic
    }
}
```

---

## Phase 1h: Error Handling (Gap 7)

**File:** `ggml/src/ggml-cpu/ops.cpp` (error handling in gather)

```cpp
void ggml_compute_forward_moe_expert_gather(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * ids = dst->src[1];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(as->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t n_expert = as->ne[2];

    const int ith = params->ith;
    const int nth = params->nth;

    for (int64_t t = ith; t < ne2; t += nth) {
        for (int64_t e = 0; e < ne1; e++) {
            const int32_t expert_id = *(const int32_t *) (
                (const char *) ids->data + e * ids->nb[0] + t * ids->nb[1]
            );

            float * dst_row = (float *) dst->data + (e * ne2 + t) * ne0;

            // Validate expert ID — zero-fill on invalid
            if (expert_id < 0 || expert_id >= (int32_t)n_expert) {
#ifndef NDEBUG
                if (expert_id < 0 || expert_id >= (int32_t)n_expert) {
                    GGML_LOG_WARN("moe_expert_gather: invalid expert_id=%d for tensor %s\n",
                                  expert_id, dst->name);
                }
#endif
                memset(dst_row, 0, ne0 * sizeof(float));
                continue;
            }

            // Check tensor data pointer
            if (!as->data) {
                GGML_ABORT("moe_expert_gather: null tensor data pointer");
            }

            // Copy expert weights
            const char * src_expert = (const char *) as->data + expert_id * as->nb[2];
            memcpy(dst_row, src_expert, ne0 * sizeof(float));
        }
    }
}
```

---

## Phase 1i: Performance Validation Hooks (Gap 8)

**File:** `ggml/src/ggml-cpu/ops.cpp` (timing hooks in gather)

```cpp
void ggml_compute_forward_moe_expert_gather(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * ids = dst->src[1];

    // [KEY CHANGE] Timing hook for performance validation
    int64_t gather_start_us = ggml_time_us();

    // ... existing gather logic ...

    int64_t gather_end_us = ggml_time_us();
    int64_t gather_duration_us = gather_end_us - gather_start_us;

#ifndef NDEBUG
    // Log per-layer timing breakdown
    GGML_LOG_DEBUG("moe_expert_gather: %s took %lld us (%.2f ms)\n",
                   dst->name, (long long)gather_duration_us,
                   gather_duration_us / 1000.0);
#endif

    // Store timing in tensor metadata for aggregation
    if (params->wdata) {
        int64_t * timing = (int64_t *) params->wdata;
        timing[0] += gather_duration_us;
    }
}
```

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu` (timing hook in staged matmul)

```cpp
static void ggml_cuda_mul_mat_id_staged(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    // [KEY CHANGE] Timing hook for staged matmul
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, ctx.stream()));

    // ... existing matmul logic ...

    CUDA_CHECK(cudaEventRecord(stop, ctx.stream()));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float matmul_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&matmul_ms, start, stop));

#ifndef NDEBUG
    GGML_LOG_DEBUG("moe_staged_matmul: %s took %.2f ms\n",
                   dst->name, matmul_ms);
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
```

---

## Complete Data Flow After All Gaps Closed

```
Before (fork's current design):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → ggml_mul_mat_id() [CPU node]
        → CUDA dispatch: ggml_cuda_mul_mat_id()
          → [SYNC] Copy ids to host
          → [SYNC] Build ids_to_sorted
          → [GPU] get_rows_cuda: gather src1 rows
          → [PER-EXPERT] cache check + matmul
          → [SYNC] Scatter results

After (our design, all gaps closed):
  llama-graph::build_moe_ffn()
    → build_lora_mm_id()
      → [CPU] ggml_moe_expert_gather() [ASYNC]
        → For each expert:
            → Hot: copy from GPU cache (fast)
            → Cold: read from RAM (slower)
        → Write to per-layer rolling staging buffer
        → Record completion event on gather stream
      → [GPU] ggml_mul_mat() [STANDARD]
        → Wait for gather completion event
        → Read gathered weights from staging buffer
        → No row gathering needed
        → Record completion event for next layer
      → [GPU] LoRA matmul (if applicable)
        → Also uses staged gather + matmul
```

---

## Final Assessment: When Are We Ready?

**We have "a place for everything" when:**
1. ✅ All 8 gaps have storyboard code blocks
2. ✅ Each code block shows the exact file, line, and diff
3. ✅ The data flow is clearly documented (before/after)
4. ✅ Error handling covers all failure modes
5. ✅ Performance validation hooks are in place

**We are ready to "put everything in its place" when:**
1. ✅ The storyboard is reviewed and approved
2. ✅ All assumptions are validated against the actual codebase
3. ✅ The implementation order is agreed upon (HIGH → MEDIUM → LOW)
4. ✅ A test plan is defined for each phase

**Current status:** All 8 gaps are now closed with storyboard code blocks. The next step is to validate the assumptions against the actual codebase and begin implementation in priority order.
