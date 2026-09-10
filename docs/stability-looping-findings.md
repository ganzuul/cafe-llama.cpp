# Stability and Looping Issues: Findings

**Date**: 2026-09-10  
**Status**: Active investigation — rolling back conversation

---

## 1. Loop Guard Problem

### Observed Behavior

The loop guard in `beellama.cpp` triggers repeatedly at the same point in hidden reasoning output:

```
loop guard force-closing hidden reasoning at token 1088 (interventions=1 token=1328 piece=''): 
ngram_dominance period=8 coverage=611 score=0.597
```

- **Region**: Hidden reasoning (not visible output)
- **Period**: 8 tokens
- **Coverage**: 611 tokens matched the pattern
- **Score**: 0.597 (loop confidence)
- **Intervention**: Force-close reasoning block

### Model Gets Stuck In

The model enters a **self-referential loop** about the implementation itself:

```
" So the flow is:
 1. GPU computes layer N on main stream
 2. CPU starts gather for layer N+1 on gather stream (async with GPU compute)
 3. CPU gather completes, records event on gather stream
 4. GPU matmul for layer N+1 waits on gather event, then runs on    "
```

This repeats almost identically, then degrades:

```
" So the flow is:
 1. GPU computes layer N on main stream
     2"
```

### Why It's Not Working

1. The loop is about **implementation details** (CUDA streams, gather events, double-buffering)
2. The model describes its own implementation → self-referential loop
3. Force-close only terminates the current reasoning block; model **re-generates** the same content
4. Context window contains the loop → model keeps **re-reading and re-generating** the same pattern

### Root Cause

The model is asked to **describe complex technical implementation details** (CUDA streams, gather events, double-buffering) while **simultaneously implementing them**. This creates a feedback loop:

1. Model describes the implementation
2. Model implements the description
3. Model describes the implementation again
4. Loop repeats

---

## 2. Numeric Sensitivity in MUL_MAT_ID Path

### Observed Behavior

From `WORKBILL-ftq-hot-cold.md`:

- **Identity expert ordering** (hot set `[0, H)`): PPL = 6.8541 ± 0.63615
- **Non-identity expert ordering** (usage-ranked, shuffled): PPL = 6.91–7.05

### Analysis

- Full byte verification + runtime probe confirms **NOT a file-generation bug**
- Router IDs at layer 0 match original EXACTLY through the permutation
- Divergence begins at **layer 1** — the PLE/engram injection layer
- Row selection at layer 1 reads the **recurrent SSM state**
- Working theory: **1-ulp numeric sensitivity** in the fork's split MUL_MAT_ID path
- Amplified by discrete PLE row-hash
- Identity ordering is immune (router IDs = slot positions, no remap)

### Implication

With flat routing histogram (~250/512 experts carry 80%), identity hot pool `[0,256)` is statistically equivalent to usage-ranked for residency purposes. Usage-ranked hot sets only worth revisiting after ordering quirk is fixed.

---

## 3. Current Implementation State

### Completed

- **MVP Step 0**: `GGML_OP_MOE_EXPERT_GATHER` op + CPU forward + graph wiring
- **Step 1**: Tensor placement overrides (`common_host_buffer_type()`)
- **Step 2**: Performance validation hooks (`GGML_DEBUG >= 20` timing logs)
- **Steps 3 & 4**: Staging buffer manager + hot/cold routing (LRU eviction)
- **Gap 3 partial**: `ggml_moe_gather_stream_ctx` struct + event recording in matmul loop

### In Progress

- **Gap 3**: CUDA stream overlap — event wait before matmul loop, gather stream integration
- Build verified clean (commit `9c4d9f382` → `mvp-performance-hooks`)

### Blocked

- None (but loop guard prevents further conversation progress)

---

## 4. Recommendations

### Immediate

1. **Switch to code-only mode** — stop asking model to describe implementation in conversation
2. **Document design separately** — this file, `cpu-mul-mat-id-storyboard.md`, `cpu-mul-mat-id-gaps.md`
3. **Roll back conversation** — truncate context to break the self-reference cycle
4. **Continue implementation via file edits only** — no conversational description

### Long-term

1. **Increase loop guard aggressiveness** — lower score threshold, more interventions
2. **Add context limits** — truncate loop from context window when detected
3. **Externalize description** — design docs in files, not generated in conversation
4. **Fix numeric sensitivity** — instrument MUL_MAT_ID inputs/outputs for permuted experts

---

## 5. Files Referenced

| File | Purpose |
|---|---|
| `docs/loop-guard-problem.md` | Original loop guard analysis |
| `docs/cpu-mul-mat-id-storyboard.md` | Implementation storyboard (Phase 1–3) |
| `docs/cpu-mul-mat-id-gaps.md` | Gap analysis vs prior art |
| `docs/cpu-mul-mat-id-gpu-expert-cache.md` | Architecture analysis |
| `docs/mvp-on-ramp.md` | MVP completion status |
| `WORKBILL-ftq-hot-cold.md` | FTQ hot/cold split work bill |
| `ggml/src/ggml-cuda/common.cuh` | Staging manager + gather stream ctx |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA dispatch + matmul loop |

---

*Documented: 2026-09-10*  
*Next action: Roll back conversation, continue via file edits only*
