# Loop Guard Problem: Hidden Reasoning Loop That Won't Break

## Problem Description

The loop guard in `beellama.cpp` is triggering at the same point in the output but **is not effectively breaking the loop**. The model gets stuck in a repeating pattern and the force-close intervention doesn't prevent it from re-entering the same loop.

## Observed Behavior

### First occurrence:
```
" So the flow is:
 1. GPU computes layer N on main stream
 2. CPU starts gather for layer N+1 on gather stream (async with GPU compute)
 3. CPU gather completes, records event on gather stream
 4. GPU matmul for layer N+1 waits on gather event, then runs on    "
```

### Second occurrence (almost identical):
```
" So the flow is:
 1. GPU computes layer N on main stream
     2"
```

The model is **stuck in a self-referential loop** about the implementation itself. It's describing the flow it's currently implementing, and gets stuck in a loop about that description.

## Loop Guard Details

- **Message:** `loop guard force-closing hidden reasoning at token 1088 (interventions=1 token=1328 piece=''): ngram_dominance period=8 coverage=611 score=0.597`
- **Region:** hidden reasoning (not visible output)
- **Period:** 8 tokens
- **Coverage:** 611 tokens matched the pattern
- **Score:** 0.597 (loop confidence)
- **Intervention:** force-close reasoning block

## Why It's Not Working

The loop guard **force-closes the reasoning block** but the model **immediately re-enters the same loop** because:

1. The loop is about the **implementation details** (GPU streams, gather events, etc.)
2. The model is describing its own implementation, creating a **self-referential loop**
3. The force-close only terminates the current reasoning block, but the model **re-generates** the same content
4. The context window contains the loop, so the model keeps **re-reading and re-generating** the same pattern

## Root Cause

The loop is caused by the model being asked to **describe complex technical implementation details** (CUDA streams, gather events, double-buffering) while **simultaneously implementing them**. This creates a feedback loop where:

1. The model describes the implementation
2. The model implements the description
3. The model describes the implementation again
4. The loop repeats

## Potential Solutions

1. **Increase loop guard aggressiveness:** More interventions, lower score threshold
2. **Break the self-reference:** Don't ask the model to describe its own implementation in real-time
3. **Add context limits:** Truncate the loop from the context window
4. **Switch to code-only mode:** Stop describing, just implement
5. **Externalize the description:** Document the design separately, don't ask the model to generate it

## Recommendation

For this session, **switch to code-only mode**. Document the design in a separate file (like this one), and stop asking the model to describe the implementation in the conversation. The loop guard will still trigger, but the model will at least **not re-enter the loop** if we don't keep feeding it the same context.

---

*Documented: 2026-01-15*
