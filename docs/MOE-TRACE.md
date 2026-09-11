# MoE Trace Instrumentation

## Overview

The fork provides lightweight, opt-in JSONL instrumentation for MoE routing.
It traces both CPU and CUDA `mul_mat_id` dispatches without requiring profiling
tools.

## Enabling

Set the environment variable before running llama.cpp:

```bash
export LLAMA_MOE_TRACE=/tmp/moe-trace.jsonl
./build/bin/llama-server ...
```

Both CPU and CUDA layers write to the same file.

## Output Fields

### CPU (`ggml-cpu.c`)

```json
{"backend":"cpu","op":"mul_mat_id","ne0":..., "ne1":..., "ne2":...,
 "n_ids":..., "n_expert":..., "routes":..., "masked":..., "unique_experts":..., "wall_us":...}
```

- `ne0`, `ne1`, `ne2` tensor shapes
- `n_ids` number of selected experts per token
- `n_expert` total expert count
- `routes` total forward selections
- `masked` masked-out experts
- `unique_experts` unique experts selected
- `wall_us` CPU wall time for the dispatch

### CUDA (`ggml-cuda.cu`)

```json
{"op":"mul_mat_id","ne0":..., "ne1":..., "ne2":...,
 "wall_us":..., "h2d_bytes":..., "d2h_bytes":..., "hot_hits":..., "cold_misses":...}
```

- `wall_us` CUDA wall time for the dispatch
- `h2d_bytes` estimated Host→Device transfer volume
- `d2h_bytes` estimated Device→Host transfer volume
- `hot_hits` staging-cache hits (GPU-hot experts)
- `cold_misses` staging-cache misses (CPU-cold experts)

## Analysis

> **Measurement caveat (2026-09, RTX 2070 SUPER / ~23 GiB RAM).** We cannot
distinguish trace-on from trace-off throughput on this host. Same recipe,
same 66-token prompt, `max_tokens=96`, greedy:
>
> | Condition | Prefill | Decode |
> |---|---|---|
> | trace on (two-flag `-ot`) | 2.97 tok/s | 2.36 tok/s |
> | trace on (one-line `-ot`) | 2.99 / 3.26 tok/s | 2.73 / 3.88 tok/s |
> | trace off | 3.67 / 2.47 tok/s | 3.10 / 4.06 tok/s |
>
> Run-to-run spread (2.4-4.1 tok/s decode) exceeds any apparent difference.
> Likely variance sources: page-cache state for the 61 GiB mmap'd GGUF,
> expert-weight page residency left over from the previous run, and 12 threads
> on a host already under memory pressure. These figures must **not** be cited
> as a measurement of `LLAMA_MOE_TRACE` cost. A defensible overhead number
> needs >=5 repetitions per arm on a warmed page cache.

Use the provided script or a custom parser:

```bash
awk '{gsub(/[{}":,]/,""); print $2,$3,$5,$6,$7,$8,$9,$10}' /tmp/moe-trace.jsonl | head
```

CPU trace helps identify:
- routing sparsity
- expert selection balance
- CPU compute stalls

CUDA trace helps identify:
- PCIe bandwidth usage
- staging-cache efficiency
- transfer vs compute ratio
