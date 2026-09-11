#!/bin/sh
# Flash-Next ranked hot/cold serving recipe for this host:
#   GPU: RTX 2070 SUPER, 8 GiB VRAM
#   Host: ~23 GiB RAM
#   Model: ~61 GiB GGUF; do not try to make the whole model resident.
#
# Placement strategy from ftq-recipe/docs/staged-expert-fetch-design.md:
#   * keep attention, norm, recurrent state, and other non-expert layers on GPU;
#   * keep ranked hot and cold expert tensors out of VRAM;
#   * use the CPU quantized mul_mat_id/cache path for expert matmuls;
#   * keep the large PLE/N-gram table on demand through mmap/SSD.
#
# Explicit tensor overrides prevent -ngl from allocating per-layer expert
# buffers on the GPU while retaining the GPU attention/rs-cache workload.
#
# Intentionally NOT used:
#   --host-moe / -hmoe: pins all expert weights in host RAM and disables the
#       memory-fitting pass. The ~61 GiB model cannot fit in ~23 GiB RAM; this
#       was observed to thrash instead of starting reliably.
#   -ngl 99: requests effectively every layer on GPU and OOMs this 8 GiB GPU.
#   --pipeline-parallel: intended for multi-device streaming; on this single
#       GPU it attempted a ~162 GiB compute allocation and only worked by
#       falling back. Leave it disabled for the stable single-GPU recipe.
#   LLAMA_MOE_EXPERT_GATHER=1: experimental f32-only gather path; this IQ2_XXS
#       artifact must use the standard quantized CPU mul_mat_id path.
#
# Active placement flags:
#   -ngl 48: measured FTQ recipe for GPU attention/norm/recurrent state.
#   -ot '...hot=CPU,...cold=CPU': keep ranked hot/cold expert tensors on the
#       host (single comma-separated value; repeated -ot is deprecated).
#   --ngram-ssd: keep the PLE table mmap/SSD-backed when present.
#   --cache-ram 0: the server's default 8192 MiB prompt/KV reuse cache competes
#       with the page cache that keeps expert weights resident. Measured A/B
#       (3 reps each, verified port ownership, non-overlapping fault ranges):
#           default : decode 2.85 +/- 0.22 tok/s, 1631 +/- 176 majflt/token
#           cram=0  : decode 3.17 +/- 0.30 tok/s, 1417 +/-  76 majflt/token
#       i.e. -13.1% major faults, +11.3% decode. This is a strict win for a
#       single-slot benchmarking recipe; multi-user serving would want a
#       nonzero value, so revisit before deploying this as a shared server.
#
# NOTE on measurement (2026-09, RTX 2070 SUPER / ~23 GiB RAM):
#   The one-line -ot form is a cleanup only. It is NOT a performance change.
#   Verified equivalent to the previous two-flag form: both resolve every
#   expert tensor to buffer type CUDA_Host (the "=CPU" value routes through
#   select_weight_buft over buft_list_cpu; it does not mean the plain CPU
#   buffer). The only observable difference is that the one-line form no
#   longer emits the DEPRECATED repeated-argument warning.
#
#   Separate A/B result that must not be over-read: we CANNOT distinguish
#   trace-on from trace-off throughput on this host. Observed decode rates:
#       trace on  (two-flag):  2.36 tok/s
#       trace on  (one-line):  2.73, 3.88 tok/s
#       trace off:             3.10, 4.06 tok/s
#   Run-to-run spread (2.4-4.1 tok/s) exceeds any apparent difference. One
#   sample per arm is not enough; a defensible overhead number needs >=5
#   repetitions per arm on a warmed page cache. Do not cite these as a
#   measurement of LLAMA_MOE_TRACE cost.
exec /run/media/nos/games/inference_engines/workstreams/llama.cpp-mtp-slot-state/build/bin/llama-server \
  --model /home/nos/models/Qwen3.8-Flash-Next-FTQ-RANKED-H256-IQ2XXS.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 0.0 \
  --flash-attn auto \
  --reasoning-budget 8192 \
  --kv-unified \
  --ctx-size 4096 \
  -np 1 \
  --cache-prompt \
  --log-colors on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -ngl 48 \
  -ot 'blk.*.ffn_(gate|up|down)_exps_hot=CPU,blk.*.ffn_(gate|up|down)_exps_cold=CPU' \
  --cache-ram 0 \
  --ngram-ssd
