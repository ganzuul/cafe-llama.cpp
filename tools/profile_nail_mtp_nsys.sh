#!/bin/sh
set -eu

# Controlled Nsight Systems capture for the lightweight Nail MTP workload.
# Usage:
#   tools/profile_nail_mtp_nsys.sh [output-prefix]
#
# The capture uses one slot and a fixed short request. It is intended to show
# CPU scheduling, CUDA kernels, H2D/D2H copies, synchronization, and idle gaps.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Nail's original artifact is compatible with the beellama build. Override
# NAIL_SERVER to profile another build, such as this fork's server.
SERVER=${NAIL_SERVER:-/run/media/nos/games/inference_engines/beellama.cpp/build/bin/llama-server}
MODEL=${NAIL_MODEL:-/home/nos/models/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_S.gguf}
SPEC_TYPE=${NAIL_SPEC_TYPE:-draft-mtp}
PORT=${NAIL_PROFILE_PORT:-18081}
MODE=${NAIL_PROFILE_MODE:-decode}
# The Nail model typically finishes initialization around 27 seconds here;
# begin collection shortly before the server becomes ready so the request is
# captured without needing privileged attach/capture controls.
NSYS_DELAY=${NAIL_NSYS_DELAY:-25}
OUT=${1:-"$ROOT/build/nsys/nail-mtp-${MODE}"}
REPORT="${OUT}.nsys-rep"
LOG="${OUT}.server.log"

mkdir -p "$(dirname -- "$OUT")"
rm -f "$REPORT" "$LOG"

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys is not installed or not on PATH" >&2
    exit 1
fi
if [ ! -x "$SERVER" ]; then
    echo "error: missing executable: $SERVER" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "error: missing model: $MODEL" >&2
    exit 1
fi

PROFILE_PID=
cleanup() {
    if [ -n "${PROFILE_PID:-}" ]; then
        # Stop the profiled server child first. This lets nsys observe a clean
        # target exit and finalize the .nsys-rep file before we reap nsys.
        pkill -TERM -P "$PROFILE_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$PROFILE_PID" 2>/dev/null || break
            sleep 1
        done
        kill "$PROFILE_PID" 2>/dev/null || true
        wait "$PROFILE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# No sampling overhead: the useful information here is CUDA/OS runtime and
# copy-engine/kernel timeline data. --stats=true emits a compact summary too.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --stats=true \
  --delay="$NSYS_DELAY" \
  --stop-on-exit=true \
  --output="$OUT" \
  --force-overwrite=true \
  "$SERVER" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --flash-attn auto \
    --reasoning-budget 1024 \
    --kv-unified \
    --ctx-size 4096 \
    -np 1 \
    --cache-prompt \
    --log-colors off \
    --spec-type "$SPEC_TYPE" \
    --spec-ngram-map-k-size-n 12 \
    --spec-ngram-map-k-size-m 48 \
    --spec-ngram-map-k-min-hits 1 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --jinja \
  >"$LOG" 2>&1 &
PROFILE_PID=$!

for _ in $(seq 1 90); do
    if grep -q "listening on" "$LOG" 2>/dev/null; then
        break
    fi
    if grep -qiE "assert|aborted|cudaMalloc failed|out of memory|exiting due|invalid argument" "$LOG" 2>/dev/null; then
        echo "server failed during startup; see $LOG" >&2
        exit 1
    fi
    sleep 1
done

if ! grep -q "listening on" "$LOG" 2>/dev/null; then
    echo "server did not become ready; see $LOG" >&2
    exit 1
fi

# Delay collection beyond model initialization, then isolate either a long
# prompt/prefill or a decode-heavy request. The server timing lines in LOG
# provide the exact prompt-eval and generation intervals for correlation with
# the Nsight Systems memory-operation timestamps.
case "$MODE" in
    prefill)
        PROMPT_TOKENS=${NAIL_PROMPT_TOKENS:-2048}
        N_PREDICT=${NAIL_N_PREDICT:-1}
        ;;
    decode)
        PROMPT_TOKENS=${NAIL_PROMPT_TOKENS:-16}
        # Longer decode amortizes profiler and graph setup overhead; override
        # with NAIL_N_PREDICT for shorter diagnostic runs.
        N_PREDICT=${NAIL_N_PREDICT:-512}
        ;;
    *)
        echo "error: NAIL_PROFILE_MODE must be prefill or decode" >&2
        exit 1
        ;;
esac

PROMPT_JSON="${OUT}.prompt.json"
python3 - "$PROMPT_JSON" "$PROMPT_TOKENS" "$N_PREDICT" <<'PY'
import json, sys
out, n, n_predict = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# Repeated deterministic text gives a stable approximate token count while
# remaining valid natural-language input for the model.
text = ("Explain the physical reason that the sky appears blue. " * max(1, n // 10))
with open(out, "w") as f:
    json.dump({"prompt": text, "n_predict": n_predict, "temperature": 0}, f)
PY

# Ensure the delayed profiler has entered collection before submitting work.
sleep 2
curl --fail --silent --show-error \
    "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    --data-binary "@$PROMPT_JSON" \
    >"${OUT}.completion.json"

cleanup
PROFILE_PID=

echo "Nsight Systems report: $REPORT"
echo "Server log:           $LOG"
echo "Completion result:    ${OUT}.completion.json"
