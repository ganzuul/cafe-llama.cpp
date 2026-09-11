#!/bin/sh
set -eu

# Controlled Nsight Compute capture for the lightweight Nail MTP workload.
# This is intentionally separate from run_llamacpp_nail-MTP.sh: profiling uses
# one slot, a short request, and a dedicated port so results are comparable.
#
# Usage:
#   tools/profile_nail_mtp_ncu.sh [output-prefix]
#
# The script does not enable --host-moe or -ngl 99. Those settings are not
# appropriate for the available memory and are not needed to measure the MTP
# CUDA/CPU transfer behavior.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Nail's original artifact is compatible with the beellama build. Override
# NAIL_SERVER to profile another build, such as this fork's server.
SERVER=${NAIL_SERVER:-/run/media/nos/games/inference_engines/beellama.cpp/build/bin/llama-server}
MODEL=${NAIL_MODEL:-/home/nos/models/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_S.gguf}
PORT=${NAIL_PROFILE_PORT:-18080}
OUT=${1:-"$ROOT/build/ncu/nail-mtp"}
REPORT="${OUT}.ncu-rep"
LOG="${OUT}.server.log"

mkdir -p "$(dirname -- "$OUT")"
rm -f "$REPORT" "$LOG"

if ! command -v ncu >/dev/null 2>&1; then
    echo "error: ncu is not installed or not on PATH" >&2
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

SERVER_PID=
cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        # Stop the profiled server child first so ncu can finalize its report.
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --target-processes all includes CUDA work spawned by the server. --set basic
# keeps the first iteration practical; use NCU_SET=full for a much heavier
# counter sweep after the baseline capture is understood.
NCU_SET=${NCU_SET:-basic}

ncu \
  --target-processes all \
  --set "$NCU_SET" \
  --launch-count 300 \
  --kernel-name-base demangled \
  --export "$OUT" \
  --force \
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
    --spec-type draft-mtp \
    --spec-ngram-map-k-size-n 12 \
    --spec-ngram-map-k-size-m 48 \
    --spec-ngram-map-k-min-hits 1 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --jinja \
  >"$LOG" 2>&1 &
SERVER_PID=$!

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

# Fixed prompt and length make repeated captures comparable. The server is
# terminated after the request, which also finalizes the .ncu-rep report.
curl --fail --silent --show-error \
    "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain why the sky appears blue in one concise sentence.","n_predict":32,"temperature":0}' \
    >"${OUT}.completion.json"

cleanup
SERVER_PID=

echo "Nsight Compute report: $REPORT"
echo "Server log:           $LOG"
echo "Completion result:    ${OUT}.completion.json"
