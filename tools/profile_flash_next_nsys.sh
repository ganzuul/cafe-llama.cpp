#!/bin/sh
set -eu

# Flash-Next Nsight Systems profile using the stable single-GPU ranked
# hot/cold recipe. This intentionally does not use --host-moe, --pipeline-
# parallel, -ngl 99, or experimental LLAMA_MOE_EXPERT_GATHER=1.
#
# Usage:
#   tools/profile_flash_next_nsys.sh prefill [output-prefix]
#   tools/profile_flash_next_nsys.sh decode  [output-prefix]

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SERVER=${FLASH_SERVER:-$ROOT/build/bin/llama-server}
MODEL=${FLASH_MODEL:-/home/nos/models/Qwen3.8-Flash-Next-FTQ-RANKED-H256-IQ2XXS.gguf}
MODE=${1:-decode}
PORT=${FLASH_PROFILE_PORT:-18082}
DELAY=${FLASH_NSYS_DELAY:-30}
OUT=${2:-"$ROOT/build/nsys/flash-next-$MODE"}
LOG="${OUT}.server.log"
PROMPT_JSON="${OUT}.prompt.json"

mkdir -p "$(dirname -- "$OUT")"
rm -f "${OUT}.nsys-rep" "${OUT}.sqlite" "$LOG" "$PROMPT_JSON" "${OUT}.completion.json"

[ -x "$SERVER" ] || { echo "error: missing executable: $SERVER" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "error: missing model: $MODEL" >&2; exit 1; }
command -v nsys >/dev/null 2>&1 || { echo "error: nsys is required" >&2; exit 1; }

case "$MODE" in
    prefill)
        PROMPT_TOKENS=${FLASH_PROMPT_TOKENS:-2048}
        N_PREDICT=${FLASH_N_PREDICT:-1}
        ;;
    decode)
        PROMPT_TOKENS=${FLASH_PROMPT_TOKENS:-16}
        N_PREDICT=${FLASH_N_PREDICT:-512}
        ;;
    *)
        echo "error: mode must be prefill or decode" >&2
        exit 2
        ;;
esac

PROFILE_PID=
cleanup() {
    if [ -n "${PROFILE_PID:-}" ]; then
        pkill -TERM -P "$PROFILE_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$PROFILE_PID" 2>/dev/null || break
            sleep 1
        done
        kill "$PROFILE_PID" 2>/dev/null || true
        wait "$PROFILE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

python3 - "$PROMPT_JSON" "$PROMPT_TOKENS" "$N_PREDICT" <<'PY'
import json, sys
out, n_tokens, n_predict = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
text = "Explain the physical reason that the sky appears blue. " * max(1, n_tokens // 10)
with open(out, "w") as f:
    json.dump({"prompt": text, "n_predict": n_predict, "temperature": 0}, f)
PY

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=true \
    --stats=true \
    --delay="$DELAY" \
    --stop-on-exit=true \
    --output="$OUT" \
    --force-overwrite=true \
    "$SERVER" \
      --model "$MODEL" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --flash-attn auto \
      --reasoning-budget 8192 \
      --kv-unified \
      --ctx-size 4096 \
      -np 1 \
      --cache-prompt \
      --log-colors off \
      --cache-type-k q4_0 \
      --cache-type-v q4_0 \
      -ngl 48 \
      -ot 'blk.*.ffn_(gate|up|down)_exps_hot=CPU' \
      -ot 'blk.*.ffn_(gate|up|down)_exps_cold=CPU' \
      --ngram-ssd \
    >"$LOG" 2>&1 &
PROFILE_PID=$!

for _ in $(seq 1 180); do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    grep -qiE "assert|aborted|cudaMalloc failed|out of memory|exiting due|invalid argument|failed to load model" "$LOG" 2>/dev/null && {
        echo "Flash-Next server failed during startup; see $LOG" >&2
        exit 1
    }
    sleep 1
done

grep -q "listening on" "$LOG" 2>/dev/null || {
    echo "Flash-Next server did not become ready; see $LOG" >&2
    exit 1
}

# The delayed capture should now be active; submit only after model readiness.
sleep 2
curl --fail --silent --show-error \
    "http://127.0.0.1:${PORT}/completion" \
    -H 'Content-Type: application/json' \
    --data-binary "@$PROMPT_JSON" \
    >"${OUT}.completion.json"

cleanup
PROFILE_PID=

echo "Nsight Systems report: ${OUT}.nsys-rep"
echo "Server log:           $LOG"
echo "Completion result:    ${OUT}.completion.json"
