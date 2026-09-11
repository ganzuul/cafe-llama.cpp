#!/bin/sh
set -eu

# Privilege-aware wrapper for NVIDIA profiler permissions.
# Nsight Systems runs unprivileged; only Nsight Compute and its target process
# are launched through sudo because restricted hardware counters require it.
# Run this wrapper as the normal user.
#
# Usage:
#   tools/profile_nail_mtp_root.sh nsys  prefill [output-prefix]
#   tools/profile_nail_mtp_root.sh nsys  decode  [output-prefix]
#   tools/profile_nail_mtp_root.sh ncu   decode  [output-prefix]
#
# Examples:
#   tools/profile_nail_mtp_root.sh nsys decode build/nsys/nail-decode
#   tools/profile_nail_mtp_root.sh ncu  decode build/ncu/nail-decode-root

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PROFILER=${1:-}
MODE=${2:-decode}
OUT=${3:-}

case "$PROFILER" in
    nsys)
        SCRIPT="$ROOT/tools/profile_nail_mtp_nsys.sh"
        DEFAULT_OUT="$ROOT/build/nsys/nail-mtp-${MODE}-root"
        ;;
    ncu)
        SCRIPT="$ROOT/tools/profile_nail_mtp_ncu.sh"
        DEFAULT_OUT="$ROOT/build/ncu/nail-mtp-${MODE}-root"
        ;;
    *)
        echo "usage: $0 {nsys|ncu} {prefill|decode} [output-prefix]" >&2
        exit 2
        ;;
esac

case "$MODE" in
    prefill|decode) ;;
    *)
        echo "error: mode must be prefill or decode" >&2
        exit 2
        ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    echo "error: run this wrapper as the normal user; it invokes sudo itself" >&2
    exit 1
fi
# Nsight Systems does not need root for our timeline/bandwidth measurements.
# Do not invoke askpass or sudo for it.
if [ "$PROFILER" = "nsys" ]; then
    exec "$SCRIPT" "$OUT"
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "error: sudo is required for Nsight Compute" >&2
    exit 1
fi

# Background tool processes do not have a terminal. Prefer the installed
# graphical askpass helper so sudo can authenticate without stdin/TTY input.
# Override SUDO_ASKPASS if a different desktop helper is preferred.
if [ -z "${SUDO_ASKPASS:-}" ]; then
    if [ -x /usr/bin/ksshaskpass ]; then
        SUDO_ASKPASS=/usr/bin/ksshaskpass
    elif [ -x /usr/bin/ssh-askpass ]; then
        SUDO_ASKPASS=/usr/bin/ssh-askpass
    else
        echo "error: no graphical askpass helper found; set SUDO_ASKPASS" >&2
        exit 1
    fi
fi
export SUDO_ASKPASS

if [ ! -x "$SCRIPT" ]; then
    echo "error: missing profiler script: $SCRIPT" >&2
    exit 1
fi

OUT=${OUT:-$DEFAULT_OUT}
USER_NAME=${USER}
USER_GROUP=$(id -gn)

# Environment is passed explicitly because sudo may sanitize PATH/HOME. The
# profiler scripts default to the compatible beellama Nail server; NAIL_SERVER
# can still be overridden by the caller.
sudo -A env \
    PATH="$PATH" \
    HOME="$HOME" \
    NAIL_MODEL="${NAIL_MODEL:-/home/nos/models/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_S.gguf}" \
    NAIL_SERVER="${NAIL_SERVER:-/run/media/nos/games/inference_engines/beellama.cpp/build/bin/llama-server}" \
    NAIL_PROFILE_MODE="$MODE" \
    NAIL_PROFILE_PORT="${NAIL_PROFILE_PORT:-}" \
    NAIL_N_PREDICT="${NAIL_N_PREDICT:-}" \
    "$SCRIPT" "$OUT"

# sudo creates root-owned reports. Return ownership to the invoking user so
# nsys-ui/ncu-ui and follow-up scripts can inspect them without sudo.
PREFIX_DIR=$(dirname -- "$OUT")
if [ -d "$PREFIX_DIR" ]; then
    sudo chown -R "$USER_NAME:$USER_GROUP" "$PREFIX_DIR"
fi

echo "Profiler artifacts are owned by $USER_NAME:$USER_GROUP"
