#!/bin/bash
set -Eeuo pipefail


EXP_NAME="${EXP_NAME:-gpt2_small_run1}"
CONFIG="${CONFIG:-configs/model/gpt2-small.yaml}"
INPUT_PATH="${INPUT_PATH:-./input.txt}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
SAVE_PATH="${SAVE_PATH:-$SAVE_DIR/${EXP_NAME}.pt}"
RESUME="${RESUME:-}"
SEED="${SEED:-69}"

mkdir -p "$SAVE_DIR" "logs"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input file not found: $INPUT_PATH"
  exit 1
fi

if ! command -v python &> /dev/null; then
  echo "Python not found in PATH"
  exit 1
fi


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/${EXP_NAME}_${TIMESTAMP}.log"
echo "Logging to $LOGFILE"


cleanup() {
    echo "Caught interrupt. Python should have saved a checkpoint."
    exit 1
}
trap cleanup SIGINT SIGTERM

CMD=(python -m src.main
    --config "$CONFIG"
    --exp_name "$EXP_NAME"
    --input "$INPUT_PATH"
    --save_path "$SAVE_PATH"
    --seed "$SEED"
)

if [[ -n "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi

echo "Launching training: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOGFILE"
