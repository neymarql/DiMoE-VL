#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/qianlong/DiMoE-VL}
MIN_STEPS=200
JOB_ID=""
STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      JOB_ID=$2
      shift 2
      ;;
    --stage)
      STAGE=$2
      shift 2
      ;;
    --min-steps)
      MIN_STEPS=$2
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$JOB_ID" || -z "$STAGE" ]]; then
  echo "usage: $0 --job-id <id> --stage <stage_name> [--min-steps 200]" >&2
  exit 2
fi

cd "$ROOT_DIR"
METRIC_FILE="artifacts/train/${STAGE}/logs/train_metrics.jsonl"

if [[ ! -f "$METRIC_FILE" ]]; then
  echo "FAIL missing metrics file: $METRIC_FILE" >&2
  exit 1
fi

STATE=$(sacct -j "$JOB_ID" --format=State -n | head -n 1 | xargs || true)
if [[ -z "$STATE" ]]; then
  STATE=$(squeue -j "$JOB_ID" -h -o "%T" | head -n 1 | xargs || true)
fi
if [[ -z "$STATE" ]]; then
  STATE="UNKNOWN"
fi

case "$STATE" in
  FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*)
    echo "FAIL job state=$STATE" >&2
    exit 1
    ;;
esac

LOG_PATH=$(scontrol show job "$JOB_ID" 2>/dev/null | sed -n 's/.*StdOut=\\([^ ]*\\).*/\\1/p' | head -n 1)
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH=""
fi

python - <<PY
import json
import math
import os
import re
from pathlib import Path

metric = Path("${METRIC_FILE}")
min_steps = int(${MIN_STEPS})
state = "${STATE}"
log_path = Path("${LOG_PATH}") if "${LOG_PATH}" else None

last_step = -1
bad_rows = 0
with metric.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        s = int(row.get("step", -1))
        last_step = max(last_step, s)
        for k in ("loss_total", "loss_branch1", "loss_branch2"):
            v = row.get(k, 0.0)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                bad_rows += 1
                break

if last_step < min_steps:
    raise SystemExit(f"FAIL insufficient steps: last_step={last_step} < {min_steps}")
if bad_rows > 0:
    raise SystemExit(f"FAIL non-finite loss rows detected: {bad_rows}")

if log_path and log_path.exists():
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"NCCL error",
        r"RuntimeError",
        r"FloatingPointError",
        r"non-finite",
        r"Traceback",
        r"CUDA out of memory",
    ]
    for p in patterns:
        if re.search(p, text):
            raise SystemExit(f"FAIL log contains pattern: {p}")

print(f"PASS state={state} last_step={last_step} metrics={metric}")
PY
