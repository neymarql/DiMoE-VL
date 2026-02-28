#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

CKPT=${CKPT:-}
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls -1 artifacts/train/stage3_lownfe/checkpoints/global_step_*.pt 2>/dev/null | tail -n 1 || true)
fi
if [[ -z "$CKPT" ]]; then
  echo "checkpoint not found; set CKPT=/path/to/ckpt.pt"
  exit 2
fi

for PRESET in fast mid best; do
  python dimoe.py eval run \
    --config configs/eval/default.yaml \
    --suite all \
    --preset "$PRESET" \
    --checkpoint "$CKPT" \
    --out-dir "artifacts/eval/${PRESET}"
done

python dimoe.py eval summarize --eval-dir artifacts/eval/best --out-dir artifacts/results/best
python dimoe.py tools export-paper-tables --eval-dir artifacts/eval/best --out-dir artifacts/results/paper

python dimoe.py diag supervision \
  --manifest artifacts/data/v1/blends/stage0_naive.jsonl \
  --out-dir artifacts/diag/supervision

python dimoe.py diag routing \
  --log artifacts/train/stage2_full/logs/routing_stats.jsonl \
  --out-dir artifacts/diag/routing

python dimoe.py diag gradient \
  --log artifacts/train/stage2_full/logs/gradient_stats.jsonl \
  --out-dir artifacts/diag/gradient

echo "[done] eval + diagnostics complete"
