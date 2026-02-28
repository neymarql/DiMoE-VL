#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

DATASET_JSONL=${DATASET_JSONL:-artifacts/data/v1/blends/stage3_lownfe.jsonl}
CKPT=${CKPT:-}
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls -1 artifacts/train/stage3_lownfe/checkpoints/global_step_*.pt 2>/dev/null | tail -n 1 || true)
fi
if [[ -z "$CKPT" ]]; then
  echo "checkpoint not found; set CKPT=/path/to/ckpt.pt"
  exit 2
fi

mkdir -p artifacts/infer
for PRESET in fast mid best; do
  python dimoe.py infer benchmark \
    --config configs/infer/default.yaml \
    --dataset-jsonl "$DATASET_JSONL" \
    --preset "$PRESET" \
    --checkpoint "$CKPT" \
    --out "artifacts/infer/benchmark.${PRESET}.json"
done

python - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path('artifacts/infer').glob('benchmark.*.json')):
    rows.append(json.loads(p.read_text()))
for r in rows:
    print(f"{r['preset']:>4s}  p50={r['latency_p50']:.4f}s  avg={r['latency_avg']:.4f}s  tok/s={r['tokens_per_sec']:.2f}  proxy={r['nfe_active_params_proxy']:.2f}")
PY
