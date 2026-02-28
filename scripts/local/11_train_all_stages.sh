#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29600}
TMP_DIR=${TMP_DIR:-/tmp/dimoe_cfgs}
mkdir -p "$TMP_DIR"

stages=(stage0_naive stage1_align stage2_full stage3_lownfe)

latest_ckpt() {
  local stage=$1
  ls -1 "artifacts/train/${stage}/checkpoints"/global_step_*.pt 2>/dev/null | tail -n 1 || true
}

for i in "${!stages[@]}"; do
  stage=${stages[$i]}
  base_cfg="configs/train/${stage}.yaml"
  run_cfg="${TMP_DIR}/${stage}.yaml"

  prev_ckpt=""
  if [[ $i -gt 0 ]]; then
    prev_stage=${stages[$((i-1))]}
    prev_ckpt=$(latest_ckpt "$prev_stage")
    if [[ -z "$prev_ckpt" ]]; then
      echo "missing previous checkpoint for ${prev_stage}, stop"
      exit 2
    fi
  fi

  python - <<PY
import yaml
from pathlib import Path
src = Path("${base_cfg}")
dst = Path("${run_cfg}")
cfg = yaml.safe_load(src.read_text())
cfg.setdefault("model", {})
cfg["model"]["init_checkpoint"] = "${prev_ckpt}"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {dst}")
PY

  torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    dimoe.py train run --stage "$stage" --config "$run_cfg"
done

echo "[done] stage0->stage3 training complete"
