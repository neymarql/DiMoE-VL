#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

STAGE=${STAGE:-stage2_full}
BASE_CFG=${BASE_CFG:-configs/train/${STAGE}.yaml}
ABL_CFG=${ABL_CFG:-configs/exp/ablations.yaml}
OUT_DIR=${OUT_DIR:-/tmp/dimoe_ablation_cfgs}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29650}
RUN_EVAL=${RUN_EVAL:-0}

mkdir -p "$OUT_DIR"
shopt -s nullglob

python - <<PY
import yaml
from pathlib import Path

abl = yaml.safe_load(Path("${ABL_CFG}").read_text())
out = Path("${OUT_DIR}")
stage = "${STAGE}"
out.mkdir(parents=True, exist_ok=True)

for name, spec in abl.get("variants", {}).items():
    cfg = yaml.safe_load(Path("${BASE_CFG}").read_text())
    cur = cfg.setdefault("curriculum", {})
    cur.update(spec.get("curriculum", {}))
    t = cfg.setdefault("train", {})
    t["resume"] = False
    cfg["output_dir"] = f"artifacts/train_ablation/{name}"
    p = out / f"{name}.{stage}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(p)
PY

cfgs=( "$OUT_DIR"/*.${STAGE}.yaml )
if [[ ${#cfgs[@]} -eq 0 ]]; then
  echo "no ablation configs generated under $OUT_DIR"
  exit 2
fi

for cfg in "${cfgs[@]}"; do
  name=$(basename "$cfg" | cut -d'.' -f1)
  echo "[ablation] ${name}"
  torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    dimoe.py train run --stage "$STAGE" --config "$cfg"

  if [[ "$RUN_EVAL" == "1" ]]; then
    ckpt=$(ls -1 artifacts/train_ablation/${name}/${STAGE}/checkpoints/global_step_*.pt 2>/dev/null | tail -n 1 || true)
    if [[ -n "$ckpt" ]]; then
      python dimoe.py eval run --config configs/eval/default.yaml --suite all --preset best --checkpoint "$ckpt" --out-dir "artifacts/eval_ablation/${name}"
    fi
  fi
done
