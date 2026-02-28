#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

DATA_IN=${DATA_IN:-/home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute}
DATA_OUT=${DATA_OUT:-artifacts/data/v1}
TOKENIZER=${TOKENIZER:-inclusionAI/LLaDA-MoE-7B-A1B-Instruct}
DATASETS_ROOT=${DATASETS_ROOT:-/home/qianlong/datasets}
STAGE_BLEND_CFG=${STAGE_BLEND_CFG:-configs/data/stage_blends.yaml}
PACK_WDS=${PACK_WDS:-0}

python dimoe.py data normalize \
  --in "$DATA_IN" \
  --out "$DATA_OUT" \
  --tokenizer "$TOKENIZER" \
  --datasets-root "$DATASETS_ROOT"

python dimoe.py data build-stage-blends --config "$STAGE_BLEND_CFG"

if [[ "$PACK_WDS" == "1" ]]; then
  for STAGE in stage0_naive stage1_align stage2_full stage3_lownfe; do
    cat > /tmp/dimoe_wds_${STAGE}.yaml <<EOF
source_jsonl: artifacts/data/v1/blends/${STAGE}.jsonl
output_dir: artifacts/data/v1/wds/${STAGE}
shard_size: 2000
include_image: true
EOF
    python dimoe.py data build-wds --config /tmp/dimoe_wds_${STAGE}.yaml
  done
fi

echo "[done] data pipeline complete"
