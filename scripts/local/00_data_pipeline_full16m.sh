#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export DATA_IN=${DATA_IN:-/home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute}
export DATA_OUT=${DATA_OUT:-artifacts/data/v1}
export TOKENIZER=${TOKENIZER:-inclusionAI/LLaDA-MoE-7B-A1B-Instruct}
export DATASETS_ROOT=${DATASETS_ROOT:-/home/qianlong/datasets}
export STAGE_BLEND_CFG=${STAGE_BLEND_CFG:-configs/data/stage_blends_full16m.yaml}
export PACK_WDS=${PACK_WDS:-0}

bash scripts/local/00_data_pipeline.sh
