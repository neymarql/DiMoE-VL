#!/usr/bin/env bash
#SBATCH -J dimoe_s2
#SBATCH -p ai_training
#SBATCH -N 2
#SBATCH --nodelist=dx-ai-node67,dx-ai-node69
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H800:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --time=144:00:00
#SBATCH -o /home/qianlong/DiMoE-VL/logs/%x-%j.out

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/qianlong/DiMoE-VL}
TMP_CFG_DIR=${TMP_CFG_DIR:-/tmp/dimoe_cfgs}
mkdir -p "${ROOT_DIR}/logs" "$TMP_CFG_DIR"
cd "$ROOT_DIR"

PREV_CKPT=${PREV_CKPT:-$(ls -1 artifacts/train/stage1_align/checkpoints/global_step_*.pt 2>/dev/null | tail -n 1 || true)}
if [[ -z "$PREV_CKPT" ]]; then
  echo "missing stage1 checkpoint; set PREV_CKPT=/path/to/ckpt.pt" >&2
  exit 2
fi

BASE_CFG=${BASE_CFG:-configs/train/stage2_full.yaml}
RUN_CFG=${TMP_CFG_DIR}/stage2_full.${SLURM_JOB_ID}.yaml

python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${BASE_CFG}").read_text())
cfg.setdefault("model", {})
cfg["model"]["init_checkpoint"] = "${PREV_CKPT}"
Path("${RUN_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "${RUN_CFG}")
PY

export STAGE=stage2_full
export CFG="$RUN_CFG"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29612}
export NPROC_PER_NODE=8

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export DEEPSPEED_TIMEOUT=${DEEPSPEED_TIMEOUT:-180}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)
  export NCCL_SOCKET_IFNAME
fi

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
  /home/qianlong/DiMoE-VL/scripts/slurm/run_worker_stage_train.sh
