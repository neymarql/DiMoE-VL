#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/qianlong/DiMoE-VL}
STAGE=${STAGE:?STAGE is required}
CFG=${CFG:?CFG is required}

if [ -x /home/qianlong/miniconda3/envs/dimoe-vl/bin/torchrun ]; then
  TORCHRUN=/home/qianlong/miniconda3/envs/dimoe-vl/bin/torchrun
elif [ -x /mnt/home/qianlong/miniconda3/envs/dimoe-vl/bin/torchrun ]; then
  TORCHRUN=/mnt/home/qianlong/miniconda3/envs/dimoe-vl/bin/torchrun
else
  TORCHRUN=$(command -v torchrun || true)
fi

if [ -z "${TORCHRUN}" ]; then
  echo "torchrun not found on node $(hostname)" >&2
  exit 127
fi

cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export DIST_TIMEOUT_SEC=${DIST_TIMEOUT_SEC:-7200}
export DEEPSPEED_TIMEOUT=${DEEPSPEED_TIMEOUT:-180}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

NODE_RANK=${SLURM_PROCID:?}
NNODES=${SLURM_NNODES:?}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:?}
MASTER_PORT=${MASTER_PORT:?}

echo "[worker $(hostname)] stage=${STAGE} cfg=${CFG} node_rank=${NODE_RANK}/${NNODES} nproc=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[worker $(hostname)] DIST_TIMEOUT_SEC=${DIST_TIMEOUT_SEC} DEEPSPEED_TIMEOUT(min)=${DEEPSPEED_TIMEOUT} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

"$TORCHRUN" \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  dimoe.py train run --stage "$STAGE" --config "$CFG"
