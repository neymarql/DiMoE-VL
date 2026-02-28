#!/usr/bin/env bash
#SBATCH -J dimoe_s0_2x4
#SBATCH -p ai_training
#SBATCH -N 2
#SBATCH --nodelist=dx-ai-node67,dx-ai-node69
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H800:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=96:00:00
#SBATCH -o /home/qianlong/DiMoE-VL/logs/%x-%j.out

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/qianlong/DiMoE-VL}
mkdir -p "${ROOT_DIR}/logs"
cd "$ROOT_DIR"

export STAGE=stage0_naive
export CFG=${CFG:-configs/train/stage0_naive.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29640}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

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
