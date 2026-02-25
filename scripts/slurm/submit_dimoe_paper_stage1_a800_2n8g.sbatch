#!/bin/bash
#SBATCH -J dimoe_paper_s1_a800
#SBATCH -p ai_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A800:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=1800G
#SBATCH -t 06:00:00
#SBATCH --nodelist=dx-ai-node2,dx-ai-node4
#SBATCH -o /home/qianlong/DiMoE-VL/logs/slurm-%j.out

set -euo pipefail

REPO_DIR="/home/qianlong/DiMoE-VL"
ENV_PREFIX="/home/qianlong/DiMoE-VL/.envs/dimoe-vl-pfx"
CONDA_SH="/home/qianlong/miniconda3/etc/profile.d/conda.sh"
RUN_NAME="paper_stage1_a800_2n8g_trial_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29601}

echo "===== Runtime ====="
echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
which python
python -V
nvidia-smi -L

# Trial-on-idle-nodes first: keep paper-level config, but cap MAX_STEPS.
export MAX_STEPS=${MAX_STEPS:-40}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export GRAD_ACCUM=${GRAD_ACCUM:-16}
export PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/home/qianlong/DiMoE-VL/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl-bf16}
export DATA_PATH=${DATA_PATH:-/home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute/stage4_joint.json}
export IMAGE_FOLDER=${IMAGE_FOLDER:-/home/qianlong/datasets}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/qianlong/DiMoE-VL/outputs/paper_stage1_a800}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-dimoe-vl}
export WANDB_DIR=${WANDB_DIR:-/home/qianlong/DiMoE-VL/wandb}

echo "===== Launch Config ====="
echo "RUN_NAME=${RUN_NAME}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "GRAD_ACCUM=${GRAD_ACCUM}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --gpus-per-node=8 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  bash -lc "source ${CONDA_SH} && conda activate ${ENV_PREFIX} && \
    export MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RANK=\${SLURM_PROCID} && \
    bash ${REPO_DIR}/train/scripts/diffusionvl_qwen3vl_moe_paper_stage1.sh ${SLURM_NNODES} 8 ${RUN_NAME} 8"

echo "===== DONE ====="

