#!/bin/bash
# Convert official Qwen3-VL-30B-A3B-Instruct HF weights into DiffusionVL format.
# Usage:
#   sbatch scripts/slurm/submit_convert_qwen3vl30b_to_diffusionvl_a800.sbatch

#SBATCH -J convert_qwen3vl30b
#SBATCH -p ai_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1800G
#SBATCH -t 24:00:00
#SBATCH -o /home/qianlong/DiMoE-VL/logs/slurm-%j.out

set -euo pipefail

REPO_DIR="/home/qianlong/DiMoE-VL"
ENV_PREFIX="/home/qianlong/DiMoE-VL/.envs/dimoe-vl-pfx"
CONDA_SH="/home/qianlong/miniconda3/etc/profile.d/conda.sh"

SOURCE_PATH="${SOURCE_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
DEST_PATH="${DEST_PATH:-/home/qianlong/DiMoE-VL/checkpoints/Qwen3-VL-30B-A3B-Instruct-DiffusionVL-bf16}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/qianlong/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${DEST_PATH}"

echo "===== Convert Qwen3-VL-30B-A3B -> DiffusionVL ====="
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "source=${SOURCE_PATH}"
echo "dest=${DEST_PATH}"
echo "dtype=${DTYPE}"
which python
python -V
nvidia-smi -L

if [ -f "${DEST_PATH}/model.safetensors" ] && [ -f "${DEST_PATH}/vision_tower.safetensors" ] && [ -f "${DEST_PATH}/projector.safetensors" ]; then
  echo "[INFO] Converted checkpoint already exists at ${DEST_PATH}, skipping conversion."
  exit 0
fi

python "${REPO_DIR}/scripts/diffusionvl_prepare/convert_qwen3vl_moe_to_diffusionvl.py" \
  --source_path "${SOURCE_PATH}" \
  --dest_path "${DEST_PATH}" \
  --dtype "${DTYPE}"

echo "===== Conversion DONE ====="
