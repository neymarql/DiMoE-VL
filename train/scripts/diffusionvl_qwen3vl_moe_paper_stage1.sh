#!/bin/bash
set -euo pipefail

# Paper-level Stage-1 training launcher for DiffusionVL-Qwen3VL-MoE.
# Usage:
#   bash train/scripts/diffusionvl_qwen3vl_moe_paper_stage1.sh <num_nodes> <gpus_per_node> [run_name] [bd3_block]

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_ENTRY="${REPO_DIR}/train/llava/train/train_mem.py"
DEEPSPEED_CFG="${REPO_DIR}/train/scripts/zero3.json"

# Runtime paths (override by env before launch)
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${REPO_DIR}/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl-bf16}"
DATA_PATH="${DATA_PATH:-/home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute/stage4_joint.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/qianlong/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/paper_stage1}"

export WANDB_PROJECT="${WANDB_PROJECT:-dimoe-vl}"
export WANDB_DIR="${WANDB_DIR:-${REPO_DIR}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

num_node=${1:?num_nodes is required}
gpu_num=${2:?gpus_per_node is required}
custom_run_name=${3:-"paper_stage1_qwen3vl_moe"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29601"}
RANK=${RANK:-"0"}

# Strong paper defaults; can be overridden by env.
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
GRAD_ACCUM=${GRAD_ACCUM:-16}          # 2*8 GPUs, per_device=1 => global batch 256
SAVE_STEPS=${SAVE_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MM_VISION_LR=${MM_VISION_LR:-2e-6}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
NCR_FOURIER_DIM=${NCR_FOURIER_DIM:-32}
LAMBDA_LB=${LAMBDA_LB:-0.05}
LAMBDA_SCR=${LAMBDA_SCR:-0.02}
SCR_EMA_DECAY=${SCR_EMA_DECAY:-0.999}

RUN_DIR="${OUTPUT_ROOT}/${custom_run_name}"
mkdir -p "${RUN_DIR}" "${WANDB_DIR}"

echo "=========================================="
echo "Paper Stage-1: DiffusionVL-Qwen3VL-MoE"
echo "=========================================="
echo "repo               : ${REPO_DIR}"
echo "checkpoint         : ${PRETRAINED_CHECKPOINT}"
echo "data               : ${DATA_PATH}"
echo "image_folder       : ${IMAGE_FOLDER}"
echo "run_dir            : ${RUN_DIR}"
echo "nodes/gpus         : ${num_node} x ${gpu_num}"
echo "master             : ${MASTER_ADDR}:${MASTER_PORT} (rank=${RANK})"
echo "max_steps/epochs   : ${MAX_STEPS} / ${NUM_TRAIN_EPOCHS}"
echo "grad_acc/global_bs : ${GRAD_ACCUM} / $((gpu_num * num_node * GRAD_ACCUM))"
echo "bd3_block          : ${BD3LM_BLOCK_SIZE}"

if [ ! -d "${PRETRAINED_CHECKPOINT}" ]; then
  echo "[ERROR] checkpoint path not found: ${PRETRAINED_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "[ERROR] data path not found: ${DATA_PATH}" >&2
  exit 2
fi
if [ ! -d "${IMAGE_FOLDER}" ]; then
  echo "[ERROR] image folder not found: ${IMAGE_FOLDER}" >&2
  exit 2
fi

cd "${REPO_DIR}/train"

torchrun \
  --nproc_per_node="${gpu_num}" \
  --nnodes="${num_node}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --node_rank="${RANK}" \
  "${TRAIN_ENTRY}" \
  --deepspeed "${DEEPSPEED_CFG}" \
  --model_name_or_path "${PRETRAINED_CHECKPOINT}" \
  --version qwen_3 \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
  --mm_vision_tower_lr "${MM_VISION_LR}" \
  --vision_tower "${PRETRAINED_CHECKPOINT}" \
  --mm_projector_type qwen_merger \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --image_aspect_ratio pad \
  --bf16 True \
  --run_name "${custom_run_name}" \
  --output_dir "${RUN_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --force_model_type diffusionvl_qwen3vl_moe \
  --bd3lm_block_aligned_eos True \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --report_to wandb \
  --dataloader_drop_last True \
  --attn_implementation sdpa \
  --use_conversation_mask False \
  --enable_bd3lm True \
  --bd3lm_block_size "${BD3LM_BLOCK_SIZE}" \
  --enable_ncr True \
  --ncr_fourier_dim "${NCR_FOURIER_DIM}" \
  --enable_bebc True \
  --lambda_lb "${LAMBDA_LB}" \
  --lambda_scr "${LAMBDA_SCR}" \
  --scr_ema_decay "${SCR_EMA_DECAY}"

