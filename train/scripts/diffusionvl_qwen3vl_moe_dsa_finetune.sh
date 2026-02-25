#!/bin/bash
# DSA alignment finetune script (MC scoring alignment) for DiffusionVL-Qwen3VL-MoE

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

PRETRAINED_CHECKPOINT="/path/to/your/stage1_or_stage2_checkpoint"
MC_DATA_PATH="/path/to/mc_dsa_stage23.jsonl"
IMAGE_FOLDER="/home/qianlong/datasets"
OUTPUT_DIR="./outputs/diffusionvl_qwen3vl_moe_dsa"

num_node=$1
gpu_num=$2
custom_run_name=${3:-"diffusionvl_qwen3vl_moe_dsa"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
  llava/train/train_mem.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path ${PRETRAINED_CHECKPOINT} \
  --version qwen_3 \
  --data_path "${MC_DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
  --mm_vision_tower_lr=2e-6 \
  --vision_tower ${PRETRAINED_CHECKPOINT} \
  --mm_projector_type qwen_merger \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --image_aspect_ratio pad \
  --bf16 True \
  --run_name ${custom_run_name} \
  --output_dir "${OUTPUT_DIR}/${custom_run_name}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 8e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --force_model_type "diffusionvl_qwen3vl_moe" \
  --bd3lm_block_aligned_eos True \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --dataloader_drop_last True \
  --attn_implementation sdpa \
  --use_conversation_mask False \
  --enable_bd3lm True \
  --bd3lm_block_size ${BD3LM_BLOCK_SIZE} \
  --enable_ncr True \
  --enable_bebc True \
  --lambda_lb 0.05 \
  --lambda_scr 0.02 \
  --enable_dsa True \
  --lambda_dsa 0.1 \
  --dsa_temperature 0.8
