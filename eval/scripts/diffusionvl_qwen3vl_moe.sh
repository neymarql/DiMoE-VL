#!/bin/bash

# ============================================
# DiffusionVL-Qwen3VL-MoE Evaluation Script
# ============================================

MODEL_PATHS=(
    "/path/to/your/qwen3vl_moe_diffusion_model"
)

OUTPUT_PATH="./eval_results_qwen3vl_moe"
TASK_NAMES="mmmu_val,mmmu_pro_standard,mmmu_pro_vision,mmstar,ai2d,mmbench_en_dev,muirbench,mme,realworldqa,chartqa"
TOTAL_GPUS=8
BLOCK_SIZE=8
STEPS=8

MODEL="llava_onevision_diffusionvl_qwen3vl_moe"
MODEL_NAME="diffusionvl_qwen3vl_moe"
CONV_TEMPLATE="qwen_3"
SCORING_MODE="loss"  # loss: argmin-compatible; neg_loss: use with multiple_choice_argmax tasks

declare -A GPU_STATUS
declare -A GPU_PIDS

for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

IFS=',' read -ra TASKS <<< "$TASK_NAMES"
declare -a TASK_QUEUE
for model_path in "${MODEL_PATHS[@]}"; do
    for task in "${TASKS[@]}"; do
        case $task in
            chartqa)
                GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"stopping_criteria\":[\"\\n\"], \"remasking_strategy\": \"low_confidence_static\"}"
                ;;
            *)
                GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"remasking_strategy\": \"low_confidence_static\"}"
                ;;
        esac
        TASK_QUEUE+=("$model_path $task $GEN_KWARGS")
    done
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
COMPLETED_TASKS=0
FINISHED_TASKS=0

while [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; do
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]}" ]]; then
            if ! kill -0 ${GPU_PIDS[$gpu]} 2>/dev/null; then
                GPU_STATUS[$gpu]=0
                unset GPU_PIDS[$gpu]
                FINISHED_TASKS=$((FINISHED_TASKS + 1))
            fi
        fi
    done

    if [ $COMPLETED_TASKS -lt $TOTAL_TASKS ]; then
        for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
            if [[ ${GPU_STATUS[$gpu]} -eq 0 && $COMPLETED_TASKS -lt $TOTAL_TASKS ]]; then
                CURRENT_TASK_STRING="${TASK_QUEUE[$COMPLETED_TASKS]}"
                read -r MODEL_PATH TASK_NAME CURRENT_GEN_KWARGS <<< "$CURRENT_TASK_STRING"

                GPU_STATUS[$gpu]=1
                MODEL_PATH_LAST=$(basename "$MODEL_PATH")
                CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$MODEL_PATH_LAST"
                mkdir -p "$CURRENT_OUTPUT_PATH"
                LOG_FILE="$CURRENT_OUTPUT_PATH/${TASK_NAME}.log"

                MODEL_ARGS_STR="pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME"
                MODEL_ARGS_STR="$MODEL_ARGS_STR,enable_bd3lm=True,bd3lm_block_size=$BLOCK_SIZE"
                MODEL_ARGS_STR="$MODEL_ARGS_STR,scoring_mode=$SCORING_MODE"

                (
                    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 -m lmms_eval \
                        --model "$MODEL" \
                        --gen_kwargs="$CURRENT_GEN_KWARGS" \
                        --model_args "$MODEL_ARGS_STR" \
                        --tasks "$TASK_NAME" \
                        --batch_size 1 \
                        --log_samples \
                        --log_samples_suffix "$TASK_NAME" \
                        --output_path "$CURRENT_OUTPUT_PATH" >> "$LOG_FILE" 2>&1
                ) &
                GPU_PIDS[$gpu]=$!

                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            fi
        done
    fi

    if [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; then
        sleep 10
    fi
done

wait
