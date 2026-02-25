#!/bin/bash
#SBATCH -J dimoe_smoke_offline
#SBATCH -p ai_training
#SBATCH --gres=gpu:H800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /home/qianlong/DiMoE-VL/logs/slurm-%j.out

set -euo pipefail

REPO_DIR="/home/qianlong/DiMoE-VL"
ENV_PREFIX="/home/qianlong/DiMoE-VL/.envs/dimoe-vl-pfx"
CONDA_SH="/home/qianlong/miniconda3/etc/profile.d/conda.sh"
MODEL_PATH="${REPO_DIR}/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"

# Disable proxy on compute nodes (cluster network cannot reach this local proxy).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Converted model not found at ${MODEL_PATH}" >&2
  exit 1
fi

echo "===== Runtime Info ====="
which python
python --version
nvidia-smi

echo "===== GPU Offline Smoke (real load + train/inference) ====="
PYTHONPATH="${REPO_DIR}/train:${PYTHONPATH:-}" python -u - <<'PY'
import torch
from llava.model.language_model.llava_diffusionvl_qwen3vl_moe import (
    DiffusionVLQwen3VLMoeConfig,
    DiffusionVLQwen3VLMoeForCausalLM,
)

model_path = "/home/qianlong/DiMoE-VL/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl"
device = "cuda:0"
torch.manual_seed(42)

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("cuda device count", torch.cuda.device_count())
print("loading config...")

config = DiffusionVLQwen3VLMoeConfig.from_pretrained(model_path)
config.enable_bd3lm = True
config.bd3lm_block_size = 4
print("loading model...")
model = DiffusionVLQwen3VLMoeForCausalLM.from_pretrained(
    model_path,
    config=config,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
model.to(device)
print("model_loaded")

seq_len = 16
vocab_size = int(config.text_config.vocab_size)
input_ids = torch.randint(low=10, high=min(vocab_size, 500), size=(1, seq_len), device=device)
labels = input_ids.clone()
attention_mask = torch.ones_like(input_ids, device=device)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
optimizer.zero_grad(set_to_none=True)
inputs_embeds = model.get_model().embed_tokens(input_ids)
outputs = model(
    inputs_embeds=inputs_embeds,
    labels=labels,
    attention_mask=attention_mask,
    use_cache=False,
)
loss = outputs.loss
print("train_smoke_loss", float(loss.detach().float().cpu()))
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print("train_smoke_step_ok")

model.eval()
with torch.no_grad():
    gen_ids = model.generate(
        inputs=input_ids,
        gen_length=8,
        steps=2,
        temperature=0.0,
        remasking_strategy="low_confidence_static",
    )

print("inference_smoke_shape", tuple(gen_ids.shape))
print("inference_smoke_ids", gen_ids[0].tolist())
print("gpu_mem_alloc_mb", round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2))
print("smoke_finished_ok")
PY

echo "===== DONE ====="
