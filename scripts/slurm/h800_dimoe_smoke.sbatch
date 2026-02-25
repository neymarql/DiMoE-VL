#!/bin/bash
#SBATCH -J dimoe_smoke
#SBATCH -p ai_training
#SBATCH --gres=gpu:H800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o /home/qianlong/DiMoE-VL/logs/slurm-%j.out

set -euo pipefail

REPO_DIR="/home/qianlong/DiMoE-VL"
ENV_NAME="dimoe-vl"
CONDA_SH="/home/qianlong/miniconda3/etc/profile.d/conda.sh"
MARKER_FILE="/home/qianlong/.cache/dimoe-vl/gpu_env_ready_v1"
MODEL_SRC="tiny-random/qwen3-vl-moe"
MODEL_DST="${REPO_DIR}/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl"

mkdir -p "${REPO_DIR}/logs" "$(dirname "${MARKER_FILE}")" "${REPO_DIR}/checkpoints"
cd "${REPO_DIR}"

if [ ! -f "${CONDA_SH}" ]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

if [ ! -f "${MARKER_FILE}" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  python -m pip install accelerate==1.10.1 transformers==4.55.0 pillow==10.4.0 requests==2.32.5 safetensors huggingface_hub loguru datasets evaluate webdataset wandb av decord hf_transfer openai tenacity sqlitedict sacrebleu pytablewriter packaging
  python -m pip install --no-deps -U -e eval/lmms-eval
  python -m pip install --no-deps -U -e train
  touch "${MARKER_FILE}"
fi

echo "===== Runtime Info ====="
which python
python --version
python -m pip --version
nvidia-smi

echo "===== Convert tiny-random Qwen3-VL-MoE to DiffusionVL format ====="
python scripts/diffusionvl_prepare/convert_qwen3vl_moe_to_diffusionvl.py \
  --source_path "${MODEL_SRC}" \
  --dest_path "${MODEL_DST}"

echo "===== Run real load + train/inference smoke ====="
PYTHONPATH="${REPO_DIR}/train:${PYTHONPATH:-}" python - <<'PY'
import os
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

torch.manual_seed(42)
device = "cuda:0"
model_path = "/home/qianlong/DiMoE-VL/checkpoints/tiny-random-qwen3-vl-moe-diffusionvl"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_name="diffusionvl_qwen3vl_moe",
    force_model_type="diffusionvl_qwen3vl_moe",
    device_map=device,
    torch_dtype="bfloat16",
    attn_implementation="sdpa",
    overwrite_config={"enable_bd3lm": True, "bd3lm_block_size": 4},
)

print("Loaded model:", model.__class__.__name__)
print("Context length:", context_len)

img = Image.new("RGB", (224, 224), color=(0, 0, 0))
proc = image_processor(images=[img], return_tensors="pt")
pixel_values = proc["pixel_values"].to(device=device)
image_grid_thw = proc.get("image_grid_thw")
if image_grid_thw is not None and hasattr(image_grid_thw, "tolist"):
    image_grid_thw = image_grid_thw.tolist()

user_query = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
assistant_answer = "A black square."

conv_prompt = conv_templates["qwen_3"].copy()
conv_prompt.append_message(conv_prompt.roles[0], user_query)
conv_prompt.append_message(conv_prompt.roles[1], None)
prompt_text = conv_prompt.get_prompt()
prompt_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

conv_full = conv_templates["qwen_3"].copy()
conv_full.append_message(conv_full.roles[0], user_query)
conv_full.append_message(conv_full.roles[1], assistant_answer)
full_text = conv_full.get_prompt()
full_ids = tokenizer_image_token(full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

labels = full_ids.clone()
labels[:, :prompt_ids.shape[1]] = IGNORE_INDEX

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
opt.zero_grad(set_to_none=True)

outputs = model(
    input_ids=full_ids,
    labels=labels,
    images=pixel_values,
    image_sizes=[[img.width, img.height]],
    image_grid_thws=image_grid_thw,
    modalities=["image"],
    use_cache=False,
)

loss = outputs.loss
print("Train smoke loss:", float(loss.detach().cpu()))
loss.backward()
opt.step()
opt.zero_grad(set_to_none=True)
print("Train smoke backward/step: OK")

model.eval()
with torch.no_grad():
    gen_ids = model.generate(
        inputs=prompt_ids,
        images=pixel_values,
        image_sizes=[[img.width, img.height]],
        image_grid_thws=image_grid_thw,
        modalities=["image"],
        gen_length=16,
        steps=4,
        temperature=0.0,
        remasking_strategy="low_confidence_static",
    )
gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print("Inference smoke generated ids shape:", tuple(gen_ids.shape))
print("Inference smoke text:", gen_text[:200])
print("Smoke run finished successfully.")
PY

echo "===== DONE ====="
