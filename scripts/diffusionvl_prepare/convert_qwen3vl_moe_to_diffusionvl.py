import argparse
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import Qwen3VLMoeForConditionalGeneration

# Add project and train roots to Python path for llava imports.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
train_root = os.path.join(project_root, "train")
for p in (project_root, train_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from llava.model.language_model.llava_diffusionvl_qwen3vl_moe import DiffusionVLQwen3VLMoeConfig


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_source_path(source_path: str) -> str:
    if os.path.isdir(source_path):
        print(f"Source path '{source_path}' is a local directory. Using it directly.")
        return source_path

    print(
        f"Source path '{source_path}' not found locally. Assuming it is a Hugging Face repo ID and downloading."
    )
    return snapshot_download(source_path)


def _copy_non_weight_files(source_local_path: str, dest_path: str):
    print("Copying configuration and tokenizer files...")
    files_to_ignore = {".git", ".gitattributes"}
    weight_extensions = (".safetensors", ".bin", ".pth")

    for filename in os.listdir(source_local_path):
        if filename in files_to_ignore:
            continue
        if filename.endswith(weight_extensions):
            continue

        src_file = os.path.join(source_local_path, filename)
        if os.path.isfile(src_file):
            print(f"  - Copying {filename}")
            shutil.copy2(src_file, dest_path)


def convert_qwen3vl_moe_to_diffusionvl(source_path: str, dest_path: str, dtype: torch.dtype):
    source_local_path = _resolve_source_path(source_path)
    os.makedirs(dest_path, exist_ok=True)

    _copy_non_weight_files(source_local_path, dest_path)

    print(f"\nLoading original Qwen3-VL-MoE model from {source_local_path} for conversion...")
    print(f"Using load dtype: {dtype}")
    original_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        source_local_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print("Assembling DiffusionVL-compatible state dictionary...")
    final_state_dict = {}

    for k, v in original_model.model.language_model.state_dict().items():
        final_state_dict[f"model.{k}"] = v

    for k, v in original_model.lm_head.state_dict().items():
        # skip tied lm_head to avoid duplicated tensor writes in safetensors
        if (
            k == "weight"
            and v.data_ptr() == original_model.model.language_model.embed_tokens.weight.data_ptr()
        ):
            print("Detected tied lm_head.weight; skipping duplicate tensor save.")
            continue
        final_state_dict[f"lm_head.{k}"] = v

    print("Saving DiffusionVL-Qwen3VL-MoE config...")
    config = DiffusionVLQwen3VLMoeConfig.from_pretrained(source_local_path)
    config.text_config = original_model.model.language_model.config
    config.tie_word_embeddings = False
    if hasattr(config, "text_config") and config.text_config is not None:
        config.text_config.tie_word_embeddings = False
    config.save_pretrained(dest_path)

    print("Saving converted model and visual component weights...")
    save_file(final_state_dict, os.path.join(dest_path, "model.safetensors"))

    original_model.model.visual.config.to_json_file(os.path.join(dest_path, "vision_config.json"))
    visual_state_dict = original_model.model.visual.state_dict()
    save_file(
        {k: v for k, v in visual_state_dict.items() if not k.startswith("merger.")},
        os.path.join(dest_path, "vision_tower.safetensors"),
    )
    save_file(
        {k.replace("merger.", ""): v for k, v in visual_state_dict.items() if k.startswith("merger.")},
        os.path.join(dest_path, "projector.safetensors"),
    )

    del original_model
    del final_state_dict

    print(f"\nConversion successful. Converted model is at: {dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Qwen3-VL-MoE model to DiffusionVL checkpoint format."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Local path or hub ID for Qwen3-VL-MoE model (e.g. Qwen/Qwen3-VL-30B-A3B-Instruct).",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        required=True,
        help="Destination directory for converted DiffusionVL checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Load dtype for source model during conversion. Default bf16 for memory-safe conversion.",
    )
    args = parser.parse_args()

    convert_qwen3vl_moe_to_diffusionvl(args.source_path, args.dest_path, DTYPE_MAP[args.dtype])
