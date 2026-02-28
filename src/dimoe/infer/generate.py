from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from dimoe.utils.config import load_yaml
from dimoe.utils.io import write_json, write_jsonl
from dimoe.utils.logging import setup_logger


PRESETS = {
    "fast": {"steps": 8, "block_length": 32, "shift_gamma": 2.0, "prefix_kv_cache": True, "prefix_route_cache": True},
    "mid": {"steps": 16, "block_length": 32, "shift_gamma": 1.2, "prefix_kv_cache": True, "prefix_route_cache": True},
    "best": {"steps": 32, "block_length": 64, "shift_gamma": 1.0, "prefix_kv_cache": False, "prefix_route_cache": False},
}


def _load_inputs(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("input must be json object or array")
    out = []
    for x in data:
        if not isinstance(x, dict):
            continue
        out.append({
            "id": str(x.get("id", "")),
            "prompt": str(x.get("prompt", "")),
            "image": str(x.get("image", "")),
            "max_new_tokens": int(x.get("max_new_tokens", 64)),
        })
    return out


def _denoise_generate(
    model,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    steps: int,
    remasking: str = "low_confidence",
) -> str:
    import torch

    tok = model.tokenizer
    device = model.device

    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}".strip()

    prompt_ids = model.encode_text(prompt, max_length=4096)
    ans_ids = [model.mask_token_id] * max_new_tokens

    ids = torch.tensor([prompt_ids + ans_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)

    answer_start = len(prompt_ids)
    token_types = torch.ones_like(ids)
    token_types[:, :answer_start] = 1
    token_types[ids == model.image_token_id] = 0
    token_types[:, answer_start:] = 3

    for s in range(steps):
        t = torch.full((1, ids.shape[1]), fill_value=min(7, int((1 - s / max(steps, 1)) * 8)), dtype=torch.long, device=device)
        out = model.forward_diffusion(ids, attn, t, token_types, image_paths=[image_path])

        logits = out["logits"][:, answer_start:, :]
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        if remasking == "low_confidence" and s < steps - 1:
            keep_ratio = (s + 1) / steps
            k = max(1, int(max_new_tokens * keep_ratio))
            top_idx = torch.topk(conf[0], k=k, largest=True).indices
            mask = torch.ones(max_new_tokens, device=device, dtype=torch.bool)
            mask[top_idx] = False
            cur = pred[0].clone()
            cur[mask] = model.mask_token_id
        else:
            cur = pred[0]

        ids[:, answer_start:] = cur

    final_ids = ids[0, answer_start:].tolist()
    return model.decode(final_ids)


def run_generate(cfg: Dict, input_json: Path, out_jsonl: Path, preset: str, checkpoint: str = ""):
    logger = setup_logger()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = PRESETS[preset]

    from dimoe.model.dimoe_model import DimoeModel

    model = DimoeModel(
        backbone=cfg["model"]["backbone"],
        vision_tower=cfg["model"]["vision_tower"],
        num_experts=int(cfg["model"]["num_experts"]),
        num_t_buckets=int(cfg["model"]["num_t_buckets"]),
        num_token_types=int(cfg["model"]["num_token_types"]),
        image_token=cfg["model"]["image_token"],
        mask_token=cfg["model"]["mask_token"],
        device=device,
        allow_dummy_vision=bool(cfg["model"].get("allow_dummy_vision", True)),
        backbone_trust_remote_code=bool(cfg["model"].get("backbone_trust_remote_code", True)),
        vision_trust_remote_code=bool(cfg["model"].get("vision_trust_remote_code", True)),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
    )
    ckpt = checkpoint or str(cfg.get("model", {}).get("checkpoint", ""))
    if ckpt:
        model.load_checkpoint(ckpt, strict=False)
    model.eval()

    rows = _load_inputs(input_json)
    out = []
    lat = []
    for r in rows:
        st = time.time()
        pred = _denoise_generate(
            model,
            prompt=r["prompt"],
            image_path=r["image"],
            max_new_tokens=r["max_new_tokens"],
            steps=int(p["steps"]),
            remasking=cfg["infer"].get("remasking", "low_confidence"),
        )
        dt = time.time() - st
        lat.append(dt)
        out.append({**r, "prediction": pred, "latency_sec": dt, "preset": preset})

    write_jsonl(out_jsonl, out)
    summary = {
        "preset": preset,
        "num_samples": len(out),
        "latency_avg": sum(lat) / max(len(lat), 1),
        "steps": p["steps"],
    }
    write_json(out_jsonl.with_suffix(".summary.json"), summary)
    logger.info("generated %d samples -> %s", len(out), out_jsonl)


def add_parser(subparsers):
    p = subparsers.add_parser("generate", help="Diffusion generation from input json")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--preset", choices=["fast", "mid", "best"], default="mid")
    p.add_argument("--checkpoint", default="", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    run_generate(cfg, Path(args.input), Path(args.out), args.preset, checkpoint=args.checkpoint)
