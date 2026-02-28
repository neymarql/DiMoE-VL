from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from dimoe.infer.generate import PRESETS, _denoise_generate
from dimoe.utils.config import load_yaml
from dimoe.utils.io import read_jsonl, write_json


def run_benchmark(cfg: Dict, dataset_jsonl: Path, out_json: Path, preset: str, checkpoint: str = ""):
    rows = list(read_jsonl(dataset_jsonl))
    rows = rows[: int(cfg["infer"].get("benchmark_samples", 128))]

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    p = PRESETS[preset]
    latencies: List[float] = []
    generated_tokens = 0

    for r in rows:
        meta = r.get("meta", {})
        prompt = str(meta.get("prompt", ""))
        image = str(r.get("image_abs", ""))
        max_new_tokens = int(cfg["infer"].get("benchmark_max_new_tokens", 64))

        st = time.time()
        pred = _denoise_generate(
            model=model,
            prompt=prompt,
            image_path=image,
            max_new_tokens=max_new_tokens,
            steps=int(p["steps"]),
            remasking=cfg["infer"].get("remasking", "low_confidence"),
        )
        dt = time.time() - st
        latencies.append(dt)
        generated_tokens += len(model.encode_text(pred, max_length=4096))

    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = sorted(latencies)[int(0.9 * (len(latencies) - 1))] if latencies else 0.0
    total_time = sum(latencies)

    out = {
        "preset": preset,
        "samples": len(rows),
        "steps": p["steps"],
        "latency_p50": p50,
        "latency_p90": p90,
        "latency_avg": total_time / max(len(rows), 1),
        "tokens_per_sec": generated_tokens / max(total_time, 1e-6),
        "nfe_active_params_proxy": p["steps"] * float(cfg["infer"].get("active_params_billion", 1.4)),
    }
    write_json(out_json, out)


def add_parser(subparsers):
    p = subparsers.add_parser("benchmark", help="Inference latency/throughput benchmark")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--dataset-jsonl", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--preset", choices=["fast", "mid", "best"], default="mid")
    p.add_argument("--checkpoint", default="", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    run_benchmark(cfg, Path(args.dataset_jsonl), Path(args.out), args.preset, checkpoint=args.checkpoint)
