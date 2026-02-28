from __future__ import annotations

import argparse
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from dimoe.infer.generate import PRESETS, _denoise_generate
from dimoe.utils.config import load_yaml


class Query(BaseModel):
    prompt: str
    image: str = ""
    max_new_tokens: int = 64
    preset: str = "mid"


class ServerState:
    model: object
    cfg: Dict


def run_server(cfg: Dict, host: str, port: int, checkpoint: str = ""):
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

    state = ServerState(model=model, cfg=cfg)

    app = FastAPI(title="DiMoE-VL Serve")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/generate")
    def generate(q: Query):
        preset = q.preset if q.preset in PRESETS else "mid"
        steps = int(PRESETS[preset]["steps"])
        text = _denoise_generate(
            model=state.model,
            prompt=q.prompt,
            image_path=q.image,
            max_new_tokens=q.max_new_tokens,
            steps=steps,
            remasking=state.cfg["infer"].get("remasking", "low_confidence"),
        )
        return {"text": text, "preset": preset, "steps": steps}

    uvicorn.run(app, host=host, port=port)


def add_parser(subparsers):
    p = subparsers.add_parser("serve", help="Serve DiMoE-VL model")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--host", default="0.0.0.0", type=str)
    p.add_argument("--port", default=30000, type=int)
    p.add_argument("--checkpoint", default="", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    run_server(cfg, host=args.host, port=args.port, checkpoint=args.checkpoint)
