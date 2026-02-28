from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

from dimoe.infer.generate import PRESETS, _denoise_generate
from dimoe.utils.config import load_yaml
from dimoe.utils.io import write_json, write_jsonl
from dimoe.utils.logging import setup_logger


_DEF_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def norm_text(x: str) -> str:
    x = x.lower().strip()
    x = _DEF_NORM_RE.sub(" ", x)
    x = " ".join(x.split())
    return x


def _parse_answer_cell(v: str) -> List[str]:
    v = (v or "").strip()
    if not v:
        return [""]
    if v.startswith("[") and v.endswith("]"):
        try:
            arr = ast.literal_eval(v)
            if isinstance(arr, list):
                return [str(x) for x in arr if str(x).strip()]
        except Exception:
            pass
    return [v]


def _parse_maybe_list_cell(v: str) -> str:
    v = (v or "").strip()
    if not v:
        return ""
    if v.startswith("[") and v.endswith("]"):
        try:
            arr = ast.literal_eval(v)
            if isinstance(arr, list):
                for x in arr:
                    sx = str(x).strip()
                    if sx:
                        return sx
        except Exception:
            pass
    return v


def load_tsv_task(task_cfg: Dict) -> List[Dict]:
    csv.field_size_limit(2**31 - 1)
    path = Path(task_cfg["path"])
    qcol = task_cfg.get("question_col", "question")
    acol = task_cfg.get("answer_col", "answer")
    icol = task_cfg.get("image_col", "image_path")
    idcol = task_cfg.get("id_col", "index")
    image_root = task_cfg.get("image_root", "")
    image_encoding = str(task_cfg.get("image_encoding", "path"))
    image_cache_dir = str(task_cfg.get("image_cache_dir", f"/tmp/dimoe_eval_img_cache/{task_cfg.get('name', 'task')}"))
    limit = int(task_cfg.get("limit", -1))

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit > 0 and i >= limit:
                break
            q = str(row.get(qcol, ""))
            if not q:
                continue
            ans = _parse_answer_cell(str(row.get(acol, "")))
            image = _parse_maybe_list_cell(str(row.get(icol, "")))
            sid = str(row.get(idcol, i))
            if image_encoding == "base64" and image:
                cache = Path(image_cache_dir)
                cache.mkdir(parents=True, exist_ok=True)
                out_path = cache / f"{sid}.jpg"
                if not out_path.exists():
                    try:
                        payload = base64.b64decode(image, validate=False)
                        out_path.write_bytes(payload)
                    except Exception:
                        out_path = cache / f"{sid}.invalid.jpg"
                        out_path.write_bytes(b"")
                image = str(out_path)
            if image_root and image and not image.startswith("/"):
                image = str((Path(image_root) / image).resolve())
            rows.append({"id": sid, "prompt": q, "answers": ans, "image": image})
    return rows


def load_jsonl_task(task_cfg: Dict) -> List[Dict]:
    path = Path(task_cfg["path"])
    qcol = task_cfg.get("question_col", "text")
    acol = task_cfg.get("answer_col", "answer")
    icol = task_cfg.get("image_col", "image")
    idcol = task_cfg.get("id_col", "question_id")
    image_root = task_cfg.get("image_root", "")
    limit = int(task_cfg.get("limit", -1))

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get(qcol, ""))
            if not q:
                continue
            ans = _parse_answer_cell(str(row.get(acol, ""))) if acol in row else []
            image = str(row.get(icol, ""))
            if image_root and image and not image.startswith("/"):
                image = str((Path(image_root) / image).resolve())
            sid = str(row.get(idcol, i))
            rows.append({"id": sid, "prompt": q, "answers": ans, "image": image})
    return rows


def score_exact(pred: str, answers: List[str]) -> float:
    if not answers:
        return float("nan")
    p = norm_text(pred)
    for a in answers:
        if p == norm_text(str(a)):
            return 1.0
    return 0.0


def eval_suite(cfg: Dict, suite: str, preset: str, out_dir: Path, checkpoint: str = ""):
    logger = setup_logger()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    tasks = cfg["eval"]["tasks"]
    if suite != "all":
        tasks = [t for t in tasks if t.get("suite", "all") in {suite, "all"}]

    summary = []
    for task in tasks:
        fmt = task.get("format", "tsv_qa")
        if fmt == "tsv_qa":
            rows = load_tsv_task(task)
        elif fmt == "jsonl_qa":
            rows = load_jsonl_task(task)
        else:
            raise ValueError(f"unsupported task format: {fmt}")

        preds = []
        scores = []
        for r in rows:
            text = _denoise_generate(
                model=model,
                prompt=r["prompt"],
                image_path=r.get("image", ""),
                max_new_tokens=int(task.get("max_new_tokens", cfg["eval"].get("max_new_tokens", 32))),
                steps=int(p["steps"]),
                remasking=cfg["infer"].get("remasking", "low_confidence"),
            )
            s = score_exact(text, r.get("answers", []))
            if s == s:
                scores.append(s)
            preds.append({
                "id": r["id"],
                "prompt": r["prompt"],
                "prediction": text,
                "answers": r.get("answers", []),
                "score": s,
                "task": task["name"],
                "preset": preset,
            })

        task_out = out_dir / f"{task['name']}.{preset}.preds.jsonl"
        write_jsonl(task_out, preds)

        task_metric = {
            "task": task["name"],
            "preset": preset,
            "num_samples": len(rows),
            "num_scored": len(scores),
            "exact_match": sum(scores) / len(scores) if scores else float("nan"),
            "pred_path": str(task_out),
        }
        write_json(out_dir / f"{task['name']}.{preset}.metrics.json", task_metric)
        summary.append(task_metric)
        logger.info("eval %s preset=%s samples=%d em=%s", task["name"], preset, len(rows), task_metric["exact_match"])

    write_json(out_dir / f"suite_{suite}.{preset}.summary.json", {"suite": suite, "preset": preset, "tasks": summary})


def add_parser(subparsers):
    p = subparsers.add_parser("run", help="Run evaluation suite")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--suite", default="all", type=str)
    p.add_argument("--preset", choices=["fast", "mid", "best"], default="best")
    p.add_argument("--out-dir", default="artifacts/eval", type=str)
    p.add_argument("--checkpoint", default="", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = load_yaml(args.config)
    eval_suite(cfg, args.suite, args.preset, Path(args.out_dir), checkpoint=args.checkpoint)
