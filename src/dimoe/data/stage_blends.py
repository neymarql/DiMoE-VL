from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from dimoe.utils.config import load_yaml
from dimoe.utils.io import read_jsonl, write_json, write_jsonl
from dimoe.utils.logging import setup_logger


def _reservoir_sample_jsonl(src: Path, max_samples: int, seed: int) -> Tuple[List[dict], int]:
    if max_samples <= 0:
        # Caller should use streaming copy for full-data mode to avoid OOM.
        rows = []
        seen = 0
        for row in read_jsonl(src):
            rows.append(row)
            seen += 1
        return rows, seen

    rng = random.Random(seed)
    sample: List[dict] = []
    seen = 0
    for row in read_jsonl(src):
        seen += 1
        if len(sample) < max_samples:
            sample.append(row)
            continue
        j = rng.randint(0, seen - 1)
        if j < max_samples:
            sample[j] = row
    return sample, seen


def build_stage_blends(config_path: Path) -> Dict[str, dict]:
    cfg = load_yaml(config_path)
    logger = setup_logger()

    stages = cfg.get("stages", {})
    if not isinstance(stages, dict) or not stages:
        raise ValueError("stages must be a non-empty mapping in config")

    out_root = Path(cfg.get("output_dir", "artifacts/data/v1/blends"))
    out_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 3407))

    report: Dict[str, dict] = {}

    for stage_name, stage_cfg in stages.items():
        src = Path(stage_cfg["source_jsonl"])
        max_samples = int(stage_cfg.get("max_samples", -1))
        shuffle = bool(stage_cfg.get("shuffle", False))
        out_path = out_root / f"{stage_name}.jsonl"

        if max_samples <= 0 and not shuffle:
            total_rows = 0
            with src.open("r", encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    # Keep one-line JSONL contract explicit.
                    row = json.loads(line)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
            kept_rows = total_rows
        else:
            rows, total_rows = _reservoir_sample_jsonl(src, max_samples=max_samples, seed=seed)
            if shuffle:
                random.Random(seed).shuffle(rows)
            write_jsonl(out_path, rows)
            kept_rows = len(rows)

        report[stage_name] = {
            "source": str(src),
            "output": str(out_path),
            "source_rows": total_rows,
            "rows": kept_rows,
            "max_samples": max_samples,
            "shuffle": shuffle,
        }
        logger.info("stage blend %s -> %s rows=%d source_rows=%d", stage_name, out_path, kept_rows, total_rows)

    write_json(out_root / "stage_blend_report.json", report)
    return report


def add_parser(subparsers):
    p = subparsers.add_parser("build-stage-blends", help="Build stage training blends from normalized jsonl")
    p.add_argument("--config", required=True, type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    build_stage_blends(Path(args.config))
