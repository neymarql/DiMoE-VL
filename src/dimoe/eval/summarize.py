from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from dimoe.utils.io import write_json


def summarize(eval_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(eval_dir.glob("*.metrics.json"))

    rows: List[Dict] = []
    for m in metrics:
        with m.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    main_csv = out_dir / "main_table.csv"
    with main_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "preset", "num_samples", "num_scored", "exact_match", "pred_path"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    by_preset: Dict[str, List[float]] = {}
    for r in rows:
        p = r.get("preset", "")
        em = r.get("exact_match", float("nan"))
        if isinstance(em, (int, float)) and em == em:
            by_preset.setdefault(p, []).append(float(em))

    summary = {
        "num_tasks": len(rows),
        "preset_avg_em": {k: (sum(v) / len(v) if v else float("nan")) for k, v in by_preset.items()},
    }

    write_json(out_dir / "summary.json", summary)


def add_parser(subparsers):
    p = subparsers.add_parser("summarize", help="Summarize evaluation outputs")
    p.add_argument("--eval-dir", default="artifacts/eval", type=str)
    p.add_argument("--out-dir", default="artifacts/results", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    summarize(Path(args.eval_dir), Path(args.out_dir))
