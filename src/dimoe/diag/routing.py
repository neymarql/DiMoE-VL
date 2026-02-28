from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from dimoe.utils.io import read_jsonl, write_json


def _cv(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    m = sum(xs) / len(xs)
    if m == 0:
        return float("nan")
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(v) / m


def analyze(log_path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(log_path))

    steps = []
    cv = []
    top1_dom = []
    for r in rows:
        gm = r.get("gate_mean", [])
        if not gm:
            continue
        steps.append(int(r.get("step", len(steps))))
        cv.append(_cv([float(x) for x in gm]))
        top1_dom.append(max(float(x) for x in gm))

    if steps:
        plt.figure(figsize=(8, 4))
        plt.plot(steps, cv, label="load_cv")
        plt.plot(steps, top1_dom, label="top1_dominance")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(out_dir / "routing_diagnostics.png")
        plt.close()

    summary = {
        "num_points": len(steps),
        "avg_cv": sum(cv) / len(cv) if cv else float("nan"),
        "avg_top1_dominance": sum(top1_dom) / len(top1_dom) if top1_dom else float("nan"),
    }
    write_json(out_dir / "routing_summary.json", summary)


def add_parser(subparsers):
    p = subparsers.add_parser("routing", help="Analyze routing logs")
    p.add_argument("--log", required=True, type=str)
    p.add_argument("--out-dir", default="artifacts/diag/routing", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    analyze(Path(args.log), Path(args.out_dir))
