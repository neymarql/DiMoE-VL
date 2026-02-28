from __future__ import annotations

import argparse
from pathlib import Path

from dimoe.utils.io import read_jsonl, write_json


def analyze(log_path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(log_path))

    steps = [int(r.get("step", i)) for i, r in enumerate(rows)]
    vg = [float(r.get("vision_grad_norm", 0.0)) for r in rows]
    eg = [float(r.get("expert_grad_norm", 0.0)) for r in rows]

    if steps:
        plt.figure(figsize=(8, 4))
        plt.plot(steps, vg, label="vision_grad_norm")
        plt.plot(steps, eg, label="expert_grad_norm")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("norm")
        plt.tight_layout()
        plt.savefig(out_dir / "gradient_diagnostics.png")
        plt.close()

    write_json(
        out_dir / "gradient_summary.json",
        {
            "num_points": len(rows),
            "vision_grad_avg": sum(vg) / len(vg) if vg else float("nan"),
            "expert_grad_avg": sum(eg) / len(eg) if eg else float("nan"),
        },
    )


def add_parser(subparsers):
    p = subparsers.add_parser("gradient", help="Analyze gradient logs")
    p.add_argument("--log", required=True, type=str)
    p.add_argument("--out-dir", default="artifacts/diag/gradient", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    analyze(Path(args.log), Path(args.out_dir))
