from __future__ import annotations

import argparse
import time
from pathlib import Path

from dimoe.utils.io import write_json


def init_experiment(name: str, root: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = root / f"{name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        exp_dir / "manifest.json",
        {
            "name": name,
            "timestamp": ts,
            "paths": {
                "checkpoints": str((exp_dir / "checkpoints").resolve()),
                "logs": str((exp_dir / "logs").resolve()),
                "results": str((exp_dir / "results").resolve()),
            },
        },
    )
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    readme = exp_dir / "README.md"
    readme.write_text(
        "# Experiment\n\n"
        f"name: {name}\n"
        f"timestamp: {ts}\n\n"
        "## Repro\n"
        "- data normalize\n"
        "- train stage0 -> stage3\n"
        "- eval + diagnostics\n",
        encoding="utf-8",
    )


def add_parser(subparsers):
    p = subparsers.add_parser("init", help="Initialize experiment directory")
    p.add_argument("--name", required=True, type=str)
    p.add_argument("--root", default="experiments", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    init_experiment(args.name, Path(args.root))
