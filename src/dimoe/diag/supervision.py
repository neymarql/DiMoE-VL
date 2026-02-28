from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dimoe.model.masking import estimate_no_supervision_ratio
from dimoe.utils.io import read_jsonl, write_json


def analyze(manifest: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    lens = []
    tasks = Counter()
    sources = Counter()
    for r in read_jsonl(manifest):
        m = r.get("meta", {})
        n = int(m.get("answer_len", 0))
        lens.append(n)
        tasks[str(m.get("task", "unknown"))] += 1
        sources[str(m.get("source", "unknown"))] += 1

    if lens:
        xs = [x for x in lens if x > 0]
        ctr = Counter(xs)
        plt.figure(figsize=(8, 4))
        plt.bar(list(ctr.keys()), list(ctr.values()))
        plt.xlabel("answer_len")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "answer_len_hist.png")
        plt.close()

    import torch

    n = torch.tensor([x for x in lens if x > 0], dtype=torch.float32)
    no_sup = estimate_no_supervision_ratio(n).mean().item() if len(n) > 0 else float("nan")

    write_json(
        out_dir / "supervision_summary.json",
        {
            "num_samples": len(lens),
            "avg_answer_len": (sum(lens) / len(lens) if lens else float("nan")),
            "est_no_supervision_ratio": no_sup,
            "task_counts": dict(tasks),
            "source_counts": dict(sources),
        },
    )


def add_parser(subparsers):
    p = subparsers.add_parser("supervision", help="Analyze answer-length supervision sparsity")
    p.add_argument("--manifest", required=True, type=str)
    p.add_argument("--out-dir", default="artifacts/diag/supervision", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    analyze(Path(args.manifest), Path(args.out_dir))
