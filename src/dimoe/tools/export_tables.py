from __future__ import annotations

import argparse
from pathlib import Path

from dimoe.eval.summarize import summarize


def add_parser(subparsers):
    p = subparsers.add_parser("export-paper-tables", help="Export paper-ready tables from eval outputs")
    p.add_argument("--eval-dir", default="artifacts/eval", type=str)
    p.add_argument("--out-dir", default="artifacts/results", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    summarize(Path(args.eval_dir), Path(args.out_dir))
