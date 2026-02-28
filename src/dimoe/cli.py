from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict, Tuple


COMMAND_REGISTRY: Dict[Tuple[str, str], str] = {
    ("exp", "init"): "dimoe.exp.init",
    ("data", "normalize"): "dimoe.data.normalize",
    ("data", "build-stage-blends"): "dimoe.data.stage_blends",
    ("data", "build-wds"): "dimoe.data.wds_pack",
    ("train", "run"): "dimoe.train.run",
    ("infer", "generate"): "dimoe.infer.generate",
    ("infer", "benchmark"): "dimoe.infer.benchmark",
    ("infer", "serve"): "dimoe.infer.serve",
    ("eval", "run"): "dimoe.eval.run",
    ("eval", "summarize"): "dimoe.eval.summarize",
    ("diag", "routing"): "dimoe.diag.routing",
    ("diag", "gradient"): "dimoe.diag.gradient",
    ("diag", "supervision"): "dimoe.diag.supervision",
    ("tools", "export-paper-tables"): "dimoe.tools.export_tables",
}

GROUPS = {
    "exp": ["init"],
    "data": ["normalize", "build-stage-blends", "build-wds"],
    "train": ["run"],
    "infer": ["generate", "benchmark", "serve"],
    "eval": ["run", "summarize"],
    "diag": ["routing", "gradient", "supervision"],
    "tools": ["export-paper-tables"],
}


def _print_top_help() -> None:
    print("usage: dimoe <group> <command> [args]")
    print("")
    print("groups:")
    for g, cmds in GROUPS.items():
        print(f"  {g:8s} {', '.join(cmds)}")


def _print_group_help(group: str) -> None:
    cmds = GROUPS.get(group)
    if not cmds:
        _print_top_help()
        return
    print(f"usage: dimoe {group} <command> [args]")
    print("")
    print("commands:")
    for c in cmds:
        print(f"  {c}")


def _build_parser(group: str, cmd: str):
    mod_path = COMMAND_REGISTRY.get((group, cmd))
    if not mod_path:
        raise ValueError(f"unknown command: {group} {cmd}")

    mod = importlib.import_module(mod_path)
    parser = argparse.ArgumentParser("dimoe")
    sub = parser.add_subparsers(dest="group", required=True)

    grp = sub.add_parser(group)
    grp_sub = grp.add_subparsers(dest="cmd", required=True)
    mod.add_parser(grp_sub)
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        _print_top_help()
        return

    group = argv[0]
    if group not in GROUPS:
        _print_top_help()
        raise SystemExit(2)

    if len(argv) == 1 or argv[1] in {"-h", "--help"}:
        _print_group_help(group)
        return

    cmd = argv[1]
    if (group, cmd) not in COMMAND_REGISTRY:
        _print_group_help(group)
        raise SystemExit(2)

    parser = _build_parser(group, cmd)
    args = parser.parse_args()
    args.func(args)
