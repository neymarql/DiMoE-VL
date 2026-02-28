from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from dimoe.data.schema import to_standard_sample
from dimoe.utils.config import load_yaml
from dimoe.utils.io import iter_json_array, write_json
from dimoe.utils.logging import setup_logger
from dimoe.utils.tokenization import tokenize_len


def _token_len_safe(tokenizer_name: str, text: str) -> int:
    try:
        return tokenize_len(tokenizer_name, text)
    except Exception:
        return len(text.split())


def _collect_input_files(in_dir: Path) -> List[Path]:
    canonical = [
        "stage1_projector.json",
        "stage2_llm.json",
        "stage3_vision.json",
        "stage4_joint.json",
    ]
    found = [in_dir / x for x in canonical if (in_dir / x).is_file()]
    if found:
        return found

    pat = re.compile(r"^stage\\d+_.*\\.json$")
    files = sorted([p for p in in_dir.glob("*.json") if p.is_file() and pat.match(p.name)])
    return files


def normalize_dir(
    in_dir: Path,
    out_dir: Path,
    tokenizer_name: str,
    datasets_root: str,
) -> Dict[str, Dict[str, int]]:
    logger = setup_logger()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    files = _collect_input_files(in_dir)
    if not files:
        raise FileNotFoundError(f"No .json files found in {in_dir}")

    for fp in files:
        stage = fp.stem
        out_jsonl = out_dir / f"{stage}.jsonl"

        n = 0
        kept = 0
        dropped = 0

        with out_jsonl.open("w", encoding="utf-8") as out_f:
            for raw in tqdm(iter_json_array(fp), desc=f"normalize:{stage}"):
                n += 1
                sample = to_standard_sample(
                    raw=raw,
                    tokenizer_name=tokenizer_name,
                    answer_len_fn=_token_len_safe,
                    datasets_root=datasets_root,
                )
                meta = sample.get("meta", {})
                if not meta.get("answer"):
                    dropped += 1
                    continue
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1

        summary[stage] = {"total": n, "kept": kept, "dropped": dropped}
        logger.info("normalized %s -> %s (kept=%d/%d)", fp.name, out_jsonl, kept, n)

    write_json(out_dir / "normalize_summary.json", summary)
    return summary


def add_parser(subparsers):
    p = subparsers.add_parser("normalize", help="Normalize training json arrays to dimoe jsonl schema")
    p.add_argument("--config", default="", type=str)
    p.add_argument("--in", dest="in_dir", default="", type=str)
    p.add_argument("--out", dest="out_dir", default="", type=str)
    p.add_argument("--tokenizer", default="", type=str)
    p.add_argument("--datasets-root", default="", type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    cfg = {}
    if args.config:
        cfg = load_yaml(args.config)

    in_dir = args.in_dir or cfg.get("in_dir", "")
    out_dir = args.out_dir or cfg.get("out_dir", "")
    tokenizer = args.tokenizer or cfg.get("tokenizer", "Qwen/Qwen2.5-7B-Instruct")
    datasets_root = args.datasets_root or cfg.get("datasets_root", "/home/qianlong/datasets")

    if not in_dir or not out_dir:
        raise ValueError("--in/--out (or config in_dir/out_dir) are required")

    normalize_dir(
        in_dir=Path(in_dir),
        out_dir=Path(out_dir),
        tokenizer_name=tokenizer,
        datasets_root=datasets_root,
    )
