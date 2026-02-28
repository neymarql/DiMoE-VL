from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from dimoe.utils.config import load_yaml
from dimoe.utils.io import read_jsonl, write_json
from dimoe.utils.logging import setup_logger


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _add_image_if_exists(tar: tarfile.TarFile, sample: dict, key: str, include_image: bool) -> bool:
    if not include_image:
        return False
    image_abs = sample.get("image_abs", "")
    if not image_abs:
        return False
    p = Path(image_abs)
    if not p.exists():
        return False
    try:
        payload = p.read_bytes()
    except Exception:
        return False
    ext = p.suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".jpg"
    info = tarfile.TarInfo(name=f"{key}{ext}")
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))
    return True


def _open_shard(out_dir: Path, shard_idx: int) -> tarfile.TarFile:
    shard_path = out_dir / f"shard-{shard_idx:06d}.tar"
    return tarfile.open(shard_path, "w")


def build_wds(config_path: Path) -> Dict[str, dict]:
    cfg = load_yaml(config_path)
    logger = setup_logger()

    source = Path(cfg["source_jsonl"])
    out_dir = Path(cfg.get("output_dir", "artifacts/data/v1/wds"))
    shard_size = int(cfg.get("shard_size", 2000))
    include_image = bool(cfg.get("include_image", True))

    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source_jsonl": str(source),
        "output_dir": str(out_dir),
        "shard_size": shard_size,
        "include_image": include_image,
        "num_rows": 0,
        "num_shards": 0,
        "images_written": 0,
    }

    image_count = 0
    shard_idx = 0
    row_in_shard = 0
    total_rows = 0

    tar: Optional[tarfile.TarFile] = None
    try:
        tar = _open_shard(out_dir, shard_idx)
        for sample in tqdm(read_jsonl(source), desc="wds-pack"):
            key = f"{shard_idx:06d}-{row_in_shard:06d}"
            _add_text(tar, f"{key}.json", json.dumps(sample, ensure_ascii=False))
            if _add_image_if_exists(tar, sample, key, include_image):
                image_count += 1

            row_in_shard += 1
            total_rows += 1

            if row_in_shard >= shard_size:
                tar.close()
                logger.info("wrote shard shard-%06d.tar rows=%d", shard_idx, row_in_shard)
                shard_idx += 1
                row_in_shard = 0
                tar = _open_shard(out_dir, shard_idx)

        if tar is not None:
            tar.close()
            tar = None

    finally:
        if tar is not None:
            tar.close()

    # If final shard is empty because we rolled exactly at boundary, remove it.
    last_path = out_dir / f"shard-{shard_idx:06d}.tar"
    if row_in_shard == 0 and last_path.exists():
        last_path.unlink()

    num_shards = len(list(out_dir.glob("shard-*.tar")))

    report["num_shards"] = num_shards
    report["num_rows"] = total_rows
    report["images_written"] = image_count
    write_json(out_dir / "wds_report.json", report)
    return report


def add_parser(subparsers):
    p = subparsers.add_parser("build-wds", help="Pack jsonl dataset into WebDataset tar shards")
    p.add_argument("--config", required=True, type=str)
    p.set_defaults(func=main)


def main(args: argparse.Namespace):
    build_wds(Path(args.config))
