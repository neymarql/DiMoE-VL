from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str = "dimoe", level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    log_level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, obj: Dict[str, Any]) -> None:
        rec = dict(obj)
        rec.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def env_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            try:
                return int(os.environ[key])
            except Exception:
                continue
    return 0
