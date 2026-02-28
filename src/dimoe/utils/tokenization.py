from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=8)
def get_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    return tok


def tokenize_len(name_or_path: str, text: str) -> int:
    tok = get_tokenizer(name_or_path)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return int(len(ids))


def tokenize_ids(name_or_path: str, text: str) -> List[int]:
    tok = get_tokenizer(name_or_path)
    return list(tok(text, add_special_tokens=False)["input_ids"])
