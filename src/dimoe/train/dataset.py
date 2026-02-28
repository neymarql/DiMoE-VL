from __future__ import annotations

import json
import os
from array import array
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class DimoeJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int,
        image_token: str,
    ):
        self.jsonl_path = str(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.image_token = image_token

        self._offsets = array("Q")
        with open(self.jsonl_path, "rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    self._offsets.append(pos)
                pos = f.tell()
                line = f.readline()

        self._fh = None
        self._fh_pid: Optional[int] = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._fh_pid = None

    def __del__(self):
        self.close()

    def _file(self):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            self.close()
            self._fh = open(self.jsonl_path, "r", encoding="utf-8")
            self._fh_pid = pid
        return self._fh

    def __len__(self) -> int:
        return len(self._offsets)

    def _encode(self, prompt: str, answer: str):
        prompt_text = prompt.strip()
        if "<image>" not in prompt_text:
            prompt_text = f"<image>\n{prompt_text}".strip()

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))

        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        prompt_mask = [1] * prompt_len + [0] * (len(input_ids) - prompt_len)
        # token types: 0 vision placeholder, 1 prompt, 2 answer
        token_type_ids = []
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        for idx, tok in enumerate(input_ids):
            if tok == image_token_id:
                token_type_ids.append(0)
            elif idx < prompt_len:
                token_type_ids.append(1)
            else:
                token_type_ids.append(2)

        attention_mask = [1] * len(input_ids)
        return input_ids, labels, prompt_mask, token_type_ids, attention_mask

    def __getitem__(self, idx: int) -> Dict:
        fh = self._file()
        fh.seek(self._offsets[idx])
        line = fh.readline()
        row = json.loads(line)

        meta = row.get("meta", {})
        prompt = str(meta.get("prompt", ""))
        answer = str(meta.get("answer", ""))

        input_ids, labels, prompt_mask, token_type_ids, attention_mask = self._encode(prompt, answer)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_mask": prompt_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "image_path": str(row.get("image_abs", "")),
            "sample_uid": str(row.get("id", f"idx-{idx}")),
            "task": str(meta.get("task", "unknown")),
            "source": str(meta.get("source", "unknown")),
            "answer_len": int(meta.get("answer_len", 0)),
        }


class DimoeCollator:
    def __init__(self, pad_token_id: int, fixed_length: Optional[int] = None):
        self.pad_token_id = int(pad_token_id)
        self.fixed_length = int(fixed_length) if fixed_length is not None else None

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
        max_len = self.fixed_length if self.fixed_length is not None else max(len(x["input_ids"]) for x in batch)

        def pad(seq, pad_val):
            if len(seq) >= max_len:
                return seq[:max_len]
            return seq + [pad_val] * (max_len - len(seq))

        out = {
            "input_ids": torch.tensor([pad(x["input_ids"], self.pad_token_id) for x in batch], dtype=torch.long),
            "labels": torch.tensor([pad(x["labels"], -100) for x in batch], dtype=torch.long),
            "prompt_mask": torch.tensor([pad(x["prompt_mask"], 0) for x in batch], dtype=torch.bool),
            "token_type_ids": torch.tensor([pad(x["token_type_ids"], 1) for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long),
            "image_paths": [x["image_path"] for x in batch],
            "sample_uids": [x["sample_uid"] for x in batch],
            "tasks": [x["task"] for x in batch],
            "sources": [x["source"] for x in batch],
            "answer_len": torch.tensor([x["answer_len"] for x in batch], dtype=torch.long),
        }
        return out
