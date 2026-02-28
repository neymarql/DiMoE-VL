from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _role(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"human", "user"}:
        return "user"
    if s in {"assistant", "gpt", "bot"}:
        return "assistant"
    return s


def infer_task(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["option", "a, b, c", "multiple choice", "choose"]):
        return "mcq"
    if any(k in t for k in ["chart", "table", "document", "ocr", "read text"]):
        return "ocr"
    if any(k in t for k in ["reason", "why", "explain", "proof"]):
        return "reasoning"
    if any(k in t for k in ["caption", "describe", "what is in"]):
        return "caption"
    return "vqa"


def infer_source(image_path: str) -> str:
    p = image_path.lower()
    for key in [
        "llava-onevision-data",
        "chartqa",
        "textvqa",
        "docvqa",
        "scienceqa",
        "mathvista",
        "vqav2",
        "seedbench",
    ]:
        if key in p:
            return key
    return "unknown"


@dataclass
class ParsedConversation:
    prompt: str
    answer: str
    prompt_turns: List[Dict[str, str]]
    answer_turn: Dict[str, str]


def parse_conversations(conv: List[Dict[str, Any]]) -> ParsedConversation:
    if not conv:
        return ParsedConversation(prompt="", answer="", prompt_turns=[], answer_turn={"from": "assistant", "value": ""})

    normalized: List[Dict[str, str]] = []
    for turn in conv:
        role = _role(str(turn.get("from", "")))
        text = str(turn.get("value", "")).strip()
        normalized.append({"from": role, "value": text})

    answer_idx = -1
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i]["from"] == "assistant":
            answer_idx = i
            break
    if answer_idx < 0:
        answer_idx = len(normalized) - 1

    answer_turn = normalized[answer_idx]
    prompt_turns = normalized[:answer_idx]
    prompt = "\n".join([t["value"] for t in prompt_turns if t["from"] == "user"]).strip()
    answer = answer_turn.get("value", "")

    return ParsedConversation(prompt=prompt, answer=answer, prompt_turns=prompt_turns, answer_turn=answer_turn)


def to_standard_sample(
    raw: Dict[str, Any],
    tokenizer_name: str,
    answer_len_fn,
    datasets_root: str,
) -> Dict[str, Any]:
    sample_id = str(raw.get("id", ""))
    image = str(raw.get("image", ""))
    conv = raw.get("conversations", [])
    if not isinstance(conv, list):
        conv = []

    parsed = parse_conversations(conv)
    answer = parsed.answer
    prompt = parsed.prompt
    answer_len = int(answer_len_fn(tokenizer_name, answer)) if answer else 0

    task = infer_task(prompt + "\n" + answer)
    source = infer_source(image)

    abs_image = str((Path(datasets_root) / image).resolve()) if image else ""

    out = {
        "id": sample_id,
        "image": image,
        "image_abs": abs_image,
        "conversations": conv,
        "meta": {
            "source": source,
            "task": task,
            "answer_len": answer_len,
            "has_image": bool(image),
            "prompt": prompt,
            "answer": answer,
        },
    }
    return out
