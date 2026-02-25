#!/usr/bin/env python3
"""Build MC-DSA training data from stage2_llm.json and stage3_vision.json.

Output format (JSONL):
{
  "id": str,
  "image": str,
  "prompt": str,
  "choices": [str],
  "answer_idx": int,
  "source": "stage2"|"stage3"
}
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional


CHOICE_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.*?)\s*$", re.MULTILINE)
ANSWER_LETTER_RE = re.compile(r"(?:answer\s*is|option)\s*[:：]?\s*([A-Z])\b", re.IGNORECASE)


def iter_json_array_items(path: str) -> Iterator[Dict]:
    """Stream items from a JSON array file.

    Uses ijson when available; falls back to full json.load otherwise.
    """
    try:
        import ijson  # type: ignore

        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
        return
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    for obj in data:
        if isinstance(obj, dict):
            yield obj


def extract_prompt_choices(question: str) -> Optional[Dict]:
    if not question:
        return None

    # Split around "Choices:" if present.
    if "Choices:" in question:
        prompt_part, choices_part = question.split("Choices:", 1)
    else:
        prompt_part, choices_part = question, question

    matches = CHOICE_LINE_RE.findall(choices_part)
    if not matches:
        return None

    letters = []
    choices = []
    for letter, text in matches:
        letters.append(letter)
        choices.append(text.strip())

    return {
        "prompt": prompt_part.strip(),
        "letters": letters,
        "choices": choices,
    }


def extract_answer_idx(answer_text: str, letters: List[str]) -> Optional[int]:
    if not answer_text:
        return None
    m = ANSWER_LETTER_RE.search(answer_text)
    if m is None:
        return None
    ans_letter = m.group(1).upper()
    try:
        return letters.index(ans_letter)
    except ValueError:
        return None


def convert_file(path: str, source_name: str, max_items: Optional[int] = None) -> Iterator[Dict]:
    n = 0
    for item in iter_json_array_items(path):
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        q = conversations[0].get("value", "")
        a = conversations[1].get("value", "")

        parsed = extract_prompt_choices(q)
        if parsed is None:
            continue

        ans_idx = extract_answer_idx(a, parsed["letters"])
        if ans_idx is None:
            continue

        yield {
            "id": str(item.get("id", f"{source_name}_{n}")),
            "image": item.get("image", ""),
            "prompt": parsed["prompt"],
            "choices": parsed["choices"],
            "answer_idx": ans_idx,
            "source": source_name,
        }

        n += 1
        if max_items is not None and n >= max_items:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", required=True, help="Path to stage2_llm.json")
    parser.add_argument("--stage3", required=True, help="Path to stage3_vision.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max_stage2", type=int, default=None)
    parser.add_argument("--max_stage3", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total = 0
    with open(args.output, "w", encoding="utf-8") as w:
        for rec in convert_file(args.stage3, "stage3", args.max_stage3):
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

        for rec in convert_file(args.stage2, "stage2", args.max_stage2):
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} MC-DSA records to {args.output}")


if __name__ == "__main__":
    main()
