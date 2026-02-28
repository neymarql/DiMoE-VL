from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def _iter_json_array_raw_decode(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    started = False
    buf = ""

    with path.open("r", encoding="utf-8") as f:
        eof = False
        while True:
            if not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buf += chunk
                else:
                    eof = True

            idx = 0
            while True:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1

                if not started:
                    if idx >= len(buf):
                        break
                    if buf[idx] != "[":
                        raise ValueError(f"Expected JSON array start '[' in {path}")
                    started = True
                    idx += 1
                    continue

                while idx < len(buf) and buf[idx].isspace():
                    idx += 1

                if idx >= len(buf):
                    break

                if buf[idx] == ",":
                    idx += 1
                    continue
                if buf[idx] == "]":
                    return

                try:
                    obj, end = decoder.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break

                if isinstance(obj, dict):
                    yield obj
                idx = end

            if idx > 0:
                buf = buf[idx:]

            if eof:
                remain = buf.strip()
                if remain in {"", "]"}:
                    return
                # final trailing object parse attempt
                if remain.startswith(","):
                    remain = remain[1:].lstrip()
                if remain.startswith("]"):
                    return
                if not remain:
                    return
                try:
                    obj, end = decoder.raw_decode(remain)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON array near EOF in {path}") from exc
                if isinstance(obj, dict):
                    yield obj
                return


def iter_json_array(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    try:
        import ijson  # type: ignore

        with p.open("rb") as f:
            for item in ijson.items(f, "item"):
                if isinstance(item, dict):
                    yield item
        return
    except Exception:
        pass

    yield from _iter_json_array_raw_decode(p)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
