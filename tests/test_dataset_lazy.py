import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DummyTok:
    def __init__(self):
        self.vocab = {"<image>": 1}

    def __call__(self, text, add_special_tokens=False):
        toks = text.replace("\n", " ").split()
        ids = [self.vocab.get(t, 10 + i) for i, t in enumerate(toks)]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, 0)


class TestDatasetLazy(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch not installed")
    def test_load_by_offsets(self):
        from dimoe.train.dataset import DimoeJsonlDataset

        rows = [
            {
                "id": "1",
                "image_abs": "/tmp/a.jpg",
                "meta": {"prompt": "<image>\nwhat", "answer": "cat", "task": "vqa", "source": "x", "answer_len": 1},
            },
            {
                "id": "2",
                "image_abs": "/tmp/b.jpg",
                "meta": {"prompt": "<image>\nwhere", "answer": "home", "task": "vqa", "source": "x", "answer_len": 1},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            ds = DimoeJsonlDataset(str(p), DummyTok(), max_length=64, image_token="<image>")
            self.assertEqual(len(ds), 2)
            x = ds[1]
            self.assertEqual(x["sample_uid"], "2")
            self.assertEqual(x["image_path"], "/tmp/b.jpg")


if __name__ == "__main__":
    unittest.main()
