import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dimoe.utils.io import iter_json_array


class TestIoStreaming(unittest.TestCase):
    def test_iter_json_array(self):
        rows = [{"id": i, "x": i * 2} for i in range(10)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "arr.json"
            p.write_text(json.dumps(rows), encoding="utf-8")

            out = list(iter_json_array(p))
            self.assertEqual(len(out), 10)
            self.assertEqual(out[0]["id"], 0)
            self.assertEqual(out[-1]["x"], 18)


if __name__ == "__main__":
    unittest.main()
