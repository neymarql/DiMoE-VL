import subprocess
import sys
import unittest
from pathlib import Path


class TestCliTopHelp(unittest.TestCase):
    def test_help(self):
        repo = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, str(repo / "dimoe.py"), "--help"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("usage: dimoe", out.stdout)


if __name__ == "__main__":
    unittest.main()
