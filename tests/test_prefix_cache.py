import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestPrefixCache(unittest.TestCase):
    def test_cache_hit_and_miss(self):
        from dimoe.model.prefix_cache import PrefixRoutingCache

        c = PrefixRoutingCache(jsd_threshold=0.02)
        p = torch.tensor([0.9, 0.1])
        q = torch.tensor([0.89, 0.11])
        r = torch.tensor([0.5, 0.5])

        c.put("s1", 0, 3, experts=torch.tensor([0]), probs=p)

        self.assertIsNotNone(c.get("s1", 0, 3, q))
        self.assertIsNone(c.get("s1", 0, 3, r))


if __name__ == "__main__":
    unittest.main()
