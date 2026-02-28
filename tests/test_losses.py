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
class TestLosses(unittest.TestCase):
    def test_masked_ce_zero_tokens_is_finite(self):
        from dimoe.model.losses import masked_ce_loss

        logits = torch.randn(2, 4, 10)
        labels = torch.randint(low=0, high=10, size=(2, 4))
        mask = torch.zeros(2, 4, dtype=torch.bool)

        out = masked_ce_loss(logits, labels, mask)
        self.assertEqual(out.token_count, 0)
        self.assertTrue(torch.isfinite(out.loss).item())
        self.assertEqual(float(out.loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
