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
class TestDMAR(unittest.TestCase):
    def test_dmar_losses_finite(self):
        from dimoe.model.dmar import dmar_losses

        b, l, e = 2, 5, 4
        gate = torch.softmax(torch.randn(b, l, e), dim=-1)
        gate2 = torch.softmax(torch.randn(b, l, e), dim=-1)
        prefix = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
        t_bucket = torch.randint(0, 8, (b, l))
        token_type = torch.randint(0, 4, (b, l))
        prior = torch.zeros(8, 4, e)

        out = dmar_losses(
            gate_probs=gate,
            gate_probs_other=gate2,
            prefix_mask=prefix,
            t_bucket=t_bucket,
            token_type=token_type,
            prior_table=prior,
            lambda_prior=0.1,
            lambda_consistency=0.1,
            lambda_balance=0.1,
            lambda_entropy=0.01,
        )

        self.assertTrue(torch.isfinite(out.prior_loss).item())
        self.assertTrue(torch.isfinite(out.consistency_loss).item())
        self.assertTrue(torch.isfinite(out.balance_loss).item())
        self.assertTrue(torch.isfinite(out.entropy_loss).item())


if __name__ == "__main__":
    unittest.main()
