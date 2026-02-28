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
class TestMasking(unittest.TestCase):
    def test_complementary_masks_disjoint_and_cover(self):
        from dimoe.model.masking import TimestepSampler, build_complementary_masks

        labels = torch.tensor([
            [-100, -100, 11, 12, 13],
            [-100, 21, 22, -100, -100],
        ])
        prompt_mask = labels.eq(-100)

        out = build_complementary_masks(
            labels=labels,
            prompt_mask=prompt_mask,
            include_prompt_ratio=0.0,
            timestep_sampler_1=TimestepSampler("uniform"),
            timestep_sampler_2=TimestepSampler("uniform"),
            force_answer_coverage=True,
        )

        self.assertTrue(torch.logical_and(out.mask1, out.mask2).sum().item() == 0)

        ans = labels.ne(-100)
        covered = torch.logical_or(out.mask1, out.mask2)
        self.assertTrue(torch.logical_and(ans, ~covered).sum().item() == 0)


if __name__ == "__main__":
    unittest.main()
