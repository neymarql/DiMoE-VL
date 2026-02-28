# DiMoE-VL

DiMoE-VL implementation repo (Diffusion + MoE + VLM) with end-to-end infra:
- Data pipeline (normalize -> stage blends -> WebDataset shards)
- Stage training (`stage0_naive -> stage1_align -> stage2_full -> stage3_lownfe`)
- Inference (fast/mid/best presets)
- Evaluation and paper table export
- Diagnostics for DMAR/EBCMC problem-existence evidence

## Quick Start

```bash
cd /home/qianlong/DiMoE-VL
pip install -e .
python dimoe.py --help
```

## Core Commands

```bash
python dimoe.py exp init --name dimoe_nips2026 --root experiments
python dimoe.py data normalize --in /home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute --out artifacts/data/v1
python dimoe.py data build-stage-blends --config configs/data/stage_blends.yaml
python dimoe.py train run --stage stage0_naive --config configs/train/stage0_naive.yaml
python dimoe.py infer benchmark --preset mid --config configs/infer/default.yaml --dataset-jsonl artifacts/data/v1/blends/stage3_lownfe.jsonl
python dimoe.py eval run --suite all --preset best --config configs/eval/default.yaml --out-dir artifacts/eval/best
```

## One-Click Scripts

```bash
bash scripts/local/00_data_pipeline.sh
bash scripts/local/11_train_all_stages.sh
bash scripts/local/20_eval_and_diag.sh
bash scripts/local/30_benchmark_grid.sh
```

## Full Repro SOP

See: `docs/DIMOE_VL_NIPS2026_RUNBOOK.md`
