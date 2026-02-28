# DiMoE-VL NIPS 2026 Runbook (Full Infra)

This repository implements the full DiMoE-VL training/eval pipeline on top of open-source diffusion infra.

## 1. Infra Baseline Selection

- Diffusion VLM infra reference: `research_repos/LaViDa` (comp-mask, prefix-cache, timestep shift ideas)
- Diffusion-MoE LLM backbone target: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`
- Optional diffusion training reference: `research_repos/SMDM`
- Optional multimodal diffusion reference: `research_repos/MMaDA`

Implemented production infra in this repo:
- Data pipeline: `dimoe data normalize/build-stage-blends/build-wds`
- Training pipeline: `dimoe train run`
- Inference: `dimoe infer generate/benchmark/serve`
- Evaluation + table export: `dimoe eval run/summarize`, `dimoe tools export-paper-tables`
- Diagnostics: `dimoe diag supervision/routing/gradient`

## 2. Environment

```bash
cd /home/qianlong/DiMoE-VL
pip install -e .
# or minimally install required runtime deps: torch transformers pyyaml tqdm pillow fastapi uvicorn matplotlib
```

## 3. Data Pipeline (using /home/qianlong/datasets)

### 3.1 Normalize stage JSON arrays -> standard JSONL

```bash
python dimoe.py data normalize \
  --in /home/qianlong/datasets/train_json_stage_blends_omvl16m_full95_v4_compute \
  --out artifacts/data/v1 \
  --tokenizer inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --datasets-root /home/qianlong/datasets
```

Output:
- `artifacts/data/v1/stage1_projector.jsonl`
- `artifacts/data/v1/stage2_llm.jsonl`
- `artifacts/data/v1/stage3_vision.jsonl`
- `artifacts/data/v1/stage4_joint.jsonl`
- `artifacts/data/v1/normalize_summary.json`

### 3.2 Build stage blends

```bash
python dimoe.py data build-stage-blends --config configs/data/stage_blends.yaml
```

Output:
- `artifacts/data/v1/blends/stage0_naive.jsonl`
- `artifacts/data/v1/blends/stage1_align.jsonl`
- `artifacts/data/v1/blends/stage2_full.jsonl`
- `artifacts/data/v1/blends/stage3_lownfe.jsonl`
- `artifacts/data/v1/blends/stage_blend_report.json`

### 3.3 Pack WebDataset shards (optional but recommended for multi-node)

```bash
for s in stage0_naive stage1_align stage2_full stage3_lownfe; do
  cat > /tmp/dimoe_wds_${s}.yaml <<EOF
source_jsonl: artifacts/data/v1/blends/${s}.jsonl
output_dir: artifacts/data/v1/wds/${s}
shard_size: 2000
include_image: true
EOF
  python dimoe.py data build-wds --config /tmp/dimoe_wds_${s}.yaml
done
```

## 4. Stage Training (DMAR + EBCMC curriculum)

### 4.1 Local single-node (8 GPU)

```bash
bash scripts/local/10_train_stage.sh stage0_naive configs/train/stage0_naive.yaml
bash scripts/local/10_train_stage.sh stage1_align configs/train/stage1_align.yaml
bash scripts/local/10_train_stage.sh stage2_full configs/train/stage2_full.yaml
bash scripts/local/10_train_stage.sh stage3_lownfe configs/train/stage3_lownfe.yaml
```

### 4.2 Local chained run (auto carry ckpt stage-by-stage)

```bash
bash scripts/local/11_train_all_stages.sh
```

### 4.3 Slurm 2x8 H800

```bash
sbatch scripts/slurm/submit_stage0_2x8_h800.sh
sbatch scripts/slurm/submit_stage1_2x8_h800.sh
sbatch scripts/slurm/submit_stage2_2x8_h800.sh
sbatch scripts/slurm/submit_stage3_2x8_h800.sh
```

Checkpoint layout:
- `artifacts/train/<stage>/checkpoints/global_step_XXXXXXXXX.pt`
- `artifacts/train/<stage>/logs/{train_metrics,routing_stats,gradient_stats}.jsonl`

## 5. Inference and Speed Profiles

### 5.1 Generation presets

- `fast`: 8 steps
- `mid`: 16 steps
- `best`: 32 steps

```bash
python dimoe.py infer generate \
  --config configs/infer/default.yaml \
  --input /path/to/prompts.json \
  --out artifacts/infer/preds.mid.jsonl \
  --preset mid \
  --checkpoint /path/to/stage3_ckpt.pt
```

### 5.2 Throughput/latency benchmark

```bash
python dimoe.py infer benchmark \
  --config configs/infer/default.yaml \
  --dataset-jsonl artifacts/data/v1/blends/stage3_lownfe.jsonl \
  --out artifacts/infer/benchmark.mid.json \
  --preset mid \
  --checkpoint /path/to/stage3_ckpt.pt
```

## 6. Evaluation

Run all tasks in `configs/eval/default.yaml`:

```bash
python dimoe.py eval run --config configs/eval/default.yaml --suite all --preset fast --checkpoint /path/to/stage3_ckpt.pt --out-dir artifacts/eval/fast
python dimoe.py eval run --config configs/eval/default.yaml --suite all --preset mid  --checkpoint /path/to/stage3_ckpt.pt --out-dir artifacts/eval/mid
python dimoe.py eval run --config configs/eval/default.yaml --suite all --preset best --checkpoint /path/to/stage3_ckpt.pt --out-dir artifacts/eval/best
```

Summarize tables:

```bash
python dimoe.py eval summarize --eval-dir artifacts/eval/best --out-dir artifacts/results/best
python dimoe.py tools export-paper-tables --eval-dir artifacts/eval/best --out-dir artifacts/results/paper
```

## 7. Problem-Existence Diagnostics (A-F)

### A/B/C: DMAR routing non-stationarity / consistency / specialization

```bash
python dimoe.py diag routing \
  --log artifacts/train/stage0_naive/logs/routing_stats.jsonl \
  --out-dir artifacts/diag/routing_stage0

python dimoe.py diag routing \
  --log artifacts/train/stage2_full/logs/routing_stats.jsonl \
  --out-dir artifacts/diag/routing_stage2
```

### D: No-answer-supervision ratio from answer length distribution

```bash
python dimoe.py diag supervision \
  --manifest artifacts/data/v1/blends/stage0_naive.jsonl \
  --out-dir artifacts/diag/supervision_stage0
```

### E/F: Vision and expert gradient imbalance

```bash
python dimoe.py diag gradient \
  --log artifacts/train/stage0_naive/logs/gradient_stats.jsonl \
  --out-dir artifacts/diag/gradient_stage0

python dimoe.py diag gradient \
  --log artifacts/train/stage2_full/logs/gradient_stats.jsonl \
  --out-dir artifacts/diag/gradient_stage2
```

## 8. Full One-Click Local Flow

```bash
bash scripts/local/00_data_pipeline.sh
bash scripts/local/11_train_all_stages.sh
bash scripts/local/20_eval_and_diag.sh
bash scripts/local/30_benchmark_grid.sh
```

## 9. Ablation Matrix to Run

Core ablations (each run `stage2_full` + `stage3_lownfe` + `eval best`):

1. Naive attach: disable DMAR and EBCMC losses (set stage0-like lambdas)
2. +DMAR only: enable prior+consistency, disable complementary curriculum behavior
3. +EBCMC only: complementary masking + stratified balance, disable consistency/prior
4. Full DiMoE-VL (DMAR + EBCMC)
5. Full + low-NFE tuning (stage3)

Extra ablations:
- Remove `lambda_cons`
- Remove `lambda_prior`
- Remove `lambda_bal`
- `include_prompt_ratio`: 0, 0.1, 0.2
- `num_t_buckets`: 4 vs 8 vs 16
- Vision resolution: 384 vs 768 variants via vision processor config

You can launch the predefined ablation variants with:

```bash
bash scripts/local/40_run_ablations.sh
```

## 10. Fairness and Reporting

Report both:
- Same wall-clock latency (`infer benchmark` outputs)
- Compute proxy `NFE * active_params_billion`

Use presets:
- Fast: 8 steps
- Mid: 16 steps
- Best: 32 steps

## 11. Files to Cite in Repro Appendix

- Configs: `configs/train/*.yaml`, `configs/eval/default.yaml`, `configs/infer/default.yaml`
- Scripts: `scripts/local/*.sh`, `scripts/slurm/*.sh`
- Logs: `artifacts/train/*/logs/*.jsonl`
- Tables: `artifacts/results/*`
