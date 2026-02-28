#!/usr/bin/env bash
#SBATCH -J dimoe_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/qianlong/DiMoE-VL}
cd "$ROOT_DIR"
mkdir -p logs

source ~/.bashrc
# conda activate <your-env>

CKPT=${CKPT:-$(ls -1 artifacts/train/stage3_lownfe/checkpoints/global_step_*.pt 2>/dev/null | tail -n 1 || true)}
if [[ -z "$CKPT" ]]; then
  echo "missing stage3 checkpoint; set CKPT=/path/to/ckpt.pt"
  exit 2
fi

for PRESET in fast mid best; do
  python dimoe.py eval run \
    --config configs/eval/default.yaml \
    --suite all \
    --preset "$PRESET" \
    --checkpoint "$CKPT" \
    --out-dir "artifacts/eval/${PRESET}"
done

python dimoe.py eval summarize --eval-dir artifacts/eval/best --out-dir artifacts/results/best
python dimoe.py tools export-paper-tables --eval-dir artifacts/eval/best --out-dir artifacts/results/paper
