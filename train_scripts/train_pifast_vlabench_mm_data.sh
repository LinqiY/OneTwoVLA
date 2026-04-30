#!/usr/bin/env bash
# VLA + VL cotrain: LaTeX_OCR / COCO / A-OKVQA (interleave_prob 0.10 / 0.15 / 0.75).
# Config: pifast_vlabench_cotrain_mm_data (parquet paths in config.py).
set -euo pipefail

export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/
export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot

cd "$(dirname "$0")/.."

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=4
batch_size=$((num_devices * single_batch_size))
echo "batch_size ${batch_size}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train_cotrain.py pifast_vlabench_cotrain_mm_data \
  --exp-name="${now_date}/${now_seconds}/pifast-vlabench-cotrain-mm-data" \
  --batch-size="${batch_size}"
