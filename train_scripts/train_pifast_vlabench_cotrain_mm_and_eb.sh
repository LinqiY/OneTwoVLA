#!/usr/bin/env bash
# VLA + VL: LaTeX_OCR / COCO / A-OKVQA (parquet) + VLABench primitive jsons_train
# (affordance, goal_description, spatial_understanding, task_planning, trajectory).
# interleave_prob: 0.02, 0.05, 0.22, 0.16, 0.10, 0.17, 0.10, 0.18
# Config: pifast_vlabench_cotrain_latex_coco_okvqa_primitive_json
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train_cotrain.py pifast_vlabench_cotrain_mm_and_eb \
  --exp-name="${now_date}/${now_seconds}/pifast_vlabench_cotrain_mm_and_eb" \
  --batch-size="${batch_size}"
