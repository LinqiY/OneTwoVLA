#!/usr/bin/env bash
# VLABench-tuned PaliGemma init (pifast_w_*): same entry as train_pifast_vlabench_test.sh
# Config: pifast_w_vlabench_pretrain_primitive_test
set -euo pipefail

export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/

cd "$(dirname "$0")/.."

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=4
batch_size=$((num_devices * single_batch_size))
echo "batch_size ${batch_size}"

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pifast_w_vlabench_pretrain_primitive_test \
  --exp-name="${now_date}/${now_seconds}/pifast-w-vlabench-test" \
  --batch-size="${batch_size}"
