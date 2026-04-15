# source .venv/bin/activate
export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/

set -euo pipefail

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=20
batch_size=$((num_devices * single_batch_size))
echo batch_size "$batch_size"

single_val_batch_size=12
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size "$val_batch_size"

# Normalization stats (single GPU; run once per machine / dataset copy).
# Writes to assets/onetwovla_vlabench_direct/vlabench/
CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py onetwovla_vlabench_direct --exp-name=computing-norm

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py onetwovla_vlabench_direct \
  --exp-name="${now_date}/${now_seconds}/onetwovla-vlabench-direct" \
  --batch-size="$batch_size" \
  --val-batch-size="$val_batch_size"
