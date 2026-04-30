# source .venv/bin/activate
export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/

# PYTHON="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin/python"

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=4
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

# single_val_batch_size=12
# val_batch_size=$((num_devices * single_val_batch_size))
# echo val_batch_size $val_batch_size

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1
DATASET_PATH=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1/bag
NORM_STATS_DIR=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1/verified_assets
CONFIG_NAME=pi0_fast_widowx
EXP_NAME=${now_date}/${now_seconds}/pifast-widowx-coffee-bean
# ensure the dataset's path is $DATASET_PATH

# normalization stats
# this can only run on a single GPU.
# this code only needs to run once.
# CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py $CONFIG_NAME \
#   --data.repo-id=$DATASET_PATH \
#   --data.assets.assets-dir=$NORM_STATS_DIR \
#   --data.assets.asset-id=widowx_bag

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py $CONFIG_NAME \
  --exp-name=$EXP_NAME \
  --batch-size=$batch_size \
  --data.repo-id=$DATASET_PATH \
  --data.assets.assets-dir=$NORM_STATS_DIR \
  --data.assets.asset-id=widowx_bag
