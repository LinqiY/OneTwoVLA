# source .venv/bin/activate
export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

num_devices=$(nvidia-smi --list-gpus | wc -l)
single_batch_size=4
batch_size=$((num_devices * single_batch_size))
echo batch_size "$batch_size"

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1
DATASET_PATH=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1/new_coffee_bag_bridge_delta
NORM_STATS_DIR=/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1/verified_assets
ASSET_ID=new_coffee_bag_bridge_delta
CONFIG_NAME=pi0_widowx_bridge_delta
EXP_NAME=${now_date}/${now_seconds}/pi0-widowx-new-coffee-bag-bridge-delta

# This dataset already stores bridge_orig-style delta EE actions:
# [dx, dy, dz, droll, dpitch, dyaw, gripper].
# The config therefore sets LeRobotWidowDataConfig(extra_delta_transform=False).

if [ "${COMPUTE_STATS:-0}" = "1" ]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py "$CONFIG_NAME" \
    --exp-name="computing-norm-$ASSET_ID" \
    --data.repo-id="$DATASET_PATH" \
    --data.assets.assets-dir="$NORM_STATS_DIR" \
    --data.assets.asset-id="$ASSET_ID"
  mkdir -p "$NORM_STATS_DIR/$ASSET_ID"
  cp "$DATASET_PATH/norm_stats.json" "$NORM_STATS_DIR/$ASSET_ID/norm_stats.json"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  --batch-size="$batch_size" \
  --data.repo-id="$DATASET_PATH" \
  --data.assets.assets-dir="$NORM_STATS_DIR" \
  --data.assets.asset-id="$ASSET_ID"
