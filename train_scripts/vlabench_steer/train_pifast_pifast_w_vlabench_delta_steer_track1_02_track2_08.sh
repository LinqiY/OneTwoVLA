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

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
# ensure the dataset's path is $LEROBOT_HOME/vlabench


XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train_cotrain.py pifast_w_vlabench_delta_steer_track1_02_track2_08 \
  --exp-name="${now_date}/${now_seconds}/pifast-w-vlabench-cotrain-mm-data" \
  --batch-size="${batch_size}" \
  --save_interval=10000

conda activate simpler
python /inspire/hdd/global_user/gongjingjing-25039/lqyin/gpu_occupy.py