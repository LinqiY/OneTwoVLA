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
echo batch_size $batch_size

# single_val_batch_size=12
# val_batch_size=$((num_devices * single_val_batch_size))
# echo val_batch_size $val_batch_size

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
# ensure the dataset's path is $LEROBOT_HOME/vlabench

# normalization stats
# this can only run on a single GPU.
# this code only needs to run once.
# CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py pi0_vlabench_pretrain_primitive --exp-name=computing-norm \
# --create_train_val_split --val_ratio=0.05 \
# --is_computing_norm_stats

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi0_vlabench_pretrain_primitive --exp-name=${now_date}/${now_seconds}/pi0-vlabench --batch-size=$batch_size --val-batch-size=$val_batch_size
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi0_vlabench_pretrain_primitive --exp-name=${now_date}/${now_seconds}/pi0-vlabench --batch-size=$batch_size
