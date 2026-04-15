# source .venv/bin/activate
# VLA + VL cotrain: scripts/train_cotrain.py, config pifast_vlabench_cotrain
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
echo batch_size "$batch_size"

export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
# LeRobot: $LEROBOT_HOME/vlabench/vlabench_pretrain_primitive (see TrainConfig repo_id)
# VL JSON/images paths are set in config pifast_vlabench_cotrain (vl_data)

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train_cotrain.py pifast_vlabench_cotrain --exp-name="${now_date}/${now_seconds}/pifast-vlabench-cotrain" --batch-size="$batch_size"
