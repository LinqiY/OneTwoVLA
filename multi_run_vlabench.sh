#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate arvla

export HF_HOME=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/
export MUJOCO_GL=egl

NUM_TRIALS=50

usage() {
    echo "Usage: $0 <save_dir> [--track <track_name>] [--task <task_name>] [--episodes <n>]"
    exit 1
}

SAVE_DIR=""
TRACK_OPT=""
TASK_OPT=""

if [ "$#" -lt 1 ]; then
    usage
fi
SAVE_DIR=$1
shift 1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            TRACK_OPT="$2"
            shift 2
            ;;
        --task)
            TASK_OPT="$2"
            shift 2
            ;;
        --episodes)
            NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

run_eval() {
    local track="$1"
    local task="$2"
    local cmd=(python examples/vlabench/eval.py --args.port 8000 --args.n-episode "$NUM_TRIALS" --args.save_dir "$SAVE_DIR")
    if [[ -n "$track" ]]; then
        cmd+=(--args.eval_track "$track")
    fi
    if [[ -n "$task" ]]; then
        cmd+=(--args.tasks "$task")
    fi

    echo "[INFO] Running: ${cmd[*]}"
    "${cmd[@]}"
}

if [[ -n "$TRACK_OPT" && -z "$TASK_OPT" ]]; then
    IFS=',' read -ra TRACKS <<< "$TRACK_OPT"
    for TRACK in "${TRACKS[@]}"; do
        run_eval "$TRACK" ""
    done
elif [[ -n "$TRACK_OPT" && -n "$TASK_OPT" ]]; then
    IFS=',' read -ra TRACKS <<< "$TRACK_OPT"
    IFS=',' read -ra TASKS <<< "$TASK_OPT"
    for TRACK in "${TRACKS[@]}"; do
        for TASK in "${TASKS[@]}"; do
            run_eval "$TRACK" "$TASK"
        done
    done
else
    ALL_TASKS=(add_condiment insert_flower select_book select_drink select_chemistry_tube select_toy select_fruit select_painting select_nth_largest_poker select_unique_type_mahjong)
    if [[ -n "$TASK_OPT" ]]; then
        IFS=',' read -ra TASKS <<< "$TASK_OPT"
    else
        TASKS=("${ALL_TASKS[@]}")
    fi
    for TASK in "${TASKS[@]}"; do
        run_eval "" "$TASK"
    done
fi

python examples/vlabench/summarize.py
