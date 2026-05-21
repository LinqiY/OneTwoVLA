#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate arvla

export HF_HOME=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/
export MUJOCO_GL=egl

NUM_TRIALS=50
POLICY_PORT=8000
BATCH_TASKS=""
RUN_SUMMARIZE=1

usage() {
    echo "Usage: $0 <save_dir> [--track <track_name>] [--task <task_name>] [--episodes <n>]"
    echo "       [--policy-port <port>] [--batch-tasks \"t1 t2 ...\"] [--no-summarize]"
    echo "  --policy-port      连接 serve_policy（默认 8000）"
    echo "  --batch-tasks      与单个 --track 联用：一次 eval 跑多个 task（空格分隔），用于多卡分片"
    echo "  --no-summarize     不写结尾 summarize.py（并行分片 worker 用）"
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
        --policy-port)
            POLICY_PORT="$2"
            shift 2
            ;;
        --batch-tasks)
            BATCH_TASKS="$2"
            shift 2
            ;;
        --no-summarize)
            RUN_SUMMARIZE=0
            shift 1
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
    local cmd=(python examples/vlabench/eval.py --args.port "${POLICY_PORT}" --args.n-episode "$NUM_TRIALS" --args.save_dir "$SAVE_DIR")
    if [[ -n "$track" ]]; then
        cmd+=(--args.eval_track "$track")
    fi
    if [[ -n "$task" ]]; then
        cmd+=(--args.tasks "$task")
    fi

    echo "[INFO] Running: ${cmd[*]}"
    "${cmd[@]}"
}

if [[ -n "$TRACK_OPT" && -n "$BATCH_TASKS" ]]; then
    if [[ "$TRACK_OPT" == *","* ]]; then
        echo "错误: --batch-tasks 模式下 --track 只能写一个 track（不要逗号）" >&2
        exit 1
    fi
    run_eval "$TRACK_OPT" "$BATCH_TASKS"
elif [[ -n "$TRACK_OPT" && -z "$TASK_OPT" ]]; then
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

if [[ "$RUN_SUMMARIZE" -eq 1 ]]; then
    python examples/vlabench/summarize.py
fi
