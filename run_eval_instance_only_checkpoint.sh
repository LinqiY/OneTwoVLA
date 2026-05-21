#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate base
# source setup_env.sh

set -euo pipefail

export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"


usage() {
    echo "Usage: $0 <checkpoint_path> [--track <track_name>] [--task <task_name>]"
    echo "       [--parallel-workers N]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path      模型 checkpoint 路径"
    echo "                       例如:"
    echo "                       /.../checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000"
    echo ""
    echo "Options:"
    echo "  --track <track_name>     指定 track，可选"
    echo "  --task <task_name>       指定 task，可选；逗号分隔多个。与 --parallel-workers>1 联用时仅跑列表内 task"
    echo "  --parallel-workers N     并行 eval 进程数，默认 1；>1 时需指定单个 --track"
    echo "                           需要 VLABENCH_ROOT；合并后仅在全部 shard 成功时跑 summarize.py"
    echo ""
    echo "Example:"
    echo "  $0 /.../checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000"
    echo "  $0 /.../checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000 --track track_1_in_distribution"
    echo "  $0 /.../checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000 --track track_1_in_distribution --parallel-workers 8"
    exit 1
}


infer_config_name_from_ckpt() {
    local ckpt_path="$1"
    local after_checkpoints=""

    # 去掉末尾 /
    ckpt_path="${ckpt_path%/}"

    # 兼容绝对路径:
    # /.../checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000
    if [[ "$ckpt_path" == *"/checkpoints/"* ]]; then
        after_checkpoints="${ckpt_path#*/checkpoints/}"

    # 兼容相对路径:
    # checkpoints/pifast_vlabench_delta_action/2026.05.14/03.30.56/pifast-vlabench/80000
    elif [[ "$ckpt_path" == checkpoints/* ]]; then
        after_checkpoints="${ckpt_path#checkpoints/}"

    else
        echo "错误: checkpoint_path 中没有找到 checkpoints/，无法自动解析 config_name: $ckpt_path" >&2
        exit 1
    fi

    # 取 checkpoints/ 后面的第一段作为 config_name
    # pifast_vlabench_delta_action/2026.05.14/... -> pifast_vlabench_delta_action
    echo "${after_checkpoints%%/*}"
}


# ==================== 参数解析 ====================

if [[ "$#" -lt 1 ]]; then
    usage
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
fi

CKPT="${1%/}"
shift 1

if [[ ! -e "$CKPT" ]]; then
    echo "错误: checkpoint_path 不存在: $CKPT" >&2
    exit 1
fi

CONFIG_NAME="$(infer_config_name_from_ckpt "$CKPT")"

TRACK_OPT=""
TASK_OPT=""
PARALLEL_WORKERS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            if [[ $# -lt 2 ]]; then
                echo "错误: --track 需要一个参数" >&2
                usage
            fi
            TRACK_OPT="$2"
            shift 2
            ;;

        --task)
            if [[ $# -lt 2 ]]; then
                echo "错误: --task 需要一个参数" >&2
                usage
            fi
            TASK_OPT="$2"
            shift 2
            ;;

        --parallel-workers)
            if [[ $# -lt 2 ]]; then
                echo "错误: --parallel-workers 需要一个参数" >&2
                usage
            fi
            PARALLEL_WORKERS="$2"
            shift 2
            ;;

        --help|-h)
            usage
            ;;

        *)
            echo "未知参数: $1" >&2
            usage
            ;;
    esac
done

if ! [[ "$PARALLEL_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "错误: --parallel-workers 必须是正整数，当前值: $PARALLEL_WORKERS" >&2
    exit 1
fi


# ==================== 路径配置 ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_SCRIPT="${SCRIPT_DIR}/serve_policy.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/multi_run_vlabench.sh"

if [[ ! -f "$POLICY_SCRIPT" ]]; then
    echo "错误: Policy server 脚本不存在: $POLICY_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "错误: Evaluation 脚本不存在: $EVAL_SCRIPT" >&2
    exit 1
fi

model_step="${CKPT##*/}"
SAVE_DIR="evaluate_results/${CONFIG_NAME}/model_${model_step}"

LOG_DIR="${SCRIPT_DIR}/logs/unified_eval_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"


echo "=========================================="
echo "统一评估脚本启动"
echo "=========================================="
echo "Checkpoint: $CKPT"
echo "配置名称: $CONFIG_NAME"
echo "模型 step: $model_step"
echo "保存目录: $SAVE_DIR"
echo "日志目录: $LOG_DIR"

if [[ -n "$TRACK_OPT" ]]; then
    echo "指定 Track: $TRACK_OPT"
fi

if [[ -n "$TASK_OPT" ]]; then
    echo "指定 Task: $TASK_OPT"
fi

if [[ "$PARALLEL_WORKERS" -gt 1 ]]; then
    echo "并行 Workers: $PARALLEL_WORKERS"
fi

echo "=========================================="


# ==================== 清理函数 ====================

cleanup() {
    echo ""
    echo "=========================================="
    echo "正在清理进程..."
    echo "=========================================="

    echo "终止 Policy Server 进程..."
    if [[ -n "${POLICY_PID:-}" ]]; then
        kill "$POLICY_PID" 2>/dev/null || true
    fi
    pkill -f "serve_policy.py" || true

    echo "终止 Evaluation 进程..."
    pkill -f "vlabench/eval.py" || true

    echo "清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM


# ==================== 第一步：启动 Policy Servers ====================

echo ""
echo "=========================================="
echo "第一步: 启动 Policy Servers"
echo "=========================================="

echo "执行命令: bash $POLICY_SCRIPT $CONFIG_NAME $CKPT"

bash "$POLICY_SCRIPT" "$CONFIG_NAME" "$CKPT" > "${LOG_DIR}/policy_servers.log" 2>&1 &
POLICY_PID=$!

echo "Policy servers 正在启动... PID: $POLICY_PID"
echo "日志文件: ${LOG_DIR}/policy_servers.log"


# ==================== 等待 Policy Servers 启动 ====================

echo ""
echo "=========================================="
echo "等待 Policy Servers 完全启动..."
echo "=========================================="

WAIT_TIME=120
echo "等待 ${WAIT_TIME} 秒..."

for i in $(seq 1 "$WAIT_TIME"); do
    printf "\r等待中: %2d/%d 秒" "$i" "$WAIT_TIME"
    sleep 1
done
echo ""


# ==================== 检查 Policy Servers 状态 ====================

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
BASE_PORT=8000

echo "验证 Policy Servers 状态..."

all_ready=true

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + gpu_id))
    if curl -s --connect-timeout 2 --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "  ✓ GPU ${gpu_id}，端口 ${port}: 运行中"
    else
        echo "  ✗ GPU ${gpu_id}，端口 ${port}: 未响应"
        all_ready=false
    fi
done

if [[ "$all_ready" = false ]]; then
    echo "警告: 部分 Policy Servers 可能未正确启动，但继续执行 evaluation..."
fi


EPAR="$PARALLEL_WORKERS"

if [[ "$EPAR" -gt "$NUM_GPUS" ]]; then
    echo "警告: parallel-workers ($EPAR) > GPU 数 ($NUM_GPUS)，使用 $NUM_GPUS 路并行"
    EPAR="$NUM_GPUS"
fi

if [[ "$EPAR" -lt 1 ]]; then
    EPAR=1
fi

cd "$SCRIPT_DIR" || exit 1


# ==================== 第二步：启动 Evaluation ====================

echo ""
echo "=========================================="
echo "第二步: 启动 Evaluation"
echo "=========================================="

EVAL_EXIT_CODE=0
SPLIT_PY="${SCRIPT_DIR}/scripts/split_track_tasks_for_shard.py"
MERGE_PY="${SCRIPT_DIR}/scripts/merge_vlabench_shard_results.py"

if [[ "$EPAR" -le 1 ]]; then
    EVAL_CMD=(bash "$EVAL_SCRIPT" "$SAVE_DIR")

    if [[ -n "$TRACK_OPT" ]]; then
        EVAL_CMD+=(--track "$TRACK_OPT")
    fi

    if [[ -n "$TASK_OPT" ]]; then
        EVAL_CMD+=(--task "$TASK_OPT")
    fi

    echo "执行命令: ${EVAL_CMD[*]}"

    if "${EVAL_CMD[@]}" > "${LOG_DIR}/evaluation.log" 2>&1; then
        EVAL_EXIT_CODE=0
    else
        EVAL_EXIT_CODE=$?
    fi

else
    if [[ -z "$TRACK_OPT" || "$TRACK_OPT" == *","* ]]; then
        echo "错误: --parallel-workers > 1 时必须指定单个 --track，不要用逗号分隔多个" >&2
        exit 1
    fi

    if [[ -n "$TASK_OPT" ]]; then
        export VLABENCH_TRACK_TASK_WHITELIST="${TASK_OPT//,/ }"
    else
        unset VLABENCH_TRACK_TASK_WHITELIST 2>/dev/null || true
    fi

    if [[ -z "${VLABENCH_ROOT:-}" ]]; then
        echo "错误: 并行模式需要环境变量 VLABENCH_ROOT" >&2
        exit 1
    fi

    if [[ ! -f "$SPLIT_PY" || ! -f "$MERGE_PY" ]]; then
        echo "错误: 缺少分片/合并脚本: $SPLIT_PY / $MERGE_PY" >&2
        exit 1
    fi

    SHARD_PARENT="${SAVE_DIR}/shards_${TRACK_OPT}"

    echo "并行模式: ${EPAR} 路 eval → 端口 8000..$((8000 + EPAR - 1))"
    echo "分片暂存目录: ${SHARD_PARENT}/shard_*"
    echo "merge 后目录: ${SAVE_DIR}/${TRACK_OPT}/"

    pids=()

    for i in $(seq 0 $((EPAR - 1))); do
        TASK_BATCH=$(python3 "$SPLIT_PY" "$TRACK_OPT" "$EPAR" "$i")

        if [[ -z "$TASK_BATCH" ]]; then
            echo "  Shard $i: 无 task，跳过"
            continue
        fi

        echo "  Shard $i: 端口 $((8000 + i))，tasks: ${TASK_BATCH:0:120}..."

        (
            bash "$EVAL_SCRIPT" "${SHARD_PARENT}/shard_${i}" \
                --track "$TRACK_OPT" \
                --batch-tasks "$TASK_BATCH" \
                --policy-port $((8000 + i)) \
                --no-summarize
        ) > "${LOG_DIR}/eval_shard_${i}.log" 2>&1 &

        pids+=($!)
    done

    if [[ ${#pids[@]} -eq 0 ]]; then
        echo "错误: 没有启动任何 eval 分片" >&2
        exit 1
    fi

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            EVAL_EXIT_CODE=1
        fi
    done

    if [[ "$EVAL_EXIT_CODE" -eq 0 ]]; then
        echo ""
        echo "所有分片成功，合并结果到 ${SAVE_DIR}/${TRACK_OPT}/ ..."

        python3 "$MERGE_PY" \
            --base-dir "${SCRIPT_DIR}/${SAVE_DIR}" \
            --track "$TRACK_OPT" \
            --num-shards "$EPAR" \
            --shard-parent "${SCRIPT_DIR}/${SHARD_PARENT}" \
            --cleanup

        echo "运行 summarize.py ..."

        if ! python examples/vlabench/summarize.py; then
            EVAL_EXIT_CODE=$?
        fi
    else
        echo "并行 eval 存在失败，跳过 merge 与 summarize。见 ${LOG_DIR}/eval_shard_*.log" >&2
    fi
fi


# ==================== 结果处理 ====================

echo ""
echo "=========================================="
echo "评估完成"
echo "=========================================="

if [[ "$EVAL_EXIT_CODE" -eq 0 ]]; then
    echo "✓ Evaluation 执行成功"
else
    echo "✗ Evaluation 执行失败，退出码: $EVAL_EXIT_CODE"
fi

echo "日志目录: $LOG_DIR"
echo "  - Policy servers 日志: ${LOG_DIR}/policy_servers.log"

if [[ "$EPAR" -le 1 ]]; then
    echo "  - Evaluation 日志: ${LOG_DIR}/evaluation.log"
else
    echo "  - Evaluation shard 日志: ${LOG_DIR}/eval_shard_*.log"
fi


# ==================== 清理 ====================

echo ""
echo "=========================================="
echo "清理资源"
echo "=========================================="

echo "终止 Policy Server 进程..."
kill "$POLICY_PID" 2>/dev/null || true
pkill -f "serve_policy.py" || true

echo "等待进程完全退出..."
sleep 5

echo ""
echo "=========================================="
echo "评估流程完成!"
echo "=========================================="
echo "配置: $CONFIG_NAME"
echo "Checkpoint: $CKPT"
echo "结果保存在: $SAVE_DIR"
echo "日志保存在: $LOG_DIR"

exit "$EVAL_EXIT_CODE"

# .venv/bin/python /inspire/hdd/global_user/gongjingjing-25039/lqyin/gpu_occupy.py