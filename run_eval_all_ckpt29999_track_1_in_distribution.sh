#!/usr/bin/env bash
# 与 run_eval_instance.sh 同目录。串行运行：对每个 checkpoints 下 step=29999 的目录，
# 使用对应训练 config（checkpoints 下第一层目录名）与 --track track_1_in_distribution。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

RUNNER="${ROOT}/run_eval_instance.sh"
TRACK="track_1_in_distribution"

if [[ ! -x "$RUNNER" ]] && [[ ! -f "$RUNNER" ]]; then
    echo "找不到: $RUNNER" >&2
    exit 1
fi

mapfile -t CKPTS < <(find "${ROOT}/checkpoints" -type d -name '29999' 2>/dev/null | LC_ALL=C sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "在 ${ROOT}/checkpoints 下未发现名为 29999 的目录。" >&2
    exit 1
fi

failed=0
for ckpt in "${CKPTS[@]}"; do
    rel="${ckpt#"${ROOT}/checkpoints/"}"
    config="${rel%%/*}"
    echo ""
    echo "================================================================================"
    echo "config:  ${config}"
    echo "ckpt:    ${ckpt}"
    echo "track:   ${TRACK}"
    echo "command: bash ${RUNNER} ${config} ${ckpt} --track ${TRACK}"
    echo "================================================================================"
    if ! bash "${RUNNER}" "${config}" "${ckpt}" --track "${TRACK}"; then
        echo "[WARN] 上一条评估失败 (exit != 0)，继续下一个。" >&2
        failed=$((failed + 1))
    fi
done

echo ""
echo "全部串行任务结束。失败次数: ${failed} / ${#CKPTS[@]}"
exit $((failed > 0 ? 1 : 0))
