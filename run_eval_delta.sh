#!/usr/bin/env bash
# 补跑 pifast_w_vlabench_delta_cotrain_mm_and_eb/model_99999 里失败或缺失的 VLABench tasks。
# --task 与 --parallel-workers 同时使用：依赖 VLABENCH_TRACK_TASK_WHITELIST（由 run_eval_instance 从 --task 导出）在分片中过滤任务。

set -euo pipefail
cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA || exit 1

export VLABENCH_ROOT=/inspire/hdd/global_user/gongjingjing-25039/lqyin/VLABench/VLABench

RUN="bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance_only_checkpoint.sh"
CKPT="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_mm_and_eb/2026.05.16/15.41.36/pifast-vlabench/99999"

TRACK1_FAILED_TASKS="add_condiment,select_painting"
TRACK2_FAILED_TASKS="add_condiment,select_drink,select_fruit,select_painting"

OPTS_1="--track track_1_in_distribution --task ${TRACK1_FAILED_TASKS} --parallel-workers 2"
OPTS_2="--track track_2_cross_category --task ${TRACK2_FAILED_TASKS} --parallel-workers 4"

${RUN} "${CKPT}" ${OPTS_1}
${RUN} "${CKPT}" ${OPTS_2}

# # 30k
# ${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_delta_action/2026.05.15/08.49.27/pifast-vlabench/99999 ${OPTS_1}
# ${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_delta_action/2026.05.15/08.49.27/pifast-vlabench/99999 ${OPTS_2}
