#!/usr/bin/env bash
# 串行调用 run_eval_instance.sh：各 config + checkpoints 29999；track 2；（去掉 insert_flower，episode 配置不足）。
# --task 与 --parallel-workers 8 同时使用：依赖 VLABENCH_TRACK_TASK_WHITELIST（由 run_eval_instance 从 --task 导出）在分片中过滤任务。

set -euo pipefail
cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA || exit 1

export VLABENCH_ROOT=/inspire/hdd/global_user/gongjingjing-25039/lqyin/VLABench/VLABench


TRACK2_TASKS="select_painting,select_fruit,select_toy,select_poker,select_mahjong,select_chemistry_tube,add_condiment,select_drink,select_book"
RUN="bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance_only_checkpoint.sh"
OPTS_1="--track track_1_in_distribution --parallel-workers 8"
# 与 track_2_cross_category.json 一致，排除 insert_flower
OPTS_2="--track track_2_cross_category --task ${TRACK2_TASKS} --parallel-workers 8"

${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_mm/2026.05.17/03.43.36/pifast-vlabench/99999 ${OPTS_1}
${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_mm/2026.05.17/03.43.36/pifast-vlabench/99999 ${OPTS_2}


# ${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_action_f/2026.05.25/15.34.25/pifast-vlabench/99999 ${OPTS_1}
# ${RUN} /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_action_f/2026.05.25/15.34.25/pifast-vlabench/99999 ${OPTS_2}