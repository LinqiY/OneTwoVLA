#!/usr/bin/env bash
# 串行调用 run_eval_instance.sh：各 config + checkpoints 29999；track 2；（去掉 insert_flower，episode 配置不足）。
# --task 与 --parallel-workers 8 同时使用：依赖 VLABENCH_TRACK_TASK_WHITELIST（由 run_eval_instance 从 --task 导出）在分片中过滤任务。

set -euo pipefail
cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA || exit 1

export VLABENCH_ROOT=/inspire/hdd/global_user/gongjingjing-25039/lqyin/VLABench/VLABench

# 与 track_2_cross_category.json 一致，排除 insert_flower
TRACK2_TASKS="select_painting,select_fruit,select_toy,select_poker,select_mahjong,select_chemistry_tube,add_condiment,select_drink,select_book"
RUN="bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh"
OPTS="--track track_2_cross_category --task ${TRACK2_TASKS} --parallel-workers 8"

${RUN} pifast_vlabench_cotrain_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_eb/2026.04.18/02.31.16/pifast-vlabench-cotrain/29999 ${OPTS}

${RUN} pifast_vlabench_cotrain_mm_and_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_mm_and_eb/2026.04.18/02.18.28/pifast_vlabench_cotrain_mm_and_eb/29999 ${OPTS}

${RUN} pifast_vlabench_pretrain_primitive /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_pretrain_primitive/2026.04.17/19.40.02/pifast-vlabench/29999 ${OPTS}


${RUN} pifast_vlabench_cotrain_mm_data /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_mm_data/2026.04.19/07.33.37/pifast-vlabench-cotrain-mm-data/29999 ${OPTS}


# ${RUN} pifast_w_vlabench_cotrain_mm_and_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_mm_and_eb/2026.04.19/13.27.08/pifast-w-vlabench-cotrain-mm-and-eb/29999 ${OPTS}


# ${RUN} pifast_w_vlabench_cotrain_mm_data /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_mm_data/2026.04.23/17.22.49/pifast-w-vlabench-cotrain-mm-data/29999 ${OPTS}


# ${RUN} pifast_w_vlabench_pretrain_primitive_test /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_pretrain_primitive_test/2026.04.19/13.27.03/pifast-w-vlabench-test/29999 ${OPTS}


# ${RUN} pifast_w_vlabench_cotrain_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_eb/2026.04.19/13.27.31/pifast-w-vlabench-cotrain-eb/29999 ${OPTS}

