#!/usr/bin/env bash
# 串行调用 run_eval_instance.sh：各 config + checkpoints 下 29999，--track track_1_in_distribution。

set -euo pipefail
cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA || exit 1

export VLABENCH_ROOT=/inspire/hdd/global_user/gongjingjing-25039/lqyin/VLABench/VLABench

# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_vlabench_cotrain_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_eb/2026.04.18/02.31.16/pifast-vlabench-cotrain/29999 --track track_1_in_distribution

# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_vlabench_cotrain_mm_and_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_mm_and_eb/2026.04.18/02.18.28/pifast_vlabench_cotrain_mm_and_eb/29999 --track track_1_in_distribution

# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_vlabench_pretrain_primitive /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_pretrain_primitive/2026.04.17/19.40.02/pifast-vlabench/29999 --track track_1_in_distribution --parallel-workers 8


# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_vlabench_cotrain_mm_data /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_cotrain_mm_data/2026.04.19/07.33.37/pifast-vlabench-cotrain-mm-data/29999 --track track_1_in_distribution --parallel-workers 8


# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_w_vlabench_cotrain_mm_and_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_mm_and_eb/2026.04.19/13.27.08/pifast-w-vlabench-cotrain-mm-and-eb/29999 --track track_1_in_distribution --parallel-workers 8


# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_w_vlabench_cotrain_mm_data /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_mm_data/2026.04.23/17.22.49/pifast-w-vlabench-cotrain-mm-data/29999 --track track_1_in_distribution --parallel-workers 8


bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_w_vlabench_pretrain_primitive_test /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_pretrain_primitive_test/2026.05.13/10.24.53/pifast-w-vlabench-test/29999 --track track_1_in_distribution --parallel-workers 8
bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_w_vlabench_pretrain_primitive_test /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_pretrain_primitive_test/2026.05.13/10.24.53/pifast-w-vlabench-test/29999 --track track_2_cross_category --parallel-workers 8
python /inspire/hdd/global_user/gongjingjing-25039/lqyin/gpu_occupy.py

# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance.sh pifast_w_vlabench_cotrain_eb /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_eb/2026.04.19/13.27.31/pifast-w-vlabench-cotrain-eb/29999 --track track_1_in_distribution --parallel-workers 8
