# bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/train_scripts/train_pifast_vlabench_delta_action/train_pifast_w_action.sh

cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA
bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/train_scripts/train_pifast_vlabench_delta_action/train_pifast_w_cotrain_mm.sh

cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA
bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/train_scripts/train_pifast_vlabench_delta_action/train_pifast_w_cotrain_mm_and_eb.sh

conda activate simpler
python /inspire/hdd/global_user/gongjingjing-25039/lqyin/gpu_occupy.py