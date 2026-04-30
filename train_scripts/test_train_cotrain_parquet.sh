# source .venv/bin/activate
export PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/.venv/bin:$PATH"
export WANDB_MODE=offline
export HF_DATASETS_CACHE=/inspire/hdd/global_user/gongjingjing-25039/lqyin/hf_cache/
export LEROBOT_HOME=/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/lerobot
# VL branch uses parquet (LaTeX OCR); path is set in config: pifast_vlabench_cotrain_latex_parquet
# python scripts/smoke_test_cotrain.py --config pifast_vlabench_cotrain_latex_parquet --only-data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/smoke_test_cotrain.py --config pifast_vlabench_cotrain_latex_parquet --only-data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/smoke_test_cotrain.py --config pifast_vlabench_cotrain_latex_parquet --only-data
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/smoke_test_cotrain.py --config pifast_vlabench_cotrain_coco_parquet --only-data
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/smoke_test_cotrain.py --config pifast_vlabench_cotrain_a_okvqa_parquet --only-data
