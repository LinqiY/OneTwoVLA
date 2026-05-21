# Evaluation TODO

This file tracks checkpoint directories under `checkpoints/` that should be evaluated but do not yet have a matching result under `evaluate_results/`.

Last synced: `2026-05-19`

Rules:

- Only checkpoint configs whose name contains `vlabench` are tracked.
- If the config name contains `delta`, it is grouped under `Delta`.
- If the config name does not contain `delta`, it is grouped under `Relative`.
- Steps `29999` and `30000` are grouped together.
- Step `99999` is grouped separately.
- A checkpoint is considered evaluated only if `evaluate_results/<config>/model_<step>` exists.

## Summary

| Group | Step 29999/30000 | Step 99999 | Total |
|---|---:|---:|---:|
| Delta | 13 | 3 | 16 |
| Relative | 5 | 2 | 7 |
| Total | 18 | 5 | 23 |

## Delta

### Step 29999/30000

| Config | Step | Run Time | Exp Dir | Checkpoint Path |
|---|---:|---|---|---|
| `pifast_vlabench_delta_cotrain_eb` | 30000 | `2026.05.16/15.47.27` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_delta_cotrain_eb/2026.05.16/15.47.27/pifast-vlabench/30000` |
| `pifast_vlabench_delta_cotrain_mm_and_eb` | 30000 | `2026.05.17/02.20.54` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_delta_cotrain_mm_and_eb/2026.05.17/02.20.54/pifast-vlabench/30000` |
| `pifast_w_vlabench_delta_action` | 30000 | `2026.05.16/07.39.56` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_action/2026.05.16/07.39.56/pifast-vlabench/30000` |
| `pifast_w_vlabench_delta_cotrain_eb` | 30000 | `2026.05.18/06.10.04` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_eb/2026.05.18/06.10.04/pifast-vlabench/30000` |
| `pifast_w_vlabench_delta_cotrain_mm` | 30000 | `2026.05.17/03.43.36` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_mm/2026.05.17/03.43.36/pifast-vlabench/30000` |
| `pifast_w_vlabench_delta_cotrain_mm_and_eb` | 30000 | `2026.05.16/15.41.36` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_mm_and_eb/2026.05.16/15.41.36/pifast-vlabench/30000` |
| `pifast_w_vlabench_delta_cotrain_task-reasoning` | 30000 | `2026.05.16/17.05.21` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_task-reasoning/2026.05.16/17.05.21/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_cotrain_task-reasoning` | 30000 | `2026.05.16/17.52.02` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_task-reasoning/2026.05.16/17.52.02/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_cotrain_trajectory` | 30000 | `2026.05.17/08.59.23` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_trajectory/2026.05.17/08.59.23/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_cotrain_trajectory` | 30000 | `2026.05.17/10.00.45` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_trajectory/2026.05.17/10.00.45/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_cotrain_understanding` | 30000 | `2026.05.18/00.55.44` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_understanding/2026.05.18/00.55.44/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_cotrain_understanding` | 30000 | `2026.05.18/02.04.11` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_understanding/2026.05.18/02.04.11/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_delta_steer_track1_09_track2_01` | 30000 | `2026.05.18/05.40.38` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_steer_track1_09_track2_01/2026.05.18/05.40.38/pifast-w-vlabench-cotrain-mm-data/30000` |

### Step 99999

| Config | Step | Run Time | Exp Dir | Checkpoint Path |
|---|---:|---|---|---|
| `pifast_w_vlabench_delta_cotrain_eb` | 99999 | `2026.05.18/06.10.04` | `pifast-vlabench` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_eb/2026.05.18/06.10.04/pifast-vlabench/99999` |
| `pifast_w_vlabench_delta_cotrain_understanding` | 99999 | `2026.05.18/00.55.44` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_understanding/2026.05.18/00.55.44/pifast-w-vlabench-cotrain-mm-data/99999` |
| `pifast_w_vlabench_delta_cotrain_understanding` | 99999 | `2026.05.18/02.04.11` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_delta_cotrain_understanding/2026.05.18/02.04.11/pifast-w-vlabench-cotrain-mm-data/99999` |

## Relative

### Step 29999/30000

| Config | Step | Run Time | Exp Dir | Checkpoint Path |
|---|---:|---|---|---|
| `pifast_w_vlabench_cotrain_eb` | 30000 | `2026.05.14/22.33.36` | `pifast-w-vlabench-cotrain-eb` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_eb/2026.05.14/22.33.36/pifast-w-vlabench-cotrain-eb/30000` |
| `pifast_w_vlabench_cotrain_mm_and_eb` | 30000 | `2026.05.15/16.43.41` | `pifast-w-vlabench-cotrain-mm-and-eb` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_cotrain_mm_and_eb/2026.05.15/16.43.41/pifast-w-vlabench-cotrain-mm-and-eb/30000` |
| `pifast_w_vlabench_relative_cotrain_trajectory` | 29999 | `2026.05.15/08.45.02` | `pifast-w-vlabench-cotrain-mm-data-libero` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_relative_cotrain_trajectory/2026.05.15/08.45.02/pifast-w-vlabench-cotrain-mm-data-libero/29999` |
| `pifast_w_vlabench_relative_steer_track1_05_track2_05` | 30000 | `2026.05.18/02.29.32` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_relative_steer_track1_05_track2_05/2026.05.18/02.29.32/pifast-w-vlabench-cotrain-mm-data/30000` |
| `pifast_w_vlabench_relative_steer_track2_1` | 30000 | `2026.05.18/02.29.58` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_relative_steer_track2_1/2026.05.18/02.29.58/pifast-w-vlabench-cotrain-mm-data/30000` |

### Step 99999

| Config | Step | Run Time | Exp Dir | Checkpoint Path |
|---|---:|---|---|---|
| `pifast_w_vlabench_relative_steer_track1_05_track2_05` | 99999 | `2026.05.18/02.29.32` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_relative_steer_track1_05_track2_05/2026.05.18/02.29.32/pifast-w-vlabench-cotrain-mm-data/99999` |
| `pifast_w_vlabench_relative_steer_track2_1` | 99999 | `2026.05.18/02.29.58` | `pifast-w-vlabench-cotrain-mm-data` | `/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_w_vlabench_relative_steer_track2_1/2026.05.18/02.29.58/pifast-w-vlabench-cotrain-mm-data/99999` |

## VLABench checkpoints to resume

These configs have existing checkpoints under `OneTwoVLA/checkpoints`, their config names contain `vlabench` and do not contain `libero`, but they do not have checkpoint `99999` yet.

| Config | Latest checkpoint | Existing exp-name | Resume script |
|---|---:|---|---|
| `pifast_vlabench_cotrain_eb` | 29999 | `2026.04.18/02.31.16/pifast-vlabench-cotrain` | `train_scripts/train_pifast_vlabench_relative_action/train_pifast_vlabench_cotrain_eb.sh` |
| `pifast_vlabench_cotrain_mm_and_eb` | 29999 | `2026.04.18/02.18.28/pifast_vlabench_cotrain_mm_and_eb` | `train_scripts/train_pifast_vlabench_relative_action/train_pifast_vlabench_cotrain_mm_and_eb.sh` |
| `pifast_vlabench_cotrain_mm_data` | 29999 | `2026.04.19/07.33.37/pifast-vlabench-cotrain-mm-data` | `train_scripts/train_pifast_vlabench_relative_action/train_pifast_vlabench_mm_data.sh` |
| `pifast_vlabench_pretrain_primitive` | 29999 | `2026.04.17/19.40.02/pifast-vlabench` | `train_scripts/train_pifast_vlabench_relative_action/train_pifast_vlabench.sh` |
| `pifast_w_vlabench_cotrain_mm_and_eb` | 30000 | `2026.05.15/16.43.41/pifast-w-vlabench-cotrain-mm-and-eb` | `train_scripts/train_pifast_vlabench_relative_action/train_pifast_w_vlabench_cotrain_mm_and_eb.sh` |
| `pifast_w_vlabench_delta_steer_track1_09_track2_01` | 90000 | `2026.05.18/05.40.38/pifast-w-vlabench-cotrain-mm-data` | `train_scripts/vlabench_steer/train_pifast_pifast_w_vlabench_delta_steer_track1_09_track2_01.sh` |
| `pifast_w_vlabench_relative_cotrain_trajectory` | 29999 | `2026.05.15/08.45.02/pifast-w-vlabench-cotrain-mm-data-libero` | `train_scripts/vlabench_emm_ablation/train_pifast_w_vlabench_trajectory.sh` |

## VLABench result issues

These runs have valid `detail_info.json` files that were not recorded in
`metrics.json`, likely because a later failed partial rerun or merge overwrote
the final metrics.

| Run | Track | Valid detail tasks missing from `metrics.json` | Failed or empty tasks |
|---|---|---|---|
| `pifast_w_vlabench_delta_cotrain_mm_and_eb/model_99999` | `track_1_in_distribution` | `insert_flower`, `select_book`, `select_chemistry_tube`, `select_drink`, `select_fruit`, `select_mahjong`, `select_poker`, `select_toy` | `add_condiment`, `select_painting` |
| `pifast_w_vlabench_delta_cotrain_mm_and_eb/model_99999` | `track_2_cross_category` | `select_book`, `select_chemistry_tube`, `select_mahjong`, `select_poker`, `select_toy` | `add_condiment`, `select_drink`, `select_fruit`, `select_painting` |
| `pifast_w_vlabench_delta_cotrain_task-reasoning/model_99999` | `track_1_in_distribution` | `add_condiment`, `insert_flower`, `select_chemistry_tube`, `select_drink`, `select_mahjong`, `select_poker` | `select_book`, `select_painting` |

This run has finite `metrics.json` values, but the current
`evaluate_results/vlabench_jax.xlsx` does not include the corresponding row, so
it only needs the summary Excel to be regenerated.

| Run | Track | Issue |
|---|---|---|
| `pifast_vlabench_pretrain_primitive/model_29999` | `track_2_cross_category` | finite metrics exist but are missing from `vlabench_jax.xlsx` |
