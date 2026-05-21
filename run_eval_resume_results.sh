#!/usr/bin/env bash
# Resume unfinished VLABench rows for:
#   pifast_vlabench_delta_cotrain_mm_and_eb/model_99999
#
# The script scans existing detail_info.json files and only reruns tasks that are
# missing, empty, malformed, or shorter than the configured episode count.

set -euo pipefail

cd /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA || exit 1

export VLABENCH_ROOT=/inspire/hdd/global_user/gongjingjing-25039/lqyin/VLABench/VLABench
export MPLCONFIGDIR=/tmp/matplotlib

RUN="bash /inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/run_eval_instance_only_checkpoint.sh"
CKPT="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/checkpoints/pifast_vlabench_delta_cotrain_mm_and_eb/2026.05.17/02.20.54/pifast-vlabench/99999"
RESULT_DIR="/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/evaluate_results/pifast_vlabench_delta_cotrain_mm_and_eb/model_99999"

NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi

workers_for_tasks() {
  local num_tasks="$1"
  if [[ "${num_tasks}" -lt 1 ]]; then
    echo 1
  elif [[ "${NUM_GPUS}" -lt "${num_tasks}" ]]; then
    echo "${NUM_GPUS}"
  else
    echo "${num_tasks}"
  fi
}

unfinished_tasks() {
  local track="$1"
  python - "${track}" "${RESULT_DIR}" "${VLABENCH_ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

track = sys.argv[1]
result_dir = Path(sys.argv[2])
vlabench_root = Path(sys.argv[3])

track_config = vlabench_root / "configs" / "evaluation" / "tracks" / f"{track}.json"
episodes_by_task = json.loads(track_config.read_text())
unfinished = []

for task, episode_configs in episodes_by_task.items():
    # run_eval_instance_only_checkpoint.sh does not override n_episode per task.
    # track_2 insert_flower only has 10 configs and would assert with default 50.
    if len(episode_configs) < 50:
        continue

    detail_path = result_dir / track / task / "detail_info.json"
    if not detail_path.exists():
        unfinished.append(task)
        continue

    try:
        episodes = json.loads(detail_path.read_text())
    except Exception:
        unfinished.append(task)
        continue

    if not isinstance(episodes, list) or len(episodes) < len(episode_configs):
        unfinished.append(task)

print(",".join(unfinished))
PY
}

TRACK1_TASKS="$(unfinished_tasks track_1_in_distribution)"
TRACK2_TASKS="$(unfinished_tasks track_2_cross_category)"

echo "Detected ${NUM_GPUS} GPU(s)."
echo "Unfinished tasks:"
echo "  track_1_in_distribution: ${TRACK1_TASKS:-none}"
echo "  track_2_cross_category: ${TRACK2_TASKS:-none}"

if [[ -n "${TRACK1_TASKS}" ]]; then
  IFS=',' read -r -a TRACK1_ARRAY <<< "${TRACK1_TASKS}"
  WORKERS_TRACK1="$(workers_for_tasks "${#TRACK1_ARRAY[@]}")"
  echo "==> Resume track_1_in_distribution (${WORKERS_TRACK1} worker(s))"
  ${RUN} "${CKPT}" \
    --track track_1_in_distribution \
    --task "${TRACK1_TASKS}" \
    --parallel-workers "${WORKERS_TRACK1}"
fi

if [[ -n "${TRACK2_TASKS}" ]]; then
  IFS=',' read -r -a TRACK2_ARRAY <<< "${TRACK2_TASKS}"
  WORKERS_TRACK2="$(workers_for_tasks "${#TRACK2_ARRAY[@]}")"
  echo "==> Resume track_2_cross_category (${WORKERS_TRACK2} worker(s))"
  ${RUN} "${CKPT}" \
    --track track_2_cross_category \
    --task "${TRACK2_TASKS}" \
    --parallel-workers "${WORKERS_TRACK2}"
fi

echo "==> Rebuild metrics.json from detail_info.json"
python - <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import math

ROOT = Path("/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/evaluate_results")
TRACKS = [
    ROOT / "pifast_vlabench_delta_cotrain_mm_and_eb/model_99999/track_1_in_distribution",
    ROOT / "pifast_vlabench_delta_cotrain_mm_and_eb/model_99999/track_2_cross_category",
]


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def summarize_task(detail_path: Path) -> dict[str, float] | None:
    try:
        episodes = json.loads(detail_path.read_text())
    except Exception:
        return None
    if not isinstance(episodes, list) or not episodes:
        return None

    success_values = []
    intention_values = []
    progress_values = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        success = episode.get("success")
        if isinstance(success, bool):
            success_values.append(1.0 if success else 0.0)
        intention = episode.get("intention_score")
        if finite_number(intention):
            intention_values.append(float(intention))
        progress = episode.get("progress_score")
        if finite_number(progress):
            progress_values.append(float(progress))

    if not success_values and not intention_values and not progress_values:
        return None

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "success_rate": mean(success_values),
        "intention_score": mean(intention_values),
        "progress_score": mean(progress_values),
    }


for track_dir in TRACKS:
    metrics = {}
    empty_or_missing = []
    if not track_dir.exists():
        print(f"skip missing {track_dir}")
        continue
    for task_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
        detail_path = task_dir / "detail_info.json"
        if not detail_path.exists():
            empty_or_missing.append(task_dir.name)
            continue
        summary = summarize_task(detail_path)
        if summary is None:
            empty_or_missing.append(task_dir.name)
            continue
        metrics[task_dir.name] = summary

    metrics_path = track_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=4) + "\n")
    print(f"wrote {metrics_path} ({len(metrics)} tasks)")
    if empty_or_missing:
        print(f"  still empty/missing: {', '.join(empty_or_missing)}")
PY

echo "==> Regenerate VLABench summary Excel/figures"
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("xlsxwriter") else 1)
PY
then
  python examples/vlabench/summarize.py \
    --root_dir evaluate_results \
    --outfile evaluate_results/vlabench_jax.xlsx \
    --figure evaluate_results/vlabench_jax_figure.png \
    --figure_stack evaluate_results/vlabench_jax_figure_stack.png
else
  echo "Skip summarize.py: missing Python package xlsxwriter in this environment." >&2
  echo "Install xlsxwriter or run summarize.py in an environment that has it." >&2
fi

echo "Done."
