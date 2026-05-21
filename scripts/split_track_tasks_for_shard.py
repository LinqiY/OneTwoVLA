#!/usr/bin/env python3
"""Print space-separated task names for shard INDEX out of NUM_WORKERS (contiguous chunks).

Optional env VLABENCH_TRACK_TASK_WHITELIST: comma/space-separated subset of track task names
(order preserved). Set by run_eval_instance.sh when --task is used with --parallel-workers > 1.

Special case: track ``track_1_in_distribution`` with the full task list (no whitelist),
``num_workers >= 6``, and contiguous packing yielding shard 4 exactly
``add_condiment insert_flower`` — splits them so shard 4 is only ``add_condiment`` and
shard 5 is only ``insert_flower`` (GPU indices 4 and 5); shards 0–3 unchanged.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: split_track_tasks_for_shard.py <track_name> <num_workers> <shard_index>",
            file=sys.stderr,
        )
        sys.exit(2)
    track_name, n_s, i_s = sys.argv[1], sys.argv[2], sys.argv[3]
    n_workers = int(n_s)
    shard_index = int(i_s)
    root = os.environ.get("VLABENCH_ROOT")
    if not root:
        print("VLABENCH_ROOT is not set", file=sys.stderr)
        sys.exit(1)
    path = os.path.join(root, "configs", "evaluation", "tracks", f"{track_name}.json")
    with open(path, encoding="utf-8") as f:
        track_keys = list(json.load(f).keys())
    whitelist_raw = os.environ.get("VLABENCH_TRACK_TASK_WHITELIST", "").strip()
    if whitelist_raw:
        requested = [
            t for t in whitelist_raw.replace(",", " ").split() if t.strip()
        ]
        unknown = [t for t in requested if t not in track_keys]
        if unknown:
            print(
                f"VLABENCH_TRACK_TASK_WHITELIST contains tasks not in {track_name}.json: {unknown}",
                file=sys.stderr,
            )
            sys.exit(1)
        tasks = requested
    else:
        tasks = track_keys
    if n_workers <= 0:
        print("num_workers must be positive", file=sys.stderr)
        sys.exit(1)
    chunk_size = max(1, (len(tasks) + n_workers - 1) // n_workers)
    chunks = [tasks[j * chunk_size : (j + 1) * chunk_size] for j in range(n_workers)]

    # track_1_in_distribution + default task list: normal packing puts add_condiment +
    # insert_flower together on shard 4 (GPU index 4). Split them onto shard 4 and 5
    # so GPU 5 is used; shards 0–3 unchanged. Requires a sixth shard slot (n_workers>=6).
    if (
        track_name == "track_1_in_distribution"
        and not whitelist_raw
        and n_workers >= 6
        and len(chunks) > 5
        and chunks[4] == ["add_condiment", "insert_flower"]
    ):
        chunks = list(chunks)
        chunks[4] = ["add_condiment"]
        chunks[5] = ["insert_flower"]

    if shard_index < 0 or shard_index >= len(chunks):
        print("", end="")
        return
    print(" ".join(chunks[shard_index]), end="")


if __name__ == "__main__":
    main()
