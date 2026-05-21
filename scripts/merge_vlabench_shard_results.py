#!/usr/bin/env python3
"""Merge per-GPU shard outputs → <base-dir>/<track>/ (tasks + metrics.json).

Reads shards from <shard-parent>/shard_<i>/<track>/.
Default shard-parent is <base-dir>/shards_<track>/ (per-track staging; avoids cross-track collision).
Optional --cleanup removes shard-parent after successful merge.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge_shards(
    base: Path,
    track: str,
    num_shards: int,
    shard_parent: Path | None,
    *,
    cleanup: bool,
) -> None:
    if shard_parent is None:
        shard_parent = base / f"shards_{track}"

    out = base / track
    out.mkdir(parents=True, exist_ok=True)
    merged_metrics: dict[str, object] = {}
    for i in range(num_shards):
        st = shard_parent / f"shard_{i}" / track
        mj = st / "metrics.json"
        if mj.is_file():
            with open(mj, encoding="utf-8") as f:
                part = json.load(f)
            overlap = set(merged_metrics) & set(part)
            if overlap:
                raise SystemExit(
                    "Overlapping tasks in shards (merge conflict): "
                    f"{sorted(overlap)[:10]}..."
                )
            merged_metrics.update(part)
        if not st.is_dir():
            continue
        for item in st.iterdir():
            if item.name == "metrics.json":
                continue
            dest = out / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(merged_metrics, f, indent=4)
    print(f"Merged metrics + task dirs → {metrics_path}")

    if cleanup and shard_parent.is_dir():
        shutil.rmtree(shard_parent)
        print(f"Removed shard staging: {shard_parent}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--track", required=True)
    p.add_argument("--num-shards", required=True, type=int)
    p.add_argument(
        "--shard-parent",
        type=Path,
        default=None,
        help="Directory containing shard_0, shard_1, ... Default: <base-dir>/shards_<track>",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="After merge, delete --shard-parent (entire staging tree)",
    )
    args = p.parse_args()
    merge_shards(
        args.base_dir,
        args.track,
        args.num_shards,
        args.shard_parent,
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    main()
