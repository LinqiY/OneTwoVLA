#!/usr/bin/env python3
"""
Print a few (question, answer) pairs from parquet VL data after the same preprocessor
as ``ParquetVQADataset`` / training.

By default only scans the first ``--max-raw-scan`` raw rows (fast), shuffles valid
samples with ``--seed``, then prints ``--num`` examples. Use ``--max-raw-scan 0`` to
scan all parquet rows and reservoir-sample ``--num`` valid pairs (slower, full pass).

Example (COCO path matches ``pifast_vlabench_cotrain_coco_parquet`` in ``config.py``)::

    uv run python scripts/inspect_parquet_vqa_samples.py \\
        --path /inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448 \\
        --source-id coco \\
        --num 5

    # omit --path to use the same default directory as above
    uv run python scripts/inspect_parquet_vqa_samples.py --source-id coco --num 5

    uv run python scripts/inspect_parquet_vqa_samples.py --source-id a_okvqa --num 8 --max-raw-scan 20000
"""

from __future__ import annotations

import argparse
import random
import sys

import pyarrow.parquet as pq

from openpi.policies.parquet_vl_dataset import PARQUET_VL_SOURCES, expand_parquet_paths
from openpi.policies.vl_parquet_common import _extract_question_answer


def _try_qa(pre, row: dict) -> tuple[str, str] | None:
    try:
        out = pre.preprocess(dict(row))
        if out is None:
            return None
        return _extract_question_answer(out)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--path",
        type=str,
        default="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448",
        help="Parquet file, glob, or directory.",
    )
    p.add_argument("--source-id", type=str, default="coco", choices=sorted(PARQUET_VL_SOURCES.keys()))
    p.add_argument("--num", type=int, default=5, help="How many samples to print.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-raw-scan",
        type=int,
        default=10_000,
        help="Max raw parquet rows to read from the start (0 = scan entire dataset).",
    )
    p.add_argument("--answer-head", type=int, default=500, help="Print at most this many chars of each answer.")
    args = p.parse_args()

    paths = expand_parquet_paths(args.path)
    if not paths:
        print(f"No parquet files under {args.path!r}", file=sys.stderr)
        sys.exit(1)

    pre = PARQUET_VL_SOURCES[args.source_id]["preprocessor"]
    rng = random.Random(args.seed)

    if args.max_raw_scan > 0:
        pool: list[tuple[str, str]] = []
        raw = 0
        outer_break = False
        for path in paths:
            if outer_break:
                break
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                if outer_break:
                    break
                for row in pf.read_row_group(rg_idx).to_pylist():
                    raw += 1
                    qa = _try_qa(pre, row)
                    if qa is not None:
                        pool.append(qa)
                    if raw >= args.max_raw_scan:
                        outer_break = True
                        break

        if not pool:
            print("No valid QA in scanned window; try larger --max-raw-scan or check path/source-id.", file=sys.stderr)
            sys.exit(1)

        rng.shuffle(pool)
        chosen = pool[: args.num]
    else:
        # Reservoir sample `num` valid (q,a) over full scan
        chosen: list[tuple[str, str]] = []
        seen = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                for row in pf.read_row_group(rg_idx).to_pylist():
                    qa = _try_qa(pre, row)
                    if qa is None:
                        continue
                    seen += 1
                    if len(chosen) < args.num:
                        chosen.append(qa)
                    else:
                        j = rng.randint(1, seen)
                        if j <= args.num:
                            chosen[j - 1] = qa

        if not chosen:
            print("No valid QA in dataset.", file=sys.stderr)
            sys.exit(1)

    h = args.answer_head
    print(f"path={args.path!r}  source_id={args.source_id!r}  showing {len(chosen)} sample(s)\n")
    for i, (q, a) in enumerate(chosen, start=1):
        a_show = a if len(a) <= h else a[:h] + "…"
        print(f"========== sample {i} ==========")
        print(f"Q ({len(q)} chars):\n  {q!r}\n")
        print(f"A ({len(a)} chars, head):\n  {a_show!r}\n")


if __name__ == "__main__":
    main()
