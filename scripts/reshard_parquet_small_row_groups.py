#!/usr/bin/env python3
"""
Rewrite Parquet VL shards so row groups are small (e.g. 128–512 rows).

Why: ParquetVQADataset reads an entire row group for each random sample
(openpi/policies/parquet_vl_dataset.py). COCO-style exports often use one
huge row group per file (~GB of image.bytes), which makes training I/O orders
of magnitude slower than LaTeX-style shards with ~100 rows per group.

This script reads each input row group once, then writes the same rows split
into multiple output row groups (peak memory ≈ largest input row group).

Example (COCO cotrain parquet):
  python scripts/reshard_parquet_small_row_groups.py \\
    --input-dir /path/to/coco/data \\
    --output-dir /path/to/coco/data_rg256 \\
    --glob 'train-*.parquet' \\
    --row-group-size 256

Then point config vl_parquet_sources path to --output-dir.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _list_inputs(input_dir: Path, glob_pat: str) -> list[Path]:
    paths = sorted(input_dir.glob(glob_pat))
    return [p for p in paths if p.is_file() and p.suffix == ".parquet"]


def reshard_one(
    src: Path,
    dst: Path,
    *,
    row_group_size: int,
    compression: str,
) -> None:
    pf = pq.ParquetFile(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            if writer is None:
                writer = pq.ParquetWriter(str(dst), table.schema, compression=compression)
            writer.write_table(table, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing source .parquet files")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write reshaped .parquet (created if missing)")
    p.add_argument(
        "--glob",
        default="train-*.parquet",
        help="Which files under input-dir to process (default: train-*.parquet)",
    )
    p.add_argument(
        "--row-group-size",
        type=int,
        default=256,
        help="Target max rows per output row group (default: 256). Try 128–512.",
    )
    p.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec for output (default: snappy). Use 'zstd' for smaller files, slower writes.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done (no writes)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in output-dir when names collide",
    )
    args = p.parse_args(argv)

    if args.row_group_size < 1:
        logger.error("--row-group-size must be >= 1")
        return 2

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        logger.error("input-dir is not a directory: %s", input_dir)
        return 2

    sources = _list_inputs(input_dir, args.glob)
    if not sources:
        logger.error("No parquet files matched %s under %s", args.glob, input_dir)
        return 2

    logger.info("Matched %d parquet file(s)", len(sources))
    for src in sources:
        dst = output_dir / src.name
        pf = pq.ParquetFile(str(src))
        md = pf.metadata
        rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        max_rg = max(rg_rows) if rg_rows else 0
        n_rows = md.num_rows
        rgs = args.row_group_size
        exp_out_rgs = math.ceil(n_rows / rgs) if n_rows and rgs else 0
        exp_max_rows = min(rgs, n_rows) if n_rows else 0

        logger.info(
            "%s -> %s | input rows=%s row_groups=%s max_input_rg_rows=%s",
            src.name,
            dst,
            n_rows,
            md.num_row_groups,
            max_rg,
        )
        logger.info(
            "  expect output (row_group_size=%s): ~%s row_groups, max %s rows/rg (last group may be smaller)",
            rgs,
            exp_out_rgs,
            exp_max_rows,
        )
        if args.dry_run:
            continue
        if dst.exists() and not args.overwrite:
            logger.error("Refusing to overwrite (use --overwrite): %s", dst)
            return 2
        if dst.exists():
            dst.unlink()
        reshard_one(src, dst, row_group_size=args.row_group_size, compression=args.compression)
        pfo = pq.ParquetFile(str(dst))
        out_rg_rows = [pfo.metadata.row_group(i).num_rows for i in range(pfo.metadata.num_row_groups)]
        logger.info(
            "  wrote %s | out_row_groups=%s max_out_rg_rows=%s",
            dst.name,
            pfo.metadata.num_row_groups,
            max(out_rg_rows) if out_rg_rows else 0,
        )

    if args.dry_run:
        logger.info("Dry run only; no files written.")
    else:
        logger.info("Done. Point vl_parquet_sources to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
