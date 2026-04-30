#!/usr/bin/env python3
"""
Resize + re-JPEG images for COCO-style VL data (CPU-only; no GPU used).

Two modes:

1) Parquet (recommended for your cotrain pipeline)
   - Reads HuggingFace-style image cells: struct { bytes, path } or raw bytes.
   - Decodes, optionally downsamples so min(w,h) <= --short-edge (never upscales),
     writes new JPEG bytes back into the same column; clears ``path`` so loaders
     prefer ``bytes`` (see openpi ``vl_parquet_common.normalize_image_cell``).

2) Image folder
   - Mirrors ``--input-images-dir`` tree under ``--output-images-dir`` for *.jpg/*.jpeg/*.png.

Hardware: **CPU only** (Pillow resize + JPEG encode). More ``--workers`` = faster until
disk or CPU saturates.

Rough wall time (order of magnitude, ~118k COCO train images):
  - SSD + 16--32 workers: often ~0.5--3 h for parquet rewrite (depends on row-group size
    and how large originals are).
  - HDD / network FS / few workers: can be many hours.

Example (parquet, match config ``vl_image_root`` for rows that use ``path``):

  python scripts/coco_parquet_resize_jpeg.py parquet \\
    --input-dir /path/to/coco/data_rg256 \\
    --output-dir /path/to/coco/data_rg256_jpeg448 \\
    --glob 'train-*.parquet' \\
    --image-root /path/to/coco \\
    --short-edge 448 \\
    --jpeg-quality 88 \\
    --workers 16 \\
    --overwrite

Then point ``vl_parquet_sources`` to ``--output-dir`` (and keep ``vl_image_root`` if any
rows still reference paths-only — after this script, bytes should be preferred).

Example (raw COCO train2017 JPEGs):

  python scripts/coco_parquet_resize_jpeg.py images \\
    --input-images-dir /data/coco/train2017 \\
    --output-images-dir /data/coco/train2017_short448 \\
    --short-edge 448 \\
    --jpeg-quality 88 \\
    --workers 16
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _normalize_image_cell(x: Any) -> Any:
    """Same contract as openpi ``vl_parquet_common.normalize_image_cell`` (subset)."""
    if not isinstance(x, dict):
        return x
    b = x.get("bytes")
    if isinstance(b, (bytes, bytearray, memoryview)) and len(bytes(b)) > 0:
        return bytes(b)
    p = x.get("path")
    if isinstance(p, str) and p.strip():
        return p
    return x


def _load_pil(x: Any, *, image_root: Path | None) -> Image.Image:
    x = _normalize_image_cell(x)
    if isinstance(x, str):
        p = Path(x).expanduser()
        if not p.is_absolute() and image_root is not None:
            p = image_root / p
        return Image.open(p).convert("RGB")
    if isinstance(x, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(x))) as img:
            return img.convert("RGB")
    raise TypeError(f"Unsupported image cell after normalize: {type(x)!r}")


def _resize_short_edge(img: Image.Image, short_edge: int, *, only_shrink: bool) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    if only_shrink and m <= short_edge:
        return img
    if not only_shrink and m == short_edge:
        return img
    scale = short_edge / m
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _encode_jpeg_rgb(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _hf_image_dict_from_bytes(jpeg_bytes: bytes) -> dict[str, Any]:
    return {"bytes": jpeg_bytes, "path": None}


def _reencode_one_cell(
    cell: Any,
    *,
    image_root: Path | None,
    short_edge: int,
    jpeg_quality: int,
    only_shrink: bool,
) -> Any:
    if cell is None:
        return None
    if isinstance(cell, (list, tuple)):
        return [
            _reencode_one_cell(x, image_root=image_root, short_edge=short_edge, jpeg_quality=jpeg_quality, only_shrink=only_shrink)
            for x in cell
        ]
    if isinstance(cell, dict) and ("bytes" in cell or "path" in cell):
        img = _load_pil(cell, image_root=image_root)
        img = _resize_short_edge(img, short_edge, only_shrink=only_shrink)
        return _hf_image_dict_from_bytes(_encode_jpeg_rgb(img, jpeg_quality))
    if isinstance(cell, (bytes, bytearray, memoryview)):
        img = _load_pil(cell, image_root=image_root)
        img = _resize_short_edge(img, short_edge, only_shrink=only_shrink)
        return _hf_image_dict_from_bytes(_encode_jpeg_rgb(img, jpeg_quality))
    raise TypeError(f"Unsupported image column value type: {type(cell)!r}")


@dataclass(frozen=True)
class _ParquetJob:
    cell: Any
    image_root: str | None
    short_edge: int
    jpeg_quality: int
    only_shrink: bool


def _parquet_worker(job: _ParquetJob) -> dict[str, Any] | list[Any] | bytes:
    root = Path(job.image_root).resolve() if job.image_root else None
    return _reencode_one_cell(
        job.cell,
        image_root=root,
        short_edge=job.short_edge,
        jpeg_quality=job.jpeg_quality,
        only_shrink=job.only_shrink,
    )


def _pick_image_column(names: list[str]) -> str:
    for c in ("image", "images"):
        if c in names:
            return c
    raise ValueError(f"No 'image' or 'images' column in parquet schema. Columns: {names!r}")


def _rewrite_parquet_file(
    src: Path,
    dst: Path,
    *,
    image_root: Path | None,
    short_edge: int,
    jpeg_quality: int,
    only_shrink: bool,
    workers: int,
    overwrite: bool,
    row_pbar: Optional[Any] = None,
    shard_pbar: Optional[Any] = None,
) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite (pass --overwrite): {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow
    col = _pick_image_column(schema.names)

    writer: pq.ParquetWriter | None = None
    all_row_groups_ok = False
    try:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            rows = table.to_pylist()

            jobs = [
                _ParquetJob(
                    cell=row[col],
                    image_root=str(image_root) if image_root is not None else None,
                    short_edge=short_edge,
                    jpeg_quality=jpeg_quality,
                    only_shrink=only_shrink,
                )
                for row in rows
            ]

            if workers <= 1:
                new_cells = [_parquet_worker(j) for j in jobs]
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    cs = max(1, min(32, len(jobs) // max(workers, 1)))
                    new_cells = list(ex.map(_parquet_worker, jobs, chunksize=cs))

            for row, new_cell in zip(rows, new_cells, strict=True):
                row[col] = new_cell

            out = pa.Table.from_pylist(rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(str(dst), schema, compression="snappy")
            writer.write_table(out)
            n = int(out.num_rows)
            if row_pbar is not None:
                row_pbar.update(n)
            if shard_pbar is not None:
                shard_pbar.set_postfix_str(src.name[:48] + ("…" if len(src.name) > 48 else ""), refresh=False)
        all_row_groups_ok = True
    finally:
        if writer is not None:
            writer.close()
    if all_row_groups_ok and shard_pbar is not None:
        shard_pbar.update(1)


def _cmd_parquet(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None

    if not input_dir.is_dir():
        logger.error("input-dir is not a directory: %s", input_dir)
        return 2

    paths = sorted(p for p in input_dir.glob(args.glob) if p.is_file() and p.suffix == ".parquet")
    if not paths:
        logger.error("No parquet matched %s under %s", args.glob, input_dir)
        return 2

    logger.info("Matched %d parquet file(s); workers=%s CPU", len(paths), args.workers)
    total_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)
    with tqdm(
        total=total_rows,
        desc="Parquet rows (all shards)",
        unit="row",
        mininterval=0.3,
        smoothing=0.05,
        position=0,
        leave=True,
    ) as row_pbar, tqdm(
        total=len(paths),
        desc="Parquet shards",
        unit="file",
        mininterval=0.2,
        position=1,
        leave=True,
    ) as shard_pbar:
        for src in paths:
            dst = output_dir / src.name
            logger.info("%s -> %s", src, dst)
            _rewrite_parquet_file(
                src,
                dst,
                image_root=image_root,
                short_edge=args.short_edge,
                jpeg_quality=args.jpeg_quality,
                only_shrink=not args.allow_upscale,
                workers=args.workers,
                overwrite=args.overwrite,
                row_pbar=row_pbar,
                shard_pbar=shard_pbar,
            )
    logger.info("Done. Output dir: %s", output_dir)
    return 0


@dataclass(frozen=True)
class _ImageFileJob:
    src: str
    dst: str
    short_edge: int
    jpeg_quality: int
    only_shrink: bool
    overwrite: bool


def _image_file_worker(job: _ImageFileJob) -> None:
    dst = Path(job.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not job.overwrite:
        raise FileExistsError(str(dst))
    with Image.open(job.src) as img:
        rgb = img.convert("RGB")
        m = min(rgb.size)
        if job.only_shrink and m <= job.short_edge:
            out = rgb
        else:
            scale = job.short_edge / m
            nw = max(1, int(round(rgb.size[0] * scale)))
            nh = max(1, int(round(rgb.size[1] * scale)))
            out = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    out.save(job.dst, format="JPEG", quality=job.jpeg_quality, optimize=True)


def _iter_images(root: Path) -> Iterator[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix in exts:
                yield Path(dirpath) / name


def _cmd_images(args: argparse.Namespace) -> int:
    input_root: Path = args.input_images_dir.expanduser().resolve()
    output_root: Path = args.output_images_dir.expanduser().resolve()
    if not input_root.is_dir():
        logger.error("input-images-dir is not a directory: %s", input_root)
        return 2

    files = list(_iter_images(input_root))
    if not files:
        logger.error("No images under %s", input_root)
        return 2

    only_shrink = not args.allow_upscale
    jobs: list[_ImageFileJob] = []
    for p in files:
        rel = p.relative_to(input_root)
        dst = output_root / rel
        dst = dst.with_suffix(".jpg")
        jobs.append(
            _ImageFileJob(
                src=str(p),
                dst=str(dst),
                short_edge=args.short_edge,
                jpeg_quality=args.jpeg_quality,
                only_shrink=only_shrink,
                overwrite=args.overwrite,
            )
        )

    logger.info("Images: %d files, workers=%s CPU", len(jobs), args.workers)
    if args.workers <= 1:
        for j in tqdm(jobs, desc="Images", unit="file", mininterval=0.3):
            _image_file_worker(j)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_image_file_worker, j) for j in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Images", unit="file", mininterval=0.3):
                f.result()

    logger.info("Done. Output: %s", output_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def _add_common_image_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--short-edge", type=int, default=448, help="Target min(w,h) in pixels after resize (default: 448)")
        sp.add_argument("--jpeg-quality", type=int, default=88, help="JPEG quality 1--95 (default: 88)")
        sp.add_argument(
            "--allow-upscale",
            action="store_true",
            help="If set, scale up images smaller than short-edge (default: only shrink)",
        )
        sp.add_argument(
            "--workers",
            type=int,
            default=max(1, (os.cpu_count() or 4) // 2),
            help="Parallel worker processes (default: about half of CPU cores)",
        )

    sp_pq = sub.add_parser("parquet", help="Rewrite parquet shards (image / images column)")
    sp_pq.add_argument("--input-dir", type=Path, required=True)
    sp_pq.add_argument("--output-dir", type=Path, required=True)
    sp_pq.add_argument("--glob", default="train-*.parquet", help="Glob under input-dir (default: train-*.parquet)")
    sp_pq.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Base dir for relative image paths in parquet (same idea as training vl_image_root)",
    )
    sp_pq.add_argument("--overwrite", action="store_true")
    _add_common_image_args(sp_pq)

    sp_im = sub.add_parser("images", help="Resize JPEG/PNG files under a directory tree")
    sp_im.add_argument("--input-images-dir", type=Path, required=True)
    sp_im.add_argument("--output-images-dir", type=Path, required=True)
    sp_im.add_argument("--overwrite", action="store_true", help="Replace existing files under output-images-dir")
    _add_common_image_args(sp_im)

    args = p.parse_args(argv)
    if args.mode == "parquet":
        return _cmd_parquet(args)
    if args.mode == "images":
        return _cmd_images(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
