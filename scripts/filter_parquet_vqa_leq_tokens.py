#!/usr/bin/env python3
"""
Copy parquet VL **raw rows** whose ``FASTTokenizer.tokenize_vqa`` length is **<=** ``--max-tokens``
into ``--out-dir`` as new ``*.parquet`` shards (same row schema as input; training still runs the
same preprocessor).

Token counting matches ``scripts/count_parquet_vqa_token_lengths.py`` (zero state, same
``source_id`` preprocessor).

Example (COCO, default path from ``pifast_vlabench_cotrain_coco_parquet``)::

    uv run python scripts/filter_parquet_vqa_leq_tokens.py \\
        --path /inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448 \\
        --source-id coco \\
        --max-tokens 512 \\
        --out-dir /inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448_leq512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

import openpi.shared.download as download
from openpi.models.tokenizer import PALIGEMMA_EOS_TOKEN
from openpi.policies.parquet_vl_dataset import PARQUET_VL_SOURCES, expand_parquet_paths
from openpi.policies.vl_parquet_common import _extract_question_answer


def _clean_text(text: str) -> str:
    text = text.replace("<image>", " ").replace("<IMAGE>", " ")
    text = text.replace("_", " ").replace("\n", " ")
    text = " ".join(text.strip().split())
    return text.lower()


def _discretize_state(state: np.ndarray) -> np.ndarray:
    return np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


def _load_paligemma_sentencepiece():
    path = download.maybe_download(
        "/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model",
        gs={"token": "anon"},
    )
    with path.open("rb") as f:
        return spm.SentencePieceProcessor(model_proto=f.read())


def vqa_sequence_length(
    sp: spm.SentencePieceProcessor,
    *,
    question: str,
    answer: str,
    state: np.ndarray,
) -> int:
    cleaned_question = _clean_text(question)
    cleaned_answer = " ".join(answer.strip().split())
    state_str = " ".join(map(str, _discretize_state(state)))
    prefix = f"Task: {cleaned_question}, State: {state_str};\n"
    prefix_tokens = sp.encode(prefix, add_bos=True)
    answer_prefix_tokens = sp.encode("Answer: ")
    answer_text_tokens = sp.encode(cleaned_answer)
    eos_tokens = [PALIGEMMA_EOS_TOKEN]
    return len(prefix_tokens) + len(answer_prefix_tokens) + len(answer_text_tokens) + len(eos_tokens)


def _total_raw_parquet_rows(parquet_paths: list[str]) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_paths)


def _default_out_dir(in_path: str, max_tokens: int) -> Path:
    p = Path(os.path.expanduser(in_path))
    if p.is_file():
        base = p.parent.name
        parent = p.parent.parent
    else:
        base = p.name
        parent = p.parent
    return parent / f"{base}_leq{max_tokens}_parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--path",
        type=str,
        default="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448",
        help="Input parquet file, glob, or directory.",
    )
    p.add_argument("--source-id", type=str, default="coco", choices=sorted(PARQUET_VL_SOURCES.keys()))
    p.add_argument("--max-tokens", type=int, default=512, help="Keep rows with sequence length <= this value.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (created). If empty, uses ``<parent>/<input_basename>_leq{max_tokens}_parquet``.",
    )
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument(
        "--shard-rows",
        type=int,
        default=20_000,
        help="Rows per output parquet shard (memory vs file count).",
    )
    p.add_argument("--no-tqdm", action="store_true")
    args = p.parse_args()

    paths = expand_parquet_paths(args.path)
    if not paths:
        print(f"No parquet files under {args.path!r}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir.strip() else _default_out_dir(args.path, args.max_tokens)
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = PARQUET_VL_SOURCES[args.source_id]["preprocessor"]
    sp = _load_paligemma_sentencepiece()
    state = np.zeros((args.state_dim,), dtype=np.float32)

    raw_total = _total_raw_parquet_rows(paths)
    pbar = tqdm(
        total=raw_total,
        unit="row",
        desc="scan",
        disable=args.no_tqdm,
        smoothing=0.05,
    )

    stats = {
        "input_path_arg": args.path,
        "num_input_parquet_files": len(paths),
        "source_id": args.source_id,
        "max_tokens": args.max_tokens,
        "state_dim": args.state_dim,
        "raw_rows_seen": 0,
        "preprocess_none": 0,
        "preprocess_error": 0,
        "kept_leq": 0,
        "dropped_gt": 0,
        "out_dir": str(out_dir),
    }

    buffer: list[dict] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        out_path = out_dir / f"filtered-{shard_idx:05d}.parquet"
        table = pa.Table.from_pylist(buffer)
        pq.write_table(table, out_path, compression="snappy")
        shard_idx += 1
        buffer = []

    try:
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                for row in pf.read_row_group(rg_idx).to_pylist():
                    stats["raw_rows_seen"] += 1
                    pbar.update(1)
                    original = dict(row)
                    try:
                        out = pre.preprocess(dict(row))
                        if out is None:
                            stats["preprocess_none"] += 1
                            continue
                        q, a = _extract_question_answer(out)
                    except Exception:
                        stats["preprocess_error"] += 1
                        continue

                    n = vqa_sequence_length(sp, question=q, answer=a, state=state)
                    if n > args.max_tokens:
                        stats["dropped_gt"] += 1
                        pbar.set_postfix_str(
                            f"kept={stats['kept_leq']} drop>{args.max_tokens}={stats['dropped_gt']}",
                            refresh=False,
                        )
                        continue

                    stats["kept_leq"] += 1
                    buffer.append(original)
                    pbar.set_postfix_str(
                        f"kept={stats['kept_leq']} drop>{args.max_tokens}={stats['dropped_gt']}",
                        refresh=False,
                    )

                    if len(buffer) >= args.shard_rows:
                        flush()
    finally:
        pbar.close()

    flush()

    manifest = out_dir / "filter_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Wrote {stats['kept_leq']} rows to {out_dir} ({shard_idx} shard(s)).")
    print(f"Manifest: {manifest}")
    print(
        f"raw={stats['raw_rows_seen']}  preprocess_none={stats['preprocess_none']}  "
        f"preprocess_error={stats['preprocess_error']}  kept<={args.max_tokens}={stats['kept_leq']}  "
        f"dropped>{args.max_tokens}={stats['dropped_gt']}"
    )

    if stats["kept_leq"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
