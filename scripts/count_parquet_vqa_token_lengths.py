#!/usr/bin/env python3
"""
Scan parquet VL data (same preprocess + token layout as FASTTokenizer.tokenize_vqa) and
count how many samples exceed a token-length threshold.

Default matches CotrainConfig ``pifast_vlabench_cotrain_coco_parquet`` / ``mm_data`` COCO path
and ``model_state_dim=action_dim`` (zeros state), same as TokenizeFASTVQAInputs.

Example (COCO path matches ``pifast_vlabench_cotrain_coco_parquet`` in ``config.py``)::

    uv run python scripts/count_parquet_vqa_token_lengths.py \\
        --path /inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448 \\
        --source-id coco \\
        --threshold 512 \\
        --state-dim 7

    # omit --path to use the same default as above
    uv run python scripts/count_parquet_vqa_token_lengths.py --threshold 512

Progress is ``tqdm`` over raw parquet rows (metadata row count); postfix shows valid rows
and how many exceed ``--threshold``. Use ``--no-tqdm`` for non-interactive logs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm

import openpi.shared.download as download
from openpi.models.tokenizer import PALIGEMMA_EOS_TOKEN
from openpi.policies.parquet_vl_dataset import PARQUET_VL_SOURCES, expand_parquet_paths
from openpi.policies.vl_parquet_common import _extract_question_answer


def _clean_text(text: str) -> str:
    """Match FASTTokenizer._clean_text (used before tokenize_vqa prefix)."""
    text = text.replace("<image>", " ").replace("<IMAGE>", " ")
    text = text.replace("_", " ").replace("\n", " ")
    text = " ".join(text.strip().split())
    return text.lower()


def _discretize_state(state: np.ndarray) -> np.ndarray:
    """Match FASTTokenizer._discretize_state."""
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
    """
    Exact token count for the concatenation built in FASTTokenizer.tokenize_vqa
    before _pad_or_truncate (prefix + ``Answer: `` + answer + EOS).
    """
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


def _iter_valid_qa_rows(
    parquet_paths: list[str],
    source_id: str,
    *,
    raw_row_pbar: tqdm,
):
    preprocessor = PARQUET_VL_SOURCES[source_id]["preprocessor"]
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            for row in pf.read_row_group(rg_idx).to_pylist():
                raw_row_pbar.update(1)
                try:
                    out = preprocessor.preprocess(dict(row))
                    if out is None:
                        continue
                    q, a = _extract_question_answer(out)
                    yield q, a
                except Exception:
                    continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--path",
        type=str,
        default="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/coco/data_rg256_jpeg448",
        help="Parquet file, glob, or directory (recursive **/*.parquet). Same expansion as training.",
    )
    p.add_argument("--source-id", type=str, default="coco", choices=sorted(PARQUET_VL_SOURCES.keys()))
    p.add_argument("--threshold", type=int, default=512)
    p.add_argument(
        "--state-dim",
        type=int,
        default=7,
        help="Length of zero state vector (Pi0FAST vlabench uses action_dim=7).",
    )
    p.add_argument("--max-rows", type=int, default=0, help="If >0, stop after this many *valid* rows (debug).")
    p.add_argument("--no-tqdm", action="store_true", help="Disable progress bar (e.g. for logs).")
    args = p.parse_args()

    paths = expand_parquet_paths(args.path)
    if not paths:
        print(f"No parquet files under {args.path!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Resolved {len(paths)} parquet files (train-* only when split shards are present).")
    sp = _load_paligemma_sentencepiece()
    state = np.zeros((args.state_dim,), dtype=np.float32)

    raw_total = _total_raw_parquet_rows(paths)
    pbar = tqdm(
        total=raw_total,
        unit="row",
        desc="parquet rows",
        smoothing=0.05,
        disable=args.no_tqdm,
    )

    total = 0
    over = 0
    lengths: list[int] = []
    max_len = -1
    max_example: tuple[str, str, int] | None = None

    try:
        for question, answer in _iter_valid_qa_rows(paths, args.source_id, raw_row_pbar=pbar):
            n = vqa_sequence_length(sp, question=question, answer=answer, state=state)
            total += 1
            lengths.append(n)
            if n > max_len:
                max_len = n
                max_example = (question[:200], answer[:200], n)
            if n > args.threshold:
                over += 1
            pbar.set_postfix_str(f"valid={total} over>{args.threshold}={over}", refresh=False)
            if args.max_rows and total >= args.max_rows:
                break
    finally:
        pbar.close()

    if total == 0:
        print("No valid rows after preprocessing.", file=sys.stderr)
        sys.exit(1)

    arr = np.asarray(lengths, dtype=np.int64)
    pct = 100.0 * over / total
    print(f"source_id={args.source_id!r}  threshold={args.threshold}")
    print(f"valid_rows={total}  over_threshold={over}  ({pct:.4f}%)")
    print(f"length min/mean/max = {arr.min()} / {arr.mean():.2f} / {arr.max()}")
    for q in (50, 90, 95, 99, 99.9):
        print(f"  p{q}: {np.percentile(arr, q):.1f}")

    bins = [256, 384, 512, 640, 768, 1024, 1536, 2048, 10_000]
    print("cdf (fraction <= L):")
    for lim in bins:
        le = int((arr <= lim).sum())
        print(f"  <= {lim:5d}: {100.0 * le / total:.2f}%")

    if max_example is not None:
        mq, ma, mn = max_example
        print(f"\nlongest sample len={mn}")
        print(f"  question[:200]={mq!r}")
        print(f"  answer[:200]={ma!r}")


if __name__ == "__main__":
    main()
