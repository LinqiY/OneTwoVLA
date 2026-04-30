from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pyarrow.parquet as pq
from torch.utils.data import Dataset

from openpi.policies.vl_parquet_common import row_to_vqa_sample
from openpi.policies.vl_parquet_preprocess import AOkvqaPreprocessor, CocoPreprocessor, LatexocrPreprocessor

# HuggingFace-style shard names, e.g. train-00000-of-00001.parquet
_SPLIT_PREFIX_RE = re.compile(r"^(train|test|validation)-", re.IGNORECASE)


def _split_shard_tag(filename: str) -> str | None:
    m = _SPLIT_PREFIX_RE.match(filename)
    return m.group(1).lower() if m else None


def _filter_train_split_parquets(paths: list[str]) -> list[str]:
    """
    If any file looks like an HF split shard (train-|test-|validation-), keep only train-*.
    Files without such a prefix are dropped in that case. If no file uses these prefixes, keep all.
    """
    if not paths:
        return paths
    tagged = [(p, _split_shard_tag(Path(p).name)) for p in paths]
    if not any(t is not None for _, t in tagged):
        return paths
    return [p for p, t in tagged if t == "train"]


def expand_parquet_paths(paths: str | Sequence[str]) -> list[str]:
    """Expand path(s) into parquet files: directories use recursive **/*.parquet; HF split shards keep train-* only."""
    if isinstance(paths, str):
        paths = [paths]
    expanded: list[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            expanded.extend(sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)))
        else:
            matches = sorted(glob.glob(p))
            if matches:
                expanded.extend(matches)
            elif os.path.isfile(p):
                expanded.append(p)
    # stable unique
    seen: set[str] = set()
    out: list[str] = []
    for x in expanded:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return _filter_train_split_parquets(out)


@dataclass(frozen=True)
class ParquetRowSpec:
    image_keys: list[str]
    target_num_images: int = 3


PARQUET_VL_SOURCES: dict[str, dict[str, Any]] = {
    "a_okvqa": {
        "preprocessor": AOkvqaPreprocessor(),
        "row_spec": ParquetRowSpec(image_keys=["images", "image"]),
    },
    "latex_ocr": {
        "preprocessor": LatexocrPreprocessor(),
        "row_spec": ParquetRowSpec(image_keys=["images", "image"]),
    },
    "coco": {
        "preprocessor": CocoPreprocessor(),
        "row_spec": ParquetRowSpec(image_keys=["images", "image"]),
    },
}


class ParquetVQADataset(Dataset):
    """Parquet VL rows to the same dict format as ShareRobotVQADataset."""

    def __init__(
        self,
        parquet_paths: Sequence[str],
        source_id: str,
        *,
        image_root: str | Path | None = None,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        row_spec: ParquetRowSpec | None = None,
    ):
        if source_id not in PARQUET_VL_SOURCES and (preprocessor is None or row_spec is None):
            raise ValueError(f"Unknown parquet VL source_id: {source_id!r}")

        self.source_id = source_id
        self.image_root = image_root

        if preprocessor is None:
            preprocessor = PARQUET_VL_SOURCES[source_id]["preprocessor"]
        if row_spec is None:
            row_spec = PARQUET_VL_SOURCES[source_id]["row_spec"]

        self.preprocessor = preprocessor
        self.row_spec = row_spec

        expanded = expand_parquet_paths(list(parquet_paths))
        if not expanded:
            raise ValueError(f"No parquet files found from: {parquet_paths!r}")

        self.parquet_paths = expanded
        self._files: list[pq.ParquetFile] = []
        self._index: list[tuple[int, int, int]] = []

        for file_idx, path in enumerate(self.parquet_paths):
            pf = pq.ParquetFile(path)
            self._files.append(pf)
            for rg_idx in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg_idx).num_rows
                for row_in_rg in range(n):
                    self._index.append((file_idx, rg_idx, row_in_rg))

    def __len__(self) -> int:
        return len(self._index)

    def _read_row(self, file_idx: int, rg_idx: int, row_in_rg: int) -> dict[str, Any]:
        pf = self._files[file_idx]
        table = pf.read_row_group(rg_idx)
        return table.slice(row_in_rg, 1).to_pylist()[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        max_retry = min(32, len(self._index))
        n = len(self._index)

        for offset in range(max_retry):
            real_idx = (idx + offset) % n
            file_idx, rg_idx, row_in_rg = self._index[real_idx]

            try:
                row = self._read_row(file_idx, rg_idx, row_in_rg)
                row = self.preprocessor.preprocess(row)
                if row is None:
                    continue

                return row_to_vqa_sample(
                    row,
                    image_keys=self.row_spec.image_keys,
                    image_root=self.image_root,
                    target_num_images=self.row_spec.target_num_images,
                )
            except Exception:
                logging.getLogger(__name__).debug(
                    "ParquetVQADataset skip idx=%s offset=%s file=%s rg=%s row=%s",
                    idx,
                    offset,
                    self.parquet_paths[file_idx],
                    rg_idx,
                    row_in_rg,
                    exc_info=True,
                )
                continue

        raise RuntimeError(f"Failed to fetch a valid parquet VL sample after {max_retry} retries (idx={idx})")


def is_parquet_vl_group(group: Any) -> bool:
    return bool(getattr(group, "parquet_source_id", None))
