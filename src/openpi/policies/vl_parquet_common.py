from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def normalize_image_cell(x: Any) -> Any:
    """
    Normalize a single image cell from parquet / HF-style tables into a decode-friendly value.

    Handles the common HuggingFace ``datasets.Image`` parquet encoding: a struct row mapped to
    ``{"bytes": ..., "path": ...}`` (prefer in-memory ``bytes`` when present, else non-empty ``path``).
    Any other value is returned unchanged for ``_to_rgb_uint8_image`` (str, ndarray, PIL, raw bytes).
    """
    if not isinstance(x, dict):
        return x
    b = x.get("bytes")
    if isinstance(b, (bytes, bytearray, memoryview)) and len(bytes(b)) > 0:
        return bytes(b)
    p = x.get("path")
    if isinstance(p, str) and p.strip():
        return p
    return x


def _to_rgb_uint8_image(x: Any, *, image_root: str | Path | None = None) -> np.ndarray:
    """Convert path/PIL/numpy/bytes-like image to RGB uint8 ndarray."""
    x = normalize_image_cell(x)

    if isinstance(x, str):
        p = Path(x).expanduser()
        if not p.is_absolute() and image_root is not None:
            p = Path(image_root) / p
        with Image.open(p) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unsupported numpy image shape: {arr.shape}")
        return arr

    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"), dtype=np.uint8)

    if isinstance(x, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(x))) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    if isinstance(x, dict):
        raise TypeError(
            "Image cell is a dict but not a usable HF-style Image struct "
            "(expected non-empty 'bytes' or 'path'); "
            f"keys={list(x.keys())!r}"
        )

    raise TypeError(f"Unsupported image type: {type(x)}")


def _extract_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    query = row.get("query", None)
    response = row.get("response", None)

    if query is not None and response is not None:
        if isinstance(response, (list, tuple)):
            response = response[0] if len(response) > 0 else ""
        return str(query), str(response)

    messages = row.get("messages", None)
    if messages is None:
        raise ValueError("Neither query/response nor messages found in row")

    user_text = None
    assistant_text = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        if role in ("user", "human"):
            user_text = content
        elif role in ("assistant", "gpt"):
            assistant_text = content

    if user_text is None or assistant_text is None:
        raise ValueError("Could not extract final user/assistant pair from messages")

    return str(user_text), str(assistant_text)


def _normalize_images(
    image_value: Any,
    *,
    image_root: str | Path | None = None,
    target_num_images: int = 3,
) -> list[np.ndarray]:
    if image_value is None:
        raise ValueError("image_value is None")

    if not isinstance(image_value, (list, tuple)):
        image_value = [image_value]

    images = [_to_rgb_uint8_image(img, image_root=image_root) for img in image_value]
    if len(images) == 0:
        raise ValueError("No images found")

    base_shape = images[0].shape
    pad_img = np.zeros(base_shape, dtype=np.uint8)

    if len(images) < target_num_images:
        images = images + [pad_img.copy() for _ in range(target_num_images - len(images))]
    elif len(images) > target_num_images:
        images = images[:target_num_images]

    return images


def row_to_vqa_sample(
    row: dict[str, Any],
    *,
    image_keys: list[str] | None = None,
    image_root: str | Path | None = None,
    target_num_images: int = 3,
) -> dict[str, Any]:
    """
    Convert a preprocessed parquet row into the training sample format:
    {
        "images": list[np.ndarray],
        "question": str,
        "answer": str,
    }
    """
    question, answer = _extract_question_answer(row)

    if image_keys is None:
        image_keys = ["images", "image"]

    image_value = None
    for key in image_keys:
        if key in row and row[key] is not None:
            image_value = row[key]
            break

    if image_value is None:
        raise ValueError(f"No image field found among keys: {image_keys}")

    images = _normalize_images(image_value, image_root=image_root, target_num_images=target_num_images)

    return {
        "images": images,
        "question": question,
        "answer": answer,
    }
