from __future__ import annotations

import ast
import json
from typing import Any, Optional


def history_to_messages(history: list, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    for q, r in history:
        messages.append({"role": "user", "content": str(q)})
        messages.append({"role": "assistant", "content": str(r)})
    return messages


class ResponsePreprocessor:
    """Normalize common VL parquet columns to query/response/messages."""

    def __init__(self, *, columns: Optional[dict[str, str]] = None):
        self.columns = columns or {}

        system_keys = ["system", "system_prompt"]
        query_keys = ["query", "prompt", "input", "instruction", "question", "problem"]
        response_keys = [
            "response",
            "answer",
            "output",
            "targets",
            "target",
            "answer_key",
            "answers",
            "solution",
            "text",
            "completion",
            "content",
        ]

        for key in system_keys:
            self.columns[key] = "system"
        for key in query_keys:
            self.columns[key] = "query"
        for key in response_keys:
            self.columns[key] = "response"

    def _rename_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in row.items():
            out[self.columns.get(k, k)] = v
        return out

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        row = self._rename_columns(dict(row))

        response = row.pop("response", None)
        if isinstance(response, (list, tuple)):
            response = response[0] if len(response) > 0 else ""

        history = row.pop("history", None) or []
        query = row.pop("query", None)
        system = row.pop("system", None)

        if isinstance(history, str):
            history = ast.literal_eval(history)

        if query is not None and response is not None:
            history = list(history)
            history.append([query, response])

        row["messages"] = history_to_messages(history, system)
        if query is not None:
            row["query"] = query
        if response is not None:
            row["response"] = response
        return row


class AOkvqaPreprocessor(ResponsePreprocessor):
    """
    A-OKVQA (HuggingFaceM4/A-OKVQA, swift/A-OKVQA export): ``question`` + ``image``;
    ``rationales`` (often a list of strings) → supervised ``response`` text.
    """

    def __init__(self) -> None:
        super().__init__(columns={"rationales": "response"})

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        row = dict(row)
        r = row.get("rationales")
        if isinstance(r, (list, tuple)) and len(r) > 1:
            parts = [str(x).strip() for x in r if str(x).strip()]
            row["rationales"] = " ".join(parts) if parts else ""
        return super().preprocess(row)


class LatexocrPreprocessor(ResponsePreprocessor):
    """LaTeX OCR parquet: fixed instruction; label from text/latex/formula columns."""

    def __init__(self) -> None:
        super().__init__(
            columns={
                "latex": "response",
                "formula": "response",
                "ground_truth": "response",
            },
        )

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        row = self._rename_columns(dict(row))
        row["query"] = "Using LaTeX to perform OCR on the image."
        return super().preprocess(row)


class CocoPreprocessor(ResponsePreprocessor):
    """COCO-style detection parquet: fixed query; response = category + bbox lines (supervisable text)."""

    category = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def preprocess(self, row: dict[str, Any]) -> dict[str, Any] | None:
        row = dict(row)
        row["query"] = "Task: Object Detection"

        raw_objects = row.get("objects")
        if raw_objects is None:
            return None
        if isinstance(raw_objects, str):
            try:
                objects = json.loads(raw_objects)
            except json.JSONDecodeError:
                return None
        elif isinstance(raw_objects, dict):
            objects = raw_objects
        else:
            return None

        cat_ids = objects.get("category", []) or []
        bboxes = objects.get("bbox", []) or []
        if not cat_ids or not bboxes:
            return None

        parts: list[str] = []
        for cat_id, bbox in zip(cat_ids, bboxes, strict=False):
            try:
                idx = int(cat_id)
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(self.category)):
                continue
            cat_name = self.category[idx]
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            parts.append(f"{cat_name}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

        if not parts:
            return None

        row["response"] = "\n".join(parts)
        return super().preprocess(row)
