import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def expand_vl_path_to_json_files(path: str) -> list[str]:
    """Resolve a VL JSON location: file -> [file]; directory -> all `*.json` (sorted) as one pool."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_file():
        if p.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {path}")
        return [str(p.resolve())]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise ValueError(f"No .json files found under directory: {path}")
        return [str(f.resolve()) for f in files]
    raise ValueError(f"Not a file or directory: {path}")


class ShareRobotVQADataset(Dataset):
    """Dataset for VL/VQA co-training from ShareRobot-style JSON."""

    def __init__(
        self,
        json_path: str | None = None,
        *,
        json_paths: Sequence[str] | None = None,
        image_root: str | None = None,
    ):
        if json_path is not None and json_paths is not None:
            raise ValueError("Pass only one of `json_path` or `json_paths`.")

        if json_paths is not None:
            paths = list(json_paths)
            if not paths:
                raise ValueError("`json_paths` must be non-empty.")
        elif json_path is not None:
            paths = expand_vl_path_to_json_files(json_path)
        else:
            raise ValueError("Provide `json_path` or `json_paths`.")

        self.json_path = paths[0] if len(paths) == 1 else None
        self.json_paths = tuple(paths)
        self.image_root = image_root

        merged: list = []
        for jp in paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected top-level JSON list in {jp}, got {type(data)}")
            merged.extend(data)

        self.data = merged

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return str(path)

        if self.image_root is None:
            return str(path)

        return str(Path(self.image_root) / path)

    def _load_image(self, path_str: str) -> np.ndarray:
        full_path = self._resolve_image_path(path_str)
        with Image.open(full_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def _extract_question_answer(self, conversations) -> tuple[str, str]:
        if not isinstance(conversations, list):
            raise ValueError("`conversations` must be a list")

        question = None
        answer = None

        for turn in conversations:
            if not isinstance(turn, dict):
                continue

            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "human" and question is None:
                question = value
            elif role == "gpt" and answer is None:
                answer = value

        if question is None:
            raise ValueError("No human question found in conversations")
        if answer is None:
            raise ValueError("No gpt answer found in conversations")

        return question, answer

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        image_paths = item.get("images", None)
        if image_paths is None:
            image_paths = item.get("image", None)

        if image_paths is None:
            raise ValueError("Sample must contain `image` or `images` field")

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if not isinstance(image_paths, list) or len(image_paths) == 0:
            raise ValueError("`image` / `images` must be a non-empty string or list")

        images = [self._load_image(p) for p in image_paths]

        conversations = item.get("conversations", None)
        if conversations is None:
            raise ValueError("Sample must contain `conversations` field")

        question, answer = self._extract_question_answer(conversations)

        return {
            "images": images,
            "question": question,
            "answer": answer,
        }