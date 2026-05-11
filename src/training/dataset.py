"""
학습 데이터셋 로더 + 분할.

여러 JSONL 파일(법령 기반 pairs, 합성 pairs, 외부 한국어 instruction)을 합쳐
train / eval 로 분할한 HF Dataset 객체를 반환한다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from datasets import Dataset, concatenate_datasets
from loguru import logger


def load_jsonl(paths: Iterable[str | Path]) -> Dataset:
    rows: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"건너뜀(없음): {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "instruction" not in row or "output" not in row:
                    continue
                rows.append(
                    {
                        "instruction": str(row["instruction"]),
                        "input": str(row.get("input", "")),
                        "output": str(row["output"]),
                        "source": str(row.get("source", "unknown")),
                    }
                )
    logger.info(f"불러온 instruction 페어: {len(rows):,}")
    return Dataset.from_list(rows)


def build_train_eval(
    train_files: list[str],
    eval_files: list[str] | None = None,
    *,
    eval_ratio: float = 0.02,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    train = load_jsonl(train_files)
    if max_samples and len(train) > max_samples:
        train = train.shuffle(seed=seed).select(range(max_samples))

    if eval_files:
        eval_ds = load_jsonl(eval_files)
        return train, eval_ds

    split = train.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]
