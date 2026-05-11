"""
정제된 문서를 토큰 친화적인 청크로 분할.

- 법령 조문: 조문 단위가 이미 적절한 청크 크기이므로 그대로 사용
- 판례/예규: 본문이 길 수 있어 의미 단위로 분할 (KSS sentence splitter)
- 최대 청크 길이: tokenizer 기준 1,536 토큰 (기본값, instruction 포맷 시 여유 확보)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from loguru import logger

try:
    import kss

    _HAS_KSS = True
except Exception:
    _HAS_KSS = False


def _split_sentences(text: str) -> list[str]:
    if _HAS_KSS:
        return [s.strip() for s in kss.split_sentences(text) if s.strip()]
    # fallback: 마침표/줄바꿈 기반
    return [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]


def chunk_text(
    text: str,
    *,
    max_chars: int = 2400,
    overlap_chars: int = 200,
) -> list[str]:
    """문자 수 기반 슬라이딩 청크 (한국어 평균 1토큰 ≈ 1.6자 가정)."""
    if len(text) <= max_chars:
        return [text]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if buf_len + len(sent) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            # overlap
            tail = []
            tail_len = 0
            for s in reversed(buf):
                if tail_len + len(s) > overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            buf = tail[:]
            buf_len = tail_len
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_jsonl(in_path: Path, out_path: Path, text_fields: list[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            for field in text_fields:
                if field not in rec or not rec[field]:
                    continue
                pieces = chunk_text(rec[field])
                if len(pieces) == 1:
                    continue
                for i, p in enumerate(pieces):
                    new_rec = {**rec, field: p, "_chunk_index": i, "_chunk_field": field}
                    fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    n_out += 1
                break  # 한 필드만 청킹 (가장 긴 본문)
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    logger.info(f"{in_path.name}: {n_in:,} → {n_out:,}")
    return n_out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--fields", nargs="+", default=["text", "holding_detail", "answer", "reasoning"])
    args = p.parse_args()
    chunk_jsonl(Path(args.in_path), Path(args.out_path), args.fields)
