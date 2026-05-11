"""
원본 크롤링 데이터(JSONL)에 대한 정제기.

기능:
- 한국어 본문에 자주 끼는 깨진 문자, HTML 잔재, 다중 공백 정리
- 개인정보(주민번호, 전화번호, 이메일, 사업자번호) 마스킹
- 너무 짧거나 의미 없는 레코드 제거
- 중복 제거 (본문 해시 기반)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

from loguru import logger


# --- 정규표현식 ---
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MULTISPACE = re.compile(r"[ \t]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")
RE_RRN = re.compile(r"\b\d{6}[-\s]?\d{7}\b")           # 주민등록번호
RE_PHONE = re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
RE_BIZ = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")          # 사업자등록번호
RE_BANK = re.compile(r"\b\d{2,6}-\d{2,6}-\d{2,8}\b")   # 계좌번호 휴리스틱
RE_CITATION_BRACKETS = re.compile(r"\[[^\]]{0,3}\]")    # [1], [편집] 등 노이즈
RE_CIRCLED_DIGITS = re.compile(r"[①-⑳]")
CIRCLED_MAP = {chr(c): f"{i+1}." for i, c in enumerate(range(0x2460, 0x2474))}


def clean_text(text: str) -> str:
    """기본 정제."""
    if not text:
        return ""
    text = RE_HTML_TAG.sub(" ", text)
    text = text.replace("​", "").replace("﻿", "")
    text = RE_CITATION_BRACKETS.sub("", text)
    text = "".join(CIRCLED_MAP.get(ch, ch) for ch in text)
    text = RE_MULTISPACE.sub(" ", text)
    text = RE_MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def mask_pii(text: str) -> str:
    """개인정보 마스킹."""
    text = RE_RRN.sub("[주민번호]", text)
    text = RE_PHONE.sub("[전화번호]", text)
    text = RE_EMAIL.sub("[이메일]", text)
    text = RE_BIZ.sub("[사업자번호]", text)
    text = RE_BANK.sub("[계좌번호]", text)
    # 이름 익명화는 너무 공격적이라 비활성화. 필요 시 KoNLPy/NER 기반 모듈 추가.
    return text


def is_meaningful(record: dict, min_chars: int = 30) -> bool:
    body_fields = ("text", "summary", "answer", "holding_detail", "reasoning")
    longest = max((len((record.get(k) or "")) for k in body_fields), default=0)
    return longest >= min_chars


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_record(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, str):
            out[k] = mask_pii(clean_text(v))
        else:
            out[k] = v
    return out


def dedupe_key(rec: dict) -> str:
    """본문 해시 기반 중복 키."""
    body = " ".join(
        str(rec.get(k, ""))
        for k in ("text", "summary", "holding_detail", "answer", "reasoning")
    )
    body = re.sub(r"\s+", "", body)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def process_file(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_in = n_out = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(in_path):
            n_in += 1
            rec = clean_record(rec)
            if not is_meaningful(rec):
                continue
            key = dedupe_key(rec)
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    logger.info(f"{in_path.name}: {n_in:,} → {n_out:,} ({100*n_out/max(n_in,1):.1f}%)")
    return n_out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    for src in raw.rglob("*.jsonl"):
        dst = out / src.relative_to(raw)
        process_file(src, dst)
