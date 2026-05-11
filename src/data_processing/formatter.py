"""
정제된 도메인 데이터를 instruction 포맷(JSONL: instruction / input / output)으로 변환.

- 법령 조문 → "○○법 ○○조의 내용을 설명해줘" 형태의 단순 명령형 페어
- 판례 → "판시사항 / 판결요지" 요약형 페어
- 예규 → 질의-회신 페어 (이미 Q&A 구조)

이 단계의 출력물은 합성 Q&A 생성기의 입력으로도 사용된다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from loguru import logger


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --- 변환기들 ---
def format_law_article(rec: dict) -> list[dict]:
    """조문 → 다양한 instruction 형태."""
    if not rec.get("text"):
        return []
    law_name = rec["law_name"]
    art = rec["article_no"]
    title = rec.get("article_title", "")
    body = rec["text"].strip()

    templates = [
        ("{law} {art}({title})의 내용을 정확히 알려줘.",
         "{law} {art}({title})\n\n{body}"),
        ("{law} {art}에 따르면 어떤 내용이 규정되어 있어?",
         "{law} {art}({title})은(는) 다음과 같이 규정합니다.\n\n{body}"),
        ("{title}에 관한 법령 조문을 알려줘. 어떤 법에 어떤 조항인지도 함께.",
         "{law} {art}({title})\n\n{body}"),
    ]
    out = []
    for instr, ans in templates:
        out.append(
            {
                "instruction": instr.format(law=law_name, art=art, title=title or "표제 없음"),
                "input": "",
                "output": ans.format(law=law_name, art=art, title=title, body=body),
                "source": "law_article",
                "law_name": law_name,
                "article_no": art,
            }
        )
    return out


def format_precedent(rec: dict) -> list[dict]:
    if not (rec.get("holding_summary") or rec.get("holding_detail")):
        return []
    case_no = rec["case_no"]
    out = []
    if rec.get("holding_summary"):
        out.append(
            {
                "instruction": f"{case_no} 판례의 판시사항을 요약해줘.",
                "input": rec.get("case_name", ""),
                "output": rec["holding_summary"],
                "source": "precedent_summary",
                "case_no": case_no,
            }
        )
    if rec.get("holding_detail"):
        out.append(
            {
                "instruction": f"{case_no} 판례의 판결요지를 알려줘.",
                "input": rec.get("case_name", ""),
                "output": rec["holding_detail"],
                "source": "precedent_detail",
                "case_no": case_no,
            }
        )
    return out


def format_nts_ruling(rec: dict) -> list[dict]:
    if not (rec.get("question") and rec.get("answer")):
        return []
    return [
        {
            "instruction": rec["question"][:1500],
            "input": "",
            "output": rec["answer"],
            "source": "nts_ruling",
            "doc_no": rec.get("doc_no", ""),
        }
    ]


def format_tribunal(rec: dict) -> list[dict]:
    if not rec.get("holding"):
        return []
    case_no = rec["case_no"]
    out = []
    if rec.get("summary"):
        out.append(
            {
                "instruction": f"조세심판원 {case_no} 사건의 결정요지를 알려줘.",
                "input": rec.get("facts", "")[:1500],
                "output": rec["summary"],
                "source": "tribunal_summary",
                "case_no": case_no,
            }
        )
    out.append(
        {
            "instruction": f"조세심판원 {case_no} 사건의 본안판단을 설명해줘.",
            "input": rec.get("facts", "")[:1500],
            "output": rec["holding"],
            "source": "tribunal_holding",
            "case_no": case_no,
        }
    )
    return out


CONVERTERS = {
    "laws": format_law_article,
    "precedents": format_precedent,
    "nts": format_nts_ruling,
    "tribunal": format_tribunal,
}


def convert(processed_dir: Path, out_path: Path, seed: int = 42) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for key, fn in CONVERTERS.items():
            for src in processed_dir.rglob(f"{key}*.jsonl"):
                logger.info(f"포맷 변환: {src}")
                for rec in iter_jsonl(src):
                    for pair in fn(rec):
                        fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        n += 1
    logger.success(f"instruction pairs: {n:,} → {out_path}")
    return n


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--out", default="data/processed/sft_base.jsonl")
    args = p.parse_args()
    convert(Path(args.processed_dir), Path(args.out))
