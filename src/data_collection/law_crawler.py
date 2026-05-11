"""
국가법령정보 공동활용 OpenAPI 기반 법령 수집기.

대상 법령:
- 상속세 및 증여세법 (법률)
- 상속세 및 증여세법 시행령 (대통령령)
- 상속세 및 증여세법 시행규칙 (기획재정부령)
- 관련 부속 법령 (국세기본법, 국세징수법 中 상속·증여 관련 조항)

API 문서: https://open.law.go.kr/LSO/openApi/guideList.do
- 본문 조회 URL: https://www.law.go.kr/DRF/lawService.do
- 목록 조회 URL: https://www.law.go.kr/DRF/lawSearch.do
- OC 파라미터에 ".env"의 LAW_API_KEY(이메일 아이디) 사용
- target=law (현행법령), type=JSON / XML 선택
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "")
BASE_LIST = "https://www.law.go.kr/DRF/lawSearch.do"
BASE_DETAIL = "https://www.law.go.kr/DRF/lawService.do"

# 수집 대상 법령명 (정확한 공식 명칭)
TARGET_LAWS = [
    "상속세 및 증여세법",
    "상속세 및 증여세법 시행령",
    "상속세 및 증여세법 시행규칙",
    "국세기본법",
    "국세기본법 시행령",
    "국세징수법",
    "조세특례제한법",  # 가업상속공제 등 관련
]


@dataclass
class LawArticle:
    """단일 조문 단위 학습 레코드."""

    law_name: str         # 예: 상속세 및 증여세법
    law_id: str           # MST(법령마스터) 번호
    article_no: str       # 예: 제13조
    article_title: str    # 예: 상속세 과세가액
    paragraph: str        # 예: ①, ② 항 단위 (없으면 빈 문자열)
    item: str             # 예: 1., 2. (없으면 빈 문자열)
    text: str             # 조문 본문
    effective_date: str   # 시행일 YYYYMMDD
    promulgation_no: str  # 공포번호

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LawCrawler:
    """국가법령정보 OpenAPI 클라이언트."""

    def __init__(self, api_key: str | None = None, *, output_dir: str | Path = "data/raw/laws"):
        self.api_key = api_key or LAW_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "LAW_API_KEY 미설정. https://open.law.go.kr 에서 OC(이메일 아이디)를 발급받아 .env 에 입력하세요."
            )
        self.client = httpx.Client(timeout=30.0, headers={"User-Agent": "inheritance-tax-llm/0.1"})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API 호출
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {"OC": self.api_key, "type": "JSON", **params}
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        # 일부 응답은 BOM 포함 — 안전하게 처리
        text = resp.text.lstrip("﻿")
        return json.loads(text)

    def search_law_id(self, law_name: str) -> str | None:
        """법령명으로 검색해 MST(마스터번호)를 얻는다."""
        data = self._get(BASE_LIST, {"target": "law", "query": law_name, "display": 20})
        rows = data.get("LawSearch", {}).get("law", [])
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            # 현행 법령 + 정확한 명칭 일치 우선
            if row.get("법령명한글", "").strip() == law_name and row.get("현행연혁코드") == "현행":
                return row.get("법령일련번호") or row.get("MST")
        # fallback: 첫 결과
        return rows[0].get("법령일련번호") if rows else None

    def fetch_law_detail(self, mst: str) -> dict[str, Any]:
        return self._get(BASE_DETAIL, {"target": "law", "MST": mst})

    # ------------------------------------------------------------------
    # 파싱
    # ------------------------------------------------------------------
    def parse_articles(self, detail: dict[str, Any]) -> list[LawArticle]:
        """API JSON 응답에서 조문 단위 레코드 추출."""
        law = detail.get("법령", {})
        basic = law.get("기본정보", {})
        law_name = basic.get("법령명_한글") or basic.get("법령명한글", "")
        law_id = str(basic.get("법령일련번호") or basic.get("MST", ""))
        effective_date = str(basic.get("시행일자", ""))
        promulgation_no = str(basic.get("공포번호", ""))

        articles_root = law.get("조문", {}).get("조문단위", [])
        if isinstance(articles_root, dict):
            articles_root = [articles_root]

        out: list[LawArticle] = []
        for art in articles_root:
            art_no = art.get("조문번호", "")
            art_branch = art.get("조문가지번호", "")
            full_no = f"제{art_no}조" + (f"의{art_branch}" if art_branch and art_branch != "0" else "")
            title = art.get("조문제목", "")
            body = (art.get("조문내용") or "").strip()

            # 조문 본문이 비어 있는 경우 항/호 단위로만 들어있을 수 있음 → 항을 평탄화
            paragraphs = art.get("항", [])
            if isinstance(paragraphs, dict):
                paragraphs = [paragraphs]

            if not paragraphs:
                if body:
                    out.append(
                        LawArticle(
                            law_name=law_name,
                            law_id=law_id,
                            article_no=full_no,
                            article_title=title,
                            paragraph="",
                            item="",
                            text=body,
                            effective_date=effective_date,
                            promulgation_no=promulgation_no,
                        )
                    )
                continue

            for para in paragraphs:
                p_no = para.get("항번호", "")
                p_text = (para.get("항내용") or "").strip()
                items = para.get("호", [])
                if isinstance(items, dict):
                    items = [items]
                if items:
                    for item in items:
                        i_no = item.get("호번호", "")
                        i_text = (item.get("호내용") or "").strip()
                        merged = f"{p_text}\n{i_text}".strip()
                        out.append(
                            LawArticle(
                                law_name=law_name,
                                law_id=law_id,
                                article_no=full_no,
                                article_title=title,
                                paragraph=p_no,
                                item=i_no,
                                text=merged,
                                effective_date=effective_date,
                                promulgation_no=promulgation_no,
                            )
                        )
                else:
                    out.append(
                        LawArticle(
                            law_name=law_name,
                            law_id=law_id,
                            article_no=full_no,
                            article_title=title,
                            paragraph=p_no,
                            item="",
                            text=p_text,
                            effective_date=effective_date,
                            promulgation_no=promulgation_no,
                        )
                    )
        return out

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    def run(self, laws: Iterable[str] | None = None) -> Path:
        laws = list(laws) if laws else TARGET_LAWS
        out_path = self.output_dir / "laws.jsonl"
        n_written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for name in laws:
                logger.info(f"법령 검색: {name}")
                mst = self.search_law_id(name)
                if not mst:
                    logger.warning(f"  → 검색 실패: {name}")
                    continue
                logger.info(f"  → MST={mst}, 본문 조회")
                detail = self.fetch_law_detail(mst)
                articles = self.parse_articles(detail)
                logger.info(f"  → 조문 {len(articles)}개 추출")
                for art in articles:
                    f.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")
                    n_written += 1
                time.sleep(1.0)  # 공동활용 OpenAPI 부하 배려
        logger.success(f"laws.jsonl 저장 완료: {n_written:,}개 레코드 → {out_path}")
        return out_path


if __name__ == "__main__":
    crawler = LawCrawler()
    crawler.run()
