"""
대법원·고등법원 상속·증여세 판례 수집기.

대법원 종합법률정보(https://glaw.scourt.go.kr) 와 국가법령정보 OpenAPI
의 판례 API (target=prec) 를 동시에 활용한다.

OpenAPI 판례 조회:
- 목록: https://www.law.go.kr/DRF/lawSearch.do?target=prec
- 본문: https://www.law.go.kr/DRF/lawService.do?target=prec&ID=<판례일련번호>

판례 본문에는 보통 다음 섹션이 포함된다:
- 판시사항 (Holding summary)
- 판결요지 (Detailed holding)
- 참조조문 / 참조판례
- 이유
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "")
BASE_LIST = "https://www.law.go.kr/DRF/lawSearch.do"
BASE_DETAIL = "https://www.law.go.kr/DRF/lawService.do"

SEARCH_KEYWORDS = [
    "상속세",
    "증여세",
    "가업상속공제",
    "명의신탁 증여의제",
    "비상장주식 평가",
    "사전증여 합산과세",
    "상속재산 분할",
]


@dataclass
class Precedent:
    case_no: str           # 사건번호 (예: 대법원 2022두12345)
    court: str             # 법원
    decision_date: str
    case_name: str
    tax_type: str
    holding_summary: str   # 판시사항
    holding_detail: str    # 판결요지
    referenced_laws: str   # 참조조문
    referenced_cases: str  # 참조판례
    reasoning: str         # 이유 (요약)
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrecedentCrawler:
    """국가법령정보 OpenAPI 기반 판례 크롤러."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        output_dir: str | Path = "data/raw/precedents",
        delay: float = 1.0,
    ):
        self.api_key = api_key or LAW_API_KEY
        if not self.api_key:
            raise RuntimeError("LAW_API_KEY 미설정 — .env 의 LAW_API_KEY 를 설정하세요.")
        self.client = httpx.Client(timeout=30.0, headers={"User-Agent": "inheritance-tax-llm/0.1"})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {"OC": self.api_key, "type": "JSON", **params}
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        text = resp.text.lstrip("﻿")
        return json.loads(text)

    # ------------------------------------------------------------------
    def search(self, keyword: str, page: int = 1, display: int = 50) -> list[dict[str, Any]]:
        data = self._get(
            BASE_LIST,
            {"target": "prec", "query": keyword, "page": page, "display": display},
        )
        rows = data.get("PrecSearch", {}).get("prec", [])
        if isinstance(rows, dict):
            rows = [rows]
        return rows

    def fetch_detail(self, prec_id: str) -> dict[str, Any]:
        return self._get(BASE_DETAIL, {"target": "prec", "ID": prec_id})

    # ------------------------------------------------------------------
    def parse(self, detail: dict[str, Any]) -> Precedent | None:
        data = detail.get("PrecService", detail.get("판례", {}))
        if not data:
            return None

        def _g(key: str) -> str:
            return str(data.get(key, "")).strip()

        case_no = _g("사건번호")
        court = _g("법원명")
        date = _g("선고일자")
        case_name = _g("사건명")

        # 본문 텍스트
        body = data.get("판례내용") or data.get("판례본문") or ""
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False)

        def _section(label: str) -> str:
            m = re.search(rf"【{label}】\s*([\s\S]+?)(?=【|\Z)", body)
            return m.group(1).strip() if m else ""

        holding_summary = _section("판시사항")
        holding_detail = _section("판결요지")
        ref_laws = _section("참조조문")
        ref_cases = _section("참조판례")
        reasoning = _section("이유")

        tax_type = "상속세" if "상속세" in body else ("증여세" if "증여세" in body else "기타")

        if not case_no:
            return None

        return Precedent(
            case_no=case_no,
            court=court,
            decision_date=date,
            case_name=case_name,
            tax_type=tax_type,
            holding_summary=holding_summary,
            holding_detail=holding_detail,
            referenced_laws=ref_laws,
            referenced_cases=ref_cases,
            reasoning=reasoning[:8000],  # 본문이 매우 길 수 있어 트림
            url=f"https://www.law.go.kr/DRF/lawService.do?target=prec&ID={data.get('판례일련번호','')}",
        )

    # ------------------------------------------------------------------
    def run(self, keywords: list[str] | None = None, max_pages: int = 10) -> Path:
        keywords = keywords or SEARCH_KEYWORDS
        out_path = self.output_dir / "precedents.jsonl"
        seen_ids: set[str] = set()
        n_written = 0

        with out_path.open("w", encoding="utf-8") as f:
            for kw in keywords:
                logger.info(f"판례 검색: {kw}")
                for page in range(1, max_pages + 1):
                    try:
                        rows = self.search(kw, page=page)
                    except Exception as e:
                        logger.warning(f"  검색 실패 p{page}: {e}")
                        break
                    if not rows:
                        break
                    for row in rows:
                        prec_id = str(row.get("판례일련번호") or row.get("판례번호") or "")
                        if not prec_id or prec_id in seen_ids:
                            continue
                        seen_ids.add(prec_id)
                        try:
                            detail = self.fetch_detail(prec_id)
                            rec = self.parse(detail)
                        except Exception as e:
                            logger.warning(f"  본문 실패 ID={prec_id}: {e}")
                            continue
                        if rec:
                            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                            n_written += 1
                        time.sleep(self.delay)
                    logger.info(f"  {kw} p{page}: 누적 {n_written}건")
        logger.success(f"precedents.jsonl 저장 완료: {n_written:,}건 → {out_path}")
        return out_path


if __name__ == "__main__":
    PrecedentCrawler().run()
