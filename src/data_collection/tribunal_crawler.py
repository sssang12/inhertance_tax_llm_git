"""
조세심판원(tt.go.kr) 심판결정례 수집기.

조세심판원 공개 결정문 중 상속세·증여세 관련 사건을 수집.
서비스 정책상 사이트 robots.txt 와 이용약관을 반드시 확인할 것.
HTML 구조는 사이트 개편에 따라 변할 수 있으므로 셀렉터를 환경에 맞게 조정해야 함.

본 모듈은 다음 두 가지 모드를 지원:
  1) 공식 OpenAPI 가 발급된 경우 (권장)
  2) 공개 검색 결과 HTML 파싱 (보조)

수집 필드:
- 사건번호 (예: 조심2023서0000)
- 결정일
- 청구취지 / 처분개요
- 본안판단 / 결정요지
- 관련 법령
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


SEARCH_KEYWORDS = [
    "상속세",
    "증여세",
    "가업상속공제",
    "사전증여",
    "명의신탁 증여의제",
    "비상장주식 평가",
    "꼬마빌딩 감정평가",
]


@dataclass
class TribunalDecision:
    case_no: str
    decision_date: str
    tax_type: str         # 상속세 | 증여세 등
    title: str
    summary: str          # 결정요지
    facts: str            # 처분개요 / 사실관계
    holding: str          # 본안판단
    related_laws: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TribunalCrawler:
    """조세심판원 결정례 크롤러 (HTML 모드)."""

    BASE = "https://www.tt.go.kr"
    SEARCH_PATH = "/main/sub05_01_01.jsp"  # 결정문 검색 페이지 (참고용 — 실제 동작은 사이트 구조에 맞춰 조정)

    def __init__(self, output_dir: str | Path = "data/raw/tribunal", delay: float = 1.5):
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; inheritance-tax-llm/0.1; "
                    "+research, respects robots.txt)"
                )
            },
            follow_redirects=True,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _get(self, url: str, params: dict[str, Any] | None = None) -> str:
        resp = self.client.get(url, params=params or {})
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------
    # 1) 검색 → 상세 URL 목록 수집
    # ------------------------------------------------------------------
    def search(self, keyword: str, page: int = 1) -> list[str]:
        """검색 결과 페이지에서 결정문 상세 URL 리스트 추출.

        주의: 실제 셀렉터는 사이트 구조에 맞게 조정해야 한다.
        아래는 일반적인 한국 정부기관 사이트의 표 기반 결과 페이지를 가정한 예시 구현.
        """
        html = self._get(
            self.BASE + self.SEARCH_PATH,
            params={"searchKeyword": keyword, "page": page, "searchKind": "01"},  # 01: 결정문
        )
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.select("table.board-list a[href]"):
            href = a.get("href", "")
            if "view" in href or "detail" in href:
                links.append(self.BASE + href if href.startswith("/") else href)
        return links

    # ------------------------------------------------------------------
    # 2) 상세 페이지 파싱
    # ------------------------------------------------------------------
    def parse_detail(self, url: str) -> TribunalDecision | None:
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        def _text(selector: str) -> str:
            el = soup.select_one(selector)
            return el.get_text(" ", strip=True) if el else ""

        case_no = _text(".case-no") or _text("dl.case dd")
        title = _text("h2.title") or _text(".board-view-title")
        decision_date = _text(".decision-date")
        body = soup.select_one(".board-view-content") or soup
        full_text = body.get_text("\n", strip=True)

        # 본문에서 구조화된 섹션 추출 (휴리스틱)
        def _section(pattern: str) -> str:
            m = re.search(pattern + r"[\s\S]+?(?=(?:【|■|◆|◇|\Z))", full_text)
            return m.group(0).strip() if m else ""

        facts = _section(r"(?:처분개요|사실관계|【사건개요】)")
        holding = _section(r"(?:본안판단|판단|【본안판단】)")
        summary = _section(r"(?:결정요지|요지|【주\s*문】|【결정요지】)")
        related = _section(r"(?:관련법령|관련규정|【관련법령】)")

        tax_type = "상속세" if "상속세" in full_text else ("증여세" if "증여세" in full_text else "기타")
        if not case_no:
            m = re.search(r"조심\d{4}[가-힣]{1,3}\d+", full_text)
            case_no = m.group(0) if m else ""

        if not case_no:
            return None

        return TribunalDecision(
            case_no=case_no,
            decision_date=decision_date,
            tax_type=tax_type,
            title=title,
            summary=summary,
            facts=facts,
            holding=holding,
            related_laws=related,
            url=url,
        )

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    def run(self, keywords: list[str] | None = None, max_pages: int = 10) -> Path:
        keywords = keywords or SEARCH_KEYWORDS
        out_path = self.output_dir / "tribunal.jsonl"
        seen: set[str] = set()
        n_written = 0

        with out_path.open("w", encoding="utf-8") as f:
            for kw in keywords:
                logger.info(f"검색어: {kw}")
                for page in range(1, max_pages + 1):
                    try:
                        urls = self.search(kw, page=page)
                    except Exception as e:
                        logger.warning(f"  검색 실패 (page={page}): {e}")
                        break
                    if not urls:
                        break
                    for url in urls:
                        if url in seen:
                            continue
                        seen.add(url)
                        try:
                            record = self.parse_detail(url)
                        except Exception as e:
                            logger.warning(f"  상세 파싱 실패: {url} ({e})")
                            continue
                        if record:
                            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                            n_written += 1
                        time.sleep(self.delay)
                    logger.info(f"  {kw} p{page}: 누적 {n_written}건")
                    time.sleep(self.delay)
        logger.success(f"tribunal.jsonl 저장 완료: {n_written:,}건 → {out_path}")
        return out_path


if __name__ == "__main__":
    TribunalCrawler().run()
