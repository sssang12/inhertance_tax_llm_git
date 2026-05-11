"""
국세법령정보시스템(txsi.hometax.go.kr) 예규·통칙·질의회신 수집기.

대상:
- 재산세제과 예규 (서면-상속증여, 서면4팀 등)
- 상속·증여세 통칙
- 법령해석 사례 (법령해석사례검색)
- 국세청 발간 「상속·증여세 안내」 PDF (별도 파이프라인에서 텍스트 추출)

본 모듈은 표준 검색 → 상세 페이지 파싱 구조의 예시 구현이며,
실제 셀렉터·파라미터는 사이트 개편에 맞춰 조정해야 한다.
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


SEARCH_QUERIES = [
    {"keyword": "상속세", "kind": "예규"},
    {"keyword": "증여세", "kind": "예규"},
    {"keyword": "가업상속공제", "kind": "예규"},
    {"keyword": "사전증여", "kind": "예규"},
    {"keyword": "비상장주식", "kind": "예규"},
    {"keyword": "상속세", "kind": "법령해석"},
    {"keyword": "증여세", "kind": "법령해석"},
]


@dataclass
class NTSRuling:
    doc_no: str          # 문서번호 (예: 서면-2023-상속증여-1234)
    issued_date: str
    kind: str            # 예규 / 통칙 / 법령해석
    tax_type: str        # 상속세 / 증여세
    title: str
    question: str        # 질의 내용
    answer: str          # 회신 내용
    related_laws: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NTSCrawler:
    """국세법령정보시스템 예규·법령해석 크롤러."""

    BASE = "https://txsi.hometax.go.kr"
    SEARCH_PATH = "/docs/customer/index.jsp"  # 실제 경로는 사이트에 맞게 조정

    def __init__(self, output_dir: str | Path = "data/raw/nts", delay: float = 1.5):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "inheritance-tax-llm/0.1 (research)"},
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
    def search(self, keyword: str, kind: str, page: int = 1) -> list[str]:
        html = self._get(
            self.BASE + self.SEARCH_PATH,
            params={"searchKeyword": keyword, "kind": kind, "page": page},
        )
        soup = BeautifulSoup(html, "lxml")
        urls = []
        for a in soup.select("table a[href*='view']"):
            href = a.get("href", "")
            urls.append(self.BASE + href if href.startswith("/") else href)
        return urls

    # ------------------------------------------------------------------
    def parse_detail(self, url: str, kind: str) -> NTSRuling | None:
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        def _text(sel: str) -> str:
            el = soup.select_one(sel)
            return el.get_text(" ", strip=True) if el else ""

        title = _text("h2") or _text(".board-title")
        body_el = soup.select_one(".board-view-content") or soup.select_one("article") or soup
        body = body_el.get_text("\n", strip=True)

        m_no = re.search(r"(서면[-\d가-힣]+|기획재정부[-\d]+|재산-\d+|상속증여-\d+)", body + " " + title)
        doc_no = m_no.group(0) if m_no else ""

        m_date = re.search(r"(20\d{2}[\.\-/]\s*\d{1,2}[\.\-/]\s*\d{1,2})", body[:500])
        issued_date = m_date.group(1) if m_date else ""

        # Q&A 분리 (예규/회신은 통상 [질의]/[회신] 구조)
        def _section(start_pat: str, end_pat: str) -> str:
            m = re.search(start_pat + r"([\s\S]+?)" + f"(?={end_pat}|\\Z)", body)
            return m.group(1).strip() if m else ""

        question = _section(r"(?:\[질의\]|【질의】|질의\s*요지|사실관계)", r"(?:\[회신\]|【회신】|회신\s*요지|【답변】)")
        answer = _section(r"(?:\[회신\]|【회신】|【답변】|회신\s*요지)", r"(?:관련법령|【관련법령】|\Z)")
        related = _section(r"(?:관련법령|【관련법령】)", r"\Z")

        tax_type = "상속세" if "상속세" in body[:1000] else ("증여세" if "증여세" in body[:1000] else "기타")

        if not (title or doc_no):
            return None

        return NTSRuling(
            doc_no=doc_no,
            issued_date=issued_date,
            kind=kind,
            tax_type=tax_type,
            title=title,
            question=question or body[:2000],
            answer=answer,
            related_laws=related,
            url=url,
        )

    # ------------------------------------------------------------------
    def run(self, queries: list[dict[str, str]] | None = None, max_pages: int = 20) -> Path:
        queries = queries or SEARCH_QUERIES
        out_path = self.output_dir / "nts.jsonl"
        seen: set[str] = set()
        n_written = 0

        with out_path.open("w", encoding="utf-8") as f:
            for q in queries:
                kw, kind = q["keyword"], q["kind"]
                logger.info(f"[{kind}] 검색어: {kw}")
                for page in range(1, max_pages + 1):
                    try:
                        urls = self.search(kw, kind=kind, page=page)
                    except Exception as e:
                        logger.warning(f"  검색 실패: {e}")
                        break
                    if not urls:
                        break
                    for url in urls:
                        if url in seen:
                            continue
                        seen.add(url)
                        try:
                            rec = self.parse_detail(url, kind=kind)
                        except Exception as e:
                            logger.warning(f"  상세 실패 {url}: {e}")
                            continue
                        if rec:
                            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                            n_written += 1
                        time.sleep(self.delay)
                    logger.info(f"  {kw}/{kind} p{page}: 누적 {n_written}건")
                    time.sleep(self.delay)
        logger.success(f"nts.jsonl 저장 완료: {n_written:,}건 → {out_path}")
        return out_path


if __name__ == "__main__":
    NTSCrawler().run()
