"""
교사 모델(GPT-4o / Claude)로 도메인 instruction 데이터를 합성한다.

전략:
1) Seed = 정제된 법령 조문, 판례, 예규
2) 각 seed 에 대해 다음 카테고리의 Q&A 를 생성:
   - 직접 질의 (해당 조문/판례에 대한 단순 설명)
   - 사례형 (특정 사실관계를 가정한 적용 판단)
   - 비교형 (유사 조항/판례와의 차이점)
   - 절차형 (신고·납부·공제 신청 절차)
   - 오개념 교정형 (흔히 잘못 알려진 상식 교정)
3) Self-instruct 스타일이 아니라 grounded synthesis — 출처를 함께 제공하여 환각 최소화
4) 라이선스: 학습된 모델 가중치 자체에는 교사 출력이 녹아들어가지만,
   생성된 데이터셋의 재배포는 교사 API 약관(예: OpenAI ToS, Anthropic ToS)을 반드시 확인할 것.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# ------------------------------------------------------------------
# 프롬프트
# ------------------------------------------------------------------
SYSTEM_PROMPT = """당신은 대한민국 상속세 및 증여세 분야의 전문 세무사이자 법률 자문가입니다.
당신의 임무는 주어진 [출처]에 근거해 학습용 질문-답변 페어를 생성하는 것입니다.

엄격한 규칙:
1. 반드시 [출처] 내용에만 근거해 답변하세요. 출처에 없는 사실이나 수치를 만들어내지 마세요.
2. 한 응답에 1~3개의 다양한 Q&A 페어를 생성합니다.
3. 각 Q&A 는 한국어로 작성하고, 실제 납세자나 세무사가 물을 법한 자연스러운 표현을 사용합니다.
4. 답변은 결론을 먼저 제시한 뒤, 근거 조문/판례를 명시하고, 마지막에 실무상 주의점을 덧붙입니다.
5. 답변 끝에 "본 정보는 일반적인 안내이며, 구체적인 사안은 세무 전문가와 상담하세요." 라는 면책 문구는 첨부하지 마세요. (별도 단계에서 일괄 부착)
6. 결과는 반드시 아래 JSON 형식의 배열로만 출력하세요. 다른 텍스트는 금지합니다.

출력 형식:
[
  {"category": "직접질의|사례형|비교형|절차형|오개념교정", "question": "...", "answer": "..."},
  ...
]
"""

USER_TEMPLATE = """[출처]
{source_label}: {source_text}

위 [출처]에 근거해 {n}개의 학습용 Q&A 페어를 생성하세요.
가능한 한 다양한 category 를 포함하세요."""


# ------------------------------------------------------------------
# 교사 모델 어댑터
# ------------------------------------------------------------------
class TeacherClient:
    """OpenAI / Anthropic 어댑터."""

    def __init__(self, provider: str = "openai", model: str | None = None):
        self.provider = provider
        if provider == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        elif provider == "anthropic":
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-haiku-4-5-20251001"
        else:
            raise ValueError(provider)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def generate(self, source_label: str, source_text: str, n: int = 3) -> list[dict]:
        user = USER_TEMPLATE.format(source_label=source_label, source_text=source_text[:6000], n=n)
        if self.provider == "openai":
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.6,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n응답은 JSON 객체로 감싸고 'pairs' 키에 배열을 넣으세요."},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content
        else:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.6,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text

        return self._parse_pairs(text)

    @staticmethod
    def _parse_pairs(text: str) -> list[dict]:
        text = text.strip()
        # JSON 객체 또는 배열 둘 다 허용
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 코드블록 제거 시도
            text = text.strip("`").lstrip("json").strip()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return []
        if isinstance(data, dict):
            data = data.get("pairs") or data.get("data") or []
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not item.get("question") or not item.get("answer"):
                continue
            out.append(
                {
                    "instruction": item["question"].strip(),
                    "input": "",
                    "output": item["answer"].strip(),
                    "category": item.get("category", "직접질의"),
                }
            )
        return out


# ------------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------------
def _iter_seed(seed_path: Path):
    with seed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _seed_to_source(rec: dict) -> tuple[str, str] | None:
    src = rec.get("source", "")
    if src == "law_article":
        label = f"{rec.get('law_name','')} {rec.get('article_no','')}"
        return label, rec["output"]
    if src in ("precedent_summary", "precedent_detail"):
        return f"판례 {rec.get('case_no','')}", rec["output"]
    if src == "nts_ruling":
        return f"국세청 예규 {rec.get('doc_no','')}", f"질의: {rec['instruction']}\n회신: {rec['output']}"
    if src in ("tribunal_summary", "tribunal_holding"):
        return f"조세심판원 {rec.get('case_no','')}", rec["output"]
    return None


async def run(
    seed_path: Path,
    out_path: Path,
    *,
    provider: str = "openai",
    model: str | None = None,
    max_concurrency: int = 4,
    pairs_per_seed: int = 3,
    max_seeds: int | None = None,
    seed: int = 42,
):
    teacher = TeacherClient(provider=provider, model=model)
    seeds = [s for s in _iter_seed(seed_path) if _seed_to_source(s)]
    rng = random.Random(seed)
    rng.shuffle(seeds)
    if max_seeds:
        seeds = seeds[:max_seeds]
    logger.info(f"합성 대상 seed: {len(seeds):,}, provider={provider}, model={teacher.model}")

    sem = asyncio.Semaphore(max_concurrency)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    fout = out_path.open("w", encoding="utf-8")
    lock = asyncio.Lock()

    async def _worker(rec: dict):
        nonlocal n_written
        label, text = _seed_to_source(rec)
        async with sem:
            try:
                pairs = await teacher.generate(label, text, n=pairs_per_seed)
            except Exception as e:
                logger.warning(f"생성 실패 ({label}): {e}")
                return
            async with lock:
                for p in pairs:
                    p["source"] = "synthetic"
                    p["seed_label"] = label
                    fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                    n_written += 1
                if n_written % 200 == 0:
                    logger.info(f"  누적 {n_written:,}개 생성")

    await asyncio.gather(*[_worker(s) for s in seeds])
    fout.close()
    logger.success(f"합성 instruction {n_written:,}개 → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="data/processed/sft_base.jsonl")
    p.add_argument("--out", default="data/synthetic/sft_synth.jsonl")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--pairs-per-seed", type=int, default=3)
    p.add_argument("--max-seeds", type=int, default=None)
    args = p.parse_args()

    asyncio.run(
        run(
            Path(args.seed),
            Path(args.out),
            provider=args.provider,
            model=args.model,
            max_concurrency=args.concurrency,
            pairs_per_seed=args.pairs_per_seed,
            max_seeds=args.max_seeds,
        )
    )
