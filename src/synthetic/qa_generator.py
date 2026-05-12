"""
교사 모델(Claude / GPT-4o)로 상속·증여세 합성 Q&A 데이터 생성.

두 가지 방식:
1. 판례 시드 기반 — sft_base.jsonl 판례를 근거로 Q&A 생성 (환각 최소화)
2. 시나리오 기반 — 실제 납세자 상황을 설정한 절세 전략 Q&A 생성

사용 예:
  # 판례 시드 기반 (500건)
  python -m src.synthetic.qa_generator --mode seed --max-seeds 500

  # 시나리오 기반 (200건)
  python -m src.synthetic.qa_generator --mode scenario --n-scenarios 200

  # 둘 다
  python -m src.synthetic.qa_generator --mode both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# ------------------------------------------------------------------
# 시스템 프롬프트
# ------------------------------------------------------------------
SEED_SYSTEM = """당신은 대한민국 상속세 및 증여세 분야의 전문 세무사입니다.
주어진 [판례]에 근거해 학습용 Q&A 페어를 생성합니다.

규칙:
1. 반드시 [판례] 내용에만 근거하세요. 없는 사실을 만들지 마세요.
2. 실제 납세자나 세무사가 물을 법한 자연스러운 질문을 만드세요.
3. 답변은 결론 → 근거 판례 → 실무 주의점 순서로 작성하세요.
4. 다양한 유형(직접질의, 사례형, 절차형, 오개념교정)을 포함하세요.
5. 반드시 아래 JSON 배열 형식으로만 출력하세요.

출력 형식:
[
  {"question": "질문 내용", "answer": "답변 내용", "category": "직접질의|사례형|절차형|오개념교정"},
  ...
]"""

SCENARIO_SYSTEM = """당신은 대한민국 상속세 및 증여세 분야의 전문 세무사입니다.
주어진 [상속·증여 시나리오]에 대해 구체적이고 실용적인 절세 전략을 답변하세요.

답변 형식:
1. 현황 분석 (과세표준, 예상 세액)
2. 절세 전략 (공제 항목, 분할 방법, 사전 증여 등)
3. 주의사항 및 신고 절차
4. 관련 조문 또는 판례 참조

반드시 아래 JSON 형식으로만 출력하세요:
{"question": "질문", "answer": "상세 답변", "category": "절세전략"}"""

# ------------------------------------------------------------------
# 시나리오 템플릿 (다양한 상속·증여 상황)
# ------------------------------------------------------------------
SCENARIO_TEMPLATES = [
    # 상속세 절세
    "피상속인 {age}세, 배우자와 자녀 {children}명. 부동산 {estate}억, 금융자산 {finance}억, 채무 {debt}억. 상속세 절세 방법은?",
    "피상속인 사망, 상속재산 총 {total}억. 배우자 단독 상속 vs 자녀 포함 분할 시 세금 차이는?",
    "상속재산 {total}억 중 부동산 비중 {ratio}%. 부동산 물납 신청 가능 여부와 절차는?",
    "피상속인이 사망 전 {years}년 이내 자녀에게 {gift}억 증여. 상속세 계산 시 합산 여부는?",
    "상속재산 {total}억, 상속인 배우자 1명 + 자녀 {children}명. 배우자 상속공제 최대 활용 방법은?",
    "중소기업 대표 사망, 가업 주식 {stock}억. 가업상속공제 요건과 적용 방법은?",
    "금융재산 {finance}억 상속 시 금융재산 상속공제 한도와 적용 방법은?",
    "피상속인 사망 후 상속 포기 시 절세 효과와 주의사항은?",

    # 증여세 절세
    "부모가 성인 자녀에게 {gift}억 현금 증여 예정. 증여세 계산과 절세 방법은?",
    "10년 주기 증여 공제를 활용한 장기 증여 계획 수립 방법은?",
    "미성년 자녀에게 {gift}억 증여 시 증여세와 증여재산공제는?",
    "배우자에게 아파트(시가 {estate}억) 증여 시 세금 계산은?",
    "부모 재산 {total}억을 자녀 {children}명에게 분산 증여하는 최적 전략은?",
    "손자녀에게 직접 증여 시 세율 할증과 절세 방법은?",
    "창업자금 증여세 과세특례 요건과 활용 방법은?",

    # 신고·납부
    "상속세 신고기한과 연부연납 신청 방법은?",
    "상속세 물납 신청 요건과 절차는?",
    "증여세 신고를 기한 내 하지 않을 경우 가산세는?",
    "상속재산 중 부동산 평가 방법과 시가 적용 기준은?",
]

def _random_scenario() -> str:
    tpl = random.choice(SCENARIO_TEMPLATES)
    return tpl.format(
        age=random.randint(55, 80),
        children=random.randint(1, 4),
        estate=random.randint(5, 100),
        finance=random.randint(1, 30),
        debt=random.randint(0, 20),
        total=random.randint(10, 150),
        ratio=random.randint(30, 90),
        years=random.randint(1, 10),
        gift=random.randint(1, 30),
        stock=random.randint(5, 100),
    )


# ------------------------------------------------------------------
# 교사 모델 클라이언트
# ------------------------------------------------------------------
class TeacherClient:
    def __init__(self, provider: str = "anthropic", model: str | None = None):
        self.provider = provider
        if provider == "anthropic":
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-haiku-4-5-20251001"
        elif provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        else:
            raise ValueError(provider)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
    async def call(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.7,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        else:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

    @staticmethod
    def parse(text: str) -> list[dict]:
        text = text.strip().strip("`").lstrip("json").strip()
        # 배열 또는 단일 객체 모두 허용
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 첫 번째 [ ... ] 혹은 { ... } 추출 시도
            import re
            m = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a and len(a) > 50:
                results.append({
                    "instruction": q,
                    "input": "",
                    "output": a,
                    "category": item.get("category", "직접질의"),
                    "source": "synthetic",
                })
        return results


# ------------------------------------------------------------------
# 생성 루프
# ------------------------------------------------------------------
async def generate_from_seeds(
    teacher: TeacherClient,
    seed_path: Path,
    out_path: Path,
    max_seeds: int = 500,
    pairs_per_seed: int = 2,
    concurrency: int = 5,
):
    """판례 시드 기반 Q&A 생성."""
    seeds = []
    with seed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    random.shuffle(seeds)
    seeds = seeds[:max_seeds]
    logger.info(f"판례 시드 {len(seeds):,}건 → Q&A 생성 시작 (provider={teacher.provider})")

    sem = asyncio.Semaphore(concurrency)
    results = []
    lock = asyncio.Lock()

    async def worker(rec: dict):
        src = rec.get("output", "")[:3000]
        user = f"[판례]\n{src}\n\n위 판례에 근거해 {pairs_per_seed}개의 Q&A 페어를 생성하세요."
        async with sem:
            try:
                text = await teacher.call(SEED_SYSTEM, user)
                pairs = TeacherClient.parse(text)
                async with lock:
                    results.extend(pairs)
                    if len(results) % 100 == 0:
                        logger.info(f"  누적 {len(results):,}개")
            except Exception as e:
                logger.warning(f"생성 실패: {e}")

    await asyncio.gather(*[worker(s) for s in seeds])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.success(f"판례 기반 Q&A {len(results):,}개 → {out_path}")
    return len(results)


async def generate_scenarios(
    teacher: TeacherClient,
    out_path: Path,
    n_scenarios: int = 200,
    concurrency: int = 5,
):
    """시나리오 기반 절세 전략 Q&A 생성."""
    logger.info(f"시나리오 {n_scenarios}건 생성 시작")
    sem = asyncio.Semaphore(concurrency)
    results = []
    lock = asyncio.Lock()

    async def worker(_):
        scenario = _random_scenario()
        user = f"[상속·증여 시나리오]\n{scenario}"
        async with sem:
            try:
                text = await teacher.call(SCENARIO_SYSTEM, user)
                pairs = TeacherClient.parse(text)
                # 시나리오는 단일 Q&A이므로 question을 시나리오로 덮어씀
                for p in pairs:
                    if not p["instruction"] or p["instruction"] == "질문":
                        p["instruction"] = scenario
                async with lock:
                    results.extend(pairs)
            except Exception as e:
                logger.warning(f"시나리오 생성 실패: {e}")

    await asyncio.gather(*[worker(i) for i in range(n_scenarios)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.success(f"시나리오 Q&A {len(results):,}개 → {out_path}")
    return len(results)


# ------------------------------------------------------------------
# 엔트리포인트
# ------------------------------------------------------------------
async def main(args):
    teacher = TeacherClient(provider=args.provider, model=args.model)
    out_path = Path(args.out)
    total = 0

    if args.mode in ("seed", "both"):
        total += await generate_from_seeds(
            teacher,
            seed_path=Path(args.seed),
            out_path=out_path,
            max_seeds=args.max_seeds,
            pairs_per_seed=args.pairs_per_seed,
            concurrency=args.concurrency,
        )

    if args.mode in ("scenario", "both"):
        total += await generate_scenarios(
            teacher,
            out_path=out_path,
            n_scenarios=args.n_scenarios,
            concurrency=args.concurrency,
        )

    logger.success(f"합성 데이터 총 {total:,}개 → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="both", choices=["seed", "scenario", "both"])
    p.add_argument("--seed", default="data/processed/sft_base.jsonl")
    p.add_argument("--out", default="data/synthetic/sft_synth.jsonl")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--max-seeds", type=int, default=500)
    p.add_argument("--pairs-per-seed", type=int, default=2)
    p.add_argument("--n-scenarios", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=5)
    args = p.parse_args()
    asyncio.run(main(args))
