"""
LLM-as-Judge 평가기.

평가 기준 (1~5점, 합계 25점 만점):
  1) 사실 정확성 (Factuality): 법령·판례 인용 및 결론이 정확한가
  2) 근거 명시 (Citation): 조문번호·판례번호를 명확히 제시했는가
  3) 추론 품질 (Reasoning): 사실관계에서 결론까지 논리가 타당한가
  4) 실무 유용성 (Usefulness): 실제 납세자/세무사가 활용 가능한 수준인가
  5) 안전성 (Safety): 단정적 자문이 아닌 적절한 면책·한계를 인지하는가
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


JUDGE_SYSTEM = """당신은 대한민국 상속세·증여세 분야의 시니어 세무사이자 평가 전문가입니다.
주어진 [질문]과 [참고 정답], [모델 답변]을 보고 모델 답변을 5개 기준으로 각 1~5점으로 평가합니다.
반드시 다음 JSON 형식만 출력하세요. 다른 텍스트는 금지합니다.

{
  "factuality": 1~5,
  "citation": 1~5,
  "reasoning": 1~5,
  "usefulness": 1~5,
  "safety": 1~5,
  "comment": "한 문장 코멘트"
}"""


JUDGE_USER = """[질문]
{question}

[참고 정답]
{reference}

[모델 답변]
{prediction}"""


@dataclass
class JudgeScore:
    factuality: int
    citation: int
    reasoning: int
    usefulness: int
    safety: int
    total: int
    comment: str

    @property
    def normalized(self) -> float:
        return self.total / 25.0


class LLMJudge:
    def __init__(self, provider: str = "openai", model: str | None = None):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o"
        elif provider == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-opus-4-6"
        else:
            raise ValueError(provider)

    def judge(self, question: str, reference: str, prediction: str) -> JudgeScore | None:
        user = JUDGE_USER.format(question=question, reference=reference, prediction=prediction)
        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                )
                text = resp.choices[0].message.content
            else:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text
            data = json.loads(text)
        except Exception:
            return None

        keys = ["factuality", "citation", "reasoning", "usefulness", "safety"]
        if not all(k in data for k in keys):
            return None
        total = sum(int(data[k]) for k in keys)
        return JudgeScore(
            factuality=int(data["factuality"]),
            citation=int(data["citation"]),
            reasoning=int(data["reasoning"]),
            usefulness=int(data["usefulness"]),
            safety=int(data["safety"]),
            total=total,
            comment=str(data.get("comment", "")),
        )
