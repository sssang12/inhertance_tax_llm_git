"""
법률 도메인 평가 메트릭.

- ROUGE-L / BLEU: 표면적 유사도 (참고용)
- Citation accuracy: 정답에 명시된 조문/판례 번호가 모델 응답에 포함되는 비율
- Factuality (LLM-as-Judge): llm_judge.py 와 연동
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


RE_ARTICLE = re.compile(r"제\d+조(?:의\d+)?(?:\s*제?\d+항)?")
RE_CASE_NO = re.compile(r"(?:\d{4}[가-힣]{1,3}\d+|조심\d{4}[가-힣]{1,3}\d+)")


@dataclass
class EvalScore:
    rouge_l: float
    bleu: float
    citation_recall: float


def score_pair(prediction: str, reference: str) -> EvalScore:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    r = scorer.score(reference, prediction)["rougeL"].fmeasure

    bleu = BLEU(effective_order=True, tokenize="char").sentence_score(prediction, [reference]).score / 100.0

    ref_cites = set(RE_ARTICLE.findall(reference)) | set(RE_CASE_NO.findall(reference))
    if ref_cites:
        pred_cites = set(RE_ARTICLE.findall(prediction)) | set(RE_CASE_NO.findall(prediction))
        recall = len(ref_cites & pred_cites) / len(ref_cites)
    else:
        recall = float("nan")

    return EvalScore(rouge_l=r, bleu=bleu, citation_recall=recall)
