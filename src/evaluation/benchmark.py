"""
파인튜닝된 모델에 대해 표면 메트릭 + LLM-as-Judge 점수를 계산.

벤치마크 셋 JSONL 형식:
  {"instruction": "...", "input": "(선택)", "reference": "정답 텍스트"}
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import score_pair
from src.training.prompt_templates import format_for_inference


def load_model(base_model: str, adapter_dir: str | None, quantize: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16)

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def run_benchmark(
    base_model: str,
    adapter_dir: str | None,
    bench_path: Path,
    out_path: Path,
    *,
    use_llm_judge: bool = True,
    judge_provider: str = "openai",
):
    model, tokenizer = load_model(base_model, adapter_dir)
    judge = LLMJudge(provider=judge_provider) if use_llm_judge else None

    rouge_l, bleu, cit_recall, judge_totals = [], [], [], []
    out_records = []

    with bench_path.open("r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    logger.info(f"평가 예제: {len(examples)}")
    for i, ex in enumerate(examples, 1):
        prompt = format_for_inference(ex["instruction"], ex.get("input", ""), tokenizer)
        pred = generate(model, tokenizer, prompt)
        sc = score_pair(pred, ex["reference"])
        rouge_l.append(sc.rouge_l)
        bleu.append(sc.bleu)
        if sc.citation_recall == sc.citation_recall:  # not NaN
            cit_recall.append(sc.citation_recall)

        judge_score = None
        if judge:
            judge_score = judge.judge(ex["instruction"], ex["reference"], pred)
            if judge_score:
                judge_totals.append(judge_score.normalized)

        out_records.append(
            {
                "instruction": ex["instruction"],
                "reference": ex["reference"],
                "prediction": pred,
                "rouge_l": sc.rouge_l,
                "bleu": sc.bleu,
                "citation_recall": sc.citation_recall,
                "judge": judge_score.__dict__ if judge_score else None,
            }
        )
        if i % 10 == 0:
            logger.info(f"  {i}/{len(examples)} | ROUGE-L={statistics.mean(rouge_l):.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "n": len(out_records),
        "rouge_l": statistics.mean(rouge_l) if rouge_l else 0,
        "bleu": statistics.mean(bleu) if bleu else 0,
        "citation_recall": statistics.mean(cit_recall) if cit_recall else 0,
        "judge_score_normalized": statistics.mean(judge_totals) if judge_totals else None,
    }
    logger.success(f"평가 결과: {summary}")
    with out_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default="outputs/checkpoints/inheritance-tax-llm-8b-qlora")
    p.add_argument("--bench", default="data/eval/benchmark.jsonl")
    p.add_argument("--out", default="outputs/eval/predictions.jsonl")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"])
    args = p.parse_args()

    run_benchmark(
        args.base_model,
        args.adapter,
        Path(args.bench),
        Path(args.out),
        use_llm_judge=not args.no_judge,
        judge_provider=args.judge_provider,
    )
