"""
HuggingFace Transformers 기반 단일 추론 인터페이스.

LoRA 어댑터를 베이스 모델에 얹어 즉시 질의응답 가능.
프로덕션 환경에서는 serve_vllm.py 사용 권장 (수십~수백 배 빠른 토큰 처리량).
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.training.prompt_templates import format_for_inference


def load_pipeline(base_model: str, adapter_dir: str | None = None, quantize: bool = True):
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
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16
        )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def chat(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    *,
    max_new_tokens: int = 768,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    prompt = format_for_inference(instruction, input_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default="outputs/checkpoints/inheritance-tax-llm-8b-qlora")
    p.add_argument("--no-quantize", action="store_true")
    args = p.parse_args()

    model, tokenizer = load_pipeline(args.base_model, args.adapter, quantize=not args.no_quantize)
    print("\n[상속·증여세 어시스턴트] 'exit' 입력 시 종료\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "exit":
            break
        ans = chat(model, tokenizer, q)
        print(f"\nA> {ans}\n")


if __name__ == "__main__":
    main()
