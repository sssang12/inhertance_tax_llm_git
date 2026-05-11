"""
vLLM 기반 OpenAI 호환 API 서버 런처 헬퍼.

1) LoRA 어댑터를 베이스 모델에 머지하여 단일 체크포인트로 저장 (선택)
2) vllm serve 로 OpenAI 호환 엔드포인트 시작

권장:
- 8B 모델은 머지 후 fp16/bf16 로 서빙하면 추론 처리량이 가장 좋다
- LoRA 어댑터를 그대로 유지하면서 다중 어댑터 핫스왑을 원하면 --enable-lora 옵션 사용
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def merge_adapter(base_model: str, adapter_dir: str, output_dir: str):
    """LoRA 가중치를 베이스 모델에 머지하여 저장."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] base={base_model}, adapter={adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge] saved → {output_dir}")


def serve(model_dir: str, *, port: int = 8000, tp: int = 1, dtype: str = "bfloat16",
          max_model_len: int = 8192):
    """vLLM OpenAI 호환 서버 실행."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--served-model-name", "inheritance-tax-llm",
    ]
    print(" ".join(cmd))
    os.execvp(cmd[0], cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--adapter", default="outputs/checkpoints/inheritance-tax-llm-8b-qlora")
    p.add_argument("--merged-dir", default="outputs/merged/inheritance-tax-llm-8b")
    p.add_argument("--skip-merge", action="store_true",
                   help="이미 머지된 경우 건너뛰고 바로 서빙")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    if not args.skip_merge:
        merge_adapter(args.base_model, args.adapter, args.merged_dir)

    serve(args.merged_dir, port=args.port, tp=args.tp, dtype=args.dtype,
          max_model_len=args.max_model_len)


if __name__ == "__main__":
    main()
