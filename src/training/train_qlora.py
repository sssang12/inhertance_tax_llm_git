"""
Llama 3.1 8B Instruct QLoRA 파인튜닝 (Unsloth 최적화 버전).

기존 bitsandbytes 버전 대비 주요 변경사항:
- Unsloth FastLanguageModel 사용 → 2~5배 빠른 학습, 40~60% VRAM 절약
- A100 40GB 기준 10k 샘플 → 약 40~60분 학습 가능
- bitsandbytes / flash-attn 별도 설치 불필요 (Unsloth 내장)
- use_gradient_checkpointing="unsloth" 으로 추가 메모리 최적화

사용 예:
  python -m src.training.train_qlora --config config/training_config.yaml

권장 환경:
  - RunPod / Colab Pro+: A100 40GB  →  10k 샘플 약 40~60분
  - Colab 무료: T4 15GB             →  10k 샘플 약 2~3시간
  - RTX 4090 24GB                   →  10k 샘플 약 60~90분

설치:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from loguru import logger

# Unsloth는 반드시 다른 transformers/peft import 보다 먼저 불러와야 합니다.
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning(
        "unsloth 미설치 — bitsandbytes fallback 모드로 실행합니다.\n"
        "  설치 명령: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

from src.training.dataset import build_train_eval
from src.training.prompt_templates import format_for_sft

load_dotenv()


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "outputs/checkpoints/inheritance-tax-llm-8b-qlora"
    train_files: list[str] = field(
        default_factory=lambda: [
            "data/processed/sft_base.jsonl",
            "data/synthetic/sft_synth.jsonl",
        ]
    )
    eval_files: list[str] | None = None
    max_samples: int | None = 10000  # 1시간 타겟: 기본 10k 샘플

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training
    seq_len: int = 2048
    epochs: float = 2.0           # Unsloth에서 2 epoch도 충분히 빠름
    micro_batch: int = 4          # Unsloth VRAM 절약 덕분에 배치 크기 4로 증가
    grad_accum: int = 4           # 실효 배치 = 16 유지
    lr: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"     # Unsloth 권장 옵티마이저
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10

    # Misc
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "wandb"      # "none" 으로 비활성화 가능


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainConfig(**raw)
    return cfg


# ------------------------------------------------------------------
# Model / Tokenizer (Unsloth 버전)
# ------------------------------------------------------------------
def load_model_and_tokenizer_unsloth(cfg: TrainConfig):
    """Unsloth FastLanguageModel 기반 로딩 — 기본 권장 방식."""
    logger.info(f"[Unsloth] 베이스 모델 로딩: {cfg.base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.seq_len,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        load_in_4bit=True,          # NF4 4-bit 양자화
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        # "unsloth" 모드: 30% 추가 VRAM 절약 + 긴 컨텍스트 지원
        use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
        random_state=cfg.seed,
        use_rslora=False,           # Rank-Stabilized LoRA (실험적)
    )
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def load_model_and_tokenizer_fallback(cfg: TrainConfig):
    """Unsloth 미설치 시 기존 bitsandbytes 방식으로 폴백."""
    logger.info(f"[Fallback] 베이스 모델 로딩: {cfg.base_model}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
def main(config_path: str):
    cfg = load_config(config_path)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.report_to == "wandb" and not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY 미설정 — report_to=none 으로 전환")
        cfg.report_to = "none"

    # 모델 로딩: Unsloth 우선, 없으면 bitsandbytes fallback
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_and_tokenizer_unsloth(cfg)
    else:
        model, tokenizer = load_model_and_tokenizer_fallback(cfg)

    train_ds, eval_ds = build_train_eval(
        cfg.train_files,
        eval_files=cfg.eval_files,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    logger.info(f"train={len(train_ds):,}, eval={len(eval_ds):,}")

    # 예상 학습 시간 안내 (A100 + Unsloth 기준)
    estimated_steps = (len(train_ds) * cfg.epochs) / (cfg.micro_batch * cfg.grad_accum)
    logger.info(
        f"예상 스텝: {estimated_steps:,.0f} "
        f"(A100+Unsloth 기준 약 {estimated_steps * 0.3 / 60:.0f}~{estimated_steps * 0.5 / 60:.0f}분)"
    )

    def _fmt(examples):
        # Unsloth는 단일 샘플(dict of scalars)과 배치(dict of lists) 두 방식으로 호출
        if isinstance(examples.get("instruction", ""), str):
            # 단일 샘플 호출 (Unsloth 초기 테스트용)
            return [format_for_sft(examples, tokenizer)]
        else:
            # 배치 호출
            texts = []
            for i in range(len(examples["instruction"])):
                ex = {k: v[i] for k, v in examples.items()}
                texts.append(format_for_sft(ex, tokenizer))
            return texts

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.micro_batch,
        per_device_eval_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_seq_length=cfg.seq_len,
        packing=False,              # DataCollatorForCompletionOnlyLM과 충돌 방지
        dataset_text_field=None,    # formatting_func 사용
        report_to=cfg.report_to,
        run_name="inheritance-tax-llm-qlora",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=_fmt,
    )

    # Unsloth 권장 방식: 응답 토큰에만 loss 적용 (DataCollatorForCompletionOnlyLM 대체)
    if UNSLOTH_AVAILABLE:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    logger.info("학습 시작")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.success(f"학습 완료. 어댑터 저장 → {cfg.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/training_config.yaml")
    args = p.parse_args()
    main(args.config)
