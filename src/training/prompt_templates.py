"""
Llama 3.1 Instruct chat template 빌더.

Llama 3.1 의 공식 chat template:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
  <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>

Tokenizer 의 apply_chat_template 을 그대로 쓰는 것이 가장 안전하지만,
SFTTrainer + DataCollatorForCompletionOnlyLM 호환을 위해
응답 시작 마커를 알아두면 학습 손실 마스킹에 활용 가능하다.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "당신은 대한민국 상속세 및 증여세 분야의 전문가 어시스턴트입니다. "
    "정확하고 검증 가능한 답변을 제공하되, 출처 조문이나 판례를 함께 제시합니다. "
    "법령 개정 시점이나 사실관계가 불분명한 경우 그 한계를 명확히 밝힙니다. "
    "본 답변은 일반적인 안내이며, 구체적인 사안은 세무사·변호사 자문을 권합니다."
)

# DataCollatorForCompletionOnlyLM 에 전달할 응답 시작 마커
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def build_messages(instruction: str, input_text: str = "", system: str | None = None) -> list[dict]:
    """user 메시지 구성. input 이 있으면 instruction 뒤에 컨텍스트로 붙임."""
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content += f"\n\n[참고 정보]\n{input_text.strip()}"
    return [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_for_sft(example: dict, tokenizer) -> str:
    """학습용 텍스트 (정답 포함) 생성."""
    messages = build_messages(example["instruction"], example.get("input", ""))
    messages.append({"role": "assistant", "content": example["output"]})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_for_inference(instruction: str, input_text: str, tokenizer) -> str:
    """추론용 prompt (응답 시작 마커까지)."""
    messages = build_messages(instruction, input_text)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
