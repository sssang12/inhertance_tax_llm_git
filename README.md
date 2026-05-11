# Inheritance & Gift Tax Legal LLM (상속·증여세 전문 LLM)

Llama 3.1 8B Instruct를 기반으로 한국 상속세 및 증여세 도메인에 특화된 Legal AI 모델을 구축하기 위한 풀스택 파인튜닝 패키지입니다.

## 핵심 구성

1. **데이터 수집 파이프라인** — 국가법령정보시스템(law.go.kr), 조세심판원, 국세청 예규·판례, 대법원 판례에서 상속·증여세 관련 데이터를 자동 수집
2. **데이터 전처리 & 합성 Q&A 생성** — 원문 정제, 청킹, 교사 모델(GPT-4o / Claude) 기반 합성 instruction 데이터셋 생성
3. **QLoRA 파인튜닝** — Llama 3.1 8B Instruct에 4-bit 양자화 + LoRA를 적용한 효율적 학습
4. **도메인 평가** — 상속·증여세 시험문제, 실제 사례 기반 벤치마크 및 LLM-as-Judge 평가
5. **추론 서버** — vLLM 기반 OpenAI-호환 API 서버

## 디렉토리 구조

```
inheritance-tax-llm/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── data_config.yaml
│   ├── training_config.yaml
│   └── eval_config.yaml
├── data/
│   ├── raw/          # 크롤링 원본 (gitignore)
│   ├── processed/    # 정제된 청크
│   ├── synthetic/    # 합성 instruction
│   └── eval/         # 평가셋
├── src/
│   ├── data_collection/
│   │   ├── law_crawler.py
│   │   ├── tribunal_crawler.py
│   │   ├── nts_crawler.py
│   │   └── precedent_crawler.py
│   ├── data_processing/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── formatter.py
│   ├── synthetic/
│   │   └── qa_generator.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── prompt_templates.py
│   │   └── train_qlora.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── llm_judge.py
│   └── inference/
│       ├── inference.py
│       └── serve_vllm.py
└── scripts/
    ├── 01_collect_data.sh
    ├── 02_process_data.sh
    ├── 03_generate_synthetic.sh
    ├── 04_train.sh
    ├── 05_evaluate.sh
    └── 06_serve.sh
```

## 빠른 시작

### 1) 환경 설정

```bash
# Python 3.10+ 권장
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Hugging Face 인증 (Llama 3.1 모델 다운로드용)
huggingface-cli login

# 환경 변수
cp .env.example .env
# OPENAI_API_KEY / ANTHROPIC_API_KEY / LAW_API_KEY 입력
```

### 2) 데이터 수집 (1~3일 소요)

```bash
bash scripts/01_collect_data.sh
```

수집되는 데이터:
- 상속세 및 증여세법 / 시행령 / 시행규칙 (조문 단위)
- 국세청 예규·통칙 (재산세제과 답변)
- 조세심판원 심판결정례 (상속·증여 관련)
- 대법원·고등법원 상속·증여세 판례
- 국세청 상속·증여세 안내문·Q&A

### 3) 전처리 + 합성 Q&A 생성

```bash
bash scripts/02_process_data.sh
bash scripts/03_generate_synthetic.sh  # 교사 모델 API 비용 발생
```

목표 데이터 규모: 30k~100k instruction pairs

### 4) QLoRA 파인튜닝

```bash
bash scripts/04_train.sh
```

권장 하드웨어: A100 40GB 1장 또는 RTX 4090 24GB 1장 (QLoRA 4-bit)
학습 시간 예상: 50k 샘플 기준 약 8~20시간

### 5) 평가 & 서빙

```bash
bash scripts/05_evaluate.sh
bash scripts/06_serve.sh  # vLLM API 서버 (포트 8000)
```

## 모델 카드 (목표)

| 항목 | 내용 |
|---|---|
| Base | meta-llama/Meta-Llama-3.1-8B-Instruct |
| 학습 방식 | QLoRA (4-bit NF4 + LoRA r=16) |
| 학습 데이터 | 상속·증여세법령 + 판례 + 예규 + 합성 Q&A |
| 컨텍스트 길이 | 8,192 토큰 |
| 라이선스 | Llama 3.1 Community License 준수 |

## 법적 고지 및 윤리 가이드라인

본 모델은 일반적인 상속·증여세 정보 제공을 위한 보조 도구이며, **법률 자문을 대체하지 않습니다**. 실제 세무 신고나 분쟁 사안은 반드시 공인된 세무사·변호사의 자문을 받아야 합니다. 학습 데이터에는 공개된 법령·판례·예규만 사용하며, 개인정보가 포함된 자료는 비식별화 처리 후 사용합니다.

## 다음 단계 (로드맵)

- v0.1: 8B QLoRA 기본 모델 출시
- v0.2: 최신 법령 RAG 통합 (법령 개정 추적)
- v0.3: 함수 호출(tool use) 통합 — 상속세 계산기, 신고서 자동 작성
- v1.0: 70B Distillation, 평가셋 공개
