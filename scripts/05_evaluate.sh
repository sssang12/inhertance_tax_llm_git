#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

ADAPTER="${ADAPTER:-outputs/checkpoints/inheritance-tax-llm-8b-qlora}"
BENCH="${BENCH:-data/eval/benchmark.jsonl}"
OUT="${OUT:-outputs/eval/predictions.jsonl}"

if [ ! -f "$BENCH" ]; then
  echo "벤치마크 셋이 없습니다: $BENCH"
  echo "샘플 형식 (한 줄당):  {\"instruction\":\"...\",\"reference\":\"...\"}"
  exit 1
fi

python -m src.evaluation.benchmark \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter "$ADAPTER" \
  --bench "$BENCH" \
  --out "$OUT"
