#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-4o-mini}"
MAX_SEEDS="${MAX_SEEDS:-20000}"
PAIRS="${PAIRS:-3}"
CONC="${CONC:-4}"

echo "교사 모델: $PROVIDER / $MODEL (seeds<=$MAX_SEEDS, pairs/seed=$PAIRS)"
python -m src.synthetic.qa_generator \
  --seed data/processed/sft_base.jsonl \
  --out data/synthetic/sft_synth.jsonl \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max-seeds "$MAX_SEEDS" \
  --pairs-per-seed "$PAIRS" \
  --concurrency "$CONC"

wc -l data/synthetic/sft_synth.jsonl
