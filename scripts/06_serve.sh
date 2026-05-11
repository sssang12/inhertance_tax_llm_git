#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

ADAPTER="${ADAPTER:-outputs/checkpoints/inheritance-tax-llm-8b-qlora}"
MERGED="${MERGED:-outputs/merged/inheritance-tax-llm-8b}"
PORT="${PORT:-8000}"

python -m src.inference.serve_vllm \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter "$ADAPTER" \
  --merged-dir "$MERGED" \
  --port "$PORT" \
  --max-model-len 8192
