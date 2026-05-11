#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] 정제 + PII 마스킹"
python -m src.data_processing.cleaner --raw-dir data/raw --out-dir data/processed

echo "[2/3] 본문 청킹 (긴 본문만)"
for f in data/processed/*.jsonl; do
  out="${f%.jsonl}.chunked.jsonl"
  python -m src.data_processing.chunker --in-path "$f" --out-path "$out"
done

echo "[3/3] instruction 포맷 변환"
python -m src.data_processing.formatter \
  --processed-dir data/processed \
  --out data/processed/sft_base.jsonl

wc -l data/processed/sft_base.jsonl
