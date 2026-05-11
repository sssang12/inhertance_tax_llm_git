#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 환경 변수 로드
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[1/4] 법령 본문 수집"
python -m src.data_collection.law_crawler

echo "[2/4] 판례 수집"
python -m src.data_collection.precedent_crawler

echo "[3/4] 국세청 예규/법령해석 수집"
python -m src.data_collection.nts_crawler || echo "  (NTS 크롤러 셀렉터 조정 필요할 수 있음)"

echo "[4/4] 조세심판원 결정례 수집"
python -m src.data_collection.tribunal_crawler || echo "  (조세심판원 셀렉터 조정 필요할 수 있음)"

echo "원본 수집 완료. data/raw/ 확인."
