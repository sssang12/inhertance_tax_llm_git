#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then set -a; source .env; set +a; fi

# Hugging Face 로그인 확인
python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1 || {
  echo "huggingface-cli login 먼저 실행 후 다시 시도하세요."; exit 1;
}

CONFIG="${CONFIG:-config/training_config.yaml}"
echo "학습 설정: $CONFIG"
python -m src.training.train_qlora --config "$CONFIG"
