#!/usr/bin/env bash
set -euo pipefail

source "$HOME/venvs/vllm-cu130/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$HOME/models/Qwen3.5-4B" \
  --served-model-name qwen3.5-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --language-model-only
