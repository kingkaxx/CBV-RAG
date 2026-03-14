#!/usr/bin/env bash
set -euo pipefail

TRACES=${1:-data/traces/hotpotqa.jsonl}
VAL_TRACES=${2:-}
OUT_BASE=${3:-experiments/reward_ablation}
SEED=${SEED:-42}

mkdir -p "${OUT_BASE}"

for TOKEN_PEN in 0.0005 0.001 0.002; do
  for RETR_PEN in 0.02 0.05 0.1; do
    NAME="tp${TOKEN_PEN}_rp${RETR_PEN}"
    OUT="${OUT_BASE}/${NAME}"
    mkdir -p "${OUT}"
    python rl/train_offline.py \
      --traces "${TRACES}" \
      ${VAL_TRACES:+--val_traces "${VAL_TRACES}"} \
      --out "${OUT}/policy_offline.pt" \
      --token_penalty "${TOKEN_PEN}" \
      --retrieval_penalty "${RETR_PEN}" \
      --seed "${SEED}" \
      --objective awr
  done
done
