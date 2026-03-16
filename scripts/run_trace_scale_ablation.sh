#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-hotpotqa}
CACHE_DIR=${CACHE_DIR:-./huggingface_cache}
LLM_DEVICE=${LLM_DEVICE:-cuda:0}
SEED=${SEED:-42}

for N in 50 100 200 500; do
  OUT_DIR="experiments/trace_scale/${DATASET}/${N}"
  mkdir -p "${OUT_DIR}"
  python rl/collect_traces.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --num_samples "${N}" --llm_device "${LLM_DEVICE}" --output "${OUT_DIR}/raw.jsonl"
  python rl/prepare_traces.py --input "${OUT_DIR}/raw.jsonl" --output_dir "${OUT_DIR}" --seed "${SEED}" --val_ratio 0.2
  python rl/train_il.py --traces "${OUT_DIR}/train.jsonl" --out "${OUT_DIR}/policy_il.pt" --seed "${SEED}"
  python rl/train_offline.py --traces "${OUT_DIR}/train.jsonl" --val_traces "${OUT_DIR}/val.jsonl" --out "${OUT_DIR}/policy_offline.pt" --seed "${SEED}" --objective awr
  python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type heuristic --output "${OUT_DIR}/eval_heuristic.json" --records_output "${OUT_DIR}/records_heuristic.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"
  python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type il --policy_ckpt "${OUT_DIR}/policy_il.pt" --output "${OUT_DIR}/eval_il.json" --records_output "${OUT_DIR}/records_il.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"
  python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type offline --policy_ckpt "${OUT_DIR}/policy_offline.pt" --output "${OUT_DIR}/eval_offline.json" --records_output "${OUT_DIR}/records_offline.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"
done
