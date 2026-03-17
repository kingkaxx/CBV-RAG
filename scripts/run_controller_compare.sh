#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-hotpotqa}
IL_CKPT=${2:-checkpoints/policy_il.pt}
OFFLINE_CKPT=${3:-checkpoints/policy_offline.pt}
CACHE_DIR=${CACHE_DIR:-./huggingface_cache}
LLM_DEVICE=${LLM_DEVICE:-cuda:0}
OUT_DIR=${OUT_DIR:-experiments/controller_compare/${DATASET}}
SEED=${SEED:-42}

mkdir -p "${OUT_DIR}"

python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type heuristic --output "${OUT_DIR}/heuristic.json" --records_output "${OUT_DIR}/heuristic.records.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"
python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type il --policy_ckpt "${IL_CKPT}" --output "${OUT_DIR}/il.json" --records_output "${OUT_DIR}/il.records.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"
python scripts/run_cbvrag_eval.py --dataset "${DATASET}" --cache_dir "${CACHE_DIR}" --controller_type offline --policy_ckpt "${OFFLINE_CKPT}" --output "${OUT_DIR}/offline.json" --records_output "${OUT_DIR}/offline.records.jsonl" --llm_device "${LLM_DEVICE}" --seed "${SEED}"

python scripts/compare_controllers.py \
  --inputs \
  heuristic="${OUT_DIR}/heuristic.records.jsonl" \
  il="${OUT_DIR}/il.records.jsonl" \
  offline="${OUT_DIR}/offline.records.jsonl" \
  --output "${OUT_DIR}/summary.json"
