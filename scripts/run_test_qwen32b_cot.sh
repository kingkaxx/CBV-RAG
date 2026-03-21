#!/bin/bash
set -e

echo "Backing up active config..."
cp config.py config_llama_active.py

echo "Activating Qwen3-32B CoT config..."
cp config_qwen32b_cot.py config.py

echo "Running Qwen3-32B + CoT heuristic eval (20 samples)..."
PYTHONPATH=. python scripts/run_cbvrag_eval_cot.py \
  --dataset hotpotqa \
  --use_oracle_context \
  --controller_type heuristic \
  --num_samples 20 \
  --output logs/eval_qwen32b_cot_20.json

echo "Restoring Llama config..."
cp config_llama_active.py config.py

echo "================================================"
echo "Results:"
python3 -c "
import json, os
new = json.load(open('logs/eval_qwen32b_cot_20.json'))
print(f'Qwen3-32B+CoT : EM={new[\"mean_em\"]:.3f}  F1={new[\"mean_f1\"]:.3f}  tokens={new.get(\"mean_tokens\",0):.0f}')
base_path = next((p for p in ['logs/eval_heuristic_nomultidraft.json','logs/eval_heuristic_hotpotqa.json'] if os.path.exists(p)), None)
if base_path:
    base = json.load(open(base_path))
    print(f'Llama-3.1-8B  : EM={base[\"mean_em\"]:.3f}  F1={base[\"mean_f1\"]:.3f}  tokens={base.get(\"mean_tokens\",0):.0f}')
    print(f'EM diff: {new[\"mean_em\"]-base[\"mean_em\"]:+.3f}')
"
