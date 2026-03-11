# CBV-RAG / CF-RAG

This repository contains a **CF-RAG** implementation (counterfactual retrieval-augmented generation) and a newer **CBV-RAG scaffold** for budget-aware, branch-based retrieval + verification with an RL-friendly control loop.

> Current status:
> - CF-RAG pipeline is runnable and integrated with dataset loaders/evaluation scripts.
> - CBV-RAG modules are implemented as a modular scaffold (tools, env, heuristic controller, RL scripts) and are intended for iterative experimentation.

---

## 1) What is in this repository

### CF-RAG (existing pipeline)
- Counterfactual query generation.
- Synergetic retrieval over original + counterfactual queries.
- Evidence clustering/sampling and explanatory answer generation.
- Interactive CLI (`main.py`) and dataset evaluation scripts.

Key files:
- `cfrag_pipeline.py`
- `retriever.py`
- `main.py`
- `run_evaluation.py`

### CBV-RAG (new scaffold)
- Token/cost instrumentation (`metrics/`).
- Modular tools (`tools/`) for LLM generation, retrieval, reranking, and context selection.
- Branch/state/action runtime (`cbvrag/`) with Gym-like environment and heuristic controller.
- RL data collection + policy training/evaluation scripts (`rl/`).
- Baseline and frontier scripts (`scripts/`).

Key files:
- `metrics/usage.py`, `metrics/cost.py`
- `tools/llm.py`, `tools/retrieve.py`, `tools/rerank.py`, `tools/select.py`
- `cbvrag/runner.py`, `cbvrag/env.py`, `cbvrag/controller_heuristic.py`
- `rl/collect_traces.py`, `rl/train_il.py`, `rl/train_offline.py`, `rl/eval_policy.py`
- `scripts/run_cfrag_baseline.py`, `scripts/run_cbvrag_eval.py`, `scripts/plot_frontier.py`

---

## 2) Requirements

- Python 3.9+ recommended
- CUDA GPU recommended for model inference
- Local Hugging Face-compatible model directories (or hub-accessible IDs)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3) Configuration

Configuration is centralized in `config.py`.

Important groups:

1. **Model paths/devices**
   - `LLM_MODEL_ID`, `RERANKER_MODEL_ID`, `EMBEDDING_MODEL_ID`
   - `LLM_DEVICE`, `RERANKER_DEVICE`, `EMBEDDING_DEVICE`

2. **CF-RAG generation/retrieval settings**
   - `MAX_NEW_TOKENS`, `EXPLANATORY_TEMPERATURE`, `COUNTERFACTUAL_TEMPERATURE`
   - `RETRIEVAL_TOP_K`, `RERANKER_TOP_K`

3. **CBV-RAG defaults**
   - `MAX_CONTEXT_CHUNKS`, `MAX_CONTEXT_TOKENS`
   - `RETRIEVAL_POOL_K0`, `RERANK_BATCH_SIZE`
   - `CBVRAG_MAX_STEPS`, `CBVRAG_MAX_BRANCHES`, `CBVRAG_MAX_RETRIEVAL_CALLS`

---

## 4) Running CF-RAG

### Interactive mode

```bash
python main.py
```

### Batch evaluation

```bash
python run_evaluation.py --dataset hotpotqa --num_samples 50
python run_evaluation.py --dataset triviaqa --num_samples 50
python run_evaluation.py --dataset popqa --num_samples 50
python run_evaluation.py --dataset musique --num_samples 50
python run_evaluation.py --dataset pubhealth --num_samples 50
```

Outputs are written as JSONL into `eval_results/` by default.

---

## 5) Running baseline + CBV-RAG experiments

### 5.1 CF-RAG baseline with cost logging

```bash
python scripts/run_cfrag_baseline.py --dataset hotpotqa --num_samples 100
```

This writes per-example logs to:
- `logs/baseline/<dataset>.jsonl`

### 5.2 CBV-RAG heuristic evaluation

```bash
python scripts/run_cbvrag_eval.py \
  --dataset hotpotqa \
  --baseline_jsonl logs/baseline/hotpotqa.jsonl \
  --output logs/cbvrag_eval_hotpotqa.json
```

### 5.3 Frontier plotting

```bash
python scripts/plot_frontier.py --input logs/cbvrag_eval_hotpotqa.json --out logs/frontier_hotpotqa.png
```

---

## 6) RL workflow

### Step 1: Collect heuristic traces

```bash
python rl/collect_traces.py --dataset hotpotqa --num_samples 500
```

Output:
- `data/traces/hotpotqa.jsonl` (default)

### Step 2: Train behavior cloning policy

```bash
python rl/train_il.py --traces data/traces/hotpotqa.jsonl --out checkpoints/policy_il.pt
```

### Step 3: Evaluate policy imitation quality

```bash
python rl/eval_policy.py --policy checkpoints/policy_il.pt --traces data/traces/hotpotqa.jsonl
```

### Step 4 (optional): Train offline policy

```bash
python rl/train_offline.py --traces data/traces/hotpotqa.jsonl --out checkpoints/policy_offline.pt
```

---

## 7) Token-efficiency design notes (CBV-RAG)

The CBV scaffold enforces/encourages:

- Retrieval pool can be large, but prompt context is selected with caps.
- Context selection bounded by:
  - `max_chunks` (default 8)
  - `max_tokens` (default 1500)
- Verification is split into:
  - cheap heuristic verification first
  - LLM verification only when needed
- Branching budget to avoid uncontrolled token growth.

---

## 8) Directory overview

```text
.
├── cfrag_pipeline.py
├── retriever.py
├── main.py
├── run_evaluation.py
├── config.py
├── model_loader.py
├── data_loader.py
├── evaluation.py
├── metrics/
├── tools/
├── cbvrag/
├── rl/
├── scripts/
├── logs/
├── eval_results/
└── requirements.txt
```

---

## 9) Practical notes / caveats

- Some scripts load large local models and may require substantial VRAM.
- Ensure model IDs/paths in `config.py` match your local environment.
- The CBV-RAG stack is structured for experimentation and iteration; you should expect to tune prompts, budgets, and rewards for your dataset/hardware constraints.

---

## 10) Quick sanity checklist

1. Verify model paths in `config.py`.
2. Confirm CUDA devices referenced in config are valid.
3. Run a small CF-RAG eval (`--num_samples 5`).
4. Run baseline logging script on same subset.
5. Run CBV heuristic eval and compare token/accuracy metrics.

