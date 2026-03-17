# CBV-RAG: Cost-Aware Budget-Verified Retrieval-Augmented Generation

CBV-RAG is a training-free, budget-aware RAG framework that combines
**attribution scoring**, **null-branch parametric detection**, and a
**two-tier NLI verifier** with an RL-trained controller to achieve high
answer quality at reduced token cost.

---

## System Overview

CBV-RAG extends retrieval-augmented generation with three principled components:

### Attribution Score (Attr)

$$
\text{Attr}(q, D) = \alpha \cdot \text{GD}(q, D) + (1-\alpha) \cdot \text{PS}(q, D, \tilde{Q})
$$

| Term | Description |
|------|-------------|
| $\text{GD}(q, D)$ | **Grounded Directness** — max NLI entailment probability over retrieved docs $D$ with query $q$ as hypothesis. Measures how strongly the evidence directly supports the question. |
| $\text{PS}(q, D, \tilde{Q})$ | **Parametric Stability** — $1 - \max_{\tilde{q} \in \tilde{Q}} \text{ent}(D, \tilde{q}) \;/\; (\text{ent}(D, q) + \varepsilon)$. Measures counterfactual resistance: how exclusively the docs support the *original* query vs adversarial alternatives. |
| $\alpha$ | Mixing coefficient (default 0.5). |

Both components are **training-free** and **model-agnostic** — no LLM log-probabilities are required.  They rely solely on a DeBERTa-large-MNLI classification head.

---

## Architecture

```
Query q
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                  CBV-RAG Episode Loop                │
│                                                      │
│  ┌────────────┐   ┌──────────┐   ┌───────────────┐  │
│  │  Retrieve  │──▶│  Select  │──▶│  Two-Tier NLI │  │
│  │  (FAISS /  │   │ Context  │   │   Verifier    │  │
│  │   BM25)    │   │ (cluster)│   │ Tier1: DeBERTa│  │
│  └────────────┘   └──────────┘   │ Tier2: LLM    │  │
│        │                         └───────┬───────┘  │
│        ▼                                 │           │
│  ┌─────────────────┐             ┌───────▼───────┐   │
│  │  Branch Manager │             │  Attr Score   │   │
│  │ (counterfactual │             │  GD + PS      │   │
│  │  hypotheses)    │             └───────┬───────┘   │
│  └─────────────────┘                     │           │
│                                          ▼           │
│                              ┌───────────────────┐   │
│                              │  Null Branch      │   │
│                              │  M(q, ∅) vs M(q,D)│   │
│                              │  Attr arbitration  │   │
│                              └────────┬──────────┘   │
└───────────────────────────────────────┼──────────────┘
                                        ▼
                              Final Answer + Episode Record
                              (parametric_hallucination_risk flag)
```

### Null Branch

The **null branch** calls the LLM with *no context* — $M(q, \emptyset)$ — to
capture the model's purely parametric answer.  After the main episode, the Attr
score of the evidence-grounded answer is compared against a threshold.  If
$\text{Attr}(q, D) < \theta$, the system flags
`parametric_hallucination_risk = True`, indicating the evidence-grounded answer
is weakly attributed and the model may be relying on unverified parametric memory.

### Two-Tier NLI Verifier

| Tier | Trigger | Model |
|------|---------|-------|
| Tier 1 (cheap) | Always | DeBERTa-large-MNLI entailment per claim |
| Tier 2 (full) | Score ∈ [0.4, 0.7] uncertain zone | LLM structured verification prompt |

Tier 2 is only triggered when Tier 1 cannot confidently classify a claim,
minimising expensive LLM calls.

### RL Controller

An MLP/GRU policy is trained in two stages:

1. **Imitation Learning (IL)** — from heuristic controller traces.
2. **Offline RL (AWR)** — using the Attr-shaped reward:

$$
R = \text{EM} - \lambda_\text{token} \cdot \frac{T}{T_{\max}} - \lambda_\text{step} \cdot S + \lambda_\text{attr} \cdot \text{Attr}(q, D)
$$

This learns a **budget-conditioned decision boundary** — when to stop
retrieving given the current attribution quality and remaining budget.

---

## Results

### Main Comparison (placeholder — fill after experiments)

| System | HotpotQA EM | HotpotQA F1 | Tokens | Ret. Calls |
|--------|-------------|-------------|--------|------------|
| Vanilla RAG | — | — | — | — |
| CF-RAG | — | — | — | — |
| VeriCite (NLI only) | — | — | — | — |
| CBV-RAG (heuristic) | — | — | — | — |
| CBV-RAG (IL) | — | — | — | — |
| CBV-RAG (offline RL) | — | — | — | — |

| System | TriviaQA EM | PopQA EM | MuSiQue EM | PubHealth EM |
|--------|-------------|----------|------------|--------------|
| Vanilla RAG | — | — | — | — |
| CF-RAG | — | — | — | — |
| CBV-RAG (offline RL) | — | — | — | — |

### Ablation Study (placeholder)

| Config | EM | F1 | Tokens | PHR Rate |
|--------|----|----|--------|----------|
| Full CBV-RAG | — | — | — | — |
| No null branch | — | — | — | — |
| GD only (α=1) | — | — | — | — |
| No NLI verifier | — | — | — | — |

*PHR = Parametric Hallucination Risk rate.*

---

## Reproducing Paper Results

### Step 0: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 1: Configure model paths

Edit `config.py` to set `LLM_MODEL_ID`, `RERANKER_MODEL_ID`, and
`EMBEDDING_MODEL_ID` to your local model directories or Hugging Face IDs.

### Step 2: Build multi-dataset knowledge base

```bash
python data/build_multidataset_kb.py \
  --datasets hotpotqa triviaqa popqa pubhealth musique \
  --split validation \
  --qa_out data/multidataset_qa.jsonl \
  --kb_out data/global_kb_chunks.jsonl

python retrieval/global_index.py \
  --mode build \
  --kb_jsonl data/global_kb_chunks.jsonl \
  --index_dir data/global_index
```

### Step 3: Run baselines

```bash
python scripts/run_baselines.py --dataset hotpotqa --num_samples 500 \
    --output logs/baselines_hotpotqa.json
```

### Step 4: Run CBV-RAG evaluation (heuristic controller)

```bash
python scripts/run_cbvrag_eval.py \
  --dataset hotpotqa \
  --controller_type heuristic \
  --output logs/eval_heuristic_hotpotqa.json
```

### Step 5: Collect traces for RL

```bash
# Collect raw traces with Attr reward labelling
python rl/collect_traces.py \
  --dataset hotpotqa \
  --num_samples 2000 \
  --use_attr_reward \
  --output data/traces/hotpotqa_attr.jsonl

# Prepare train/val split
python rl/prepare_traces.py \
  --input data/traces/hotpotqa_attr.jsonl \
  --output_dir data/traces/hotpotqa_prepared \
  --val_ratio 0.15 \
  --seed 42
```

### Step 6: Train IL → AWR policy

```bash
# Stage 1: Imitation learning
python rl/train_il.py \
  --traces data/traces/hotpotqa_prepared/train.jsonl \
  --val_traces data/traces/hotpotqa_prepared/val.jsonl \
  --out checkpoints/policy_il.pt \
  --policy_type mlp_residual

# Stage 2: Offline RL with Attr reward shaping
python rl/train_offline.py \
  --traces data/traces/hotpotqa_prepared/train.jsonl \
  --val_traces data/traces/hotpotqa_prepared/val.jsonl \
  --init_policy checkpoints/policy_il.pt \
  --out checkpoints/policy_offline.pt \
  --objective awr \
  --use_attr_shaping \
  --lambda_token 0.1 \
  --lambda_step 0.05 \
  --attr_bonus 0.2 \
  --token_budget 4096
```

### Step 7: Evaluate learned controller

```bash
python scripts/run_cbvrag_eval.py \
  --dataset hotpotqa \
  --controller_type offline \
  --policy_ckpt checkpoints/policy_offline.pt \
  --policy_mode greedy \
  --output logs/eval_offline_hotpotqa.json
```

### Step 8: Run ablation study

```bash
python scripts/run_ablation.py \
  --dataset hotpotqa \
  --num_samples 200 \
  --output logs/ablation_hotpotqa.json
```

### Step 9: Plot Pareto frontier

```bash
python scripts/plot_frontier.py \
  --inputs \
    logs/baselines_hotpotqa.json \
    logs/eval_heuristic_hotpotqa.json \
    logs/eval_offline_hotpotqa.json \
    logs/ablation_hotpotqa.json \
  --out logs/frontier_hotpotqa.png \
  --title HotpotQA
```

### Multi-dataset rollout

```bash
for ds in hotpotqa triviaqa popqa musique pubhealth; do
  python scripts/run_cbvrag_eval.py \
    --dataset $ds \
    --controller_type offline \
    --policy_ckpt checkpoints/policy_offline.pt \
    --output logs/eval_offline_${ds}.json
done
```

---

## Requirements

- Python 3.9+
- CUDA GPU recommended for model inference
- PyTorch ≥ 2.0
- `transformers` ≥ 4.38 (for DeBERTa-large-MNLI)
- `faiss-cpu` or `faiss-gpu`
- `matplotlib` ≥ 3.7 (for PDF frontier plots)

```bash
pip install -r requirements.txt
```

---

## Repository Layout

```
CBV-RAG/
├── cbvrag/
│   ├── attribution.py          # Attr score: GD + PS components
│   ├── runner.py               # Episode loop + null branch arbitration
│   ├── env.py                  # Gym-like RL environment
│   ├── actions.py              # 11-action discrete space
│   ├── state.py                # EpisodeState, Branch, EvidenceItem
│   ├── features.py             # Observation feature extraction (v5)
│   ├── reward.py               # Step-level reward computation
│   ├── prompts.py              # LLM prompt templates
│   ├── controller_heuristic.py # Rule-based policy
│   └── controller_learned.py   # Learned policy loader
├── tools/
│   ├── llm.py                  # LLMEngine wrapper
│   ├── retrieve.py             # RetrieverTool (cached)
│   ├── rerank.py               # CrossEncoderReranker
│   ├── select.py               # Cluster-aware context selection
│   └── verify.py               # Two-tier NLI verifier
├── rl/
│   ├── collect_traces.py       # Trace collection (+ --use_attr_reward)
│   ├── train_il.py             # Imitation learning
│   ├── train_offline.py        # Offline RL (AWR) + Attr reward shaping
│   ├── prepare_traces.py       # Train/val split without qid leakage
│   ├── eval_policy.py          # Policy evaluation
│   └── policy.py               # MLP / GRU policy architectures
├── scripts/
│   ├── run_cbvrag_eval.py      # CBV-RAG end-to-end evaluation
│   ├── run_baselines.py        # Vanilla RAG / CF-RAG / VeriCite baselines
│   ├── run_ablation.py         # 4-config ablation study
│   ├── plot_frontier.py        # Pareto frontier (PNG + PDF)
│   └── compare_controllers.py  # Side-by-side controller comparison
├── metrics/
│   ├── usage.py                # Token usage tracker
│   └── cost.py                 # Cost estimation
├── cfrag_pipeline.py           # CF-RAG legacy pipeline
├── retriever.py                # Knowledge base retriever
├── config.py                   # Centralised configuration
├── evaluation.py               # EM / F1 metrics
├── data_loader.py              # Dataset loading utilities
└── tests/
    ├── test_attribution.py     # Unit tests for attribution module
    └── test_runner_stop_semantics.py
```

---

## Citation

```bibtex
@inproceedings{cbvrag2025,
  title     = {CBV-RAG: Cost-Aware Budget-Verified Retrieval-Augmented Generation},
  author    = {[Authors]},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```
