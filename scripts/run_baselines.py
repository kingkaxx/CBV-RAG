"""Baseline evaluation script for the NeurIPS comparison table.

Evaluates three named baselines on a given dataset and saves per-example
records plus aggregate metrics to ``logs/baselines_<dataset>.json``.

Baselines
---------
1. **vanilla_rag** — Single retrieval pass, no verification.  Retrieves the
   top-K passages once and generates an answer directly.  Represents the
   simplest possible RAG system.

2. **cfrag** — The existing CF-RAG pipeline (``cfrag_pipeline.py``).
   Counterfactual query generation + synergetic retrieval + explanatory
   answer generation.

3. **vericite** — Simulated VeriCite-style pipeline: single retrieval pass
   followed by NLI-only answer verification (no counterfactual queries, no
   branching).  Uses the two-tier NLI verifier from ``tools/verify.py`` to
   flag unverified claims.

Metrics reported per baseline
------------------------------
* **EM** (exact match) — using ``evaluation.smart_exact_match_score``.
* **F1** — token-overlap F1.
* **avg_total_tokens** — mean tokens used per example.
* **avg_retrieval_calls** — mean retrieval calls per example.
* **avg_latency_ms** — mean wall-clock latency per example.

Usage
-----
    python scripts/run_baselines.py \\
        --dataset hotpotqa \\
        --num_samples 100 \\
        --output logs/baselines_hotpotqa.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from evaluation import f1_score, smart_exact_match_score


# ---------------------------------------------------------------------------
# Helper: aggregate records → summary dict
# ---------------------------------------------------------------------------

def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(records))
    return {
        "em": sum(r["em"] for r in records) / n,
        "f1": sum(r["f1"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r.get("retrieval_calls", 0) for r in records) / n,
        "avg_latency_ms": sum(r.get("latency_ms", 0.0) for r in records) / n,
        "count": n,
    }


# ---------------------------------------------------------------------------
# Baseline 1: Vanilla RAG
# ---------------------------------------------------------------------------

def run_vanilla_rag(
    data: List[dict],
    tools: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Single-retrieval, no-verification baseline.

    Retrieves *top_k* passages and generates an answer with a simple prompt.
    No reranking, no verification, no branching.
    """
    llm = tools["llm"]
    retriever = tools["retrieve"]
    records = []

    for i, ex in enumerate(data):
        q = ex["question"]
        golds: List[str] = ex.get("answer") or [""]
        t0 = time.time()

        cands = retriever.retrieve(q, top_k)
        passages = [c.get("text", "") for c in cands[:top_k]]
        context = "\n".join(f"[{j+1}] {p[:300]}" for j, p in enumerate(passages))
        prompt = (
            "Answer the question using the passages below. Be concise.\n\n"
            f"Question: {q}\n\nPassages:\n{context}\n\nAnswer:"
        )
        pred, usage = llm.generate(prompt, max_new_tokens=96, temperature=0.0, name="vanilla_rag")
        latency_ms = (time.time() - t0) * 1000

        em = max(smart_exact_match_score(pred, g, q) for g in golds)
        f1 = max(f1_score(pred, g) for g in golds)
        tokens = int(usage.get("total_tokens", 0))

        records.append({
            "qid": str(i),
            "baseline": "vanilla_rag",
            "question": q,
            "prediction": pred,
            "gold": golds,
            "em": float(em),
            "f1": float(f1),
            "total_tokens": tokens,
            "retrieval_calls": 1,
            "latency_ms": latency_ms,
        })

    return records


# ---------------------------------------------------------------------------
# Baseline 2: CF-RAG
# ---------------------------------------------------------------------------

def run_cfrag(
    data: List[dict],
    models: Dict[str, Any],
    retriever_obj: Any,
) -> List[Dict[str, Any]]:
    """CF-RAG pipeline baseline (existing ``cfrag_pipeline.py``)."""
    from cfrag_pipeline import CFRAGPipeline

    pipeline = CFRAGPipeline(models, retriever_obj)
    records = []

    for i, ex in enumerate(data):
        q = ex["question"]
        golds: List[str] = ex.get("answer") or [""]
        t0 = time.time()

        pred = pipeline.run(q)
        latency_ms = (time.time() - t0) * 1000

        # Estimate tokens from tokenizer (pipeline is not per-call instrumented).
        tok = models.get("llm_tokenizer") or models.get("tokenizer")
        if tok is not None:
            tokens = len(tok(q).input_ids) + len(tok(pred).input_ids)
        else:
            tokens = len(q.split()) + len(pred.split())

        em = max(smart_exact_match_score(pred, g, q) for g in golds)
        f1 = max(f1_score(pred, g) for g in golds)

        records.append({
            "qid": str(i),
            "baseline": "cfrag",
            "question": q,
            "prediction": pred,
            "gold": golds,
            "em": float(em),
            "f1": float(f1),
            "total_tokens": tokens,
            "retrieval_calls": 1,
            "latency_ms": latency_ms,
        })

    return records


# ---------------------------------------------------------------------------
# Baseline 3: VeriCite-style (NLI verify only)
# ---------------------------------------------------------------------------

def run_vericite(
    data: List[dict],
    tools: Dict[str, Any],
    top_k: int = 5,
    cheap_threshold: float = 0.7,
    uncertain_low: float = 0.4,
) -> List[Dict[str, Any]]:
    """Simulated VeriCite-style pipeline: single-retrieval + NLI verification.

    Retrieves passages once and generates an answer (same as vanilla RAG), then
    runs the two-tier NLI verifier from ``tools/verify.py`` to flag claims that
    are not supported by the retrieved passages.  Unlike CBV-RAG there is no
    branching, no counterfactual queries, and no null-branch arbitration.
    """
    from tools.verify import verify_answer

    llm = tools["llm"]
    retriever = tools["retrieve"]
    records = []

    for i, ex in enumerate(data):
        q = ex["question"]
        golds: List[str] = ex.get("answer") or [""]
        t0 = time.time()

        cands = retriever.retrieve(q, top_k)
        passages = [c.get("text", "") for c in cands[:top_k]]
        context = "\n".join(f"[{j+1}] {p[:300]}" for j, p in enumerate(passages))
        prompt = (
            "Answer the question using the passages below. Be concise.\n\n"
            f"Question: {q}\n\nPassages:\n{context}\n\nAnswer:"
        )
        pred, usage = llm.generate(prompt, max_new_tokens=96, temperature=0.0, name="vericite")

        # NLI verification pass (Tier 1 only — no LLM tier for baseline purity).
        verify_result = verify_answer(
            answer=pred,
            docs=passages,
            llm=None,  # Tier 2 disabled for this baseline
            cheap_threshold=cheap_threshold,
            uncertain_low=uncertain_low,
        )
        latency_ms = (time.time() - t0) * 1000

        em = max(smart_exact_match_score(pred, g, q) for g in golds)
        f1 = max(f1_score(pred, g) for g in golds)
        tokens = int(usage.get("total_tokens", 0))

        records.append({
            "qid": str(i),
            "baseline": "vericite",
            "question": q,
            "prediction": pred,
            "gold": golds,
            "em": float(em),
            "f1": float(f1),
            "total_tokens": tokens,
            "retrieval_calls": 1,
            "latency_ms": latency_ms,
            "nli_overall_score": float(verify_result["overall_score"]),
            "nli_num_verified": int(verify_result["num_verified"]),
            "nli_num_rejected": int(verify_result["num_rejected"]),
        })

    return records


# ---------------------------------------------------------------------------
# Formatted table printing
# ---------------------------------------------------------------------------

def _print_table(results: Dict[str, Dict]) -> None:
    header = ["Baseline", "EM", "F1", "Tokens", "Ret.Calls", "Latency(ms)"]
    col_widths = [max(16, max(len(k) for k in results)), 6, 6, 8, 10, 12]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    def _row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"

    print(sep)
    print(_row(header))
    print(sep)
    for name, agg in results.items():
        print(_row([
            name,
            f"{agg['em']:.3f}",
            f"{agg['f1']:.3f}",
            f"{agg['avg_total_tokens']:.0f}",
            f"{agg['avg_retrieval_calls']:.2f}",
            f"{agg['avg_latency_ms']:.1f}",
        ]))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate named baselines for the CBV-RAG comparison table.")
    ap.add_argument("--dataset", required=True, help="Dataset name (e.g. hotpotqa).")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path.  Defaults to logs/baselines_<dataset>.json.",
    )
    ap.add_argument(
        "--baselines",
        nargs="+",
        choices=["vanilla_rag", "cfrag", "vericite"],
        default=["vanilla_rag", "cfrag", "vericite"],
        help="Which baselines to evaluate (default: all three).",
    )
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    args = ap.parse_args()

    output = Path(args.output or f"logs/baselines_{args.dataset}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    data = list(
        __import__("data_loader", fromlist=["load_and_process_data"]).load_and_process_data(
            args.dataset, args.cache_dir, args.num_samples
        )
    )

    import model_loader
    from retriever import KnowledgeBaseRetriever
    from tools.llm import LLMEngine
    from tools.rerank import CrossEncoderReranker
    from tools.retrieve import RetrieverTool

    models = model_loader.load_all_models()
    kb = KnowledgeBaseRetriever(models["embedding_model"])
    tools = {
        "llm": LLMEngine(
            getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path),
            device="cpu",
        ),
        "retrieve": RetrieverTool(kb),
        "rerank": CrossEncoderReranker(),
    }

    all_results: Dict[str, Any] = {}

    if "vanilla_rag" in args.baselines:
        print("[baselines] Running: vanilla_rag", flush=True)
        recs = run_vanilla_rag(data, tools, top_k=args.retrieval_top_k)
        all_results["vanilla_rag"] = {"aggregate": _aggregate(recs), "per_example": recs}
        agg = all_results["vanilla_rag"]["aggregate"]
        print(f"  EM={agg['em']:.3f}  F1={agg['f1']:.3f}  Tokens={agg['avg_total_tokens']:.0f}", flush=True)

    if "cfrag" in args.baselines:
        print("[baselines] Running: cfrag", flush=True)
        recs = run_cfrag(data, models, kb)
        all_results["cfrag"] = {"aggregate": _aggregate(recs), "per_example": recs}
        agg = all_results["cfrag"]["aggregate"]
        print(f"  EM={agg['em']:.3f}  F1={agg['f1']:.3f}  Tokens={agg['avg_total_tokens']:.0f}", flush=True)

    if "vericite" in args.baselines:
        print("[baselines] Running: vericite", flush=True)
        recs = run_vericite(data, tools, top_k=args.retrieval_top_k)
        all_results["vericite"] = {"aggregate": _aggregate(recs), "per_example": recs}
        agg = all_results["vericite"]["aggregate"]
        print(f"  EM={agg['em']:.3f}  F1={agg['f1']:.3f}  Tokens={agg['avg_total_tokens']:.0f}", flush=True)

    output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}\n")

    _print_table({k: v["aggregate"] for k, v in all_results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
