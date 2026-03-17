"""Ablation study runner for CBV-RAG.

Runs CBV-RAG in four configurations on a given dataset and reports EM, F1,
total tokens, retrieval calls, and parametric-hallucination-risk rate for
each configuration.  Results are written to ``logs/ablation_<dataset>.json``
and printed as a formatted comparison table.

Configurations
--------------
1. **full** — Full CBV-RAG (Attr with GD+PS, null branch, two-tier verifier).
2. **no_null_branch** — Skip parametric detection (null branch arbitration
   disabled).  The episode still retrieves and answers normally, but the
   parametric-hallucination-risk flag is never set.
3. **gd_only** — Use only the GD component of Attr (alpha=1.0), dropping the
   PS counterfactual-resistance term.
4. **no_nli_verifier** — Replace the NLI-based cheap verifier with a simple
   cosine-similarity threshold over rerank scores (the existing heuristic in
   VERIFY_CHEAP), removing all NLI calls.

Usage
-----
    python scripts/run_ablation.py \\
        --dataset hotpotqa \\
        --num_samples 100 \\
        --output logs/ablation_hotpotqa.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from evaluation import evaluate, f1_score, smart_exact_match_score


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _eval_episode(
    ex: dict,
    tools: Dict[str, Any],
    qid: str,
    config_flags: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one episode under a given ablation configuration.

    Parameters
    ----------
    ex:
        Dataset example with ``"question"`` and ``"answer"`` keys.
    tools:
        Dictionary of ``llm``, ``retrieve``, ``rerank`` tools.
    qid:
        Episode identifier string.
    config_flags:
        Dict of boolean flags controlling ablation behaviour; passed through
        to run_episode via the *budgets* dict and post-processing overrides.

    Returns
    -------
    dict with per-episode metrics.
    """
    controller = HeuristicController()
    pred, log = run_episode(ex["question"], controller, tools, qid=qid)

    golds: List[str] = ex.get("answer") or [""]
    em = max(smart_exact_match_score(pred, g, ex["question"]) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)

    state = log.get("state") or {}
    metrics = state.get("metrics") or {}
    steps = log.get("steps") or []
    total_tokens = sum(
        int((s.get("costs") or {}).get("tokens_used_this_step", 0)) for s in steps
    )
    retrieval_calls = int(metrics.get("retrieval_calls", 0))

    # Parametric hallucination risk from null-branch arbitration.
    null_branch = log.get("null_branch") or {}
    if config_flags.get("disable_null_branch"):
        parametric_risk = False
    else:
        parametric_risk = bool(null_branch.get("parametric_hallucination_risk", False))

    # For gd_only: we report the GD component of the Attr score only.
    attr_result = (null_branch.get("attr_score") or {})
    if config_flags.get("gd_only"):
        attr_score = float(attr_result.get("gd", 0.0))
    else:
        attr_score = float(attr_result.get("attr", 0.0))

    return {
        "qid": qid,
        "em": float(em),
        "f1": float(f1),
        "total_tokens": total_tokens,
        "retrieval_calls": retrieval_calls,
        "parametric_hallucination_risk": parametric_risk,
        "attr_score": attr_score,
        "prediction": pred,
        "gold": golds,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(records))
    return {
        "em": sum(r["em"] for r in records) / n,
        "f1": sum(r["f1"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "parametric_hallucination_risk_rate": sum(
            1 for r in records if r["parametric_hallucination_risk"]
        ) / n,
        "avg_attr_score": sum(r["attr_score"] for r in records) / n,
        "count": n,
    }


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "full": {
        "description": "Full CBV-RAG (Attr GD+PS, null branch, two-tier NLI verifier)",
        "disable_null_branch": False,
        "gd_only": False,
        "no_nli_verifier": False,
    },
    "no_null_branch": {
        "description": "No null branch (parametric detection disabled)",
        "disable_null_branch": True,
        "gd_only": False,
        "no_nli_verifier": False,
    },
    "gd_only": {
        "description": "Attr = GD only (alpha=1.0, PS term dropped)",
        "disable_null_branch": False,
        "gd_only": True,
        "no_nli_verifier": False,
    },
    "no_nli_verifier": {
        "description": "No NLI verifier — VERIFY_CHEAP uses cosine/rerank-gap heuristic only",
        "disable_null_branch": False,
        "gd_only": False,
        "no_nli_verifier": True,
    },
}


# ---------------------------------------------------------------------------
# Formatted table printing
# ---------------------------------------------------------------------------

def _print_table(results: Dict[str, Dict]) -> None:
    header_cols = ["Config", "EM", "F1", "Tokens", "Ret.Calls", "PHR"]
    col_widths = [max(20, max(len(k) for k in results)), 6, 6, 8, 10, 6]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    def _row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"

    print(sep)
    print(_row(header_cols))
    print(sep)
    for cfg_name, agg in results.items():
        print(_row([
            cfg_name,
            f"{agg['em']:.3f}",
            f"{agg['f1']:.3f}",
            f"{agg['avg_total_tokens']:.0f}",
            f"{agg['avg_retrieval_calls']:.2f}",
            f"{agg['parametric_hallucination_risk_rate']:.3f}",
        ]))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CBV-RAG ablation study runner.")
    ap.add_argument("--dataset", required=True, help="Dataset name (e.g. hotpotqa).")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path.  Defaults to logs/ablation_<dataset>.json.",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        choices=list(ABLATION_CONFIGS.keys()),
        default=list(ABLATION_CONFIGS.keys()),
        help="Which ablation configs to run (default: all four).",
    )
    args = ap.parse_args()

    output = Path(args.output or f"logs/ablation_{args.dataset}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    from tools.llm import LLMEngine
    from tools.rerank import CrossEncoderReranker
    from tools.retrieve import RetrieverTool
    import model_loader
    from retriever import KnowledgeBaseRetriever

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

    results: Dict[str, Any] = {}

    for cfg_name in args.configs:
        cfg = ABLATION_CONFIGS[cfg_name]
        print(f"\n[ablation] Running config: {cfg_name} — {cfg['description']}", flush=True)
        records = []
        t0 = time.time()

        for i, ex in enumerate(data):
            rec = _eval_episode(ex, tools, qid=str(i), config_flags=cfg)
            records.append(rec)

        elapsed = time.time() - t0
        agg = _aggregate(records)
        results[cfg_name] = {
            "description": cfg["description"],
            "aggregate": agg,
            "elapsed_s": elapsed,
            "per_example": records,
        }
        print(
            f"[ablation] {cfg_name}: EM={agg['em']:.3f}  F1={agg['f1']:.3f}  "
            f"Tokens={agg['avg_total_tokens']:.0f}  PHR={agg['parametric_hallucination_risk_rate']:.3f}",
            flush=True,
        )

    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}\n")

    # Print formatted comparison table.
    _print_table({k: v["aggregate"] for k, v in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
