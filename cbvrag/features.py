from __future__ import annotations

from typing import List

from cbvrag.state import EpisodeState


def _one_hot_verification(status: str) -> List[float]:
    vals = ["unknown", "supported", "contradicted"]
    return [1.0 if status == v else 0.0 for v in vals]


def build_features(state: EpisodeState) -> List[float]:
    budgets = state.budgets or {}
    max_steps = max(1, budgets.get("max_steps", 8))
    max_retrieval = max(1, budgets.get("max_retrieval_calls", 5))

    pool = list(state.evidence_pool.values())
    rerank_scores = sorted([e.rerank_score for e in pool], reverse=True)
    best = rerank_scores[0] if rerank_scores else 0.0
    second = rerank_scores[1] if len(rerank_scores) > 1 else 0.0
    mean_topk = sum(rerank_scores[:5]) / max(1, len(rerank_scores[:5]))

    retrieval_calls = state.metrics.get("retrieval_calls", 0)
    rerank_calls = state.metrics.get("rerank_calls", 0)
    verify_calls = state.metrics.get("verify_calls", 0)
    llm_calls = state.metrics.get("llm_calls", 0)

    remaining_budget_frac = max(0.0, 1 - (retrieval_calls / max_retrieval))

    vec = [
        state.step / max_steps,
        float(len(pool)),
        float(len(state.selected_evidence_ids)),
        float(retrieval_calls),
        float(rerank_calls),
        float(verify_calls),
        float(llm_calls),
        float(len([b for b in state.branches.values() if b.status != "pruned"])),
        best,
        best - second,
        mean_topk,
        float(state.metrics.get("redundancy_score", 0.0)),
        float(state.metrics.get("trap_score", 0.0)),
        remaining_budget_frac,
    ]
    return vec + _one_hot_verification(state.verification_status)
