from __future__ import annotations

from typing import List

from cbvrag.state import EpisodeState


def _one_hot_verification(status: str) -> List[float]:
    vals = ["unknown", "supported", "contradicted"]
    return [1.0 if status == v else 0.0 for v in vals]


def build_features(state: EpisodeState) -> List[float]:
    budgets = state.budgets or {}
    max_steps = max(1, int(budgets.get("max_steps", 8)))
    max_retrieval = max(1, int(budgets.get("max_retrieval_calls", 5)))
    max_branches = max(1, int(budgets.get("max_branches", 3)))
    max_context_chunks = max(1, int(budgets.get("max_context_chunks", 8)))

    pool = list(state.evidence_pool.values())
    rerank_scores = sorted([e.rerank_score for e in pool], reverse=True)
    best = rerank_scores[0] if rerank_scores else 0.0
    second = rerank_scores[1] if len(rerank_scores) > 1 else 0.0
    mean_topk = sum(rerank_scores[:5]) / max(1, len(rerank_scores[:5]))

    retrieval_calls = int(state.metrics.get("retrieval_calls", 0))
    rerank_calls = int(state.metrics.get("rerank_calls", 0))
    verify_calls = int(state.metrics.get("verify_calls", 0))
    llm_calls = int(state.metrics.get("llm_calls", 0))

    selected_count = len(state.selected_evidence_ids)
    pool_count = len(pool)
    active_branches = len([b for b in state.branches.values() if b.status != "pruned"])
    last_action = int(state.metrics.get("last_action", -1))
    no_progress_streak = int(state.metrics.get("no_progress_streak", 0))
    selected_changed = float(state.metrics.get("selected_evidence_changed", 0))
    pool_changed = float(state.metrics.get("evidence_pool_changed", 0))

    # existing/core features first (backward-compat ordering)
    vec = [
        state.step / max_steps,
        float(pool_count),
        float(selected_count),
        float(retrieval_calls),
        float(rerank_calls),
        float(verify_calls),
        float(llm_calls),
        float(active_branches),
        best,
        best - second,
        mean_topk,
        float(state.metrics.get("redundancy_score", 0.0)),
        float(state.metrics.get("trap_score", 0.0)),
        max(0.0, 1.0 - (retrieval_calls / max_retrieval)),
    ] + _one_hot_verification(state.verification_status)

    # appended richer control features
    selected_frac_of_pool = selected_count / max(1, pool_count)
    branch_frac = active_branches / max_branches
    retrieval_frac = retrieval_calls / max_retrieval
    step_frac = state.step / max_steps

    unique_titles = set()
    for eid in state.selected_evidence_ids:
        ev = state.evidence_pool.get(eid)
        if ev and getattr(ev, "title", ""):
            t = (ev.title or "").strip()
            if t:
                unique_titles.add(t)

    has_summary = 1.0 if (state.global_summary or "").strip() else 0.0
    has_final_answer = 1.0 if (state.final_answer or "").strip() else 0.0
    cheap_verified = 1.0 if verify_calls > 0 else 0.0
    llm_verified = 1.0 if verify_calls > 1 else 0.0
    multi_branch = 1.0 if active_branches > 1 else 0.0
    near_retrieval_exhaustion = 1.0 if retrieval_calls >= max(1, max_retrieval - 1) else 0.0
    near_step_exhaustion = 1.0 if state.step >= max(1, max_steps - 1) else 0.0
    context_pressure = selected_count / max_context_chunks
    score_std_proxy = abs(best - mean_topk)
    conflicting_evidence = 1.0 if (best - second) < 0.05 and pool_count >= 2 else 0.0

    last_action_norm = (last_action / max(1, len(_one_hot_verification("unknown")) + 8)) if last_action >= 0 else -1.0
    last_action_onehot = [1.0 if i == last_action else 0.0 for i in range(11)]

    vec.extend(
        [
            retrieval_frac,
            branch_frac,
            float(selected_count),
            float(pool_count),
            has_final_answer,
            has_summary,
            cheap_verified,
            llm_verified,
            multi_branch,
            selected_frac_of_pool,
            near_retrieval_exhaustion,
            near_step_exhaustion,
            context_pressure,
            float(len(unique_titles)),
            score_std_proxy,
            conflicting_evidence,
            step_frac,
            float(no_progress_streak) / max_steps,
            selected_changed,
            pool_changed,
            last_action_norm,
        ]
    )
    vec.extend(last_action_onehot)
    return [float(v) for v in vec]
