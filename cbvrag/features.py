from __future__ import annotations

from typing import List

from cbvrag.actions import Action
from cbvrag.state import EpisodeState


FEATURE_SCHEMA_VERSION = "cbvrag_features_v4"


def _one_hot_verification(status: str) -> List[float]:
    vals = ["unknown", "supported", "contradicted"]
    return [1.0 if status == v else 0.0 for v in vals]


def _safe_div(a: float, b: float) -> float:
    return float(a) / max(1.0, float(b))


def build_features(state: EpisodeState) -> List[float]:
    budgets = state.budgets or {}
    max_steps = max(1, int(budgets.get("max_steps", 8)))
    max_retrieval = max(1, int(budgets.get("max_retrieval_calls", 5)))
    max_branches = max(1, int(budgets.get("max_branches", 3)))
    max_context_chunks = max(1, int(budgets.get("max_context_chunks", 8)))

    pool = list(state.evidence_pool.values())
    rerank_scores = sorted([float(e.rerank_score) for e in pool], reverse=True)
    best = rerank_scores[0] if rerank_scores else 0.0
    second = rerank_scores[1] if len(rerank_scores) > 1 else 0.0
    mean_topk = sum(rerank_scores[:5]) / max(1, len(rerank_scores[:5]))
    min_topk = min(rerank_scores[:5]) if rerank_scores else 0.0

    retrieval_calls = int(state.metrics.get("retrieval_calls", 0))
    rerank_calls = int(state.metrics.get("rerank_calls", 0))
    verify_calls = int(state.metrics.get("verify_calls", 0))
    llm_calls = int(state.metrics.get("llm_calls", 0))

    selected_count = len(state.selected_evidence_ids)
    pool_count = len(pool)
    active_branches = len([b for b in state.branches.values() if getattr(b, "status", "active") != "pruned"])

    last_action = int(state.metrics.get("last_action", -1))
    second_last_action = int(state.metrics.get("second_last_action", -1))
    no_progress_streak = int(state.metrics.get("no_progress_streak", 0))
    selected_changed = float(state.metrics.get("selected_evidence_changed", 0))
    pool_changed = float(state.metrics.get("evidence_pool_changed", 0))
    branch_count_changed = float(state.metrics.get("branch_count_changed", 0))
    verification_status_changed = float(state.metrics.get("verification_status_changed", 0))
    previous_selected_count = int(state.metrics.get("previous_selected_count", 0))
    forced_action_count = int(state.metrics.get("forced_action_count", 0))
    illegal_action_requested = int(state.metrics.get("illegal_action_requested", 0))
    explicit_stop_used = int(state.metrics.get("explicit_stop_used", 0))
    forced_stop_used = int(state.metrics.get("forced_stop_used", 0))
    fallback_stop_was_used = int(state.metrics.get("fallback_stop_was_used", 0))

    selected_evidence = [state.evidence_pool[eid] for eid in state.selected_evidence_ids if eid in state.evidence_pool]
    selected_scores = sorted([float(e.rerank_score) for e in selected_evidence], reverse=True)
    selected_best = selected_scores[0] if selected_scores else 0.0
    selected_mean = sum(selected_scores) / max(1, len(selected_scores))
    selected_gap = selected_scores[0] - selected_scores[1] if len(selected_scores) > 1 else selected_best

    unique_titles = set()
    for ev in selected_evidence:
        t = (getattr(ev, "title", "") or "").strip()
        if t:
            unique_titles.add(t)

    selected_title_diversity = _safe_div(len(unique_titles), max(1, selected_count))
    selected_frac_of_pool = _safe_div(selected_count, max(1, pool_count))
    branch_frac = _safe_div(active_branches, max_branches)
    retrieval_frac = _safe_div(retrieval_calls, max_retrieval)
    step_frac = _safe_div(state.step, max_steps)
    context_pressure = _safe_div(selected_count, max_context_chunks)

    has_global_summary = 1.0 if (state.global_summary or "").strip() else 0.0
    active_branch = state.branches.get(state.active_branch_id)
    has_branch_summary = 1.0 if (getattr(active_branch, "summary", "") or "").strip() else 0.0
    has_final_answer = 1.0 if (state.final_answer or "").strip() else 0.0

    selected_nonempty = 1.0 if selected_count > 0 else 0.0
    retrieval_nonempty = 1.0 if retrieval_calls > 0 else 0.0
    pool_nonempty = 1.0 if pool_count > 0 else 0.0
    multi_branch = 1.0 if active_branches > 1 else 0.0

    near_retrieval_exhaustion = 1.0 if retrieval_calls >= max(1, max_retrieval - 1) else 0.0
    near_step_exhaustion = 1.0 if state.step >= max(1, max_steps - 1) else 0.0
    conflicting_evidence = 1.0 if (best - second) < 0.05 and pool_count >= 2 else 0.0
    score_std_proxy = abs(best - mean_topk)

    # Better proxies for control readiness.
    can_stop_now = 1.0 if (selected_count > 0 and (retrieval_calls >= 2 or state.verification_status == "supported")) else 0.0
    can_select_now = 1.0 if pool_count > 0 else 0.0
    can_verify_cheap_now = 1.0 if pool_count > 0 else 0.0
    can_verify_llm_now = 1.0 if selected_count > 0 else 0.0
    can_spawn_branch = 1.0 if active_branches < max_branches else 0.0
    can_prune_branch = 1.0 if active_branches > 1 else 0.0
    retrieval_budget_left = max(0.0, 1.0 - retrieval_frac)
    step_budget_left = max(0.0, 1.0 - step_frac)

    # Keep the original/core block first for backward-ish stability.
    vec = [
        step_frac,
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
        retrieval_budget_left,
    ] + _one_hot_verification(state.verification_status)

    # Richer control and quality features.
    vec.extend(
        [
            retrieval_frac,
            branch_frac,
            has_final_answer,
            has_global_summary,
            has_branch_summary,
            multi_branch,
            selected_frac_of_pool,
            near_retrieval_exhaustion,
            near_step_exhaustion,
            context_pressure,
            selected_nonempty,
            retrieval_nonempty,
            pool_nonempty,
            float(len(unique_titles)),
            selected_title_diversity,
            score_std_proxy,
            conflicting_evidence,
            float(no_progress_streak) / max_steps,
            selected_changed,
            pool_changed,
            branch_count_changed,
            verification_status_changed,
            _safe_div(previous_selected_count, max_context_chunks),
            _safe_div(forced_action_count, max_steps),
            _safe_div(illegal_action_requested, max_steps),
            float(explicit_stop_used),
            float(forced_stop_used),
            float(fallback_stop_was_used),
            selected_best,
            selected_mean,
            selected_gap,
            min_topk,
            can_stop_now,
            can_select_now,
            can_verify_cheap_now,
            can_verify_llm_now,
            can_spawn_branch,
            can_prune_branch,
            retrieval_budget_left,
            step_budget_left,
            1.0 if second_last_action == last_action and last_action >= 0 else 0.0,
        ]
    )

    # Safer action encoding: derive dimensionality from Action enum.
    act_dim = len(Action)
    last_action_norm = (_safe_div(last_action, max(1, act_dim - 1)) if last_action >= 0 else -1.0)
    second_last_action_norm = (_safe_div(second_last_action, max(1, act_dim - 1)) if second_last_action >= 0 else -1.0)
    vec.extend([last_action_norm, second_last_action_norm])

    last_action_onehot = [1.0 if i == last_action else 0.0 for i in range(act_dim)]
    second_last_action_onehot = [1.0 if i == second_last_action else 0.0 for i in range(act_dim)]
    vec.extend(last_action_onehot)
    vec.extend(second_last_action_onehot)

    return [float(v) for v in vec]


def feature_schema_version() -> str:
    return FEATURE_SCHEMA_VERSION
