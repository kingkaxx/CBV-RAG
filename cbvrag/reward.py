from __future__ import annotations


def _to_bool_flag(x) -> float:
    return 1.0 if bool(x) else 0.0


def compute_reward(state, action, terminal_correct=None, step_costs=None) -> float:
    """
    Reward aligned with the current runner/env semantics.

    Goals:
    - Strongly reward correct terminal answers.
    - Penalize expensive / wasteful behavior.
    - Reward progress signals without letting them dominate the terminal objective.
    - Penalize rescue behavior when the runner/env had to force actions.
    """
    step_costs = step_costs or {}
    metrics = getattr(state, "metrics", {}) or {}

    reward = 0.0

    # Cost penalties
    reward += -0.0005 * float(step_costs.get("tokens_used_this_step", 0))
    reward += -0.08 * float(step_costs.get("retrieval_calls_this_step", 0))
    reward += -0.06 * float(step_costs.get("new_branch_created", 0))

    # Dense progress rewards
    reward += 0.03 * float(metrics.get("selected_evidence_changed", 0))
    reward += 0.02 * float(metrics.get("evidence_pool_changed", 0))
    reward += 0.02 * float(metrics.get("branch_count_changed", 0))
    reward += 0.03 * float(metrics.get("verification_status_changed", 0))

    # Verification shaping
    if getattr(state, "verification_status", "unknown") == "supported":
        reward += 0.08
    elif getattr(state, "verification_status", "unknown") == "contradicted":
        reward -= 0.05

    # Penalize stagnation and rescue behavior.
    reward += -0.04 * float(metrics.get("no_progress_streak", 0))
    reward += -0.03 * float(metrics.get("forced_action_count", 0) > 0)
    reward += -0.03 * float(metrics.get("illegal_action_requested", 0) > 0)

    # Gentle pressure not to exhaust budgets.
    retrieval_calls = float(metrics.get("retrieval_calls", 0))
    max_retrieval = max(1.0, float((getattr(state, "budgets", {}) or {}).get("max_retrieval_calls", 5)))
    reward += -0.03 * max(0.0, retrieval_calls / max_retrieval - 0.6)

    step_idx = float(getattr(state, "step", 0))
    max_steps = max(1.0, float((getattr(state, "budgets", {}) or {}).get("max_steps", 8)))
    reward += -0.02 * max(0.0, step_idx / max_steps - 0.75)

    # Terminal reward should dominate.
    if terminal_correct is not None:
        if terminal_correct:
            reward += 2.5
            if getattr(state, "verification_status", "unknown") == "supported":
                reward += 0.25
            if int(metrics.get("retrieval_calls", 0)) <= 2:
                reward += 0.10
        else:
            reward -= 1.5
            if not (getattr(state, "final_answer", "") or "").strip():
                reward -= 0.25

    return float(reward)
