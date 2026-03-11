from __future__ import annotations


def compute_reward(state, action, terminal_correct=None, step_costs=None) -> float:
    step_costs = step_costs or {}
    reward = 0.0
    reward += -0.001 * float(step_costs.get("tokens_used_this_step", 0))
    reward += -0.05 * float(step_costs.get("retrieval_calls_this_step", 0))
    reward += -0.05 * float(step_costs.get("new_branch_created", 0))

    if terminal_correct is not None:
        reward += 1.0 if terminal_correct else 0.0
    if state.verification_status == "supported":
        reward += 0.1
    return reward
