from __future__ import annotations

from typing import Any, Dict, List

from cbvrag.actions import Action


class HeuristicController:
    def __init__(self) -> None:
        self.trace: List[Dict[str, Any]] = []

    def act(self, obs, state) -> int:
        step = state.step
        if step == 0:
            action = Action.RETRIEVE_MORE_SMALL
        elif step == 1:
            action = Action.SELECT_CONTEXT
        elif step == 2:
            pool_scores = sorted([e.rerank_score for e in state.evidence_pool.values()], reverse=True)
            gap = (pool_scores[0] - pool_scores[1]) if len(pool_scores) > 1 else 0.0
            if gap > 0.25 and len(state.selected_evidence_ids) >= 2:
                action = Action.STOP_AND_ANSWER
            else:
                action = Action.RETRIEVE_MORE_LARGE
        elif step == 3:
            action = Action.SELECT_CONTEXT
        elif step == 4 and len(state.branches) < state.budgets.get("max_branches", 3):
            action = Action.SPAWN_COUNTERFACTUAL
        elif step == 5:
            action = Action.VERIFY_CHEAP
        elif step == 6:
            action = Action.VERIFY_LLM if state.verification_status == "unknown" else Action.STOP_AND_ANSWER
        else:
            action = Action.STOP_AND_ANSWER

        self.trace.append({"obs": list(obs), "action": int(action), "reward": 0.0, "done": False, "info": {}})
        return int(action)
