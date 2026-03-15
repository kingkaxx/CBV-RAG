from __future__ import annotations

from typing import Any, Dict, List

from cbvrag.actions import Action


class HeuristicController:
    def __init__(self) -> None:
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _pool_scores(state) -> List[float]:
        return sorted([e.rerank_score for e in state.evidence_pool.values()], reverse=True)

    @staticmethod
    def _selected_unique_titles(state) -> int:
        titles = set()
        for eid in state.selected_evidence_ids:
            ev = state.evidence_pool.get(eid)
            if not ev:
                continue
            title = (getattr(ev, "title", "") or "").strip()
            if title:
                titles.add(title)
        return len(titles)

    def _is_extremely_strong_evidence(self, state) -> bool:
        # Very strict gate for rare early stop cases.
        scores = self._pool_scores(state)
        gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
        selected = len(state.selected_evidence_ids)
        unique_titles = self._selected_unique_titles(state)
        return gap >= 0.8 and selected >= 5 and unique_titles >= 4

    def _is_weak_or_low_diversity(self, state) -> bool:
        scores = self._pool_scores(state)
        gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
        selected = len(state.selected_evidence_ids)
        unique_titles = self._selected_unique_titles(state)

        weak_gap = gap < 0.22
        too_few_selected = selected < 3
        low_diversity = unique_titles < 2
        return weak_gap or too_few_selected or low_diversity

    def _verification_calls(self, state) -> int:
        return int((state.metrics or {}).get("verify_calls", 0))

    def act(self, obs, state) -> int:
        step = state.step

        # Stronger staged expert policy.
        if step == 0:
            action = Action.RETRIEVE_MORE_LARGE
        elif step == 1:
            action = Action.SELECT_CONTEXT
        elif step == 2:
            if self._is_extremely_strong_evidence(state):
                action = Action.STOP_AND_ANSWER
            elif self._is_weak_or_low_diversity(state):
                action = Action.RETRIEVE_MORE_LARGE
            elif len(state.branches) < state.budgets.get("max_branches", 3):
                action = Action.SPAWN_COUNTERFACTUAL
            else:
                action = Action.RETRIEVE_MORE_LARGE
        elif step == 3:
            action = Action.SELECT_CONTEXT
        elif step == 4:
            action = Action.VERIFY_CHEAP
        elif step == 5:
            action = Action.VERIFY_LLM if state.verification_status == "unknown" else Action.STOP_AND_ANSWER
        else:
            action = Action.STOP_AND_ANSWER

        # Require at least one verification step before stopping in most cases.
        if action == Action.STOP_AND_ANSWER and self._verification_calls(state) == 0 and step >= 4:
            action = Action.VERIFY_CHEAP

        self.trace.append({"obs": list(obs), "action": int(action), "reward": 0.0, "done": False, "info": {}})
        return int(action)
