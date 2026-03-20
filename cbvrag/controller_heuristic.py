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
    def _rerank_gap(state) -> float:
        scores = HeuristicController._pool_scores(state)
        return float((scores[0] - scores[1]) if len(scores) > 1 else 0.0)

    @staticmethod
    def _selected_count(state) -> int:
        return len(state.selected_evidence_ids)

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

    @staticmethod
    def _retrieval_calls(state) -> int:
        return int((state.metrics or {}).get("retrieval_calls", 0))

    @staticmethod
    def _verification_calls(state) -> int:
        return int((state.metrics or {}).get("verify_calls", 0))

    def _is_strong_early_stop(self, state) -> bool:
        return (
            self._selected_count(state) >= 3
            and self._selected_unique_titles(state) >= 2
            and self._rerank_gap(state) > 0.35
        )

    def _has_verification_signal(self, state) -> bool:
        return self._verification_calls(state) > 0 or state.verification_status in {"supported", "contradicted"}

    def act(self, obs, state, action_mask=None) -> int:
        step = state.step
        selected_count = self._selected_count(state)
        unique_title_count = self._selected_unique_titles(state)
        rerank_gap = self._rerank_gap(state)
        retrieval_calls = self._retrieval_calls(state)
        branch_count = len(state.branches)
        max_branches = int(state.budgets.get("max_branches", 3))
        verification_status = state.verification_status

        pool_nonempty = len(state.evidence_pool) > 0
        selected_nonempty = selected_count > 0
        verify_calls = self._verification_calls(state)

        # PRIORITY RULE 1: After any retrieval, always select context first
        if retrieval_calls > 0 and pool_nonempty and not selected_nonempty:
            action = Action.SELECT_CONTEXT

        # PRIORITY RULE 2: After context selection, always verify before stopping
        elif selected_nonempty and verification_status == "unknown" and verify_calls == 0:
            action = Action.VERIFY_CHEAP

        # PRIORITY RULE 3: Strong evidence + verified = stop
        elif (
            verification_status == "supported"
            and selected_count >= 2
            and unique_title_count >= 2
        ):
            action = Action.STOP_AND_ANSWER

        # PRIORITY RULE 4: Need more evidence after first retrieval
        elif retrieval_calls < 2 and selected_count < 2:
            action = Action.RETRIEVE_MORE_LARGE

        # PRIORITY RULE 5: Need second retrieval for multi-hop
        elif retrieval_calls < 2 and unique_title_count < 2:
            action = Action.RETRIEVE_MORE_SMALL

        # PRIORITY RULE 6: Contradicted — spawn counterfactual branch
        elif verification_status == "contradicted" and branch_count < max_branches:
            action = Action.SPAWN_COUNTERFACTUAL

        # PRIORITY RULE 7: Unknown after verification — escalate to LLM verify
        elif selected_nonempty and verification_status == "unknown" and verify_calls > 0:
            action = Action.VERIFY_LLM

        # DEFAULT: stop and answer
        else:
            action = Action.STOP_AND_ANSWER

        # Do not allow early STOP before verification unless evidence is very strong.
        if action == Action.STOP_AND_ANSWER and self._verification_calls(state) == 0 and not self._is_strong_early_stop(state):
            action = Action.VERIFY_CHEAP

        self.trace.append(
            {
                "obs": list(obs),
                "action": action.value,
                "reward": 0.0,
                "done": False,
                "info": {
                    "step": step,
                    "rerank_gap": rerank_gap,
                    "selected_evidence_count": selected_count,
                    "unique_title_count": unique_title_count,
                    "retrieval_calls": retrieval_calls,
                    "verification_status": verification_status,
                    "branch_count": branch_count,
                },
            }
        )
        return action.value
