from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional

from cbvrag.actions import Action
from cbvrag.controller_heuristic import HeuristicController


class TraceMixtureController:
    """
    Trace-collection controller:
    - mostly follows the heuristic
    - sometimes samples other legal actions
    - boosts underrepresented but useful actions

    This is for generating diverse IL / offline-RL traces, not for final deployment.
    """

    def __init__(
        self,
        seed: int = 42,
        heuristic_prob: float = 0.65,
        random_legal_prob: float = 0.20,
        rare_action_boost_prob: float = 0.15,
    ) -> None:
        self.trace: List[Dict[str, Any]] = []
        self.heuristic = HeuristicController()
        self.rng = random.Random(seed)

        total = heuristic_prob + random_legal_prob + rare_action_boost_prob
        self.heuristic_prob = heuristic_prob / total
        self.random_legal_prob = random_legal_prob / total
        self.rare_action_boost_prob = rare_action_boost_prob / total

        self.action_counter = Counter()

    @staticmethod
    def _legal_actions(action_mask: Optional[List[bool]]) -> List[Action]:
        if action_mask is None:
            return list(Action)
        return [Action(i) for i, ok in enumerate(action_mask) if ok]

    @staticmethod
    def _is_terminal_action(a: Action) -> bool:
        return a in {Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT}

    @staticmethod
    def _rare_priority_actions() -> List[Action]:
        return [
            Action.SPAWN_COUNTERFACTUAL,
            Action.VERIFY_CHEAP,
            Action.VERIFY_LLM,
            Action.SUMMARIZE_STATE,
            Action.PRUNE_BRANCH,
            Action.MERGE_BRANCHES,
            Action.STOP_AND_ANSWER,
            Action.ANSWER_DIRECT,
        ]

    def _pick_rare_boosted(self, legal: List[Action], state) -> Action:
        """
        Prefer rare but strategically meaningful actions when legal.
        """
        candidates = [a for a in self._rare_priority_actions() if a in legal]

        if not candidates:
            return self.rng.choice(legal)

        # Weight lower-count actions higher.
        weights = []
        for a in candidates:
            cnt = self.action_counter[int(a)]
            w = 1.0 / (1.0 + cnt)

            # Mild state-aware shaping.
            if a == Action.SPAWN_COUNTERFACTUAL and len(state.branches) < int(state.budgets.get("max_branches", 3)):
                w *= 1.5
            if a == Action.VERIFY_CHEAP and len(state.evidence_pool) > 0:
                w *= 1.4
            if a == Action.VERIFY_LLM and len(state.selected_evidence_ids) > 0:
                w *= 1.4
            if a in {Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT} and len(state.selected_evidence_ids) > 0:
                w *= 1.2
            if a == Action.SUMMARIZE_STATE and len(state.selected_evidence_ids) > 0:
                w *= 1.2

            weights.append(max(w, 1e-6))

        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for a, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                return a
        return candidates[-1]

    def act(self, obs, state, action_mask=None) -> int:
        legal = self._legal_actions(action_mask)
        if not legal:
            action = Action.STOP_AND_ANSWER
        else:
            heuristic_action = Action(self.heuristic.act(obs, state, action_mask=action_mask))
            mode_roll = self.rng.random()

            if heuristic_action not in legal:
                action = self.rng.choice(legal)
            elif mode_roll < self.heuristic_prob:
                action = heuristic_action
            elif mode_roll < self.heuristic_prob + self.random_legal_prob:
                non_terminal_legal = [a for a in legal if not self._is_terminal_action(a)]
                action = self.rng.choice(non_terminal_legal if non_terminal_legal else legal)
            else:
                action = self._pick_rare_boosted(legal, state)

        self.action_counter[int(action)] += 1

        self.trace.append(
            {
                "obs": list(obs),
                "action": int(action),
                "reward": 0.0,
                "done": False,
                "info": {
                    "step": state.step,
                    "legal_actions": [int(a) for a in legal],
                    "selected_evidence_count": len(state.selected_evidence_ids),
                    "retrieval_calls": int((state.metrics or {}).get("retrieval_calls", 0)),
                    "verification_status": state.verification_status,
                    "branch_count": len(state.branches),
                },
            }
        )
        return int(action)