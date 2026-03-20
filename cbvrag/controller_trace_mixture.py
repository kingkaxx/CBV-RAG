from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional

from cbvrag.actions import Action
from cbvrag.controller_heuristic import HeuristicController


class TraceMixtureController:
    """
    Trace-collection controller:
    - follows heuristic often enough to stay grounded
    - injects legal exploration
    - boosts underrepresented actions in state-appropriate situations
    - discourages collapse to only dominant retrieval/select patterns

    This is for generating diverse IL/offline-RL traces, not for deployment.
    """

    def __init__(
        self,
        seed: int = 42,
        heuristic_prob: float = 0.50,
        random_legal_prob: float = 0.15,
        rare_action_boost_prob: float = 0.35,
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
    def _retrieval_actions() -> List[Action]:
        return [Action.RETRIEVE_MORE_SMALL, Action.RETRIEVE_MORE_LARGE]

    @staticmethod
    def _dominant_actions() -> List[Action]:
        # These are the actions that tend to dominate your traces.
        return [
            Action.RETRIEVE_MORE_SMALL,
            Action.RETRIEVE_MORE_LARGE,
            Action.SELECT_CONTEXT,
        ]

    def _state_features(self, state) -> Dict[str, Any]:
        retrieval_calls = int((state.metrics or {}).get("retrieval_calls", 0))
        verify_calls = int((state.metrics or {}).get("verify_calls", 0))
        no_progress = int((state.metrics or {}).get("no_progress_streak", 0))
        selected_count = len(state.selected_evidence_ids)
        pool_count = len(state.evidence_pool)
        branch_count = len(state.branches)
        max_branches = int(state.budgets.get("max_branches", 3))
        return {
            "step": int(state.step),
            "retrieval_calls": retrieval_calls,
            "verify_calls": verify_calls,
            "no_progress": no_progress,
            "selected_count": selected_count,
            "pool_count": pool_count,
            "branch_count": branch_count,
            "can_branch_more": branch_count < max_branches,
            "verification_status": state.verification_status,
            "selected_nonempty": selected_count > 0,
            "pool_nonempty": pool_count > 0,
        }

    def _heuristic_bias(self, step: int) -> float:
        # Early steps should still stay grounded; later steps can diversify more.
        if step <= 1:
            return 1.20
        if step <= 3:
            return 1.00
        return 0.80

    def _anti_collapse_penalty(self, action: Action) -> float:
        """
        Penalize actions that are already overrepresented globally in this controller instance.
        """
        cnt = self.action_counter[int(action)]
        if action in self._dominant_actions():
            return 1.0 / (1.0 + 0.20 * cnt)
        return 1.0 / (1.0 + 0.05 * cnt)

    def _state_appropriate_weight(self, action: Action, s: Dict[str, Any]) -> float:
        """
        Reward actions that make sense in the current state.
        """
        w = 1.0

        # Retrieve is useful early and when pool is thin.
        if action in self._retrieval_actions():
            if s["retrieval_calls"] == 0:
                w *= 1.60
            elif s["retrieval_calls"] == 1:
                w *= 1.25
            elif s["retrieval_calls"] >= 2:
                w *= 0.75
            if not s["pool_nonempty"]:
                w *= 1.20

        # Select is useful only once some pool exists.
        if action == Action.SELECT_CONTEXT:
            if s["pool_nonempty"]:
                w *= 1.25
            else:
                w *= 0.50
            if s["selected_count"] > 0:
                w *= 0.90

        # Cheap verify is useful once evidence exists.
        if action == Action.VERIFY_CHEAP:
            if s["pool_nonempty"]:
                w *= 1.35
            else:
                w *= 0.40
            if s["verification_status"] == "unknown":
                w *= 1.20

        # LLM verify is useful when selected evidence exists and cheap verify may be insufficient.
        if action == Action.VERIFY_LLM:
            if s["selected_nonempty"]:
                w *= 1.40
            else:
                w *= 0.35
            if s["verification_status"] == "unknown":
                w *= 1.10

        # Summarize becomes more useful later with selected evidence.
        if action == Action.SUMMARIZE_STATE:
            if s["selected_nonempty"] and s["step"] >= 3:
                w *= 1.35
            else:
                w *= 0.70

        # Branching should happen only when there is uncertainty and room to branch.
        if action == Action.SPAWN_COUNTERFACTUAL:
            if s["can_branch_more"]:
                w *= 1.20
            else:
                w *= 0.20
            if s["verification_status"] in {"unknown", "contradicted"} and s["retrieval_calls"] >= 1:
                w *= 1.30
            if not s["pool_nonempty"]:
                w *= 0.60

        # Stop / answer should happen only once selected evidence exists.
        if action in {Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT}:
            if s["selected_nonempty"]:
                w *= 1.15
            else:
                w *= 0.20
            if s["verification_status"] == "supported":
                w *= 1.25
            if s["retrieval_calls"] >= 2:
                w *= 1.10

        # Branch maintenance actions are useful only with multiple branches.
        if action in {Action.PRUNE_BRANCH, Action.MERGE_BRANCHES}:
            if s["branch_count"] > 1:
                w *= 1.25
            else:
                w *= 0.25

        # If no progress, push toward non-dominant strategic actions.
        if s["no_progress"] >= 1:
            if action in {Action.VERIFY_CHEAP, Action.VERIFY_LLM, Action.SPAWN_COUNTERFACTUAL, Action.SUMMARIZE_STATE}:
                w *= 1.20
            if action in self._dominant_actions():
                w *= 0.80

        return max(w, 1e-6)

    def _pick_weighted(self, legal: List[Action], weights: List[float]) -> Action:
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(legal)
        r = self.rng.random() * total
        acc = 0.0
        for a, w in zip(legal, weights):
            acc += w
            if acc >= r:
                return a
        return legal[-1]

    def _pick_rare_boosted(self, legal: List[Action], state) -> Action:
        s = self._state_features(state)

        # Candidate set: emphasize strategically meaningful actions, but keep legal retrieval/select if needed.
        candidate_priority = [
            Action.VERIFY_CHEAP,
            Action.VERIFY_LLM,
            Action.SPAWN_COUNTERFACTUAL,
            Action.SUMMARIZE_STATE,
            Action.STOP_AND_ANSWER,
            Action.ANSWER_DIRECT,
            Action.PRUNE_BRANCH,
            Action.MERGE_BRANCHES,
            Action.SELECT_CONTEXT,
            Action.RETRIEVE_MORE_SMALL,
            Action.RETRIEVE_MORE_LARGE,
        ]
        candidates = [a for a in candidate_priority if a in legal]
        if not candidates:
            return self.rng.choice(legal)

        weights = []
        for a in candidates:
            cnt = self.action_counter[int(a)]

            # Stronger rare-action boost than before.
            rarity = 1.0 / (1.0 + 0.75 * cnt)

            state_fit = self._state_appropriate_weight(a, s)
            anti_collapse = self._anti_collapse_penalty(a)

            # Encourage strategic diversity after early retrieval phase.
            step_factor = 1.0
            if s["step"] >= 2 and a in {Action.VERIFY_CHEAP, Action.VERIFY_LLM, Action.SPAWN_COUNTERFACTUAL, Action.SUMMARIZE_STATE}:
                step_factor *= 1.25
            if s["step"] <= 1 and a in {Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT, Action.PRUNE_BRANCH, Action.MERGE_BRANCHES}:
                step_factor *= 0.30

            w = rarity * state_fit * anti_collapse * step_factor
            weights.append(max(w, 1e-6))

        return self._pick_weighted(candidates, weights)

    def _filter_state_valid(self, legal, state):
        """Remove actions invalid for current state."""
        retrieval_calls = int((state.metrics or {}).get('retrieval_calls', 0))
        selected_empty = len(state.selected_evidence_ids) == 0
        from cbvrag.actions import Action as _A
        invalid = set()
        if retrieval_calls == 0 or selected_empty:
            invalid.update([int(_A.SPAWN_COUNTERFACTUAL), int(_A.PRUNE_BRANCH), int(_A.MERGE_BRANCHES)])
        if selected_empty:
            invalid.update([int(_A.VERIFY_CHEAP), int(_A.VERIFY_LLM), int(_A.STOP_AND_ANSWER), int(_A.ANSWER_DIRECT), int(_A.SUMMARIZE_STATE)])
        if retrieval_calls >= 2 and selected_empty:
            invalid.update([int(_A.RETRIEVE_MORE_SMALL), int(_A.RETRIEVE_MORE_LARGE)])
        filtered = [a for a in legal if int(a) not in invalid]
        return filtered if filtered else legal

    def _pick_random_legal(self, legal: List[Action], state) -> Action:
        s = self._state_features(state)

        # Random legal, but still somewhat state-aware and anti-collapse.
        weights = []
        for a in legal:
            w = self._state_appropriate_weight(a, s) * self._anti_collapse_penalty(a)

            # Avoid too much random terminal behavior early.
            if s["step"] <= 1 and self._is_terminal_action(a):
                w *= 0.20

            # Encourage non-dominant exploration.
            if a not in self._dominant_actions():
                w *= 1.15

            weights.append(max(w, 1e-6))

        return self._pick_weighted(legal, weights)

    def act(self, obs, state, action_mask=None) -> int:
        legal = self._legal_actions(action_mask)
        if not legal:
            action = Action.STOP_AND_ANSWER
        else:
            s = self._state_features(state)
            heuristic_action = Action(self.heuristic.act(obs, state, action_mask=action_mask))
            mode_roll = self.rng.random()

            if heuristic_action not in legal:
                action = self._pick_random_legal(self._filter_state_valid(legal, state), state)
            elif mode_roll < self.heuristic_prob * self._heuristic_bias(s["step"]):
                action = heuristic_action
            elif mode_roll < self.heuristic_prob * self._heuristic_bias(s["step"]) + self.random_legal_prob:
                action = self._pick_random_legal(self._filter_state_valid(legal, state), state)
            else:
                action = self._pick_rare_boosted(self._filter_state_valid(legal, state), state)

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
                    "action_histogram_snapshot": dict(self.action_counter),
                },
            }
        )
        return int(action)