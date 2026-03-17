from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cbvrag.actions import Action
from cbvrag.features import build_features
from cbvrag.reward import compute_reward
from cbvrag.runner import (
    _make_state,
    choose_valid_action,
    compute_action_mask,
    default_budgets,
    execute_action,
)


class CBVRAGEnv:
    """
    RL environment aligned with runner.py semantics.

    Key fixes vs the older version:
    - Uses the same action mask and fallback resolution as run_episode.
    - Treats both ANSWER_DIRECT and STOP_AND_ANSWER as terminal actions.
    - Tracks whether the requested action was illegal and had to be forced.
    - Surfaces richer info so RL/debugging can detect policy rescue behavior.
    """

    def __init__(self, tools: Dict[str, Any], budgets: Optional[Dict] = None) -> None:
        self.tools = tools
        self.budgets = {**default_budgets(), **(budgets or {})}
        self.state = None
        self.gold = None

    def reset(self, qid: str, question: str, gold: str):
        self.state = _make_state(question, qid, self.budgets)
        self.gold = gold
        return build_features(self.state)

    def step(self, action: int) -> Tuple[list, float, bool, Dict]:
        if self.state is None:
            raise RuntimeError("Environment must be reset() before step().")

        requested_action = int(action)
        action_mask = compute_action_mask(self.state)
        resolved_action, action_was_forced, requested_action = choose_valid_action(
            requested_action,
            self.state,
            action_mask,
        )

        # Match runner.py behavior: don't allow an effectively premature stop.
        if (
            resolved_action == Action.STOP_AND_ANSWER
            and self.state.verification_status == "unknown"
            and self.state.step < self.budgets["max_steps"] - 1
            and (
                len(self.state.selected_evidence_ids) == 0
                or int(self.state.metrics.get("retrieval_calls", 0)) == 0
                or self.state.step == 0
            )
        ):
            if action_mask[int(Action.VERIFY_CHEAP)]:
                resolved_action = Action.VERIFY_CHEAP
                action_was_forced = True

        costs = execute_action(self.state, resolved_action, controller=None, tools=self.tools)

        pred = (self.state.final_answer or "").strip().lower()
        terminal = (
            resolved_action in (Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT)
            or self.state.step >= self.budgets["max_steps"]
            or int(self.state.metrics.get("retrieval_calls", 0)) >= self.budgets["max_retrieval_calls"]
        )

        terminal_correct = None
        if terminal:
            terminal_correct = self.gold.strip().lower() in pred if self.gold else False

        reward = compute_reward(
            self.state,
            int(resolved_action),
            terminal_correct=terminal_correct,
            step_costs=costs,
        )
        obs = build_features(self.state)
        info = {
            "costs": costs,
            "terminal_correct": terminal_correct,
            "requested_action": requested_action,
            "executed_action": int(resolved_action),
            "executed_action_name": resolved_action.name,
            "action_was_forced": bool(action_was_forced),
            "action_mask": list(action_mask),
            "verification_status": self.state.verification_status,
            "retrieval_calls": int(self.state.metrics.get("retrieval_calls", 0)),
            "no_progress_streak": int(self.state.metrics.get("no_progress_streak", 0)),
        }
        return obs, reward, terminal, info
