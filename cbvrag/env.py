from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cbvrag.actions import Action
from cbvrag.features import build_features
from cbvrag.reward import compute_reward
from cbvrag.runner import _make_state, default_budgets, execute_action


class CBVRAGEnv:
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
        costs = execute_action(self.state, Action(action), controller=None, tools=self.tools)
        pred = (self.state.final_answer or "").strip().lower()
        terminal = Action(action) == Action.STOP_AND_ANSWER or self.state.step >= self.budgets["max_steps"]
        terminal_correct = None
        if terminal:
            terminal_correct = self.gold.strip().lower() in pred if self.gold else False
        reward = compute_reward(self.state, action, terminal_correct=terminal_correct, step_costs=costs)
        obs = build_features(self.state)
        info = {"costs": costs, "terminal_correct": terminal_correct}
        return obs, reward, terminal, info
