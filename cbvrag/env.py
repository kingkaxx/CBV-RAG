from __future__ import annotations

import re
import string
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


# ---------------------------------------------------------------------------
# Answer extraction + EM — self-contained, no circular import
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract_answer(pred: str) -> str:
    """Extract the answer span from raw LLM output.

    FIX: Old code used `gold.strip().lower() in pred` which is a raw substring
    match on the full unprocessed output. This caused two bugs:
      1. Template text matching gold ("1755" never in "...[your concise answer]...")
         → always False, EM=0 even when correct
      2. Short gold strings matching anywhere in a long prediction
         → False positives (e.g. "no" in any long negative answer)
    """
    if not pred:
        return ""
    # Strip echoed "Answer:" prefix
    pred = re.sub(r"^Answer:\s*", "", pred.strip(), flags=re.IGNORECASE)
    # Drop Reasoning block
    if "\nReasoning:" in pred:
        pred = pred.split("\nReasoning:")[0]
    if "\nAnswer:" in pred:
        pred = pred.split("\nAnswer:")[0]
    # First non-empty line
    for line in pred.split("\n"):
        line = line.strip()
        if line:
            return re.sub(r"[.!?]+$", "", line).strip()
    return pred.strip()


def _em_correct(pred: str, gold: str) -> bool:
    """Exact match after normalization — used for terminal reward signal."""
    return _normalize(_extract_answer(pred)) == _normalize(gold)


def _any_em_correct(pred: str, golds) -> bool:
    """Max-over-golds EM, handles both str and list gold."""
    if isinstance(golds, str):
        golds = [golds]
    pred_norm = _normalize(_extract_answer(pred))
    return any(pred_norm == _normalize(str(g)) for g in golds if str(g).strip())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CBVRAGEnv:
    """
    RL environment aligned with runner.py semantics.

    Key fixes vs the older version:
    - Uses the same action mask and fallback resolution as run_episode.
    - Treats both ANSWER_DIRECT and STOP_AND_ANSWER as terminal actions.
    - Tracks whether the requested action was illegal and had to be forced.
    - terminal_correct uses normalized EM (not raw substring match).
    - gold can be str or List[str].
    """

    def __init__(self, tools: Dict[str, Any], budgets: Optional[Dict] = None) -> None:
        self.tools = tools
        self.budgets = {**default_budgets(), **(budgets or {})}
        self.state = None
        self.gold = None

    def reset(self, qid: str, question: str, gold):
        """
        Parameters
        ----------
        gold : str or List[str]
            Accepted answer(s). Stored as-is; _any_em_correct handles both.
        """
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

        # Match runner.py behavior: don't allow premature stop.
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

        pred = (self.state.final_answer or "").strip()
        terminal = (
            resolved_action in (Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT)
            or self.state.step >= self.budgets["max_steps"]
            or int(self.state.metrics.get("retrieval_calls", 0)) >= self.budgets["max_retrieval_calls"]
        )

        # FIX: use normalized EM instead of raw substring match
        terminal_correct = None
        if terminal:
            if self.gold:
                terminal_correct = _any_em_correct(pred, self.gold)
            else:
                terminal_correct = False

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
            "prediction_extracted": _extract_answer(pred),  # debug
        }
        return obs, reward, terminal, info