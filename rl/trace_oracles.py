from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from cbvrag.actions import Action


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    gc = {}
    for t in g:
        gc[t] = gc.get(t, 0) + 1
    for t in p:
        if gc.get(t, 0) > 0:
            common += 1
            gc[t] -= 1
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def qa_em(pred: str, gold: str) -> bool:
    ng = normalize_answer(gold)
    np = normalize_answer(pred)
    return bool(ng) and ng in np


def _rerank_gap(state: Any) -> float:
    scores = sorted([e.rerank_score for e in state.evidence_pool.values()], reverse=True)
    return float((scores[0] - scores[1]) if len(scores) > 1 else 0.0)


def _selected_count(state: Any) -> int:
    return len(state.selected_evidence_ids)


def _unique_selected_titles(state: Any) -> int:
    titles = set()
    for eid in state.selected_evidence_ids:
        ev = state.evidence_pool.get(eid)
        if ev and getattr(ev, "title", ""):
            titles.add((ev.title or "").strip())
    return len([t for t in titles if t])


def estimate_case_profile(state: Any) -> str:
    retrieval_calls = int((state.metrics or {}).get("retrieval_calls", 0))
    branches = len(state.branches)
    selected = _selected_count(state)
    uniq_titles = _unique_selected_titles(state)
    gap = _rerank_gap(state)

    if selected >= 3 and uniq_titles >= 2 and gap > 0.28 and retrieval_calls <= 1:
        return "easy"
    if gap < 0.1 or uniq_titles < 2:
        return "ambiguous"
    if branches >= max(1, int(state.budgets.get("max_branches", 3)) - 1) or retrieval_calls >= max(1, int(state.budgets.get("max_retrieval_calls", 5)) - 1):
        return "hard_overloaded"
    return "standard"


class OracleControllerBase:
    oracle_name = "base"

    def __init__(self, seed: int = 42) -> None:
        self.trace: List[Dict[str, Any]] = []
        self.rng = random.Random(seed)

    def _log(self, obs: List[float], action: Action, state: Any, reason: str) -> int:
        self.trace.append(
            {
                "obs": list(obs),
                "action": int(action),
                "reward": 0.0,
                "done": False,
                "info": {
                    "oracle_name": self.oracle_name,
                    "case_profile": estimate_case_profile(state),
                    "action_reason": reason,
                    "rerank_gap": _rerank_gap(state),
                    "selected_count": _selected_count(state),
                    "unique_title_count": _unique_selected_titles(state),
                },
            }
        )
        return int(action)


class EfficientOracle(OracleControllerBase):
    oracle_name = "efficient"

    def act(self, obs, state) -> int:
        step = state.step
        if step == 0:
            return self._log(obs, Action.RETRIEVE_MORE_SMALL, state, "cheap_initial_retrieval")
        if step == 1:
            return self._log(obs, Action.SELECT_CONTEXT, state, "pack_minimal_context")
        if step == 2 and _selected_count(state) >= 2 and _rerank_gap(state) > 0.25:
            return self._log(obs, Action.ANSWER_DIRECT, state, "high_confidence_direct_answer")
        if state.verification_status == "unknown" and step <= 3:
            return self._log(obs, Action.VERIFY_CHEAP, state, "fast_verification")
        return self._log(obs, Action.STOP_AND_ANSWER, state, "finish_efficiently")


class SafeOracle(OracleControllerBase):
    oracle_name = "safe"

    def act(self, obs, state) -> int:
        step = state.step
        if step == 0:
            return self._log(obs, Action.RETRIEVE_MORE_LARGE, state, "safe_broad_retrieval")
        if step == 1:
            return self._log(obs, Action.SELECT_CONTEXT, state, "select_after_broad_retrieval")
        if step == 2 and _selected_count(state) < 2:
            return self._log(obs, Action.RETRIEVE_MORE_SMALL, state, "top_up_for_insufficient_evidence")
        if step in {2, 3}:
            return self._log(obs, Action.VERIFY_CHEAP, state, "cheap_safety_check")
        if state.verification_status == "unknown" and step <= 5:
            return self._log(obs, Action.VERIFY_LLM, state, "escalate_verification")
        return self._log(obs, Action.STOP_AND_ANSWER, state, "safe_finish")


class ExploratoryOracle(OracleControllerBase):
    oracle_name = "exploratory"

    def act(self, obs, state) -> int:
        step = state.step
        max_branches = int(state.budgets.get("max_branches", 3))
        if step == 0:
            return self._log(obs, Action.RETRIEVE_MORE_LARGE, state, "explore_with_broad_retrieval")
        if step == 1:
            return self._log(obs, Action.SELECT_CONTEXT, state, "select_for_ambiguity_check")
        if len(state.branches) < max_branches and (_rerank_gap(state) < 0.15 or _unique_selected_titles(state) < 2):
            return self._log(obs, Action.SPAWN_COUNTERFACTUAL, state, "branch_on_ambiguity")
        if len(state.branches) > 1 and _rerank_gap(state) > 0.2:
            return self._log(obs, Action.PRUNE_BRANCH, state, "prune_dominated_branch")
        if len(state.branches) > 1 and _selected_count(state) >= 3:
            return self._log(obs, Action.MERGE_BRANCHES, state, "merge_complementary_branches")
        if state.verification_status == "unknown":
            return self._log(obs, Action.VERIFY_CHEAP, state, "cheap_verify_after_exploration")
        return self._log(obs, Action.STOP_AND_ANSWER, state, "finish_exploratory")


class DeliberativeOracle(OracleControllerBase):
    oracle_name = "deliberative"

    def act(self, obs, state) -> int:
        step = state.step
        if step == 0:
            return self._log(obs, Action.RETRIEVE_MORE_LARGE, state, "deliberative_broad_retrieval")
        if step == 1:
            return self._log(obs, Action.SELECT_CONTEXT, state, "deliberative_context_pack")
        if _selected_count(state) >= 5 or len(state.evidence_pool) >= 20:
            return self._log(obs, Action.SUMMARIZE_STATE, state, "compress_large_state")
        if state.verification_status == "unknown":
            return self._log(obs, Action.VERIFY_LLM, state, "llm_verify_for_hard_case")
        if _selected_count(state) < 2:
            return self._log(obs, Action.RETRIEVE_MORE_SMALL, state, "top_up_before_finalize")
        return self._log(obs, Action.STOP_AND_ANSWER, state, "deliberative_finish")


ORACLE_REGISTRY = {
    "efficient": EfficientOracle,
    "safe": SafeOracle,
    "exploratory": ExploratoryOracle,
    "deliberative": DeliberativeOracle,
}


def parse_oracle_mix(mix: str | None) -> Dict[str, float]:
    if not mix:
        return {"efficient": 0.2, "safe": 0.45, "exploratory": 0.2, "deliberative": 0.15}
    mix = mix.strip()
    if mix.startswith("{"):
        obj = json.loads(mix)
        return {str(k): float(v) for k, v in obj.items() if k in ORACLE_REGISTRY}
    out: Dict[str, float] = {}
    for part in mix.split(","):
        if not part.strip() or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        if k in ORACLE_REGISTRY:
            out[k] = float(v)
    return out or parse_oracle_mix(None)


def sample_oracle_name(case_profile: str, oracle_mix: Dict[str, float], rng: random.Random) -> str:
    adjusted = dict(oracle_mix)
    if case_profile == "easy":
        adjusted["efficient"] = adjusted.get("efficient", 0.0) + 0.2
    elif case_profile == "ambiguous":
        adjusted["exploratory"] = adjusted.get("exploratory", 0.0) + 0.25
    elif case_profile == "hard_overloaded":
        adjusted["deliberative"] = adjusted.get("deliberative", 0.0) + 0.25

    names = [k for k in adjusted.keys() if k in ORACLE_REGISTRY and adjusted[k] > 0]
    weights = [adjusted[n] for n in names]
    total = sum(weights)
    if total <= 0:
        return "safe"
    weights = [w / total for w in weights]
    return rng.choices(names, weights=weights, k=1)[0]


def build_oracle_controller(oracle_name: str, seed: int = 42):
    if oracle_name not in ORACLE_REGISTRY:
        raise ValueError(f"Unknown oracle_name={oracle_name}")
    return ORACLE_REGISTRY[oracle_name](seed=seed)


@dataclass
class TrajectoryScoreConfig:
    success_reward: float = 3.0
    em_reward: float = 1.5
    f1_reward: float = 1.0
    support_reward: float = 0.75
    token_penalty: float = 0.001
    step_penalty: float = 0.03
    branch_penalty: float = 0.06
    redundant_verify_penalty: float = 0.03


def score_trajectory(
    *,
    success: bool,
    em: float,
    f1: float,
    support_hit: float,
    tokens_used: int,
    steps: int,
    branches: int,
    verify_calls: int,
    cfg: TrajectoryScoreConfig,
) -> float:
    return (
        (cfg.success_reward if success else 0.0)
        + cfg.em_reward * float(em)
        + cfg.f1_reward * float(f1)
        + cfg.support_reward * float(support_hit)
        - cfg.token_penalty * float(tokens_used)
        - cfg.step_penalty * float(steps)
        - cfg.branch_penalty * float(max(0, branches - 1))
        - cfg.redundant_verify_penalty * float(max(0, verify_calls - 1))
    )


def compute_episode_quality(pred: str, golds: List[str]) -> Tuple[float, float, bool]:
    em = max((1.0 if qa_em(pred, g) else 0.0) for g in (golds or [""]))
    f1 = max(qa_f1(pred, g) for g in (golds or [""]))
    return em, f1, bool(em > 0)
