from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    correctness_reward: float = 1.0
    token_penalty: float = 0.001
    retrieval_penalty: float = 0.05
    branch_penalty: float = 0.1
    verify_bonus: float = 0.02
    early_stop_bonus: float = 0.05


def compute_reward_components(
    *,
    terminal_correct: bool,
    tokens_used: int,
    retrieval_calls: int,
    branches_created: int,
    verify_calls: int,
    early_stop: bool,
    cfg: RewardConfig,
) -> Dict[str, float]:
    correctness = cfg.correctness_reward if terminal_correct else 0.0
    token_cost = -cfg.token_penalty * float(tokens_used)
    retrieval_cost = -cfg.retrieval_penalty * float(retrieval_calls)
    branch_cost = -cfg.branch_penalty * float(max(0, branches_created))
    verify = cfg.verify_bonus * float(verify_calls)
    early_bonus = cfg.early_stop_bonus if early_stop else 0.0
    total = correctness + token_cost + retrieval_cost + branch_cost + verify + early_bonus
    return {
        "correctness": correctness,
        "token_cost": token_cost,
        "retrieval_cost": retrieval_cost,
        "branch_cost": branch_cost,
        "verify_bonus": verify,
        "early_stop_bonus": early_bonus,
        "total": total,
    }
