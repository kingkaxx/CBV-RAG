from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    correctness_reward: float = 1.0
    token_penalty: float = 0.001
    retrieval_penalty: float = 0.03
    branch_penalty: float = 0.05
    verify_bonus: float = 0.03
    early_stop_bonus: float = 0.03
    support_hit_reward: float = 0.15
    support_full_reward: float = 0.25
    support_rank_reward: float = 0.1
    discrimination_reward: float = 0.1
    contradiction_bonus: float = 0.1
    use_support_reward: bool = True
    use_verification_reward: bool = True
    use_efficiency_penalty: bool = True
    use_counterfactual_discrimination_reward: bool = True


def compute_reward_components(
    *,
    terminal_correct: bool,
    tokens_used: int,
    retrieval_calls: int,
    branches_created: int,
    verify_calls: int,
    early_stop: bool,
    cfg: RewardConfig,
    support_pages_hit: int = 0,
    support_pages_total: int = 0,
    support_best_rank: int | None = None,
    discrimination_gain: float = 0.0,
    contradiction_sensitive_verification: bool = False,
) -> Dict[str, float]:
    correctness = cfg.correctness_reward if terminal_correct else 0.0

    token_cost = -cfg.token_penalty * float(tokens_used) if cfg.use_efficiency_penalty else 0.0
    retrieval_cost = -cfg.retrieval_penalty * float(retrieval_calls) if cfg.use_efficiency_penalty else 0.0
    branch_cost = -cfg.branch_penalty * float(max(0, branches_created)) if cfg.use_efficiency_penalty else 0.0

    verify = cfg.verify_bonus * float(verify_calls) if cfg.use_verification_reward else 0.0
    early_bonus = cfg.early_stop_bonus if early_stop else 0.0

    support_hit = 0.0
    support_full = 0.0
    support_rank = 0.0
    if cfg.use_support_reward and support_pages_total > 0:
        if support_pages_hit > 0:
            support_hit = cfg.support_hit_reward
        if support_pages_hit >= support_pages_total:
            support_full = cfg.support_full_reward
        if support_best_rank is not None and support_best_rank > 0:
            support_rank = cfg.support_rank_reward / float(support_best_rank)

    discrimination = (
        cfg.discrimination_reward * float(discrimination_gain)
        if cfg.use_counterfactual_discrimination_reward
        else 0.0
    )
    contradiction = cfg.contradiction_bonus if contradiction_sensitive_verification else 0.0

    total = (
        correctness
        + token_cost
        + retrieval_cost
        + branch_cost
        + verify
        + early_bonus
        + support_hit
        + support_full
        + support_rank
        + discrimination
        + contradiction
    )
    return {
        "correctness": correctness,
        "token_cost": token_cost,
        "retrieval_cost": retrieval_cost,
        "branch_cost": branch_cost,
        "verify_bonus": verify,
        "early_stop_bonus": early_bonus,
        "support_hit_reward": support_hit,
        "support_full_reward": support_full,
        "support_rank_reward": support_rank,
        "counterfactual_discrimination_reward": discrimination,
        "contradiction_bonus": contradiction,
        "total": total,
    }
