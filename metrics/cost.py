from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class CostMetrics:
    retrieval_calls: int = 0
    rerank_calls: int = 0
    verify_calls: int = 0
    branch_count: int = 1
    steps: int = 0


class CostTracker:
    def __init__(self) -> None:
        self.metrics = CostMetrics()

    def inc_retrieval(self, n: int = 1) -> None:
        self.metrics.retrieval_calls += n

    def inc_rerank(self, n: int = 1) -> None:
        self.metrics.rerank_calls += n

    def inc_verify(self, n: int = 1) -> None:
        self.metrics.verify_calls += n

    def inc_steps(self, n: int = 1) -> None:
        self.metrics.steps += n

    def set_branch_count(self, branch_count: int) -> None:
        self.metrics.branch_count = max(0, int(branch_count))

    def to_dict(self) -> Dict:
        return asdict(self.metrics)
