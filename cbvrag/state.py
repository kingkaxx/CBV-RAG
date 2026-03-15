from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EvidenceItem:
    evidence_id: str
    doc_id: str
    chunk_id: str
    retriever_score: float
    rerank_score: float
    short_claim: str
    branch_id: str
    title: str = ""


@dataclass
class Branch:
    branch_id: str
    parent_id: Optional[str]
    hypothesis: str
    status: str = "active"
    confidence: float = 0.0
    step_created: int = 0
    cost_spent: float = 0.0
    summary: str = ""


@dataclass
class EpisodeState:
    question: str
    qid: str
    branches: Dict[str, Branch]
    active_branch_id: str
    evidence_pool: Dict[str, EvidenceItem] = field(default_factory=dict)
    selected_evidence_ids: List[str] = field(default_factory=list)
    global_summary: str = ""
    step: int = 0
    budgets: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    final_answer: str = ""
    verification_status: str = "unknown"

    def add_edge(self, src: str, dst: str, edge_type: str) -> None:
        self.edges.setdefault(src, []).append((dst, edge_type))
