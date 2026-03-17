from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def _norm(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    x = float(x)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _support_strength(ev: Any) -> float:
    rerank = float(getattr(ev, "rerank_score", 0.0))
    retriever = float(getattr(ev, "retriever_score", 0.0))
    # Conservative heuristic normalization.
    rerank_term = _norm(rerank)
    retriever_term = _norm(retriever)
    return _norm(0.7 * rerank_term + 0.3 * retriever_term)


def _source_key(ev: Any) -> str:
    title = str(getattr(ev, "title", "") or "").strip().lower()
    if title:
        return f"title::{title}"
    doc_id = str(getattr(ev, "doc_id", "") or "").strip().lower()
    if doc_id:
        return f"doc::{doc_id}"
    evidence_id = str(getattr(ev, "evidence_id", "") or "").strip().lower()
    return f"eid::{evidence_id.split('::')[0]}" if evidence_id else "unknown::default"


def score_evidence_specificity(question: str, evidence_items: Iterable[Any], selected_ids: Iterable[str] | None = None) -> Dict[str, Any]:
    del question  # Phase-1 heuristic only; question-aware scoring comes later.
    selected_ids = set(selected_ids or [])
    items = list(evidence_items)

    if not items:
        return {
            "chunk_scores": {},
            "summary": {
                "best_specificity_score": 0.0,
                "mean_specificity": 0.0,
                "mean_specificity_selected": 0.0,
                "best_support_strength": 0.0,
                "mean_genericity": 0.0,
            },
        }

    freq = Counter(_source_key(ev) for ev in items)

    chunk_scores: Dict[str, Dict[str, float]] = {}
    specificities = []
    support_strengths = []
    genericities = []
    selected_specificities = []

    for ev in items:
        evidence_id = str(getattr(ev, "evidence_id", ""))
        support = _support_strength(ev)

        source_count = freq[_source_key(ev)]
        # More repeated title/doc => more generic.
        genericity = _norm((source_count - 1) / max(1, len(items) - 1)) if len(items) > 1 else 0.0
        novelty = 1.0 - genericity

        if evidence_id in selected_ids:
            novelty = _norm(novelty + 0.05)

        specificity = _norm(support + 0.35 * novelty - 0.35 * genericity)

        scores = {
            "specificity": float(specificity),
            "genericity": float(genericity),
            "support_strength": float(support),
            "novelty": float(novelty),
        }
        chunk_scores[evidence_id] = scores

        specificities.append(scores["specificity"])
        support_strengths.append(scores["support_strength"])
        genericities.append(scores["genericity"])
        if evidence_id in selected_ids:
            selected_specificities.append(scores["specificity"])

    return {
        "chunk_scores": chunk_scores,
        "summary": {
            "best_specificity_score": float(max(specificities) if specificities else 0.0),
            "mean_specificity": float(sum(specificities) / max(1, len(specificities))),
            "mean_specificity_selected": float(
                sum(selected_specificities) / max(1, len(selected_specificities))
            ),
            "best_support_strength": float(max(support_strengths) if support_strengths else 0.0),
            "mean_genericity": float(sum(genericities) / max(1, len(genericities))),
        },
    }