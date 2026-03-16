from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict


def _clip01(x: float) -> float:
    if x != x:
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _safe_str(v: Any) -> str:
    return str(v or "").strip()


def score_evidence_specificity(question: str, evidence_items: list) -> Dict[str, Any]:
    items = [it for it in (evidence_items or []) if isinstance(it, dict)]
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

    title_keys = [(_safe_str(it.get("title")) or _safe_str(it.get("doc_id")) or "unknown") for it in items]
    counts = Counter(title_keys)
    max_repeat = max(1, max(counts.values()))
    reranks = [float(it.get("rerank_score", 0.0)) for it in items]
    retrs = [float(it.get("retriever_score", 0.0)) for it in items]
    rr_min, rr_max = min(reranks), max(reranks)
    rt_min, rt_max = min(retrs), max(retrs)

    def _norm(v: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        return _clip01((v - lo) / (hi - lo))

    chunk_scores: Dict[str, Dict[str, float]] = {}
    specificities, genericities, supports = [], [], []
    selected_specificities = []

    for it, tk in zip(items, title_keys):
        eid = _safe_str(it.get("evidence_id"))
        if not eid:
            continue
        rr = _norm(float(it.get("rerank_score", 0.0)), rr_min, rr_max)
        rt = _norm(float(it.get("retriever_score", 0.0)), rt_min, rt_max)
        support = _clip01(0.7 * rr + 0.3 * rt)

        repeat_frac = counts[tk] / max_repeat
        genericity = _clip01(0.8 * repeat_frac + (0.2 if counts[tk] > 1 else 0.0))
        novelty = _clip01(1.0 - repeat_frac)
        specificity = _clip01(support + 0.5 * novelty - 0.6 * genericity)

        rec = {
            "specificity": float(specificity),
            "genericity": float(genericity),
            "support_strength": float(support),
            "novelty": float(novelty),
        }
        chunk_scores[eid] = rec
        specificities.append(specificity)
        genericities.append(genericity)
        supports.append(support)
        if bool(it.get("is_selected", False)):
            selected_specificities.append(specificity)

    summary = {
        "best_specificity_score": float(max(specificities) if specificities else 0.0),
        "mean_specificity": float(mean(specificities) if specificities else 0.0),
        "mean_specificity_selected": float(mean(selected_specificities) if selected_specificities else 0.0),
        "best_support_strength": float(max(supports) if supports else 0.0),
        "mean_genericity": float(mean(genericities) if genericities else 0.0),
    }
    return {"chunk_scores": chunk_scores, "summary": summary}
