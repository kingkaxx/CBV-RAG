from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict


def _safe(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    if v == float("inf"):
        return 1.0
    if v == float("-inf"):
        return -1.0
    return float(max(-1.0, min(1.0, v)))


def score_evidence_specificity(
    original_question: str,
    question_variants: dict,
    retrieved_by_query: dict,
    reranker=None,
) -> Dict[str, Any]:
    # lightweight approximation: relevance is taken from rerank/retriever scores already available.
    chunk_obs: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: {"orig": [], "alt": [], "cf": [], "cluster": []})

    for qkey, items in (retrieved_by_query or {}).items():
        qtype = str(qkey)
        for it in items or []:
            if not isinstance(it, dict):
                continue
            eid = str(it.get("evidence_id", ""))
            if not eid:
                continue
            rel = float(it.get("rerank_score", it.get("retriever_score", 0.0)))
            if qtype == "original":
                chunk_obs[eid]["orig"].append(rel)
            elif "counter" in qtype or "disprove" in qtype or "contrast" in qtype:
                chunk_obs[eid]["cf"].append(rel)
                chunk_obs[eid]["alt"].append(rel)
            else:
                chunk_obs[eid]["alt"].append(rel)
            cid = str(it.get("cluster_id", ""))
            if cid:
                chunk_obs[eid]["cluster"].append(cid)

    chunk_scores: Dict[str, Dict[str, float]] = {}
    cluster_aggr: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: {"specificity": [], "counterfactual_resistance": [], "support": [], "genericity": []})

    for eid, parts in chunk_obs.items():
        orig_rel = float(mean(parts["orig"]) if parts["orig"] else 0.0)
        alt_rel = float(mean(parts["alt"]) if parts["alt"] else 0.0)
        cf_rel = float(mean(parts["cf"]) if parts["cf"] else alt_rel)
        specificity = _safe(orig_rel - alt_rel)
        resistance = _safe(orig_rel - cf_rel)
        genericity = _safe(alt_rel)
        rec = {
            "orig_rel": _safe(orig_rel),
            "alt_rel_mean": _safe(alt_rel),
            "cf_rel_mean": _safe(cf_rel),
            "specificity": specificity,
            "counterfactual_resistance": resistance,
            "genericity": genericity,
        }
        chunk_scores[eid] = rec
        for cid in set(parts["cluster"]):
            if not cid:
                continue
            cluster_aggr[cid]["specificity"].append(specificity)
            cluster_aggr[cid]["counterfactual_resistance"].append(resistance)
            cluster_aggr[cid]["support"].append(orig_rel)
            cluster_aggr[cid]["genericity"].append(genericity)

    cluster_scores: Dict[str, Dict[str, float]] = {}
    for cid, ag in cluster_aggr.items():
        cluster_scores[cid] = {
            "specificity": _safe(mean(ag["specificity"]) if ag["specificity"] else 0.0),
            "counterfactual_resistance": _safe(mean(ag["counterfactual_resistance"]) if ag["counterfactual_resistance"] else 0.0),
            "support_strength": _safe(mean(ag["support"]) if ag["support"] else 0.0),
            "genericity": _safe(mean(ag["genericity"]) if ag["genericity"] else 0.0),
        }

    return {
        "chunk_scores": chunk_scores,
        "cluster_scores": cluster_scores,
    }
