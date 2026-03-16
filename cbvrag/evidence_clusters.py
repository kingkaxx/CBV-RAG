from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _cluster_key(item: Dict[str, Any]) -> str:
    title = _norm(str(item.get("title") or ""))
    if title:
        return f"title::{title}"

    doc_id = _norm(str(item.get("doc_id") or ""))
    if doc_id:
        return f"doc::{doc_id}"

    evidence_id = str(item.get("evidence_id") or "")
    if "::" in evidence_id:
        return f"eid_prefix::{evidence_id.split('::')[0]}"
    if evidence_id:
        return f"eid::{evidence_id}"
    return "unknown"


def cluster_evidence_items(evidence_items: list) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in evidence_items or []:
        item = raw if isinstance(raw, dict) else {}
        buckets[_cluster_key(item)].append(item)

    clusters: List[Dict[str, Any]] = []
    for idx, (_, members) in enumerate(buckets.items()):
        reranks = [float(m.get("rerank_score", 0.0)) for m in members]
        retrs = [float(m.get("retriever_score", 0.0)) for m in members]
        selected = [m for m in members if bool(m.get("is_selected", False))]
        clusters.append(
            {
                "cluster_id": f"c{idx}",
                "member_ids": [str(m.get("evidence_id", "")) for m in members if str(m.get("evidence_id", ""))],
                "doc_ids": sorted({str(m.get("doc_id", "")) for m in members if str(m.get("doc_id", ""))}),
                "titles": sorted({str(m.get("title", "")) for m in members if str(m.get("title", ""))}),
                "size": int(len(members)),
                "mean_rerank": float(mean(reranks) if reranks else 0.0),
                "max_rerank": float(max(reranks) if reranks else 0.0),
                "mean_retriever": float(mean(retrs) if retrs else 0.0),
                "selected_count": int(len(selected)),
            }
        )

    clusters.sort(key=lambda c: (c.get("mean_rerank", 0.0), c.get("max_rerank", 0.0), c.get("size", 0)), reverse=True)
    for i, c in enumerate(clusters):
        c["cluster_id"] = f"c{i}"
    return clusters


def summarize_cluster_stats(clusters: list[dict]) -> Dict[str, float]:
    clusters = list(clusters or [])
    if not clusters:
        return {
            "num_clusters": 0,
            "largest_cluster_size": 0,
            "largest_cluster_frac": 0.0,
            "top_cluster_mean_rerank": 0.0,
            "second_cluster_mean_rerank": 0.0,
            "cluster_gap": 0.0,
            "selected_cluster_count": 0,
            "selected_cluster_diversity": 0.0,
            "selected_same_cluster_frac": 0.0,
            "evidence_redundancy_proxy": 0.0,
            "multi_cluster_support_flag": 0.0,
        }

    sizes = [int(c.get("size", 0)) for c in clusters]
    total = max(1, sum(sizes))
    mean_reranks = [float(c.get("mean_rerank", 0.0)) for c in clusters]
    selected_sizes = [int(c.get("selected_count", 0)) for c in clusters]
    selected_total = sum(selected_sizes)
    selected_cluster_count = sum(1 for s in selected_sizes if s > 0)
    selected_same_cluster_frac = (max(selected_sizes) / max(1, selected_total)) if selected_total > 0 else 0.0

    return {
        "num_clusters": int(len(clusters)),
        "largest_cluster_size": int(max(sizes) if sizes else 0),
        "largest_cluster_frac": float((max(sizes) if sizes else 0) / total),
        "top_cluster_mean_rerank": float(mean_reranks[0] if mean_reranks else 0.0),
        "second_cluster_mean_rerank": float(mean_reranks[1] if len(mean_reranks) > 1 else 0.0),
        "cluster_gap": float((mean_reranks[0] - mean_reranks[1]) if len(mean_reranks) > 1 else (mean_reranks[0] if mean_reranks else 0.0)),
        "selected_cluster_count": int(selected_cluster_count),
        "selected_cluster_diversity": float(selected_cluster_count / max(1, len(clusters))),
        "selected_same_cluster_frac": float(selected_same_cluster_frac),
        "evidence_redundancy_proxy": float(selected_same_cluster_frac),
        "multi_cluster_support_flag": float(1.0 if selected_cluster_count >= 2 else 0.0),
    }
