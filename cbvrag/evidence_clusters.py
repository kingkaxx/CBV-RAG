from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List


def _cluster_key(item: Dict[str, Any]) -> str:
    title = str(item.get("title") or "").strip().lower()
    if title:
        return f"title::{title}"
    return f"doc::{item.get('doc_id', '')}"


def cluster_evidence_items(
    evidence_items: list,
    embedding_model=None,
    enable_semantic_merge: bool = False,
) -> List[Dict[str, Any]]:
    # Phase-1 robust clustering: group by title/doc_id. semantic merge intentionally optional/no-op for now.
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in evidence_items or []:
        item = raw if isinstance(raw, dict) else {}
        buckets[_cluster_key(item)].append(item)

    clusters: List[Dict[str, Any]] = []
    for idx, (_, members) in enumerate(sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)):
        reranks = [float(m.get("rerank_score", 0.0)) for m in members]
        retrs = [float(m.get("retriever_score", 0.0)) for m in members]
        selected = [m for m in members if bool(m.get("is_selected", False))]
        branch_ids = sorted({str(m.get("branch_id", "")) for m in members if str(m.get("branch_id", ""))})
        clusters.append(
            {
                "cluster_id": f"c{idx}",
                "member_ids": [str(m.get("evidence_id", "")) for m in members if str(m.get("evidence_id", ""))],
                "doc_ids": sorted({str(m.get("doc_id", "")) for m in members if str(m.get("doc_id", ""))}),
                "titles": sorted({str(m.get("title", "")) for m in members if str(m.get("title", ""))}),
                "size": len(members),
                "mean_rerank": float(mean(reranks) if reranks else 0.0),
                "max_rerank": float(max(reranks) if reranks else 0.0),
                "mean_retriever": float(mean(retrs) if retrs else 0.0),
                "selected_count": len(selected),
                "branch_ids": branch_ids,
            }
        )

    # deterministic order by rerank support then size.
    clusters.sort(key=lambda c: (c.get("mean_rerank", 0.0), c.get("max_rerank", 0.0), c.get("size", 0)), reverse=True)
    for i, c in enumerate(clusters):
        c["cluster_id"] = f"c{i}"
    return clusters


def summarize_cluster_stats(clusters) -> Dict[str, float]:
    clusters = list(clusters or [])
    if not clusters:
        return {
            "num_clusters": 0.0,
            "largest_cluster_size": 0.0,
            "largest_cluster_frac": 0.0,
            "top_cluster_mean_rerank": 0.0,
            "second_cluster_mean_rerank": 0.0,
            "cluster_gap": 0.0,
            "selected_cluster_count": 0.0,
            "selected_cluster_diversity": 0.0,
        }

    sizes = [int(c.get("size", 0)) for c in clusters]
    total = max(1, sum(sizes))
    mean_reranks = [float(c.get("mean_rerank", 0.0)) for c in clusters]
    selected_cluster_count = sum(1 for c in clusters if int(c.get("selected_count", 0)) > 0)
    return {
        "num_clusters": float(len(clusters)),
        "largest_cluster_size": float(max(sizes) if sizes else 0),
        "largest_cluster_frac": float((max(sizes) if sizes else 0) / total),
        "top_cluster_mean_rerank": float(mean_reranks[0] if mean_reranks else 0.0),
        "second_cluster_mean_rerank": float(mean_reranks[1] if len(mean_reranks) > 1 else 0.0),
        "cluster_gap": float((mean_reranks[0] - mean_reranks[1]) if len(mean_reranks) > 1 else (mean_reranks[0] if mean_reranks else 0.0)),
        "selected_cluster_count": float(selected_cluster_count),
        "selected_cluster_diversity": float(selected_cluster_count / max(1, len(clusters))),
    }
