from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def _norm_text(x: Any) -> str:
    return str(x or "").strip().lower()


def _cluster_key(ev: Any) -> str:
    title = _norm_text(getattr(ev, "title", ""))
    if title:
        return f"title::{title}"
    doc_id = _norm_text(getattr(ev, "doc_id", ""))
    if doc_id:
        return f"doc::{doc_id}"
    evidence_id = _norm_text(getattr(ev, "evidence_id", ""))
    if evidence_id:
        prefix = evidence_id.split("::")[0]
        return f"eid::{prefix}"
    return "unknown::default"


def cluster_evidence_items(evidence_items: Iterable[Any], selected_ids: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    selected_ids = set(selected_ids or [])
    grouped: Dict[str, List[Any]] = defaultdict(list)

    for ev in evidence_items:
        grouped[_cluster_key(ev)].append(ev)

    clusters: List[Dict[str, Any]] = []
    for cid, items in grouped.items():
        member_ids = [str(getattr(ev, "evidence_id", "")) for ev in items]
        doc_ids = sorted({str(getattr(ev, "doc_id", "")) for ev in items if getattr(ev, "doc_id", "")})
        titles = sorted({str(getattr(ev, "title", "")).strip() for ev in items if str(getattr(ev, "title", "")).strip()})
        rerank_scores = [float(getattr(ev, "rerank_score", 0.0)) for ev in items]
        retriever_scores = [float(getattr(ev, "retriever_score", 0.0)) for ev in items]
        selected_count = sum(1 for mid in member_ids if mid in selected_ids)

        clusters.append(
            {
                "cluster_id": cid,
                "member_ids": member_ids,
                "doc_ids": doc_ids,
                "titles": titles,
                "size": len(items),
                "mean_rerank": float(sum(rerank_scores) / max(1, len(rerank_scores))),
                "max_rerank": float(max(rerank_scores) if rerank_scores else 0.0),
                "mean_retriever": float(sum(retriever_scores) / max(1, len(retriever_scores))),
                "selected_count": int(selected_count),
            }
        )

    clusters.sort(key=lambda c: (c["mean_rerank"], c["max_rerank"], c["size"]), reverse=True)
    return clusters


def summarize_cluster_stats(clusters: List[Dict[str, Any]]) -> Dict[str, float]:
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
            "selected_same_cluster_frac": 0.0,
            "evidence_redundancy_proxy": 0.0,
            "multi_cluster_support_flag": 0.0,
        }

    total_items = sum(int(c["size"]) for c in clusters)
    selected_total = sum(int(c["selected_count"]) for c in clusters)

    largest_cluster_size = max(int(c["size"]) for c in clusters)
    top_mean = float(clusters[0]["mean_rerank"])
    second_mean = float(clusters[1]["mean_rerank"]) if len(clusters) > 1 else 0.0

    selected_cluster_count = sum(1 for c in clusters if int(c["selected_count"]) > 0)
    selected_same_cluster_frac = (
        max(int(c["selected_count"]) for c in clusters) / max(1, selected_total) if selected_total > 0 else 0.0
    )

    return {
        "num_clusters": float(len(clusters)),
        "largest_cluster_size": float(largest_cluster_size),
        "largest_cluster_frac": float(largest_cluster_size / max(1, total_items)),
        "top_cluster_mean_rerank": top_mean,
        "second_cluster_mean_rerank": second_mean,
        "cluster_gap": float(top_mean - second_mean),
        "selected_cluster_count": float(selected_cluster_count),
        "selected_cluster_diversity": float(selected_cluster_count / max(1, len(clusters))),
        "selected_same_cluster_frac": float(selected_same_cluster_frac),
        "evidence_redundancy_proxy": float(largest_cluster_size / max(1, total_items)),
        "multi_cluster_support_flag": 1.0 if selected_cluster_count >= 2 else 0.0,
    }