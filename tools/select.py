from __future__ import annotations

import re
from typing import Any, Dict, List


def _token_len(tokenizer, text: str) -> int:
    text = text or ""
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


def _build_cluster_lookup(pool: List[Dict[str, Any]], cluster_info: List[Dict[str, Any]] | None) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if cluster_info:
        for cluster in cluster_info:
            cid = str(cluster.get("cluster_id", ""))
            for eid in cluster.get("member_ids", []):
                lookup[str(eid)] = cid

    if lookup:
        return lookup

    for item in pool:
        title = str(item.get("title", "") or "").strip().lower()
        if title:
            lookup[str(item.get("evidence_id", ""))] = f"title::{title}"
        else:
            lookup[str(item.get("evidence_id", ""))] = f"doc::{str(item.get('doc_id', '')).strip().lower()}"
    return lookup


def select_context_cluster_aware(
    question: str,
    pool: List[Dict[str, Any]],
    tokenizer,
    max_chunks: int,
    max_tokens: int,
    cluster_info: List[Dict[str, Any]] | None = None,
    per_cluster_soft_cap: int = 2,
) -> List[Dict[str, Any]]:
    if not pool:
        return []
    question_entities = {
        tok.lower()
        for tok in re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", question or "")
        if len(tok) > 2
    }

    cluster_lookup = _build_cluster_lookup(pool, cluster_info)
    cluster_selected_count: Dict[str, int] = {}

    scored: List[tuple[float, Dict[str, Any]]] = []
    for item in pool:
        rerank = float(item.get("rerank_score", 0.0))
        specificity = float(item.get("specificity", 0.0))
        novelty = float(item.get("novelty", 0.0))
        genericity = float(item.get("genericity", 0.0))
        text_l = str(item.get("text", "")).lower()
        entity_hits = sum(1 for ent in question_entities if ent in text_l)
        entity_bonus = 0.06 * min(3, entity_hits)
        base_score = 1.0 * rerank + 0.35 * novelty + 0.25 * specificity - 0.40 * genericity + entity_bonus
        scored.append((base_score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_tokens = 0
    selected_docs: set[str] = set()

    for _, item in scored:
        if len(selected) >= max_chunks:
            break

        text = str(item.get("text", "") or "")
        item_tokens = _token_len(tokenizer, text)
        if used_tokens + item_tokens > max_tokens:
            continue

        eid = str(item.get("evidence_id", ""))
        cid = cluster_lookup.get(eid, "unknown::default")
        already = cluster_selected_count.get(cid, 0)

        redundancy_penalty = 0.35 * max(0, already)
        adjusted_score = (
            1.0 * float(item.get("rerank_score", 0.0))
            + 0.35 * float(item.get("novelty", 0.0))
            + 0.25 * float(item.get("specificity", 0.0))
            - 0.40 * redundancy_penalty
        )

        if already >= per_cluster_soft_cap and adjusted_score < 0.75:
            continue

        doc_key = str(item.get("doc_id", "")).strip().lower()
        if len(selected) < 2 and selected_docs and doc_key in selected_docs:
            continue

        selected.append(item)
        used_tokens += item_tokens
        selected_docs.add(doc_key)
        cluster_selected_count[cid] = already + 1

    return selected


def select_context(
    question: str,
    pool: List[Dict[str, Any]],
    tokenizer,
    max_chunks: int,
    max_tokens: int,
    cluster_info: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    return select_context_cluster_aware(
        question=question,
        pool=pool,
        tokenizer=tokenizer,
        max_chunks=max_chunks,
        max_tokens=max_tokens,
        cluster_info=cluster_info,
    )
