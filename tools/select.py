from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from cbvrag.evidence_clusters import cluster_evidence_items


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _title(cand: Dict) -> str:
    return (cand.get("title") or "").strip()


def select_context(
    query: str,
    candidates: List[Dict],
    tokenizer,
    max_chunks: int = 8,
    max_tokens: int = 1500,
    strategy: str = "mmr",
) -> List[Dict]:
    # cluster-aware first; deterministic fallback to legacy strategy if needed.
    clusters = cluster_evidence_items(candidates)
    selected = select_context_cluster_aware(
        question=query,
        pool=candidates,
        cluster_info=clusters,
        tokenizer=tokenizer,
        max_chunks=max_chunks,
        max_tokens=max_tokens,
    )
    if selected:
        return selected

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("rerank_score", c.get("retriever_score", 0.0)),
        reverse=True,
    )
    selected = []
    used_tokens = 0
    per_title_count = defaultdict(int)

    for cand in sorted_candidates:
        if len(selected) >= max_chunks:
            break
        text = cand.get("text", "")
        tks = len(tokenizer(text, add_special_tokens=False).input_ids)
        if used_tokens + tks > max_tokens:
            continue
        if strategy == "mmr" and selected:
            max_sim = max(_jaccard(text, s.get("text", "")) for s in selected)
            if max_sim > 0.85:
                continue
        title = _title(cand)
        if title and per_title_count[title] >= max(2, max_chunks // 2):
            continue
        selected.append(cand)
        used_tokens += tks
        if title:
            per_title_count[title] += 1

    return selected



def select_context_cluster_aware(
    question: str,
    pool: List[Dict],
    cluster_info,
    tokenizer,
    max_chunks: int,
    max_tokens: int,
    per_cluster_soft_cap: int = 2,
    rerank_weight: float = 1.0,
    specificity_weight: float = 0.7,
    resistance_weight: float = 0.5,
    diversity_bonus: float = 0.2,
    redundancy_penalty: float = 0.2,
) -> List[Dict]:
    selected: List[Dict] = []
    used_tokens = 0
    per_cluster_count = defaultdict(int)

    cluster_by_eid = {}
    for c in cluster_info or []:
        cid = c.get("cluster_id", "")
        for eid in c.get("member_ids", []) or []:
            cluster_by_eid[str(eid)] = cid

    def _score(cand: Dict) -> float:
        rerank = float(cand.get("rerank_score", cand.get("retriever_score", 0.0)))
        spec = float(cand.get("specificity", 0.0))
        resist = float(cand.get("counterfactual_resistance", 0.0))
        cid = cluster_by_eid.get(str(cand.get("evidence_id", "")), "")
        div = diversity_bonus if per_cluster_count[cid] == 0 else 0.0
        red = redundancy_penalty * max(0, per_cluster_count[cid] - per_cluster_soft_cap + 1)
        return rerank_weight * rerank + specificity_weight * spec + resistance_weight * resist + div - red

    ranked = sorted(pool, key=_score, reverse=True)
    for cand in ranked:
        if len(selected) >= max_chunks:
            break
        text = cand.get("text", "")
        tks = len(tokenizer(text, add_special_tokens=False).input_ids)
        if used_tokens + tks > max_tokens:
            continue
        cid = cluster_by_eid.get(str(cand.get("evidence_id", "")), "")
        if per_cluster_count[cid] >= max(1, per_cluster_soft_cap + max_chunks // 4):
            continue
        selected.append(cand)
        used_tokens += tks
        per_cluster_count[cid] += 1

    return selected
