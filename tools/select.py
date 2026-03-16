from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from cbvrag.evidence_clusters import cluster_evidence_items
from cbvrag.evidence_specificity import score_evidence_specificity


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _title(cand: Dict) -> str:
    return (cand.get("title") or "").strip()


def select_context_cluster_aware(
    question,
    pool,
    tokenizer,
    max_chunks,
    max_tokens,
    cluster_info=None,
    per_cluster_soft_cap: int = 2,
):
    selected: List[Dict] = []
    used_tokens = 0
    per_cluster_count = defaultdict(int)

    clusters = cluster_info if cluster_info is not None else cluster_evidence_items(pool)
    eid_to_cluster = {}
    for c in clusters:
        cid = str(c.get("cluster_id", ""))
        for eid in c.get("member_ids", []) or []:
            eid_to_cluster[str(eid)] = cid

    spec = score_evidence_specificity(question, pool)
    chunk_scores = spec.get("chunk_scores", {}) if isinstance(spec, dict) else {}

    def _score(cand: Dict) -> float:
        rerank = float(cand.get("rerank_score", cand.get("retriever_score", 0.0)))
        eid = str(cand.get("evidence_id", ""))
        srec = chunk_scores.get(eid, {}) if isinstance(chunk_scores.get(eid, {}), dict) else {}
        novelty = float(srec.get("novelty", 0.0))
        specificity = float(srec.get("specificity", 0.0))
        cid = eid_to_cluster.get(eid, "")
        redundancy_penalty = float(max(0, per_cluster_count[cid] - per_cluster_soft_cap + 1))
        return (1.0 * rerank) + (0.35 * novelty) + (0.25 * specificity) - (0.40 * redundancy_penalty)

    ranked = sorted(pool, key=_score, reverse=True)
    for cand in ranked:
        if len(selected) >= max_chunks:
            break
        text = cand.get("text", "")
        tks = len(tokenizer(text, add_special_tokens=False).input_ids)
        if used_tokens + tks > max_tokens:
            continue
        eid = str(cand.get("evidence_id", ""))
        cid = eid_to_cluster.get(eid, "")
        if per_cluster_count[cid] >= max(1, per_cluster_soft_cap + max_chunks // 4):
            continue
        if selected and max(_jaccard(text, s.get("text", "")) for s in selected) > 0.9:
            continue
        selected.append(cand)
        used_tokens += tks
        per_cluster_count[cid] += 1

    return selected


def select_context(
    query: str,
    candidates: List[Dict],
    tokenizer,
    max_chunks: int = 8,
    max_tokens: int = 1500,
    strategy: str = "mmr",
) -> List[Dict]:
    # cluster-aware first; fallback keeps prior behavior shape.
    selected = select_context_cluster_aware(
        question=query,
        pool=candidates,
        tokenizer=tokenizer,
        max_chunks=max_chunks,
        max_tokens=max_tokens,
    )
    if selected:
        return selected

    sorted_candidates = sorted(candidates, key=lambda c: c.get("rerank_score", c.get("retriever_score", 0.0)), reverse=True)
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
