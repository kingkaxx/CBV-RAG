from __future__ import annotations

from typing import Dict, List


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def select_context(
    query: str,
    candidates: List[Dict],
    tokenizer,
    max_chunks: int = 8,
    max_tokens: int = 1500,
    strategy: str = "mmr",
) -> List[Dict]:
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("rerank_score", c.get("retriever_score", 0.0)),
        reverse=True,
    )
    selected: List[Dict] = []
    used_tokens = 0

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

        selected.append(cand)
        used_tokens += tks

    return selected
