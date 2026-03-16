from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


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
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("rerank_score", c.get("retriever_score", 0.0)),
        reverse=True,
    )
    selected: List[Dict] = []
    used_tokens = 0

    titles_in_pool = {_title(c) for c in sorted_candidates if _title(c)}
    target_min_titles = 2 if len(titles_in_pool) >= 2 else 1
    per_title_count = defaultdict(int)

    def _try_add(cand: Dict) -> bool:
        nonlocal used_tokens
        if len(selected) >= max_chunks:
            return False
        text = cand.get("text", "")
        tks = len(tokenizer(text, add_special_tokens=False).input_ids)
        if used_tokens + tks > max_tokens:
            return False
        if strategy == "mmr" and selected:
            max_sim = max(_jaccard(text, s.get("text", "")) for s in selected)
            if max_sim > 0.85:
                return False
        # avoid one title crowding everything out
        title = _title(cand)
        if title and per_title_count[title] >= max(2, max_chunks // 2):
            return False
        selected.append(cand)
        used_tokens += tks
        if title:
            per_title_count[title] += 1
        return True

    # pass 1: diversify by title among top-ranked evidence
    if target_min_titles >= 2:
        seen_titles = set()
        for cand in sorted_candidates:
            if len(seen_titles) >= target_min_titles or len(selected) >= max_chunks:
                break
            title = _title(cand)
            if not title or title in seen_titles:
                continue
            if _try_add(cand):
                seen_titles.add(title)

    # pass 2: fill remaining slots by score, with redundancy/title checks.
    for cand in sorted_candidates:
        if len(selected) >= max_chunks:
            break
        if cand in selected:
            continue
        _try_add(cand)

    return selected
