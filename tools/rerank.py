from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from metrics.cost import CostTracker
from tools.retrieve import Candidate


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: str = "./cache/rerank",
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cross_encoder = CrossEncoder(model_name, device=device, trust_remote_code=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = cost_tracker

    def _cache_key(self, query: str, candidates: List[Candidate]) -> str:
        ids = [f"{c.get('doc_id')}::{c.get('chunk_id')}" for c in candidates]
        raw = json.dumps({"q": query, "ids": ids, "m": self.model_name}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def rerank(self, query: str, candidates: List[Candidate], top_n: Optional[int] = None) -> List[Candidate]:
        if not candidates:
            return []
        key = self._cache_key(query, candidates)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            ranked = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = self.cross_encoder.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            ranked = []
            for cand, score in zip(candidates, scores):
                item = dict(cand)
                item["rerank_score"] = float(score)
                ranked.append(item)
            ranked.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
            cache_file.write_text(json.dumps(ranked, ensure_ascii=False), encoding="utf-8")

        if self.cost_tracker:
            self.cost_tracker.inc_rerank(1)
        return ranked[:top_n] if top_n is not None else ranked
