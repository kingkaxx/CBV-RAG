from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


Candidate = Dict[str, Any]


class RetrieverTool:
    def __init__(self, retriever: Any, cache_dir: str = "./cache/retrieval", dataset_filter: Optional[str] = None) -> None:
        self.retriever = retriever
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_filter = dataset_filter

    def _cache_key(self, query: str, pool_k: int, mode: str) -> str:
        raw = json.dumps({"q": query, "k": pool_k, "m": mode, "dataset_filter": self.dataset_filter}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def retrieve(self, query: str, pool_k: int, mode: str = "dense") -> List[Candidate]:
        key = self._cache_key(query, pool_k, mode)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        try:
            docs = self.retriever.search(query, top_k=pool_k, dataset_filter=self.dataset_filter)
        except TypeError:
            docs = self.retriever.search(query, top_k=pool_k)
        candidates: List[Candidate] = []
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict):
                text = doc.get("text", "")
                doc_id = str(doc.get("doc_id", idx))
                chunk_id = str(doc.get("chunk_id", idx))
                score = float(doc.get("score", doc.get("retriever_score", 0.0)))
                meta = doc.get("meta", {})
                title = str(doc.get("title", ""))
                dataset = str(doc.get("dataset", ""))
            else:
                text = str(doc)
                doc_id = str(idx)
                chunk_id = str(idx)
                score = 0.0
                meta = {}
                title = ""
                dataset = ""
            candidates.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "retriever_score": score,
                    "meta": meta,
                    "title": title,
                    "dataset": dataset,
                }
            )

        cache_file.write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
        return candidates
