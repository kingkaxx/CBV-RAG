from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np


class GlobalChunkRetriever:
    def __init__(self, embedding_model: Any) -> None:
        self.embedding_model = embedding_model
        self.rows: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def build_from_jsonl(self, kb_path: str, batch_size: int = 64) -> None:
        self.rows = self._read_jsonl(kb_path)
        texts = [r.get("text", "") for r in self.rows]
        embs = self.embedding_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
        embs = embs.astype(np.float32)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def save(self, out_dir: str) -> None:
        if self.index is None:
            raise RuntimeError("index is empty; call build_from_jsonl first")
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "global.index"))
        (p / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows) + "\n", encoding="utf-8")

    def load(self, out_dir: str) -> None:
        p = Path(out_dir)
        self.index = faiss.read_index(str(p / "global.index"))
        self.rows = self._read_jsonl(str(p / "rows.jsonl"))

    def search(self, query: str, top_k: int = 20, dataset_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("index not initialized")
        q = self.embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        k = min(max(1, top_k), len(self.rows))
        scores, idxs = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            row = self.rows[int(idx)]
            if dataset_filter and row.get("dataset") != dataset_filter:
                continue
            out.append({**row, "score": float(score)})
        return out


def retrieval_diagnostics(
    retriever: GlobalChunkRetriever,
    qa_rows: Iterable[Dict[str, Any]],
    top_k: int = 20,
    dataset_filter: Optional[str] = None,
) -> Dict[str, float]:
    n = 0
    support_hit = 0
    rr_sum = 0.0
    for ex in qa_rows:
        supports = set(ex.get("support_titles") or [])
        if not supports:
            continue
        n += 1
        hits = retriever.search(ex["question"], top_k=top_k, dataset_filter=dataset_filter)
        titles = [h.get("title") for h in hits]
        if any(t in supports for t in titles):
            support_hit += 1
        rank = None
        for i, t in enumerate(titles, start=1):
            if t in supports:
                rank = i
                break
        if rank is not None:
            rr_sum += 1.0 / rank
    return {
        "n_with_support": n,
        "support_doc_recall_at_k": support_hit / max(1, n),
        "support_doc_mrr": rr_sum / max(1, n),
    }


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/load/search global chunk index with diagnostics.")
    ap.add_argument("--mode", choices=["build", "diagnose"], required=True)
    ap.add_argument("--kb_jsonl", default="data/global_kb_chunks.jsonl")
    ap.add_argument("--qa_jsonl", default="data/multidataset_qa.jsonl")
    ap.add_argument("--index_dir", default="data/global_index")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--dataset_filter", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)
    retriever = GlobalChunkRetriever(model)

    if args.mode == "build":
        retriever.build_from_jsonl(args.kb_jsonl)
        retriever.save(args.index_dir)
        print(json.dumps({"kb_rows": len(retriever.rows), "index_dir": args.index_dir}, indent=2))
        return 0

    retriever.load(args.index_dir)
    qa_rows = _read_jsonl(args.qa_jsonl)
    metrics = retrieval_diagnostics(retriever, qa_rows, top_k=args.top_k, dataset_filter=args.dataset_filter)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
