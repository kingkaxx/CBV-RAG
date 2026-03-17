from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from evaluation import smart_exact_match_score
from tools.llm import LLMEngine
from tools.retrieve import RetrieverTool
from tools.rerank import CrossEncoderReranker

import model_loader


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(records))
    return {
        "accuracy": sum(r["correct"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "avg_steps": sum(r["steps"] for r in records) / n,
        "avg_branches": sum(r["num_branches"] for r in records) / n,
        "pct_early": sum(1 for r in records if r["early_exit"]) / n,
        "count": len(records),
    }


class GlobalIndexAdapter:
    """
    Minimal adapter so RetrieverTool can call:
        retriever.search(query, top_k=..., dataset_filter=...)
    """

    def __init__(self, index_dir: str, kb_jsonl: str, embedding_model: Any | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_dir = Path(index_dir)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(self.index_dir / "global.index"))

        self.rows = load_jsonl(kb_jsonl)

        # Optional metadata sidecar if it exists
        meta_path = self.index_dir / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.meta = None

    def _embed(self, query: str) -> np.ndarray:
        emb = self.embedding_model.encode([query], convert_to_numpy=True)
        return np.asarray(emb, dtype="float32")

    def search(self, query: str, top_k: int = 10, dataset_filter: str | None = None) -> List[Dict[str, Any]]:
        q = self._embed(query)
        scores, ids = self.index.search(q, top_k * 5 if dataset_filter else top_k)

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.rows[int(idx)]

            if dataset_filter and row.get("dataset") != dataset_filter:
                continue

            out.append(
                {
                    "doc_id": str(row.get("doc_id", idx)),
                    "chunk_id": str(row.get("chunk_id", idx)),
                    "text": row.get("text", ""),
                    "score": float(score),
                    "retriever_score": float(score),
                    "title": str(row.get("title", "")),
                    "dataset": str(row.get("dataset", "")),
                    "meta": row.get("metadata", {}),
                }
            )
            if len(out) >= top_k:
                break

        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qa_jsonl", default="data/multidataset_qa.jsonl")
    ap.add_argument("--kb_jsonl", default="data/global_kb_chunks.jsonl")
    ap.add_argument("--index_dir", default="data/global_index")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--output", default="logs/cbvrag_eval_global.json")
    args = ap.parse_args()

    all_rows = load_jsonl(args.qa_jsonl)
    data = [r for r in all_rows if r.get("dataset") == args.dataset]
    if args.num_samples is not None:
        data = data[: args.num_samples]

    if not data:
        raise SystemExit(f"No rows found for dataset={args.dataset} in {args.qa_jsonl}")

    models = model_loader.load_all_models()

    global_retriever = GlobalIndexAdapter(
        index_dir=args.index_dir,
        kb_jsonl=args.kb_jsonl,
        embedding_model=models["embedding_model"],
    )

    tools = {
        "llm": LLMEngine(
            getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path),
            device="cuda:0",
            model=models["llm_model"],
            tokenizer=models["llm_tokenizer"],
        ),
        "retrieve": RetrieverTool(global_retriever, dataset_filter=args.dataset),
        "rerank": CrossEncoderReranker(),
    }

    cbv_records: List[Dict[str, Any]] = []
    prev_total_tokens = 0

    for i, ex in enumerate(data):
        controller = HeuristicController()
        pred, log = run_episode(ex["question"], controller, tools, qid=str(ex.get("qid", i)))

        usage = tools["llm"].usage_tracker.summary()
        running_total = int(usage.get("total_tokens", 0))
        example_total_tokens = max(0, running_total - prev_total_tokens)
        prev_total_tokens = running_total

        retrieval_calls = int(log["state"]["metrics"]["retrieval_calls"])
        steps = len(log.get("steps", []))
        num_branches = len((log.get("state") or {}).get("branches", {}))
        golds = ex.get("answers", [])
        correct = any(smart_exact_match_score(pred, g, ex["question"]) for g in golds)

        cbv_records.append(
            {
                "qid": str(ex.get("qid", i)),
                "question": ex["question"],
                "prediction": pred,
                "gold_answers": golds,
                "correct": bool(correct),
                "total_tokens": example_total_tokens,
                "retrieval_calls": retrieval_calls,
                "steps": steps,
                "num_branches": num_branches,
                "early_exit": retrieval_calls <= 1,
            }
        )

    out = {f"cbvrag_{args.dataset}": evaluate_records(cbv_records)}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    records_path = output_path.with_suffix(".records.jsonl")
    with records_path.open("w", encoding="utf-8") as f:
        for rec in cbv_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps(out, indent=2))
    print(f"records_path={records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())