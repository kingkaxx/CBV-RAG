from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from evaluation import smart_exact_match_score


def evaluate_records(records):
    n = max(1, len(records))
    return {
        "accuracy": sum(r["correct"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "pct_early": sum(1 for r in records if r["early_exit"]) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--baseline_jsonl", default=None)
    ap.add_argument("--output", default="logs/cbvrag_eval.json")
    args = ap.parse_args()

    data = load_and_process_data(args.dataset, args.cache_dir)

    # Lightweight placeholder tools for dry-run evaluation integration.
    from tools.llm import LLMEngine
    from tools.retrieve import RetrieverTool
    from tools.rerank import CrossEncoderReranker
    import model_loader
    from retriever import KnowledgeBaseRetriever

    models = model_loader.load_all_models()
    kb = KnowledgeBaseRetriever(models["embedding_model"])
    tools = {
        "llm": LLMEngine(getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path), device="cpu"),
        "retrieve": RetrieverTool(kb),
        "rerank": CrossEncoderReranker(),
    }

    cbv_records = []
    for i, ex in enumerate(data):
        controller = HeuristicController()
        pred, log = run_episode(ex["question"], controller, tools, qid=str(i))
        total_tokens = tools["llm"].usage_tracker.summary()["total_tokens"]
        retrieval_calls = log["state"]["metrics"]["retrieval_calls"]
        correct = any(smart_exact_match_score(pred, g, ex["question"]) for g in ex["answer"])
        cbv_records.append(
            {
                "qid": str(i),
                "correct": bool(correct),
                "total_tokens": total_tokens,
                "retrieval_calls": retrieval_calls,
                "early_exit": retrieval_calls <= 1,
            }
        )

    out = {"cbvrag_heuristic": evaluate_records(cbv_records)}
    if args.baseline_jsonl:
        rows = [json.loads(l) for l in Path(args.baseline_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        out["cfrag_baseline"] = {
            "accuracy": sum(1 for r in rows if r["correct"]) / max(1, len(rows)),
            "avg_total_tokens": sum(r["total_tokens"] for r in rows) / max(1, len(rows)),
            "avg_retrieval_calls": sum(r["retrieval_calls"] for r in rows) / max(1, len(rows)),
            "pct_early": sum(1 for r in rows if r.get("retrieval_calls", 99) <= 1) / max(1, len(rows)),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
