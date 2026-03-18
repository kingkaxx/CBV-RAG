from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
import model_loader
from cfrag_pipeline import CFRAGPipeline
from data_loader import load_and_process_data
from evaluation import evaluate, smart_exact_match_score
from metrics.cost import CostTracker
from metrics.usage import UsageTracker
from retriever import KnowledgeBaseRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--cache_dir", default="./huggingface_cache")
    parser.add_argument("--output_dir", default="logs/baseline")
    parser.add_argument("--output", default=None,
                        help="Path for the summary JSON output. "
                             "Defaults to <output_dir>/<dataset>_summary.json.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.dataset}.jsonl"

    usage = UsageTracker()
    cost = CostTracker()

    models = model_loader.load_all_models()
    retriever = KnowledgeBaseRetriever(models["embedding_model"])
    pipeline = CFRAGPipeline(models, retriever)

    data = load_and_process_data(args.dataset, args.cache_dir, num_samples=args.num_samples)

    records = []
    for i, ex in enumerate(data):
        qid = str(ex.get("id", i))
        q = ex["question"]
        golds = ex["answer"]
        start = time.time()
        pred = pipeline.run(q)
        latency_ms = (time.time() - start) * 1000
        # Fallback estimate since legacy pipeline is not instrumented per-call.
        prompt_tokens = len(models["llm_tokenizer"](q).input_ids)
        completion_tokens = len(models["llm_tokenizer"](pred).input_ids)
        usage.track("cfrag_baseline", prompt_tokens, completion_tokens)
        cost.inc_retrieval(1)
        cost.inc_steps(1)

        em, f1 = evaluate(pred, golds, q)
        correct = bool(em)
        rec = {
            "qid": qid,
            "question": q,
            "prediction": pred,
            "gold": golds,
            "correct": bool(correct),
            "em": float(em),
            "f1": float(f1),
            "total_tokens": prompt_tokens + completion_tokens,
            "retrieval_calls": cost.metrics.retrieval_calls,
            "rerank_calls": cost.metrics.rerank_calls,
            "latency_ms": latency_ms,
        }
        records.append(rec)

    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = max(1, len(records))
    agg = {
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "mean_em": sum(r["em"] for r in records) / n,
        "mean_f1": sum(r["f1"] for r in records) / n,
        "mean_tokens": sum(r["total_tokens"] for r in records) / n,
        "mean_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "count": len(records),
    }
    summary_path = Path(args.output) if args.output else (out_dir / f"{args.dataset}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
