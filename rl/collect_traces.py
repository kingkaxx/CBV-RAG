from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from tools.llm import LLMEngine
from tools.rerank import CrossEncoderReranker
from tools.retrieve import RetrieverTool
import model_loader
from retriever import KnowledgeBaseRetriever


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    args = ap.parse_args()

    output = Path(args.output or f"data/traces/{args.dataset}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    models = model_loader.load_all_models()
    retriever = KnowledgeBaseRetriever(models["embedding_model"])
    tools = {
        "llm": LLMEngine(model_name_or_path=getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path), device="cpu"),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)
    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            controller = HeuristicController()
            pred, log = run_episode(ex["question"], controller, tools, qid=str(i))
            gold = ex["answer"][0] if ex.get("answer") else ""
            correct = gold.lower() in pred.lower()
            for t, tr in enumerate(controller.trace):
                row = {
                    "qid": str(i),
                    "t": t,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": tr.get("reward", 0.0),
                    "done": t == len(controller.trace) - 1,
                    "info": log.get("steps", []),
                    "terminal_correct": bool(correct),
                }
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
