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


def _trajectory_score(log: dict, correct: bool) -> float:
    state = (log.get("state") or {})
    metrics = state.get("metrics") or {}
    steps = log.get("steps") or []
    retrieval_calls = int(metrics.get("retrieval_calls", 0))
    llm_calls = int(metrics.get("llm_calls", 0))
    no_progress = int(metrics.get("no_progress_streak", 0))
    explicit_stop = int(metrics.get("explicit_stop_used", 0))
    fallback_stop = int(metrics.get("fallback_stop_was_used", 0))

    score = 1.0 if correct else 0.0
    score += 0.15 * explicit_stop
    score -= 0.05 * retrieval_calls
    score -= 0.03 * llm_calls
    score -= 0.05 * no_progress
    score -= 0.10 * fallback_stop
    score -= 0.02 * max(0, len(steps) - 4)
    return float(score)


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
        "llm": LLMEngine(
            model_name_or_path=getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path),
            device="cpu",
        ),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            controller = HeuristicController()
            pred, log = run_episode(ex["question"], controller, tools, qid=str(i))

            golds = ex.get("answer") or [""]
            correct = any(str(g).strip().lower() in pred.strip().lower() for g in golds if str(g).strip())
            traj_score = _trajectory_score(log, correct)

            step_logs = log.get("steps", [])
            for t, tr in enumerate(controller.trace):
                step_info = step_logs[t] if t < len(step_logs) else {}
                row = {
                    "qid": str(i),
                    "t": t,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": float(step_info.get("costs", {}).get("new_branch_created", 0) * 0.01),
                    "done": t == len(controller.trace) - 1,
                    "terminal_correct": bool(correct),
                    "trajectory_score": traj_score,
                    "question": ex["question"],
                    "pred": pred,
                    "gold_answers": golds,
                    "step_info": step_info,
                    "trace_info": tr.get("info", {}),
                }
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
