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
    forced_stop = int(metrics.get("forced_stop_used", 0))
    forced_actions = int(metrics.get("forced_action_count", 0))
    illegal_requested = int(metrics.get("illegal_action_requested", 0))

    score = 1.0 if correct else 0.0
    score += 0.15 * explicit_stop
    score -= 0.05 * retrieval_calls
    score -= 0.03 * llm_calls
    score -= 0.05 * no_progress
    score -= 0.10 * fallback_stop
    score -= 0.08 * forced_stop
    score -= 0.04 * forced_actions
    score -= 0.04 * illegal_requested
    score -= 0.02 * max(0, len(steps) - 4)
    return float(score)


def _maybe_build_temp_index(retriever: KnowledgeBaseRetriever, ex: dict, qid: str) -> None:
    context_docs = ex.get("context")

    if context_docs is None:
        return

    try:
        retriever.build_temp_index_from_docs(context_docs)
    except Exception as e:
        print(f"Warning: failed to build temp index for qid={qid}: {e}", flush=True)


def _maybe_clear_temp_index(retriever: KnowledgeBaseRetriever) -> None:
    # Support several possible retriever implementations without breaking.
    try:
        if hasattr(retriever, "clear_temp_index"):
            retriever.clear_temp_index()
            return
        if hasattr(retriever, "temp_index"):
            retriever.temp_index = None
        if hasattr(retriever, "temp_documents"):
            retriever.temp_documents = []
    except Exception:
        pass


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

    # Only datasets like ARC-C need the global FAISS index.
    # HotpotQA and similar context-provided datasets should use per-example temp indexes.
    if args.dataset == "arc_c":
        try:
            if hasattr(retriever, "load_index"):
                retriever.load_index()
        except Exception as e:
            print(f"Warning: failed to load global index for dataset={args.dataset}: {e}", flush=True)

    tools = {
        "llm": LLMEngine(
            model_name_or_path=getattr(
                models["llm_model"],
                "name_or_path",
                models["llm_model"].config._name_or_path,
            ),
            device="cpu",
        ),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(i)

            _maybe_clear_temp_index(retriever)
            _maybe_build_temp_index(retriever, ex, qid=qid)

            controller = HeuristicController()
            pred, log = run_episode(ex["question"], controller, tools, qid=qid)

            golds = ex.get("answer") or [""]
            pred_norm = pred.strip().lower()
            correct = any(str(g).strip().lower() in pred_norm for g in golds if str(g).strip())
            traj_score = _trajectory_score(log, correct)

            step_logs = log.get("steps", [])
            for t, tr in enumerate(controller.trace):
                step_info = step_logs[t] if t < len(step_logs) else {}
                row = {
                    "qid": qid,
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

    _maybe_clear_temp_index(retriever)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())