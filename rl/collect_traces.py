from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch

import config
import model_loader
from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from retriever import KnowledgeBaseRetriever
from rl.reward import RewardConfig, compute_reward_components
from tools.llm import LLMEngine
from tools.rerank import CrossEncoderReranker
from tools.retrieve import RetrieverTool


def resolve_llm_device(requested: str | None) -> str:
    if requested:
        device = requested
    elif getattr(config, "LLM_DEVICE", None):
        device = config.LLM_DEVICE
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested GPU device '{device}' but CUDA is unavailable.")
        if not model_loader.check_device_availability(device):
            raise RuntimeError(f"Requested GPU device '{device}' is unavailable on this host.")
    return device


def fmt_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--llm_device", default=None)
    ap.add_argument("--keep_only_successful", action="store_true")
    ap.add_argument("--min_episode_reward", type=float, default=None)
    ap.add_argument("--max_episode_tokens", type=int, default=None)
    ap.add_argument("--max_episode_steps", type=int, default=None)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--correctness_reward", type=float, default=1.0)
    ap.add_argument("--token_penalty", type=float, default=0.001)
    ap.add_argument("--retrieval_penalty", type=float, default=0.05)
    ap.add_argument("--branch_penalty", type=float, default=0.1)
    ap.add_argument("--verify_bonus", type=float, default=0.02)
    ap.add_argument("--early_stop_bonus", type=float, default=0.05)
    args = ap.parse_args()

    llm_device = resolve_llm_device(args.llm_device)
    print(f"[collect_traces] LLM device: {llm_device}", flush=True)

    output = Path(args.output or f"data/traces/{args.dataset}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    reward_cfg = RewardConfig(
        correctness_reward=args.correctness_reward,
        token_penalty=args.token_penalty,
        retrieval_penalty=args.retrieval_penalty,
        branch_penalty=args.branch_penalty,
        verify_bonus=args.verify_bonus,
        early_stop_bonus=args.early_stop_bonus,
    )

    models = model_loader.load_all_models()
    retriever = KnowledgeBaseRetriever(models["embedding_model"])
    tools = {
        "llm": LLMEngine(
            model_name_or_path=getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path),
            device=llm_device,
        ),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)
    total = len(data)
    rows_written = 0
    episodes_completed = 0
    success_count = 0
    step_counts: List[int] = []
    token_counts: List[int] = []
    retrieval_counts: List[int] = []
    prev_total_tokens = 0

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(ex.get("id", i))

            # For datasets like HotpotQA that include per-example context, build a temporary
            # in-memory retrieval index so trace collection does not depend on knowledge_base/.
            context_docs = ex.get("context") if isinstance(ex, dict) else None
            if context_docs:
                retriever.build_temp_index_from_docs(context_docs)
                # Avoid cross-example cache collisions: use per-qid retrieval cache dir.
                tools["retrieve"] = RetrieverTool(retriever, cache_dir=f"./cache/retrieval/{args.dataset}/{qid}")
                print(
                    f"[collect_traces] built temporary retrieval index for qid={qid} "
                    f"with {len(context_docs)} context docs",
                    flush=True,
                )
            else:
                retriever.clear_temp_index()
                tools["retrieve"] = RetrieverTool(retriever)

            controller = HeuristicController()
            pred, log = run_episode(ex["question"], controller, tools, qid=qid)
            golds = ex.get("answer") or [""]
            correct = any(g.lower() in pred.lower() for g in golds)

            state_metrics = log.get("state", {}).get("metrics", {})
            running_total_tokens = int(tools["llm"].usage_tracker.summary().get("total_tokens", 0))
            total_tokens = max(0, running_total_tokens - prev_total_tokens)
            prev_total_tokens = running_total_tokens
            retrieval_calls = int(state_metrics.get("retrieval_calls", 0))
            verify_calls = int(state_metrics.get("verify_calls", 0))
            llm_calls = int(state_metrics.get("llm_calls", 0))
            episode_steps = len(controller.trace)
            num_branches = len(log.get("state", {}).get("branches", {}))
            early_stop = episode_steps < log.get("state", {}).get("budgets", {}).get("max_steps", episode_steps)

            rewards = compute_reward_components(
                terminal_correct=bool(correct),
                tokens_used=total_tokens,
                retrieval_calls=retrieval_calls,
                branches_created=max(0, num_branches - 1),
                verify_calls=verify_calls,
                early_stop=early_stop,
                cfg=reward_cfg,
            )
            episode_total_reward = rewards["total"]

            if args.keep_only_successful and not correct:
                continue
            if args.min_episode_reward is not None and episode_total_reward < args.min_episode_reward:
                continue
            if args.max_episode_tokens is not None and total_tokens > args.max_episode_tokens:
                continue
            if args.max_episode_steps is not None and episode_steps > args.max_episode_steps:
                continue

            episodes_completed += 1
            success_count += int(bool(correct))
            step_counts.append(episode_steps)
            token_counts.append(total_tokens)
            retrieval_counts.append(retrieval_calls)

            for t, tr in enumerate(controller.trace):
                row = {
                    "qid": qid,
                    "episode_id": f"{qid}::0",
                    "episode_index": i,
                    "episode_step_index": t,
                    "episode_num_steps": episode_steps,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": tr.get("reward", 0.0),
                    "reward_components": rewards,
                    "done": t == len(controller.trace) - 1,
                    "success": bool(correct),
                    "terminal_correct": bool(correct),
                    "info": log.get("steps", []),
                    "episode_total_reward": episode_total_reward,
                    "episode_total_tokens": total_tokens,
                    "episode_total_retrieval_calls": retrieval_calls,
                    "episode_total_verify_calls": verify_calls,
                    "episode_total_llm_calls": llm_calls,
                    "episode_num_branches": num_branches,
                }
                f.write(json.dumps(row) + "\n")
                rows_written += 1

            if (i + 1) % max(1, args.progress_every) == 0 or i + 1 == total:
                print(
                    "[collect_traces] "
                    f"example={i + 1}/{total} rows_written={rows_written} episodes={episodes_completed} "
                    f"terminal_correct_pct={100.0 * fmt_ratio(success_count, episodes_completed):.1f} "
                    f"avg_steps={mean(step_counts) if step_counts else 0:.2f} "
                    f"avg_tokens={mean(token_counts) if token_counts else 0:.1f} "
                    f"avg_retrieval_calls={mean(retrieval_counts) if retrieval_counts else 0:.2f}",
                    flush=True,
                )

    summary = {
        "dataset": args.dataset,
        "output": str(output),
        "total_examples": total,
        "episodes_written": episodes_completed,
        "rows_written": rows_written,
        "terminal_correct_pct": 100.0 * fmt_ratio(success_count, episodes_completed),
        "avg_steps": mean(step_counts) if step_counts else 0.0,
        "avg_tokens": mean(token_counts) if token_counts else 0.0,
        "avg_retrieval_calls": mean(retrieval_counts) if retrieval_counts else 0.0,
        "llm_device": llm_device,
    }
    print("[collect_traces] Final summary:")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
