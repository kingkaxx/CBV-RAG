from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbvrag.controller_trace_mixture import TraceMixtureController
from cbvrag.evidence_clusters import cluster_evidence_items, summarize_cluster_stats
from cbvrag.evidence_specificity import score_evidence_specificity
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from rl.train_offline import shape_reward_with_attr
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
    ap = argparse.ArgumentParser(
        description="Collect heuristic controller traces for IL/offline-RL training."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    # ---- Attr reward re-scoring ----
    ap.add_argument(
        "--use_attr_reward",
        action="store_true",
        help=(
            "Re-score each episode's reward using the Attr-based reward shaping "
            "from rl/train_offline.py instead of the raw trajectory_score. "
            "This ensures the IL traces used for behavioural cloning are labelled "
            "with the same reward the offline RL policy will optimise, making the "
            "two-stage IL→AWR pipeline self-consistent. "
            "The 'attr_score' field (from the null-branch arbitration record) is "
            "read from the episode log; episodes without this field receive 0.0."
        ),
    )
    ap.add_argument(
        "--attr_lambda_token", type=float, default=0.1,
        help="Token penalty weight for Attr-shaped reward (default 0.1).",
    )
    ap.add_argument(
        "--attr_lambda_step", type=float, default=0.05,
        help="Step penalty weight for Attr-shaped reward (default 0.05).",
    )
    ap.add_argument(
        "--attr_bonus", type=float, default=0.2,
        help="Attribution bonus weight for Attr-shaped reward (default 0.2).",
    )
    ap.add_argument(
        "--token_budget", type=int, default=4096,
        help="Token budget denominator for normalisation (default 4096).",
    )
    args = ap.parse_args()

    output = Path(args.output or f"data/traces/{args.dataset}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    models = model_loader.load_all_models()
    retriever = KnowledgeBaseRetriever(models["embedding_model"])

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
            device="cuda:0",
        ),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    # Accumulators for summary statistics.
    summary_em: list = []
    summary_tokens: list = []
    summary_attr: list = []

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(i)

            _maybe_clear_temp_index(retriever)
            _maybe_build_temp_index(retriever, ex, qid=qid)

            controller = TraceMixtureController(seed=1000 + i)
            pred, log = run_episode(ex["question"], controller, tools, qid=qid)

            golds = ex.get("answer") or [""]
            pred_norm = pred.strip().lower()
            correct = any(str(g).strip().lower() in pred_norm for g in golds if str(g).strip())
            traj_score = _trajectory_score(log, correct)

            # Extract episode-level Attr score from null-branch arbitration record.
            null_branch_record = log.get("null_branch") or {}
            attr_result = null_branch_record.get("attr_score") or {}
            episode_attr_score = float(attr_result.get("attr", 0.0))

            # Token usage for this episode (sum over all steps).
            state_dict = (log.get("state") or {})
            evidence_pool_dict = state_dict.get("evidence_pool") or {}
            selected_ids = state_dict.get("selected_evidence_ids") or []
            step_logs = log.get("steps", [])
            episode_tokens = sum(
                int((s.get("costs") or {}).get("tokens_used_this_step", 0))
                for s in step_logs
            )

            class SimpleEvidence:
                def __init__(self, evidence_id: str, payload: dict):
                    self.evidence_id = evidence_id
                    self.doc_id = payload.get("doc_id", "")
                    self.title = payload.get("title", "")
                    self.rerank_score = float(payload.get("rerank_score", 0.0))
                    self.retriever_score = float(payload.get("retriever_score", 0.0))

            evs = [SimpleEvidence(eid, payload) for eid, payload in evidence_pool_dict.items()]
            cluster_stats = summarize_cluster_stats(cluster_evidence_items(evs, selected_ids=selected_ids))
            specificity_summary = score_evidence_specificity(ex["question"], evs, selected_ids=selected_ids)["summary"]

            for t, tr in enumerate(controller.trace):
                step_info = step_logs[t] if t < len(step_logs) else {}

                if args.use_attr_reward:
                    # Build a synthetic row that shape_reward_with_attr can consume.
                    synthetic_row = {
                        "terminal_correct": bool(correct),
                        "step_info": step_info,
                        "attr_score": episode_attr_score,
                        "t": t,
                    }
                    row_reward = shape_reward_with_attr(
                        synthetic_row,
                        lambda_token=args.attr_lambda_token,
                        lambda_step=args.attr_lambda_step,
                        attr_bonus=args.attr_bonus,
                        token_budget=args.token_budget,
                    )
                else:
                    row_reward = float(step_info.get("costs", {}).get("new_branch_created", 0) * 0.01)

                row = {
                    "qid": qid,
                    "t": t,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": row_reward,
                    "done": t == len(controller.trace) - 1,
                    "terminal_correct": bool(correct),
                    "trajectory_score": traj_score,
                    # Store Attr score so downstream offline-RL can re-use it.
                    "attr_score": episode_attr_score,
                    "question": ex["question"],
                    "pred": pred,
                    "gold_answers": golds,
                    "step_info": step_info,
                    "trace_info": tr.get("info", {}),
                    "cluster_stats": cluster_stats,
                    "specificity_summary": specificity_summary,
                }
                f.write(json.dumps(row) + "\n")

            # Accumulate summary stats once per episode (at episode boundary).
            summary_em.append(1.0 if correct else 0.0)
            summary_tokens.append(episode_tokens)
            summary_attr.append(episode_attr_score)

    _maybe_clear_temp_index(retriever)

    # Print summary statistics so operators can verify trace quality.
    n = max(1, len(summary_em))
    summary = {
        "dataset": args.dataset,
        "num_episodes": n,
        "use_attr_reward": args.use_attr_reward,
        "mean_em": float(sum(summary_em) / n),
        "mean_token_use": float(sum(summary_tokens) / n),
        "mean_attr_score": float(sum(summary_attr) / n),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())