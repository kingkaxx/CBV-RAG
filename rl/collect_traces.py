from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch

import config
import model_loader
from cbvrag.actions import Action
from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from retriever import KnowledgeBaseRetriever
from rl.reward import RewardConfig, compute_reward_components
from rl.trace_oracles import (
    TrajectoryScoreConfig,
    build_oracle_controller,
    compute_episode_quality,
    estimate_case_profile_from_example,
    parse_oracle_mix,
    sample_oracle_name,
    score_trajectory,
)
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


def _action_sequence(log: Dict) -> List[int]:
    return [int(s.get("action")) for s in log.get("steps", []) if isinstance(s, dict) and s.get("action") is not None]


def _explicit_early_stop(log: Dict) -> bool:
    if "explicit_stop_used" in log:
        return bool(log.get("explicit_stop_used", False))

    # Backward-compatible fallback for older logs.
    steps = log.get("steps", [])
    if not steps:
        return False
    budgets = (log.get("state") or {}).get("budgets", {})
    max_steps = int(budgets.get("max_steps", len(steps)))
    last = steps[-1]
    action_name = str(last.get("action_name", ""))
    action_idx = int(last.get("action", -1)) if last.get("action") is not None else -1
    explicit_terminate = action_name in {"STOP_AND_ANSWER", "ANSWER_DIRECT"} or action_idx in {
        int(Action.STOP_AND_ANSWER),
        int(Action.ANSWER_DIRECT),
    }
    return explicit_terminate and len(steps) < max_steps


def _behavior_signature(cand: Dict) -> str:
    return "|".join(
        [
            f"s{cand['steps']}",
            f"r{cand['retrieval_calls']}",
            f"b{cand['num_branches']}",
            ",".join(map(str, cand.get("action_sequence", []))),
        ]
    )


def _last_action_is(cand: Dict, action: Action) -> bool:
    seq = cand.get("action_sequence", [])
    return bool(seq) and int(seq[-1]) == int(action)


def _is_successful_explicit_stop(cand: Dict) -> bool:
    if not cand.get("success", False):
        return False
    return _last_action_is(cand, Action.STOP_AND_ANSWER) or _last_action_is(cand, Action.ANSWER_DIRECT)


def _is_short_successful_stop_bucket(cand: Dict) -> bool:
    # keep compact successful trajectories that terminate explicitly soon after useful context setup
    if not _is_successful_explicit_stop(cand):
        return False
    if not cand.get("had_selected_evidence", False):
        return False
    seq = cand.get("action_sequence", [])
    if len(seq) <= 3:
        return True
    return len(seq) <= 4 and int(Action.VERIFY_CHEAP) in seq




def _is_preferred_short_stop_sequence(seq: List[int]) -> bool:
    preferred = {
        (int(Action.RETRIEVE_MORE_LARGE), int(Action.SELECT_CONTEXT), int(Action.ANSWER_DIRECT)),
        (int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.ANSWER_DIRECT)),
        (int(Action.SELECT_CONTEXT), int(Action.ANSWER_DIRECT)),
        (int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.VERIFY_CHEAP), int(Action.ANSWER_DIRECT)),
        (int(Action.RETRIEVE_MORE_LARGE), int(Action.SELECT_CONTEXT), int(Action.STOP_AND_ANSWER)),
        (int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.STOP_AND_ANSWER)),
        (int(Action.SELECT_CONTEXT), int(Action.STOP_AND_ANSWER)),
        (int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.VERIFY_CHEAP), int(Action.STOP_AND_ANSWER)),
    }
    return tuple(seq) in preferred

def _pick_shortest_successful_terminal(pool: List[Dict], action: Action) -> Dict | None:
    matches = [c for c in pool if c.get("success", False) and _last_action_is(c, action)]
    if not matches:
        return None
    return min(matches, key=lambda c: (c.get("steps", 10**9), c.get("tokens", 10**9), -c.get("trajectory_score", 0.0)))


def _pick_diverse_candidates(cands: List[Dict], keep_n: int, allow_near_success: bool) -> List[Dict]:
    if not cands:
        return []

    successful = [c for c in cands if c.get("success")]
    near = [c for c in cands if c.get("f1", 0.0) >= 0.5 or c.get("em", 0.0) >= 1.0]
    pool = successful or (near if allow_near_success else cands)
    if not pool:
        pool = cands

    sorted_by_score = sorted(pool, key=lambda c: (c["trajectory_score"], c["f1"], -c["tokens"]), reverse=True)
    sorted_by_short = sorted(pool, key=lambda c: (c["steps"], c["tokens"], -c["trajectory_score"]))
    branch_pool = successful if successful else pool
    sorted_by_branch = sorted(branch_pool, key=lambda c: (c["num_branches"], c["trajectory_score"]), reverse=True)

    best = sorted_by_score[0]
    shortest = sorted_by_short[0]
    chosen: List[Dict] = []

    def _add_if_new(c: Dict | None) -> None:
        if c is None:
            return
        if all(_behavior_signature(c) != _behavior_signature(x) for x in chosen):
            chosen.append(c)

    # Retention bucket: preserve short successful explicit-stop traces even if a longer trace scores slightly higher.
    short_successful_stops = [c for c in pool if _is_short_successful_stop_bucket(c)]
    preferred_short_stops = [c for c in short_successful_stops if _is_preferred_short_stop_sequence(c.get("action_sequence", []))]
    for c in sorted(preferred_short_stops, key=lambda x: (x["steps"], x["tokens"], -x["trajectory_score"])):
        if len(chosen) >= keep_n:
            break
        _add_if_new(c)
    for c in sorted(short_successful_stops, key=lambda x: (x["steps"], x["tokens"], -x["trajectory_score"])):
        if len(chosen) >= keep_n:
            break
        _add_if_new(c)

    # Keep at least one shortest successful terminal trace when available.
    stop_shortest = _pick_shortest_successful_terminal(pool, Action.STOP_AND_ANSWER)
    answer_direct_shortest = _pick_shortest_successful_terminal(pool, Action.ANSWER_DIRECT)
    if stop_shortest is not None and answer_direct_shortest is not None:
        preferred = stop_shortest if (stop_shortest["steps"], stop_shortest["tokens"]) <= (answer_direct_shortest["steps"], answer_direct_shortest["tokens"]) else answer_direct_shortest
        _add_if_new(preferred)
    else:
        _add_if_new(stop_shortest or answer_direct_shortest)

    if keep_n <= 1:
        if chosen:
            return chosen[:1]
        close_in_score = (best["trajectory_score"] - shortest["trajectory_score"]) <= 0.4
        return [shortest if close_in_score else best]

    _add_if_new(shortest)
    _add_if_new(best)

    if keep_n >= 3:
        for c in sorted_by_branch:
            if c["num_branches"] <= 1:
                continue
            if len(chosen) >= keep_n:
                break
            _add_if_new(c)

    for c in sorted_by_score:
        if len(chosen) >= keep_n:
            break
        _add_if_new(c)

    return chosen[: max(1, keep_n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--llm_device", default=None)
    ap.add_argument("--trace_policy", choices=["heuristic", "mixture"], default="heuristic")
    ap.add_argument("--num_candidates_per_example", type=int, default=4)
    ap.add_argument("--oracle_mix", default=None)
    ap.add_argument("--keep_top_n_per_example", type=int, default=1)
    ap.add_argument("--allow_near_success", action="store_true")

    ap.add_argument("--keep_only_successful", action="store_true")
    ap.add_argument("--min_episode_reward", type=float, default=None)
    ap.add_argument("--max_episode_tokens", type=int, default=None)
    ap.add_argument("--max_episode_steps", type=int, default=None)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--correctness_reward", type=float, default=1.0)
    ap.add_argument("--token_penalty", type=float, default=0.001)
    ap.add_argument("--retrieval_penalty", type=float, default=0.05)
    ap.add_argument("--branch_penalty", type=float, default=0.1)
    ap.add_argument("--verify_bonus", type=float, default=0.02)
    ap.add_argument("--early_stop_bonus", type=float, default=0.05)

    ap.add_argument("--score_success_reward", type=float, default=3.0)
    ap.add_argument("--score_em_reward", type=float, default=1.5)
    ap.add_argument("--score_f1_reward", type=float, default=1.0)
    ap.add_argument("--score_support_reward", type=float, default=0.75)
    ap.add_argument("--score_token_penalty", type=float, default=0.001)
    ap.add_argument("--score_step_penalty", type=float, default=0.03)
    ap.add_argument("--score_branch_penalty", type=float, default=0.06)
    ap.add_argument("--score_redundant_verify_penalty", type=float, default=0.03)
    args = ap.parse_args()

    rng = random.Random(args.seed)
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
    score_cfg = TrajectoryScoreConfig(
        success_reward=args.score_success_reward,
        em_reward=args.score_em_reward,
        f1_reward=args.score_f1_reward,
        support_reward=args.score_support_reward,
        token_penalty=args.score_token_penalty,
        step_penalty=args.score_step_penalty,
        branch_penalty=args.score_branch_penalty,
        redundant_verify_penalty=args.score_redundant_verify_penalty,
    )
    oracle_mix = parse_oracle_mix(args.oracle_mix)

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

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(ex.get("id", i))

            context_docs = ex.get("context") if isinstance(ex, dict) else None
            if context_docs:
                retriever.build_temp_index_from_docs(context_docs)
                tools["retrieve"] = RetrieverTool(retriever, cache_dir=f"./cache/retrieval/{args.dataset}/{qid}")
            else:
                retriever.clear_temp_index()
                tools["retrieve"] = RetrieverTool(retriever)

            num_candidates = 1 if args.trace_policy == "heuristic" else max(1, args.num_candidates_per_example)
            case_profile = estimate_case_profile_from_example(ex) if args.trace_policy == "mixture" else "standard"
            candidate_infos = []

            for cand_idx in range(num_candidates):
                running_before = int(tools["llm"].usage_tracker.summary().get("total_tokens", 0))

                if args.trace_policy == "heuristic":
                    controller = HeuristicController()
                    oracle_name = "heuristic"
                else:
                    oracle_name = sample_oracle_name(case_profile, oracle_mix, rng)
                    controller = build_oracle_controller(oracle_name, seed=args.seed + i * 97 + cand_idx)

                pred, log = run_episode(ex["question"], controller, tools, qid=f"{qid}-{cand_idx}")
                running_after = int(tools["llm"].usage_tracker.summary().get("total_tokens", 0))
                cand_tokens = max(0, running_after - running_before)

                state_metrics = log.get("state", {}).get("metrics", {})
                retrieval_calls = int(state_metrics.get("retrieval_calls", 0))
                verify_calls = int(state_metrics.get("verify_calls", 0))
                llm_calls = int(state_metrics.get("llm_calls", 0))
                episode_steps = len(controller.trace)
                num_branches = len(log.get("state", {}).get("branches", {}))

                action_sequence = _action_sequence(log)
                explicit_early_stop = _explicit_early_stop(log)

                golds = ex.get("answer") or [""]
                em, f1, success = compute_episode_quality(pred, golds)
                support_titles = set(ex.get("support_titles") or [])
                retrieved_titles = {
                    (ev.get("title") or "").strip()
                    for ev in (log.get("state", {}).get("evidence_pool", {}) or {}).values()
                    if isinstance(ev, dict)
                }
                support_hit = 1.0 if (support_titles and (retrieved_titles & support_titles)) else 0.0

                rewards = compute_reward_components(
                    terminal_correct=bool(success),
                    tokens_used=cand_tokens,
                    retrieval_calls=retrieval_calls,
                    branches_created=max(0, num_branches - 1),
                    verify_calls=verify_calls,
                    early_stop=explicit_early_stop,
                    cfg=reward_cfg,
                )
                episode_total_reward = rewards["total"]

                trajectory_score = score_trajectory(
                    success=bool(success),
                    em=em,
                    f1=f1,
                    support_hit=support_hit,
                    tokens_used=cand_tokens,
                    steps=episode_steps,
                    branches=num_branches,
                    verify_calls=verify_calls,
                    cfg=score_cfg,
                )

                final_state = log.get("state", {}) or {}
                had_selected_evidence = len(final_state.get("selected_evidence_ids", []) or []) > 0
                final_action = action_sequence[-1] if action_sequence else -1
                explicit_terminal = final_action in {int(Action.STOP_AND_ANSWER), int(Action.ANSWER_DIRECT)}
                verify_steps = sum(1 for a in action_sequence if a == int(Action.VERIFY_CHEAP))
                verify_no_improvement = sum(
                    1
                    for s in (log.get("steps", []) or [])
                    if isinstance(s, dict)
                    and int(s.get("action", -1)) == int(Action.VERIFY_CHEAP)
                    and int(s.get("verification_status_changed", 0)) == 0
                )

                # stronger preference for efficient, explicit, successful terminations after context selection
                if success:
                    trajectory_score += 0.25 * (1.0 if explicit_early_stop else 0.0)
                    if explicit_terminal and had_selected_evidence:
                        trajectory_score += 0.45
                        trajectory_score += 0.10
                        trajectory_score -= 0.07 * max(0, verify_steps - 1)

                # penalize repeated VERIFY_CHEAP especially when status does not improve
                trajectory_score -= 0.04 * max(0, verify_steps - 1)
                trajectory_score -= 0.12 * float(verify_no_improvement)
                trajectory_score -= 0.04 * max(0, episode_steps - 5)
                trajectory_score -= 0.06 * max(0, num_branches - 2)

                candidate_infos.append(
                    {
                        "cand_idx": cand_idx,
                        "pred": pred,
                        "log": log,
                        "controller": controller,
                        "tokens": cand_tokens,
                        "retrieval_calls": retrieval_calls,
                        "verify_calls": verify_calls,
                        "llm_calls": llm_calls,
                        "steps": episode_steps,
                        "num_branches": num_branches,
                        "explicit_early_stop": explicit_early_stop,
                        "success": bool(success),
                        "em": float(em),
                        "f1": float(f1),
                        "support_hit": support_hit,
                        "rewards": rewards,
                        "episode_total_reward": episode_total_reward,
                        "trajectory_score": trajectory_score,
                        "oracle_name": oracle_name,
                        "case_profile": case_profile,
                        "fallback_stop_was_used": bool(log.get("fallback_stop_was_used", False)),
                        "explicit_stop_used": bool(log.get("explicit_stop_used", False)),
                        "forced_stop_used": bool(log.get("forced_stop_used", False)),
                        "had_selected_evidence": had_selected_evidence,
                        "verify_steps": verify_steps,
                        "verify_no_improvement": verify_no_improvement,
                        "final_action": final_action,
                        "action_sequence": action_sequence,
                    }
                )

            kept = _pick_diverse_candidates(
                candidate_infos,
                keep_n=max(1, args.keep_top_n_per_example),
                allow_near_success=bool(args.allow_near_success),
            )

            for rank, cand in enumerate(kept, start=1):
                if args.keep_only_successful and not cand["success"]:
                    continue
                if args.min_episode_reward is not None and cand["episode_total_reward"] < args.min_episode_reward:
                    continue
                if args.max_episode_tokens is not None and cand["tokens"] > args.max_episode_tokens:
                    continue
                if args.max_episode_steps is not None and cand["steps"] > args.max_episode_steps:
                    continue

                episodes_completed += 1
                success_count += int(bool(cand["success"]))
                step_counts.append(cand["steps"])
                token_counts.append(cand["tokens"])
                retrieval_counts.append(cand["retrieval_calls"])

                trace = cand["controller"].trace
                for t, tr in enumerate(trace):
                    info = tr.get("info", {}) if isinstance(tr, dict) else {}
                    if not isinstance(info, dict):
                        info = {}
                    row = {
                        "qid": qid,
                        "episode_id": f"{qid}::{cand['cand_idx']}",
                        "episode_index": i,
                        "episode_step_index": t,
                        "episode_num_steps": cand["steps"],
                        "obs": tr["obs"],
                        "action": tr["action"],
                        "reward": tr.get("reward", 0.0),
                        "reward_components": cand["rewards"],
                        "done": t == len(trace) - 1,
                        "success": bool(cand["success"]),
                        "terminal_correct": bool(cand["success"]),
                        "info": info,
                        "episode_trace": cand["log"].get("steps", []),
                        "episode_total_reward": cand["episode_total_reward"],
                        "episode_total_tokens": cand["tokens"],
                        "episode_total_retrieval_calls": cand["retrieval_calls"],
                        "episode_total_verify_calls": cand["verify_calls"],
                        "episode_total_llm_calls": cand["llm_calls"],
                        "episode_num_branches": cand["num_branches"],
                        "episode_final_fallback_stop": cand["fallback_stop_was_used"],
                        "episode_final_explicit_stop": cand.get("explicit_stop_used", False),
                        "episode_final_forced_stop": cand.get("forced_stop_used", False),
                        "explicit_early_stop": cand["explicit_early_stop"],
                        "action_sequence": cand["action_sequence"],
                        # richer optional schema
                        "oracle_name": cand["oracle_name"],
                        "case_profile": cand["case_profile"],
                        "candidate_rank": rank,
                        "trajectory_score": cand["trajectory_score"],
                        "state_features": tr.get("obs", []),
                        "action_reason": info.get("action_reason", ""),
                        "tokens_before": None,
                        "tokens_after": None,
                        "branches_before": None,
                        "branches_after": None,
                        "retrieval_calls_before": None,
                        "retrieval_calls_after": None,
                        "was_successful": bool(cand["success"]),
                        "final_em": cand["em"],
                        "final_f1": cand["f1"],
                        "steps": cand["steps"],
                        "retrieval_calls": cand["retrieval_calls"],
                        "num_branches": cand["num_branches"],
                        "verify_calls": cand["verify_calls"],
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
        "trace_policy": args.trace_policy,
        "num_candidates_per_example": args.num_candidates_per_example,
        "keep_top_n_per_example": args.keep_top_n_per_example,
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
