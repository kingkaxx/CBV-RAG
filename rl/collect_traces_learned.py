from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from cbvrag.actions import Action, get_num_actions
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from rl.policy import PolicyConfig, build_policy
from rl.train_offline import shape_reward_with_attr
from tools.llm import LLMEngine
from tools.rerank import CrossEncoderReranker
from tools.retrieve import RetrieverTool

import model_loader
from retriever import KnowledgeBaseRetriever


# ── Reuse helpers from collect_traces ─────────────────────────────────────────

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


def _validate_traces(output_path: Path) -> None:
    num_actions = get_num_actions()
    records_by_qid: dict[str, int] = defaultdict(int)
    all_action_values: list[int] = []
    attr_scores: list[float] = []
    traj_scores: list[float] = []
    correct_flags: list[float] = []

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("qid", "")
            records_by_qid[qid] += 1
            for step in rec.get("trajectory", []):
                all_action_values.append(int(step.get("action", -1)))
            attr_scores.append(float(rec.get("episode_attr_score", 0.0)))
            traj_scores.append(float(rec.get("trajectory_score", 0.0)))
            correct_flags.append(1.0 if rec.get("terminal_correct") else 0.0)

    total_episodes = sum(records_by_qid.values())
    unique_qids = len(records_by_qid)
    records_per_qid = total_episodes / unique_qids if unique_qids else 0.0
    action_min = min(all_action_values) if all_action_values else -1
    action_max = max(all_action_values) if all_action_values else -1
    n = max(1, len(attr_scores))

    print("\n--- Trace Validation (DAgger) ---")
    print(f"  Total episodes written : {total_episodes}")
    print(f"  Unique qids            : {unique_qids}")
    print(f"  Records per qid        : {records_per_qid:.1f}  (must be 1.0)")
    print(f"  Action value range     : [{action_min}, {action_max}]  (valid: [0, {num_actions - 1}])")
    print(f"  Mean episode_attr_score: {sum(attr_scores) / n:.4f}")
    print(f"  Mean trajectory_score  : {sum(traj_scores) / n:.4f}")
    print(f"  % terminal_correct     : {100.0 * sum(correct_flags) / n:.1f}%")

    if records_per_qid != 1.0:
        print("  WARNING: records_per_qid != 1.0 — duplicate or missing episodes!")
    if action_min < 0 or action_max >= num_actions:
        print(f"  WARNING: action values outside valid range [0, {num_actions - 1}]!")
    print("---------------------------------\n")


# ── Learned controller with epsilon-greedy exploration ────────────────────────

class LearnedController:
    """
    Wraps a trained IL/offline-RL policy checkpoint with epsilon-greedy
    exploration for DAgger-style trace collection.

    At each step:
      - With probability `epsilon`: pick a random *valid* action (respects
        action_mask and state-validity constraints).
      - Otherwise: run the policy forward pass, mask out invalid actions,
        and take the argmax.

    The `trace` list accumulates (obs, action) pairs in the same format
    as TraceMixtureController so downstream code is compatible.
    """

    def __init__(self, policy_path: str, epsilon: float = 0.15, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.trace: List[Dict[str, Any]] = []

        # Load checkpoint
        ckpt = torch.load(policy_path, map_location="cpu")
        arch = ckpt.get("arch", {})
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])

        cfg = PolicyConfig(
            policy_type=arch.get("policy_type", "mlp_residual"),
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=int(arch.get("hidden_dim", 128)),
            num_layers=int(arch.get("num_layers", 2)),
            dropout=float(arch.get("dropout", 0.1)),
            history_len=int(arch.get("history_len", 1)),
        )
        self.model = build_policy(cfg)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()

        print(
            f"[LearnedController] Loaded policy from {policy_path} "
            f"(obs_dim={obs_dim}, act_dim={act_dim}, "
            f"policy_type={cfg.policy_type}, hidden_dim={cfg.hidden_dim}, "
            f"epsilon={epsilon})",
            flush=True,
        )

    def _valid_indices(self, action_mask: Optional[List[bool]]) -> List[int]:
        """Return list of valid action indices from action_mask."""
        if action_mask is None:
            return list(range(get_num_actions()))
        return [i for i, ok in enumerate(action_mask) if ok]

    def act(self, obs, state, action_mask: Optional[List[bool]] = None) -> int:
        valid = self._valid_indices(action_mask)
        if not valid:
            valid = list(range(get_num_actions()))

        if self.rng.random() < self.epsilon:
            # Random valid action (DAgger exploration)
            action_idx = self.rng.choice(valid)
        else:
            # Policy argmax over valid actions only
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(obs_tensor)[0]  # shape: (act_dim,)

            # Mask invalid actions to -inf
            mask_tensor = torch.full((len(logits),), float("-inf"))
            for v in valid:
                mask_tensor[v] = logits[v]

            action_idx = int(mask_tensor.argmax().item())

        self.trace.append({
            "obs": list(obs),
            "action": action_idx,
            "reward": 0.0,
            "done": False,
            "info": {
                "step": state.step,
                "selected_evidence_count": len(state.selected_evidence_ids),
                "retrieval_calls": int((state.metrics or {}).get("retrieval_calls", 0)),
                "verification_status": state.verification_status,
                "epsilon_triggered": self.rng.random() < self.epsilon,
            },
        })
        return action_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "DAgger-style trace collection using a trained IL/offline-RL policy "
            "with epsilon-greedy exploration. Produces traces in the same format "
            "as collect_traces.py for use in offline RL training."
        )
    )
    ap.add_argument("--policy", required=True,
                    help="Path to trained policy checkpoint (.pt file).")
    ap.add_argument("--epsilon", type=float, default=0.15,
                    help="Epsilon for greedy exploration (default 0.15).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--llm_device", default="cuda:0")
    ap.add_argument("--use_attr_reward", action="store_true")
    ap.add_argument("--attr_lambda_token", type=float, default=0.1)
    ap.add_argument("--attr_lambda_step", type=float, default=0.05)
    ap.add_argument("--attr_bonus", type=float, default=0.2)
    ap.add_argument("--token_budget", type=int, default=4096)
    args = ap.parse_args()

    output = Path(args.output or f"data/traces/{args.dataset}_dagger.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    models = model_loader.load_all_models()
    retriever = KnowledgeBaseRetriever(models["embedding_model"])

    tools = {
        "llm": LLMEngine(
            model_name_or_path=getattr(
                models["llm_model"],
                "name_or_path",
                models["llm_model"].config._name_or_path,
            ),
            device=args.llm_device,
        ),
        "retrieve": RetrieverTool(retriever),
        "rerank": CrossEncoderReranker(),
    }

    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    summary_em: list = []
    summary_tokens: list = []
    summary_attr: list = []

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(i)

            _maybe_clear_temp_index(retriever)
            _maybe_build_temp_index(retriever, ex, qid=qid)

            # Fresh controller per episode
            controller = LearnedController(
                policy_path=args.policy,
                epsilon=args.epsilon,
                seed=2000 + i,
            )

            pred, log = run_episode(ex["question"], controller, tools, qid=qid)

            golds = ex.get("answer") or [""]
            pred_norm = pred.strip().lower()
            correct = any(str(g).strip().lower() in pred_norm for g in golds if str(g).strip())
            traj_score = _trajectory_score(log, correct)

            null_branch_record = log.get("null_branch") or {}
            attr_result = null_branch_record.get("attr_score") or {}
            episode_attr_score = float(attr_result.get("attr", 0.0))

            step_logs = log.get("steps", [])
            episode_tokens = sum(
                int((s.get("costs") or {}).get("tokens_used_this_step", 0))
                for s in step_logs
            )

            trajectory = []
            for t, tr in enumerate(controller.trace):
                step_info = step_logs[t] if t < len(step_logs) else {}

                if args.use_attr_reward:
                    synthetic_row = {
                        "terminal_correct": bool(correct),
                        "step_info": step_info,
                        "attr_score": episode_attr_score,
                        "t": t,
                    }
                    step_reward = shape_reward_with_attr(
                        synthetic_row,
                        lambda_token=args.attr_lambda_token,
                        lambda_step=args.attr_lambda_step,
                        attr_bonus=args.attr_bonus,
                        token_budget=args.token_budget,
                    )
                else:
                    step_reward = float(
                        (step_info.get("costs") or {}).get("new_branch_created", 0) * 0.01
                    )

                trajectory.append({
                    "t": t,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": step_reward,
                    "attr_score": episode_attr_score,
                    "step_info": step_info,
                })

            episode_record = {
                "qid": qid,
                "question": ex["question"],
                "gold_answers": golds,
                "trajectory": trajectory,
                "terminal_correct": bool(correct),
                "trajectory_score": traj_score,
                "total_tokens": episode_tokens,
                "episode_attr_score": episode_attr_score,
                "num_steps": len(trajectory),
                "dagger": True,
                "epsilon": args.epsilon,
            }
            f.write(json.dumps(episode_record) + "\n")

            summary_em.append(1.0 if correct else 0.0)
            summary_tokens.append(episode_tokens)
            summary_attr.append(episode_attr_score)

    _maybe_clear_temp_index(retriever)

    n = max(1, len(summary_em))
    summary = {
        "dataset": args.dataset,
        "num_episodes": n,
        "epsilon": args.epsilon,
        "use_attr_reward": args.use_attr_reward,
        "mean_em": float(sum(summary_em) / n),
        "mean_token_use": float(sum(summary_tokens) / n),
        "mean_attr_score": float(sum(summary_attr) / n),
    }
    print(json.dumps(summary, indent=2))
    _validate_traces(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())