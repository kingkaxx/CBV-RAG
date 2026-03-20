"""Offline RL training for the CBV-RAG controller.

This module trains a policy using **Advantage-Weighted Regression (AWR)** on
pre-collected episode traces.  It extends the imitation-learning (IL) baseline
with an Attr-based reward shaping signal that ties training directly to the
attribution quality of the final answer.

Why Attr-shaped rewards are better than a pure heuristic threshold
------------------------------------------------------------------
A heuristic threshold (e.g., "accept the answer if Attr > 0.3") provides a
single fixed decision boundary that cannot adapt to varying token budgets or
question difficulty.  By incorporating Attr into a *learnable* reward signal:

    R = EM_score
        - lambda_token  * (tokens_used / token_budget)
        - lambda_step   * num_steps
        + attr_bonus    * attr_score_of_final_answer

the policy learns a **budget-conditioned decision boundary** — it discovers
when it is worth spending additional tokens to reach a better-attributed answer
versus stopping early.  This is the key distinction from static thresholds:
the policy internalises the cost-quality tradeoff through gradient signal
rather than a hand-tuned constant.  The lambda hyperparameters control the
Pareto tradeoff between accuracy/attribution and computational efficiency.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cbvrag.actions import ACTION_ENUM_VERSION, Action, action_names
from cbvrag.features import FEATURE_SCHEMA_VERSION
from rl.policy import PolicyConfig, build_policy
from rl.reward import RewardConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_rows(path: str, min_score: Optional[float] = None) -> List[dict]:
    """Load trace rows from a JSONL file, optionally filtering by trajectory score."""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if min_score is not None and float(row.get("trajectory_score", 0.0)) < min_score:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No traces loaded from {path}. Adjust filters or input file.")
    return rows


def shape_reward_with_attr(
    row: dict,
    lambda_token: float = 0.04,
    lambda_step: float = 0.02,
    attr_bonus: float = 0.12,
    token_budget: int = 4096,
) -> float:
    """Compute the Attr-shaped reward for a single trace row.

    Formula
    -------
        R = EM_score
            - lambda_token * (tokens_used / token_budget)
            - lambda_step  * num_steps
            + attr_bonus   * attr_score_of_final_answer

    Parameters
    ----------
    row:
        A single JSONL trace record.  Expected keys (all optional with safe
        defaults):

        * ``"terminal_correct"`` (bool) — EM signal.
        * ``"step_info"`` (dict) — contains ``"costs"`` with token counts.
        * ``"attr_score"`` (float) — pre-computed Attr score stored by
          ``collect_traces.py --use_attr_reward``; falls back to 0.0.
        * ``"t"`` (int) — step index within the episode.
    lambda_token:
        Penalty weight for normalised token usage (default 0.1).
    lambda_step:
        Penalty weight per step (default 0.05).
    attr_bonus:
        Bonus weight for the Attr score of the final answer (default 0.2).
    token_budget:
        Denominator for token normalisation (default 4096).

    Returns
    -------
    float
        The shaped scalar reward for this transition.
    """
    em_score = 2.0 if bool(row.get("terminal_correct", False)) else 0.0

    step_info = row.get("step_info") or {}
    costs = step_info.get("costs") or {}
    tokens_used = int(costs.get("tokens_used_this_step", 0))
    token_fraction = tokens_used / max(1, token_budget)

    num_steps = int(row.get("t", 0)) + 1  # 1-indexed

    # Support both the old per-step key ("attr_score") written by collect_traces.py
    # when --use_attr_reward is set, and the new episode-level key ("episode_attr_score")
    # written into flattened step rows by prepare_traces.py _flatten_episode().
    attr_score = float(row.get("attr_score") or row.get("episode_attr_score") or 0.0)

    reward = (
        em_score
        - lambda_token * token_fraction
        - lambda_step * num_steps
        + attr_bonus * attr_score
    )
    return float(reward)


def build_reward_tensor(
    rows: List[dict],
    reward_cfg: RewardConfig,
    success_bonus: float = 0.0,
    # Attr shaping parameters (applied on top of base reward)
    use_attr_shaping: bool = False,
    lambda_token: float = 0.1,
    lambda_step: float = 0.05,
    attr_bonus: float = 0.2,
    token_budget: int = 4096,
) -> torch.Tensor:
    """Build a reward tensor for a list of trace rows.

    If ``use_attr_shaping=True``, each row's reward is replaced by the
    Attr-shaped reward from :func:`shape_reward_with_attr`.  Otherwise the
    raw ``trajectory_score`` field is used (compatible with heuristic traces).

    Parameters
    ----------
    rows:
        List of trace dictionaries.
    reward_cfg:
        RewardConfig instance (used only when ``use_attr_shaping=False``).
    success_bonus:
        Additional scalar bonus for terminal-correct episodes.
    use_attr_shaping:
        Whether to apply Attr-based reward shaping.
    lambda_token, lambda_step, attr_bonus, token_budget:
        Attr shaping hyperparameters; ignored if ``use_attr_shaping=False``.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(len(rows),)``.
    """
    rewards = []
    for row in rows:
        if use_attr_shaping:
            r = shape_reward_with_attr(
                row,
                lambda_token=lambda_token,
                lambda_step=lambda_step,
                attr_bonus=attr_bonus,
                token_budget=token_budget,
            )
        else:
            r = float(row.get("trajectory_score", 0.0))

        if success_bonus > 0 and bool(row.get("terminal_correct", False)):
            r += success_bonus
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline RL (AWR) training for the CBV-RAG controller."
    )
    # ---- I/O ----
    ap.add_argument("--traces", required=True, help="Path to train JSONL traces.")
    ap.add_argument("--val_traces", default=None, help="Path to validation JSONL traces.")
    ap.add_argument("--out", default="checkpoints/policy_offline.pt")
    ap.add_argument("--init_policy", default=None, help="Optional IL checkpoint to warm-start from.")

    # ---- Architecture ----
    ap.add_argument("--policy_type", choices=["mlp", "mlp_residual", "gru_policy"], default="mlp_residual")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--history_len", type=int, default=1)

    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", choices=["awr", "bc"], default="awr",
                    help="'awr' = Advantage-Weighted Regression; 'bc' = pure behaviour cloning.")
    ap.add_argument("--bc_coef", type=float, default=0.1,
                    help="BC coefficient (only for AWR objective).")
    ap.add_argument("--adv_temperature", type=float, default=1.0,
                    help="Temperature for AWR advantage weighting.")
    ap.add_argument("--entropy_coef", type=float, default=0.0,
                    help="Entropy regularisation coefficient.")
    ap.add_argument("--success_bonus", type=float, default=0.0,
                    help="Extra reward bonus for terminal-correct episodes.")
    ap.add_argument("--filter_min_trajectory_score", type=float, default=None)

    # ---- Attr-based reward shaping ----
    ap.add_argument(
        "--lambda_token",
        type=float,
        default=0.04,
        help=(
            "Token efficiency penalty weight in the Attr-shaped reward: "
            "R -= lambda_token * (tokens_used / token_budget). "
            "The policy learns a budget-conditioned decision boundary — higher "
            "values push the policy to stop earlier."
        ),
    )
    ap.add_argument(
        "--lambda_step",
        type=float,
        default=0.02,
        help=(
            "Per-step penalty weight: R -= lambda_step * num_steps. "
            "Encourages fewer reasoning steps without sacrificing accuracy."
        ),
    )
    ap.add_argument(
        "--attr_bonus",
        type=float,
        default=0.12,
        help=(
            "Attribution score bonus weight: R += attr_bonus * attr_score. "
            "Rewards the policy for producing well-attributed final answers."
        ),
    )
    ap.add_argument(
        "--token_budget",
        type=int,
        default=4096,
        help=(
            "Denominator for token normalisation in the shaped reward "
            "(default 4096). Set to the max token budget used during collection."
        ),
    )
    ap.add_argument(
        "--use_attr_shaping",
        action="store_true",
        help=(
            "Replace the raw trajectory_score reward with the Attr-shaped "
            "reward function. Requires traces to have 'attr_score' fields "
            "(produced by collect_traces.py --use_attr_reward)."
        ),
    )

    args = ap.parse_args()

    # Auto-enable use_attr_shaping when any Attr shaping param is set to a
    # non-default value but --use_attr_shaping was not explicitly passed.
    # This prevents the silent bug where --attr_bonus / --lambda_token /
    # --lambda_step are provided but shaping stays disabled.
    _attr_param_defaults = {"lambda_token": 0.04, "lambda_step": 0.02, "attr_bonus": 0.12, "token_budget": 4096}
    _user_set_attr_param = any(
        getattr(args, k) != default for k, default in _attr_param_defaults.items()
    )
    if _user_set_attr_param and not args.use_attr_shaping:
        print(json.dumps({
            "event": "auto_enable_attr_shaping",
            "reason": (
                "One or more Attr shaping params (--attr_bonus, --lambda_token, "
                "--lambda_step, --token_budget) were set to non-default values "
                "but --use_attr_shaping was not passed. Enabling automatically."
            ),
        }))
        args.use_attr_shaping = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dim = len(Action)

    rows = load_rows(args.traces, min_score=args.filter_min_trajectory_score)

    val_rows: List[dict] = []
    if args.val_traces:
        val_rows = load_rows(args.val_traces)
        if args.filter_min_trajectory_score is not None:
            val_rows = [
                r for r in val_rows
                if float(r.get("trajectory_score", 0.0)) >= args.filter_min_trajectory_score
            ]

    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    act = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)

    if int(act.max().item()) >= act_dim or int(act.min().item()) < 0:
        raise ValueError(
            f"Train traces contain action ids outside Action enum range [0, {act_dim - 1}]."
        )

    reward_cfg = RewardConfig()
    rew = build_reward_tensor(
        rows,
        reward_cfg,
        success_bonus=args.success_bonus,
        use_attr_shaping=args.use_attr_shaping,
        lambda_token=args.lambda_token,
        lambda_step=args.lambda_step,
        attr_bonus=args.attr_bonus,
        token_budget=args.token_budget,
    )

    val_obs = val_act = val_rew = None
    if val_rows:
        val_obs = torch.tensor([r["obs"] for r in val_rows], dtype=torch.float32)
        val_act = torch.tensor([r["action"] for r in val_rows], dtype=torch.long)
        if int(val_act.max().item()) >= act_dim or int(val_act.min().item()) < 0:
            raise ValueError(
                f"Val traces contain action ids outside Action enum range [0, {act_dim - 1}]."
            )
        val_rew = build_reward_tensor(
            val_rows,
            reward_cfg,
            success_bonus=args.success_bonus,
            use_attr_shaping=args.use_attr_shaping,
            lambda_token=args.lambda_token,
            lambda_step=args.lambda_step,
            attr_bonus=args.attr_bonus,
            token_budget=args.token_budget,
        )

    obs_dim = int(obs.shape[1])
    cfg = PolicyConfig(
        policy_type=args.policy_type,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        history_len=args.history_len,
    )
    model = build_policy(cfg).to(device)

    # Warm-start from IL checkpoint if provided.
    if args.init_policy:
        ckpt = torch.load(args.init_policy, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(json.dumps({"event": "loaded_init_policy", "path": args.init_policy}))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = TensorDataset(obs, act, rew)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    val_dl = None
    if val_obs is not None:
        val_ds = TensorDataset(val_obs, val_act, val_rew)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    ce = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for xb, ab, rb in dl:
            xb = xb.to(device)
            ab = ab.to(device)
            rb = rb.to(device)

            logits = model(xb)
            logp = torch.log_softmax(logits, dim=-1)
            chosen = logp[torch.arange(len(ab)), ab]

            if args.objective == "awr":
                # AWR: weight each step by exp(advantage / temperature).
                adv = rb - rb.mean()
                weights = torch.exp(
                    torch.clamp(adv / max(args.adv_temperature, 1e-6), -3.0, 3.0)
                ).detach()
                awr_loss = -(weights * chosen).mean()

                # Optional BC regularisation: keeps policy close to data distribution.
                bc_loss = ce(logits, ab).mean() if args.bc_coef > 0 else torch.tensor(0.0, device=device)
                loss = awr_loss + args.bc_coef * bc_loss
            else:
                # Pure behaviour cloning (uniform weights).
                loss = ce(logits, ab).mean()

            if args.entropy_coef > 0:
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * logp).sum(dim=-1).mean()
                loss = loss - args.entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))

        log_rec: dict = {
            "epoch": epoch,
            "offline_loss": sum(losses) / max(1, len(losses)),
            "objective": args.objective,
            "use_attr_shaping": args.use_attr_shaping,
        }

        if val_dl is not None:
            model.eval()
            val_losses = []
            val_correct = val_total = 0
            with torch.no_grad():
                for xb, ab, rb in val_dl:
                    xb, ab, rb = xb.to(device), ab.to(device), rb.to(device)
                    logits = model(xb)
                    adv = rb - rb.mean()
                    weights = torch.exp(torch.clamp(adv / max(args.adv_temperature, 1e-6), -3.0, 3.0)).detach()
                    chosen = torch.log_softmax(logits, dim=-1)[torch.arange(len(ab)), ab]
                    val_losses.append(float((-(weights * chosen)).mean().item()))
                    val_correct += int((logits.argmax(dim=-1) == ab).sum().item())
                    val_total += int(ab.numel())
            log_rec["val_loss"] = sum(val_losses) / max(1, len(val_losses))
            log_rec["val_acc"] = val_correct / max(1, val_total)

        print(json.dumps(log_rec))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "arch": {
                "policy_type": args.policy_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "history_len": args.history_len,
            },
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "action_enum_version": ACTION_ENUM_VERSION,
            "action_names": action_names(),
            "git_commit": get_git_commit(),
            "hparams": vars(args),
        },
        out,
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
