from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from rl.policy import PolicyConfig, build_policy
from rl.reward import RewardConfig, compute_reward_components


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_rows(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_reward_tensor(rows: List[Dict], cfg: RewardConfig, success_bonus: float = 0.0) -> torch.Tensor:
    out = []
    for r in rows:
        comp = r.get("reward_components")
        if comp and "total" in comp:
            total = float(comp["total"])
        else:
            total = compute_reward_components(
                terminal_correct=bool(r.get("terminal_correct", False)),
                tokens_used=int(r.get("episode_total_tokens", 0)),
                retrieval_calls=int(r.get("episode_total_retrieval_calls", 0)),
                branches_created=max(0, int(r.get("episode_num_branches", 1)) - 1),
                verify_calls=int(r.get("episode_total_verify_calls", 0)),
                early_stop=bool(r.get("done", False)),
                cfg=cfg,
            )["total"]
        if bool(r.get("terminal_correct", False)):
            total += success_bonus
        out.append(total)
    return torch.tensor(out, dtype=torch.float32)


def eval_metrics(model, obs, act, rew) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(obs)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp[torch.arange(len(act)), act]
        pred = logits.argmax(dim=-1)
    return {
        "action_match": float((pred == act).float().mean().item()),
        "avg_logp_chosen": float(chosen.mean().item()),
        "avg_reward": float(rew.mean().item()) if len(rew) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--val_traces", default=None)
    ap.add_argument("--out", default="checkpoints/policy_offline.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy_type", choices=["mlp", "mlp_residual", "gru_policy"], default="mlp")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--history_len", type=int, default=1)
    ap.add_argument("--objective", choices=["wbc", "awr", "cql_like"], default="awr")
    ap.add_argument("--bc_coef", type=float, default=0.1)
    ap.add_argument("--adv_temperature", type=float, default=1.0)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--success_bonus", type=float, default=0.0)
    ap.add_argument("--init_policy", default=None)
    ap.add_argument("--correctness_reward", type=float, default=1.0)
    ap.add_argument("--token_penalty", type=float, default=0.001)
    ap.add_argument("--retrieval_penalty", type=float, default=0.05)
    ap.add_argument("--branch_penalty", type=float, default=0.1)
    ap.add_argument("--verify_bonus", type=float, default=0.02)
    ap.add_argument("--early_stop_bonus", type=float, default=0.05)
    ap.add_argument("--support_hit_reward", type=float, default=0.15)
    ap.add_argument("--support_full_reward", type=float, default=0.25)
    ap.add_argument("--support_rank_reward", type=float, default=0.1)
    ap.add_argument("--discrimination_reward", type=float, default=0.1)
    ap.add_argument("--contradiction_bonus", type=float, default=0.1)
    ap.add_argument("--disable_support_reward", action="store_true")
    ap.add_argument("--disable_verification_reward", action="store_true")
    ap.add_argument("--disable_efficiency_penalty", action="store_true")
    ap.add_argument("--disable_counterfactual_discrimination_reward", action="store_true")
    ap.add_argument("--filter_min_trajectory_score", type=float, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    rows = load_rows(args.traces)
    if args.filter_min_trajectory_score is not None:
        rows = [r for r in rows if float(r.get("trajectory_score", 0.0)) >= args.filter_min_trajectory_score]
    if not rows:
        raise ValueError("No traces left after filtering; lower --filter_min_trajectory_score or check input")
    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    act = torch.tensor([r["action"] for r in rows], dtype=torch.long)

    reward_cfg = RewardConfig(
        correctness_reward=args.correctness_reward,
        token_penalty=args.token_penalty,
        retrieval_penalty=args.retrieval_penalty,
        branch_penalty=args.branch_penalty,
        verify_bonus=args.verify_bonus,
        early_stop_bonus=args.early_stop_bonus,
        support_hit_reward=args.support_hit_reward,
        support_full_reward=args.support_full_reward,
        support_rank_reward=args.support_rank_reward,
        discrimination_reward=args.discrimination_reward,
        contradiction_bonus=args.contradiction_bonus,
        use_support_reward=not args.disable_support_reward,
        use_verification_reward=not args.disable_verification_reward,
        use_efficiency_penalty=not args.disable_efficiency_penalty,
        use_counterfactual_discrimination_reward=not args.disable_counterfactual_discrimination_reward,
    )
    rew = build_reward_tensor(rows, reward_cfg, success_bonus=args.success_bonus)

    val_obs = val_act = val_rew = None
    if args.val_traces:
        val_rows = load_rows(args.val_traces)
        if args.filter_min_trajectory_score is not None:
            val_rows = [r for r in val_rows if float(r.get("trajectory_score", 0.0)) >= args.filter_min_trajectory_score]
        if val_rows:
            val_obs = torch.tensor([r["obs"] for r in val_rows], dtype=torch.float32)
            val_act = torch.tensor([r["action"] for r in val_rows], dtype=torch.long)
            val_rew = build_reward_tensor(val_rows, reward_cfg, success_bonus=args.success_bonus)

    act_dim = int(max(act.max().item(), val_act.max().item() if val_act is not None else 0)) + 1
    cfg = PolicyConfig(
        policy_type=args.policy_type,
        obs_dim=obs.shape[1],
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        history_len=args.history_len,
    )
    model = build_policy(cfg)
    if args.init_policy:
        init_ckpt = torch.load(args.init_policy, map_location="cpu")
        model.load_state_dict(init_ckpt["state_dict"], strict=False)
        print(json.dumps({"init_policy": args.init_policy, "status": "loaded"}))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = []
    best_metric = float("-inf")
    for epoch in range(1, args.epochs + 1):
        logits = model(obs)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp[torch.arange(len(act)), act]
        probs = F.softmax(logits, dim=-1)

        bc_loss = -chosen.mean()
        if args.objective == "wbc":
            w = torch.clamp(rew - rew.min() + 1e-3, max=20.0).detach()
            pg_loss = -(w * chosen).mean()
            cql_penalty = torch.tensor(0.0)
        elif args.objective == "awr":
            adv = rew - rew.mean()
            w = torch.exp(torch.clamp(adv / max(1e-6, args.adv_temperature), -5, 5)).detach()
            pg_loss = -(w * chosen).mean()
            cql_penalty = torch.tensor(0.0)
        else:  # cql_like
            adv = rew - rew.mean()
            w = torch.exp(torch.clamp(adv / max(1e-6, args.adv_temperature), -5, 5)).detach()
            pg_loss = -(w * chosen).mean()
            lse = torch.logsumexp(logits, dim=-1)
            cql_penalty = (lse - chosen).mean()

        entropy = -(probs * logp).sum(dim=-1).mean()
        loss = pg_loss + args.bc_coef * bc_loss + cql_penalty - args.entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        rec = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "cql_penalty": float(cql_penalty.item()),
            "entropy": float(entropy.item()),
        }
        rec.update({f"train_{k}": v for k, v in eval_metrics(model, obs, act, rew).items()})
        if val_obs is not None and val_act is not None and val_rew is not None:
            rec.update({f"val_{k}": v for k, v in eval_metrics(model, val_obs, val_act, val_rew).items()})
        metrics.append(rec)
        print(json.dumps(rec))

        monitor = rec.get("val_avg_reward", rec.get("train_avg_reward", 0.0))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        latest_path = out.with_name(out.stem + ".latest" + out.suffix)
        torch.save({"state_dict": model.state_dict(), "obs_dim": obs.shape[1], "act_dim": act_dim, "epoch": epoch}, latest_path)
        if monitor > best_metric:
            best_metric = monitor
            best_path = out.with_name(out.stem + ".best" + out.suffix)
            torch.save({"state_dict": model.state_dict(), "obs_dim": obs.shape[1], "act_dim": act_dim, "epoch": epoch, "best_metric": best_metric}, best_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs.shape[1],
            "act_dim": act_dim,
            "arch": {
                "policy_type": args.policy_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "history_len": args.history_len,
            },
            "dataset": args.traces,
            "seed": args.seed,
            "git_commit": get_git_commit(),
            "hparams": {
                "epochs": args.epochs,
                "lr": args.lr,
                "objective": args.objective,
                "bc_coef": args.bc_coef,
                "adv_temperature": args.adv_temperature,
                "entropy_coef": args.entropy_coef,
                "success_bonus": args.success_bonus,
            },
            "reward_config": reward_cfg.__dict__,
        },
        out,
    )
    out.with_suffix(".metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    out.with_suffix(".config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
