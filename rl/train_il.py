from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cbvrag.actions import ACTION_ENUM_VERSION, Action, action_names
from cbvrag.features import FEATURE_SCHEMA_VERSION
from rl.policy import PolicyConfig, build_policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_xy(path: str, min_score: float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    obs, acts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if min_score is not None and float(row.get("trajectory_score", 0.0)) < min_score:
                continue
            obs.append(row["obs"])
            acts.append(int(row["action"]))
    if not obs:
        raise ValueError(f"No traces loaded from {path}. Adjust filters or input file.")
    return torch.tensor(obs, dtype=torch.float32), torch.tensor(acts, dtype=torch.long)


def _eval(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "acc": float(correct / max(1, total)),
        "n": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--val_traces", default=None)
    ap.add_argument("--out", default="checkpoints/policy_il.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy_type", choices=["mlp", "mlp_residual", "gru_policy"], default="mlp")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--history_len", type=int, default=1)
    ap.add_argument("--use_action_weights", action="store_true")
    ap.add_argument("--terminal_action_boost", type=float, default=1.0)
    ap.add_argument("--filter_min_trajectory_score", type=float, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = _load_xy(args.traces, min_score=args.filter_min_trajectory_score)
    obs_dim = int(x_train.shape[1])
    act_dim = len(Action)

    if int(y_train.max().item()) >= act_dim or int(y_train.min().item()) < 0:
        raise ValueError(f"Trace actions must be within [0, {act_dim - 1}], got min={int(y_train.min().item())}, max={int(y_train.max().item())}")

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_dl = None
    if args.val_traces:
        x_val, y_val = _load_xy(args.val_traces)
        if int(x_val.shape[1]) != obs_dim:
            raise ValueError(f"val_traces obs_dim={int(x_val.shape[1])} does not match train obs_dim={obs_dim}")
        val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights = None
    if args.use_action_weights:
        counts = torch.bincount(y_train, minlength=act_dim).float()
        weights = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
        if weights.sum() > 0:
            weights = weights / weights.mean().clamp_min(1e-6)

    if args.terminal_action_boost != 1.0:
        if args.terminal_action_boost <= 0:
            raise ValueError("--terminal_action_boost must be > 0")
        if weights is None:
            weights = torch.ones(act_dim, dtype=torch.float32)
        for terminal_action in (int(Action.ANSWER_DIRECT), int(Action.STOP_AND_ANSWER)):
            weights[terminal_action] *= float(args.terminal_action_boost)

    if weights is not None:
        if weights.mean() > 0:
            weights = weights / weights.mean().clamp_min(1e-6)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
        }
        if val_dl is not None:
            val_metrics = _eval(model, val_dl, device)
            rec["val_loss"] = val_metrics["loss"]
            rec["val_acc"] = val_metrics["acc"]
        metrics.append(rec)
        print(json.dumps(rec))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
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
        "dataset": args.traces,
        "val_dataset": args.val_traces,
        "seed": args.seed,
        "git_commit": get_git_commit(),
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_action_weights": bool(args.use_action_weights),
            "terminal_action_boost": args.terminal_action_boost,
            "filter_min_trajectory_score": args.filter_min_trajectory_score,
        },
    }
    torch.save(ckpt, out)
    out.with_suffix(".metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    out.with_suffix(".config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
