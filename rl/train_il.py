from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
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
    args = ap.parse_args()

    set_seed(args.seed)

    obs, acts = [], []
    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            obs.append(row["obs"])
            acts.append(row["action"])

    x = torch.tensor(obs, dtype=torch.float32)
    y = torch.tensor(acts, dtype=torch.long)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    act_dim = int(y.max().item()) + 1 if y.numel() else 11
    cfg = PolicyConfig(
        policy_type=args.policy_type,
        obs_dim=x.shape[1],
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        history_len=args.history_len,
    )
    model = build_policy(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    metrics = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses) if losses else 0.0)}
        metrics.append(rec)
        print(json.dumps(rec))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "obs_dim": x.shape[1],
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
        "hparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
    }
    torch.save(ckpt, out)
    out.with_suffix(".metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    out.with_suffix(".config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
