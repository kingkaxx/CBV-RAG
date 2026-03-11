from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="checkpoints/policy_il.pt")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    obs, acts = [], []
    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            obs.append(row["obs"])
            acts.append(row["action"])

    x = torch.tensor(obs, dtype=torch.float32)
    y = torch.tensor(acts, dtype=torch.long)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = PolicyMLP(obs_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": x.shape[1]}, out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
