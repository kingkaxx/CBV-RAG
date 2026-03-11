from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from rl.train_il import PolicyMLP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="checkpoints/policy_offline.pt")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.traces).read_text(encoding="utf-8").splitlines() if l.strip()]
    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    act = torch.tensor([r["action"] for r in rows], dtype=torch.long)
    rew = torch.tensor([r.get("reward", 0.0) + (1.0 if r.get("terminal_correct") else 0.0) for r in rows], dtype=torch.float32)

    model = PolicyMLP(obs_dim=obs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(args.epochs):
        logits = model(obs)
        logp = torch.log_softmax(logits, dim=-1)
        chosen = logp[torch.arange(len(act)), act]
        # Simple AWR-style weighting by positive-centered rewards.
        adv = rew - rew.mean()
        w = torch.exp(torch.clamp(adv, -2, 2)).detach()
        loss = -(w * chosen).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": obs.shape[1]}, out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
