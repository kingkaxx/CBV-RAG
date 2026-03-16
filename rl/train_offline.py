from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rl.policy import PolicyConfig, build_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="checkpoints/policy_offline.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy_type", choices=["mlp", "mlp_residual", "gru_policy"], default="mlp_residual")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--history_len", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.traces).read_text(encoding="utf-8").splitlines() if l.strip()]
    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    act = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)

    # Reward shaping for offline preference weighting.
    rew = []
    for r in rows:
        v = float(r.get("reward", 0.0))
        v += 0.75 * float(bool(r.get("terminal_correct", False)))
        v += 0.25 * float(r.get("trajectory_score", 0.0))
        rew.append(v)
    rew = torch.tensor(rew, dtype=torch.float32)

    cfg = PolicyConfig(
        policy_type=args.policy_type,
        obs_dim=int(obs.shape[1]),
        act_dim=int(act.max().item()) + 1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        history_len=args.history_len,
    )
    model = build_policy(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = TensorDataset(obs, act, rew)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, ab, rb in dl:
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=-1)
            chosen = logp[torch.arange(len(ab)), ab]

            adv = rb - rb.mean()
            weights = torch.exp(torch.clamp(adv, -2.0, 2.0)).detach()

            loss = -(weights * chosen).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "offline_loss": sum(losses) / max(1, len(losses))}))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": int(obs.shape[1]),
            "act_dim": cfg.act_dim,
            "arch": {
                "policy_type": args.policy_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "history_len": args.history_len,
            },
        },
        out,
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
