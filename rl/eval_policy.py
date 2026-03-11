from __future__ import annotations

import argparse
import json

import torch

from rl.train_il import PolicyMLP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--traces", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.policy, map_location="cpu")
    model = PolicyMLP(obs_dim=ckpt["obs_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total = 0
    correct = 0
    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            obs = torch.tensor(row["obs"], dtype=torch.float32).unsqueeze(0)
            pred = int(model(obs).argmax(dim=-1).item())
            correct += int(pred == int(row["action"]))
            total += 1

    print(json.dumps({"action_match": correct / max(1, total), "n": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
