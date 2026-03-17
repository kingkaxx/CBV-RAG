from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from rl.policy import build_policy, policy_config_from_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--traces", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.policy, map_location="cpu")
    cfg = policy_config_from_checkpoint(ckpt)
    model = build_policy(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total = 0
    correct = 0
    per_true_total = defaultdict(int)
    per_true_correct = defaultdict(int)
    confusion = [[0 for _ in range(cfg.act_dim)] for _ in range(cfg.act_dim)]
    terminal_total = 0
    terminal_correct = 0

    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            obs = torch.tensor(row["obs"], dtype=torch.float32).unsqueeze(0)
            pred = int(model(obs).argmax(dim=-1).item())
            gold = int(row["action"])

            correct += int(pred == gold)
            total += 1
            per_true_total[gold] += 1
            per_true_correct[gold] += int(pred == gold)
            confusion[gold][pred] += 1

            is_terminal = bool(row.get("done", False)) or gold in {5, 10}
            if is_terminal:
                terminal_total += 1
                terminal_correct += int(pred == gold)

    per_action_acc = {
        str(a): (per_true_correct[a] / per_true_total[a]) if per_true_total[a] > 0 else None
        for a in range(cfg.act_dim)
    }

    print(
        json.dumps(
            {
                "action_match": correct / max(1, total),
                "terminal_action_match": terminal_correct / max(1, terminal_total),
                "n": total,
                "policy_type": cfg.policy_type,
                "obs_dim": cfg.obs_dim,
                "act_dim": cfg.act_dim,
                "per_action_acc": per_action_acc,
                "confusion_matrix": confusion,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
