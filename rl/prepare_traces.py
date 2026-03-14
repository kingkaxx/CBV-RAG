from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_passes(row: Dict, args: argparse.Namespace) -> bool:
    if args.only_successful and not bool(row.get("success", row.get("terminal_correct", False))):
        return False
    if args.min_episode_reward is not None and float(row.get("episode_total_reward", 0.0)) < args.min_episode_reward:
        return False
    if args.max_episode_tokens is not None and int(row.get("episode_total_tokens", 0)) > args.max_episode_tokens:
        return False
    return True


def split_qids(qids: List[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    val_n = int(round(len(shuffled) * val_ratio))
    val = set(shuffled[:val_n])
    train = set(shuffled[val_n:])
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_successful", action="store_true")
    ap.add_argument("--min_episode_reward", type=float, default=None)
    ap.add_argument("--max_episode_tokens", type=int, default=None)
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("qid", "unknown"))].append(row)

    qids = sorted(grouped.keys())
    train_qids, val_qids = split_qids(qids, args.val_ratio, args.seed)

    train_rows, val_rows = [], []
    filtered_episodes = 0
    for qid, qrows in grouped.items():
        exemplar = qrows[0]
        if not episode_passes(exemplar, args):
            filtered_episodes += 1
            continue
        if qid in train_qids:
            train_rows.extend(qrows)
        else:
            val_rows.extend(qrows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    summary_path = out_dir / "summary.json"

    train_path.write_text("\n".join(json.dumps(r) for r in train_rows) + ("\n" if train_rows else ""), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(r) for r in val_rows) + ("\n" if val_rows else ""), encoding="utf-8")

    def agg(rr: List[Dict], key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rr if r.get("done", False)]
        return mean(vals) if vals else 0.0

    summary = {
        "input": args.input,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_input_rows": len(rows),
        "num_input_qids": len(qids),
        "num_filtered_episodes": filtered_episodes,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_qids": len({r['qid'] for r in train_rows}) if train_rows else 0,
        "val_qids": len({r['qid'] for r in val_rows}) if val_rows else 0,
        "train_avg_episode_reward": agg(train_rows, "episode_total_reward"),
        "val_avg_episode_reward": agg(val_rows, "episode_total_reward"),
        "train_success_rate": agg(train_rows, "success"),
        "val_success_rate": agg(val_rows, "success"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
