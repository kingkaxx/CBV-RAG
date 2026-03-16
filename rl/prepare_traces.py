from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
    if args.filter_min_trajectory_score is not None and float(row.get("trajectory_score", 0.0)) < args.filter_min_trajectory_score:
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


def build_episode_groups(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        eid = str(row.get("episode_id") or f"{row.get('qid', 'unknown')}::0")
        grouped[eid].append(row)
    return grouped


def _action_hist(rows: List[Dict]) -> Dict[str, int]:
    c = Counter(int(r.get("action", -1)) for r in rows)
    return {str(k): int(v) for k, v in sorted(c.items()) if k >= 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_successful", action="store_true")
    ap.add_argument("--min_episode_reward", type=float, default=None)
    ap.add_argument("--max_episode_tokens", type=int, default=None)

    ap.add_argument("--min_action_count_for_keep", type=int, default=None)
    ap.add_argument("--max_traces_per_qid", type=int, default=None)
    ap.add_argument("--emit_action_histogram", action="store_true")
    ap.add_argument("--filter_min_trajectory_score", type=float, default=None)
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    episodes = build_episode_groups(rows)

    # Filter at episode level.
    kept_episodes = {}
    filtered_episodes = 0
    for eid, erows in episodes.items():
        exemplar = erows[0]
        if not episode_passes(exemplar, args):
            filtered_episodes += 1
            continue
        kept_episodes[eid] = erows

    # Optional cap per qid, keeping top trajectory_score episodes.
    if args.max_traces_per_qid is not None:
        by_qid = defaultdict(list)
        for eid, erows in kept_episodes.items():
            qid = str(erows[0].get("qid", "unknown"))
            score = float(erows[0].get("trajectory_score", 0.0))
            by_qid[qid].append((score, eid))

        selected_eids = set()
        for qid, arr in by_qid.items():
            arr.sort(reverse=True)
            for _, eid in arr[: max(1, args.max_traces_per_qid)]:
                selected_eids.add(eid)

        kept_episodes = {eid: kept_episodes[eid] for eid in selected_eids}

    # Episode-aware action filter.
    if args.min_action_count_for_keep is not None:
        global_hist = Counter(int(r.get("action", -1)) for er in kept_episodes.values() for r in er)
        allowed_actions = {a for a, c in global_hist.items() if a >= 0 and c >= args.min_action_count_for_keep}
        kept_episodes = {
            eid: er
            for eid, er in kept_episodes.items()
            if any(int(r.get("action", -1)) in allowed_actions for r in er)
        }

    qids = sorted({str(er[0].get("qid", "unknown")) for er in kept_episodes.values()})
    train_qids, val_qids = split_qids(qids, args.val_ratio, args.seed)

    train_rows, val_rows = [], []
    for _, erows in kept_episodes.items():
        qid = str(erows[0].get("qid", "unknown"))
        if qid in train_qids:
            train_rows.extend(erows)
        else:
            val_rows.extend(erows)

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

    train_hist = _action_hist(train_rows)
    val_hist = _action_hist(val_rows)
    inv_weights = {}
    if args.emit_action_histogram and train_hist:
        for a, c in train_hist.items():
            inv_weights[a] = float(1.0 / max(1, c))

    summary = {
        "input": args.input,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_input_rows": len(rows),
        "num_input_episodes": len(episodes),
        "num_filtered_episodes": filtered_episodes,
        "num_kept_episodes": len(kept_episodes),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_qids": len({r['qid'] for r in train_rows}) if train_rows else 0,
        "val_qids": len({r['qid'] for r in val_rows}) if val_rows else 0,
        "train_avg_episode_reward": agg(train_rows, "episode_total_reward"),
        "val_avg_episode_reward": agg(val_rows, "episode_total_reward"),
        "train_success_rate": agg(train_rows, "success"),
        "val_success_rate": agg(val_rows, "success"),
    }
    if args.emit_action_histogram:
        summary["train_action_histogram"] = train_hist
        summary["val_action_histogram"] = val_hist
        summary["train_inverse_action_weights"] = inv_weights

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
