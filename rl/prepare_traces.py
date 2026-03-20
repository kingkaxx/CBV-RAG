from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from cbvrag.actions import Action


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_episode_format(rows: List[Dict]) -> bool:
    """Return True if *rows* are in the new episode-level format from collect_traces.py."""
    return bool(rows) and "trajectory" in rows[0]


def _step_sort_key(row: Dict) -> tuple[int, int]:
    idx = row.get("episode_step_index")
    step = row.get("step")
    idx_v = int(idx) if idx is not None else 10**9
    step_v = int(step) if step is not None else idx_v
    return idx_v, step_v


def get_terminal_row(erows: List[Dict]) -> Dict:
    done_rows = [r for r in erows if bool(r.get("done", False))]
    if done_rows:
        return max(done_rows, key=_step_sort_key)
    return max(erows, key=_step_sort_key)


def episode_passes(terminal_row: Dict, args: argparse.Namespace) -> bool:
    if args.only_successful and not bool(terminal_row.get("success", terminal_row.get("terminal_correct", False))):
        return False
    if args.min_episode_reward is not None and float(terminal_row.get("episode_total_reward", 0.0)) < args.min_episode_reward:
        return False
    if args.max_episode_tokens is not None and int(terminal_row.get("episode_total_tokens", 0)) > args.max_episode_tokens:
        return False
    if args.filter_min_trajectory_score is not None and float(terminal_row.get("trajectory_score", 0.0)) < args.filter_min_trajectory_score:
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


def _terminal_action_hist(episodes: Dict[str, List[Dict]]) -> Dict[str, int]:
    c = Counter()
    for erows in episodes.values():
        trow = get_terminal_row(erows)
        c[int(trow.get("action", -1))] += 1
    return {str(k): int(v) for k, v in sorted(c.items()) if k >= 0}





def _warn_terminal_action_disappearance(kept_terminal_hist: Dict[str, int], train_terminal_hist: Dict[str, int]) -> None:
    for action_idx in (int(Action.ANSWER_DIRECT), int(Action.STOP_AND_ANSWER)):
        a = str(action_idx)
        if int(kept_terminal_hist.get(a, 0)) > 0 and int(train_terminal_hist.get(a, 0)) == 0:
            print(
                f"[prepare_traces][warn] terminal action {action_idx} present in kept episodes but missing from train terminal split",
                flush=True,
            )

def _episode_passes_new(ep: Dict, args: argparse.Namespace) -> bool:
    """Filter predicate for episode-level records (new format)."""
    if args.only_successful and not bool(ep.get("terminal_correct", False)):
        return False
    traj_score = float(ep.get("trajectory_score", 0.0))
    if args.filter_min_trajectory_score is not None and traj_score < args.filter_min_trajectory_score:
        return False
    if args.min_episode_reward is not None and traj_score < args.min_episode_reward:
        return False
    if args.max_episode_tokens is not None and int(ep.get("total_tokens", 0)) > args.max_episode_tokens:
        return False
    return True


def _flatten_episode(ep: Dict) -> List[Dict]:
    """Convert one episode record into a list of step rows for train_il.py."""
    qid = str(ep.get("qid", "unknown"))
    traj_score = float(ep.get("trajectory_score", 0.0))
    terminal_correct = bool(ep.get("terminal_correct", False))
    episode_attr_score = float(ep.get("episode_attr_score", 0.0))
    il_weight = float(ep.get("il_weight", 1.0))
    num_steps = int(ep.get("num_steps", len(ep.get("trajectory", []))))
    last_t = num_steps - 1

    rows: List[Dict] = []
    for step in ep.get("trajectory", []):
        t = int(step.get("t", 0))
        rows.append({
            "qid": qid,
            "obs": step["obs"],
            "action": step["action"],
            "reward": step["reward"],
            "trajectory_score": traj_score,
            "terminal_correct": terminal_correct,
            "episode_attr_score": episode_attr_score,
            "il_weight": il_weight,
            "done": t == last_t,
        })
    return rows


def _process_episode_format(rows: List[Dict], args: argparse.Namespace, out_dir: Path) -> int:
    """Handle the new episode-level trace format end-to-end.

    Steps:
    1. Apply episode-level filters.
    2. Optionally cap traces per qid.
    3. Split by qid (train/val) BEFORE flattening to prevent leakage.
    4. Flatten each episode's trajectory into step rows.
    5. Write train.jsonl, val.jsonl, summary.json.

    Returns 0 on success.
    """
    total_episodes = len(rows)
    kept: List[Dict] = []
    filtered_episodes = 0
    for ep in rows:
        if _episode_passes_new(ep, args):
            kept.append(ep)
        else:
            filtered_episodes += 1

    by_qid: Dict[str, List[Dict]] = defaultdict(list)
    for ep in kept:
        by_qid[str(ep.get("qid", "unknown"))].append(ep)

    capped: List[Dict] = []
    for eps_for_qid in by_qid.values():
        eps_for_qid.sort(key=lambda e: float(e.get("trajectory_score", 0.0)), reverse=True)
        correct_eps = [e for e in eps_for_qid if bool(e.get("terminal_correct", False))]
        wrong_eps = [e for e in eps_for_qid if not bool(e.get("terminal_correct", False))]

        max_correct = args.max_correct_traces_per_qid
        if max_correct is None and args.max_traces_per_qid is not None:
            max_correct = args.max_traces_per_qid
        if max_correct is not None:
            correct_eps = correct_eps[: max(1, max_correct)]

        max_wrong = args.max_wrong_traces_per_qid
        if max_wrong is not None:
            wrong_eps = wrong_eps[: max(0, max_wrong)]

        for ep in correct_eps + wrong_eps:
            if bool(ep.get("terminal_correct", False)) and args.correct_il_weight > 1.0:
                ep = dict(ep)
                ep["il_weight"] = float(args.correct_il_weight)
            capped.append(ep)
    kept = capped

    # Split by qid BEFORE flattening.
    all_qids = sorted({str(ep.get("qid", "unknown")) for ep in kept})
    train_qids, val_qids = split_qids(all_qids, args.val_ratio, args.seed)

    train_eps = [ep for ep in kept if str(ep.get("qid", "unknown")) in train_qids]
    val_eps = [ep for ep in kept if str(ep.get("qid", "unknown")) in val_qids]

    # Flatten each episode's trajectory into step rows.
    train_rows: List[Dict] = []
    for ep in train_eps:
        train_rows.extend(_flatten_episode(ep))

    val_rows: List[Dict] = []
    for ep in val_eps:
        val_rows.extend(_flatten_episode(ep))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    (out_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val_rows) + ("\n" if val_rows else ""),
        encoding="utf-8",
    )

    kept_scores = [float(ep.get("trajectory_score", 0.0)) for ep in kept]
    mean_traj_score = mean(kept_scores) if kept_scores else 0.0
    pct_terminal_correct = (
        100.0 * sum(1 for ep in kept if bool(ep.get("terminal_correct", False))) / len(kept)
        if kept else 0.0
    )

    summary = {
        "total_episodes": total_episodes,
        "kept_episodes": len(kept),
        "filtered_episodes": filtered_episodes,
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
        "train_steps": len(train_rows),
        "val_steps": len(val_rows),
        "mean_trajectory_score": round(mean_traj_score, 6),
        "pct_terminal_correct": round(pct_terminal_correct, 2),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


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
    ap.add_argument("--max_correct_traces_per_qid", type=int, default=None)
    ap.add_argument("--max_wrong_traces_per_qid", type=int, default=2)
    ap.add_argument("--correct_il_weight", type=float, default=1.5)
    args = ap.parse_args()

    rows = load_rows(Path(args.input))

    # Detect format and branch.
    if _is_episode_format(rows):
        return _process_episode_format(rows, args, Path(args.output_dir))

    # ---- Legacy step-level format (unchanged below) ----
    episodes = build_episode_groups(rows)

    kept_episodes: Dict[str, List[Dict]] = {}
    filtered_episodes = 0
    missing_terminal_warning_count = 0
    episode_meta: Dict[str, Dict] = {}

    for eid, erows in episodes.items():
        sorted_rows = sorted(erows, key=_step_sort_key)
        terminal_row = get_terminal_row(sorted_rows)
        if not any(bool(r.get("done", False)) for r in sorted_rows):
            missing_terminal_warning_count += 1
            print(f"[prepare_traces][warn] episode={eid} has no done=True row; using latest step as terminal row", flush=True)

        if not episode_passes(terminal_row, args):
            filtered_episodes += 1
            continue

        kept_episodes[eid] = sorted_rows
        episode_meta[eid] = {
            "qid": str(terminal_row.get("qid", "unknown")),
            "trajectory_score": float(terminal_row.get("trajectory_score", 0.0)),
        }

    if args.max_traces_per_qid is not None:
        by_qid = defaultdict(list)
        for eid in kept_episodes:
            by_qid[episode_meta[eid]["qid"]].append((episode_meta[eid]["trajectory_score"], eid))

        selected_eids = set()
        for _, arr in by_qid.items():
            arr.sort(reverse=True)
            for _, eid in arr[: max(1, args.max_traces_per_qid)]:
                selected_eids.add(eid)

        kept_episodes = {eid: kept_episodes[eid] for eid in selected_eids}
        episode_meta = {eid: episode_meta[eid] for eid in selected_eids}

    if args.min_action_count_for_keep is not None:
        global_hist = Counter(int(r.get("action", -1)) for er in kept_episodes.values() for r in er)
        allowed_actions = {a for a, c in global_hist.items() if a >= 0 and c >= args.min_action_count_for_keep}
        kept_episodes = {
            eid: er
            for eid, er in kept_episodes.items()
            if any(int(r.get("action", -1)) in allowed_actions for r in er)
        }
        episode_meta = {eid: episode_meta[eid] for eid in kept_episodes}

    qids = sorted({episode_meta[eid]["qid"] for eid in kept_episodes})
    train_qids, val_qids = split_qids(qids, args.val_ratio, args.seed)

    train_rows, val_rows = [], []
    train_eps, val_eps = {}, {}
    for eid, erows in kept_episodes.items():
        qid = episode_meta[eid]["qid"]
        if qid in train_qids:
            train_rows.extend(erows)
            train_eps[eid] = erows
        else:
            val_rows.extend(erows)
            val_eps[eid] = erows

    kept_raw_hist = _action_hist([r for er in kept_episodes.values() for r in er])
    train_hist = _action_hist(train_rows)
    val_hist = _action_hist(val_rows)
    train_terminal_hist = _terminal_action_hist(train_eps)
    val_terminal_hist = _terminal_action_hist(val_eps)
    kept_terminal_hist = _terminal_action_hist(kept_episodes)

    kept_actions = {int(a) for a in kept_raw_hist}
    train_actions = {int(a) for a in train_hist}
    disappeared_actions = sorted(kept_actions - train_actions)
    if disappeared_actions:
        print(
            f"[prepare_traces][warn] actions present in kept episodes disappeared from train split: {disappeared_actions}",
            flush=True,
        )
    _warn_terminal_action_disappearance(kept_terminal_hist, train_terminal_hist)

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
        "num_missing_terminal_rows": missing_terminal_warning_count,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_qids": len({r['qid'] for r in train_rows}) if train_rows else 0,
        "val_qids": len({r['qid'] for r in val_rows}) if val_rows else 0,
        "train_avg_episode_reward": agg(train_rows, "episode_total_reward"),
        "val_avg_episode_reward": agg(val_rows, "episode_total_reward"),
        "train_success_rate": agg(train_rows, "success"),
        "val_success_rate": agg(val_rows, "success"),
        "kept_raw_action_histogram": kept_raw_hist,
        "kept_terminal_action_histogram": kept_terminal_hist,
        "train_action_histogram": train_hist,
        "val_action_histogram": val_hist,
        "train_terminal_action_histogram": train_terminal_hist,
        "val_terminal_action_histogram": val_terminal_hist,
    }
    if args.emit_action_histogram:
        summary["train_inverse_action_weights"] = inv_weights

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
