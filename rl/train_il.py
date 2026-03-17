from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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


def _load_rows(path: str, min_score: float | None = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if min_score is not None and float(row.get("trajectory_score", 0.0)) < min_score:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No traces loaded from {path}. Adjust filters or input file.")
    return rows


def _rows_to_tensors(rows: List[dict], act_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    acts = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)

    if int(acts.max().item()) >= act_dim or int(acts.min().item()) < 0:
        raise ValueError(
            f"Trace actions must be within [0, {act_dim - 1}], "
            f"got min={int(acts.min().item())}, max={int(acts.max().item())}"
        )

    weights = []
    for r in rows:
        w = 1.0
        w += 0.25 * float(r.get("trajectory_score", 0.0))
        if bool(r.get("terminal_correct", False)):
            w += 0.50
        if bool(r.get("done", False)):
            w += 0.10
        weights.append(max(0.05, float(w)))

    sample_w = torch.tensor(weights, dtype=torch.float32)
    sample_w = sample_w / sample_w.mean().clamp_min(1e-6)
    return obs, acts, sample_w


def _macro_action_acc(y_true: np.ndarray, y_pred: np.ndarray, act_dim: int) -> float:
    vals = []
    for a in range(act_dim):
        mask = (y_true == a)
        if mask.sum() == 0:
            continue
        vals.append(float((y_pred[mask] == y_true[mask]).mean()))
    return float(np.mean(vals) if vals else 0.0)


def _eval(model: torch.nn.Module, dl: DataLoader, device: torch.device, ce: nn.Module, act_dim: int) -> dict:
    model.eval()
    losses, correct, total = [], 0, 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb, sw in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            sw = sw.to(device)

            logits = model(xb)
            per_ex = ce(logits, yb)
            loss = (per_ex * sw).mean()

            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

            all_true.append(yb.cpu())
            all_pred.append(pred.cpu())

    if all_true:
        y_true = torch.cat(all_true).numpy()
        y_pred = torch.cat(all_pred).numpy()
        macro_acc = _macro_action_acc(y_true, y_pred, act_dim)
    else:
        macro_acc = 0.0

    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "acc": float(correct / max(1, total)),
        "macro_action_acc": float(macro_acc),
        "n": total,
    }


def _split_rows_by_qid(rows: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    qid_to_rows: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        qid_to_rows[str(r.get("qid", "unknown"))].append(r)

    qids = list(qid_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)

    if len(qids) < 2 or val_ratio <= 0:
        return rows, []

    val_qid_count = max(1, int(round(len(qids) * val_ratio)))
    val_qids = set(qids[:val_qid_count])

    train_rows, val_rows = [], []
    for qid, qrows in qid_to_rows.items():
        if qid in val_qids:
            val_rows.extend(qrows)
        else:
            train_rows.extend(qrows)

    if not train_rows:
        train_rows, val_rows = rows, []

    return train_rows, val_rows


def _make_weighted_sampler(y_train: torch.Tensor, sample_w_train: torch.Tensor, act_dim: int) -> WeightedRandomSampler:
    counts = torch.bincount(y_train, minlength=act_dim).float()
    class_sampling_w = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
    class_sampling_w = class_sampling_w / class_sampling_w.mean().clamp_min(1e-6)

    row_w = class_sampling_w[y_train] * sample_w_train
    row_w = row_w / row_w.mean().clamp_min(1e-6)

    return WeightedRandomSampler(
        weights=row_w.double(),
        num_samples=len(row_w),
        replacement=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--val_traces", default=None)
    ap.add_argument("--out", default="checkpoints/policy_il.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy_type", choices=["mlp", "mlp_residual", "gru_policy"], default="mlp_residual")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--history_len", type=int, default=1)
    ap.add_argument("--use_action_weights", action="store_true")
    ap.add_argument("--terminal_action_boost", type=float, default=1.0)
    ap.add_argument("--filter_min_trajectory_score", type=float, default=None)
    ap.add_argument("--auto_val_ratio", type=float, default=0.1)
    ap.add_argument("--use_weighted_sampler", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dim = len(Action)

    all_rows = _load_rows(args.traces, min_score=args.filter_min_trajectory_score)
    obs_dim = len(all_rows[0]["obs"])

    if args.val_traces:
        train_rows = all_rows
        val_rows = _load_rows(args.val_traces)
    else:
        train_rows, val_rows = _split_rows_by_qid(all_rows, args.auto_val_ratio, args.seed)

    x_train, y_train, w_train = _rows_to_tensors(train_rows, act_dim=act_dim)
    train_ds = TensorDataset(x_train, y_train, w_train)

    val_ds = None
    if val_rows:
        x_val, y_val, w_val = _rows_to_tensors(val_rows, act_dim=act_dim)
        if int(x_val.shape[1]) != obs_dim:
            raise ValueError(f"val_traces obs_dim={int(x_val.shape[1])} does not match train obs_dim={obs_dim}")
        val_ds = TensorDataset(x_val, y_val, w_val)

    sampler = None
    if args.use_weighted_sampler:
        sampler = _make_weighted_sampler(y_train, w_train, act_dim)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None

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

    class_weights = None
    if args.use_action_weights:
        counts = torch.bincount(y_train, minlength=act_dim).float()
        class_weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.mean().clamp_min(1e-6)

    if args.terminal_action_boost != 1.0:
        if args.terminal_action_boost <= 0:
            raise ValueError("--terminal_action_boost must be > 0")
        if class_weights is None:
            class_weights = torch.ones(act_dim, dtype=torch.float32)
        for terminal_action in (int(Action.ANSWER_DIRECT), int(Action.STOP_AND_ANSWER)):
            class_weights[terminal_action] *= float(args.terminal_action_boost)

    if class_weights is not None and class_weights.mean() > 0:
        class_weights = class_weights / class_weights.mean().clamp_min(1e-6)

    ce = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        reduction="none",
    )

    metrics = []
    best_metric = float("-inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_correct, train_total = [], 0, 0

        for xb, yb, sw in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            sw = sw.to(device)

            logits = model(xb)
            per_ex = ce(logits, yb)
            loss = (per_ex * sw).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1)
            train_correct += int((pred == yb).sum().item())
            train_total += int(yb.numel())

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else 0.0),
            "train_acc": float(train_correct / max(1, train_total)),
        }
        score_for_selection = rec["train_acc"]

        if val_dl is not None:
            val_metrics = _eval(model, val_dl, device, ce, act_dim)
            rec["val_loss"] = val_metrics["loss"]
            rec["val_acc"] = val_metrics["acc"]
            rec["val_macro_action_acc"] = val_metrics["macro_action_acc"]
            score_for_selection = 0.5 * rec["val_acc"] + 0.5 * rec["val_macro_action_acc"]

        if score_for_selection > best_metric:
            best_metric = score_for_selection
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rec["is_best"] = True
            rec["selection_metric"] = float(score_for_selection)
        else:
            rec["is_best"] = False
            rec["selection_metric"] = float(score_for_selection)

        metrics.append(rec)
        print(json.dumps(rec))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_out = out.with_name(out.stem + "_best" + out.suffix)

    def _build_ckpt(state_dict: dict, tag: str) -> dict:
        return {
            "state_dict": state_dict,
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
            "checkpoint_tag": tag,
            "best_selection_metric": best_metric,
            "hparams": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "use_action_weights": bool(args.use_action_weights),
                "terminal_action_boost": args.terminal_action_boost,
                "filter_min_trajectory_score": args.filter_min_trajectory_score,
                "auto_val_ratio": args.auto_val_ratio,
                "use_weighted_sampler": bool(args.use_weighted_sampler),
            },
        }

    torch.save(_build_ckpt(model.state_dict(), "last"), out)
    if best_state is not None:
        torch.save(_build_ckpt(best_state, "best"), best_out)

    out.with_suffix(".metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8")
    out.with_suffix(".config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Saved last checkpoint to {out}")
    if best_state is not None:
        print(f"Saved best checkpoint to {best_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())