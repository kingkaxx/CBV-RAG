"""Recompute EM and F1 from a .records.jsonl file using the corrected metric functions.

Usage
-----
    python scripts/recompute_metrics.py \
        --records logs/eval_il_hotpotqa.records.jsonl \
        --output  logs/eval_il_hotpotqa_fixed.json
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path


# ---------------------------------------------------------------------------
# Standard HotpotQA / SQuAD EM + F1
# (identical to the copies in run_cbvrag_eval.py — kept self-contained so
#  this script can be run independently without importing the eval script)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, remove articles, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(pred: str, golds: list[str]) -> tuple[float, float]:
    """Return (EM, F1) against a list of gold answers.

    Uses first 150 characters of *pred* to avoid penalising reasoning chains.
    Takes the max over all gold answers.
    """
    pred_short = pred[:150] if pred else ""
    em = max(
        float(normalize_answer(pred_short) == normalize_answer(g))
        for g in golds
    ) if golds else 0.0
    f1 = max(token_f1(pred_short, g) for g in golds) if golds else 0.0
    return em, f1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute EM/F1 from a .records.jsonl file."
    )
    ap.add_argument(
        "--records", required=True,
        help="Path to a .records.jsonl file written by run_cbvrag_eval.py.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path for the corrected summary JSON.",
    )
    args = ap.parse_args()

    records_path = Path(args.records)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    corrected: list[dict] = []
    for rec in records:
        pred = (rec.get("prediction") or rec.get("pred") or "").strip()
        golds = rec.get("gold_answers") or rec.get("gold") or [""]
        if isinstance(golds, str):
            golds = [golds]

        em, f1 = compute_metrics(pred, golds)
        corrected.append({**rec, "em": em, "f1": f1})

    n = max(1, len(corrected))
    summary = {
        "source_records": str(records_path),
        "num_examples": len(corrected),
        "mean_em": sum(r["em"] for r in corrected) / n,
        "mean_f1": sum(r["f1"] for r in corrected) / n,
        "mean_tokens": sum(int(r.get("total_tokens", 0)) for r in corrected) / n,
        "mean_steps": sum(int(r.get("steps", 0)) for r in corrected) / n,
        "mean_attr_score": sum(float(r.get("attr_score", 0.0)) for r in corrected) / n,
        "mean_retrieval_calls": sum(int(r.get("retrieval_calls", 0)) for r in corrected) / n,
        "pct_parametric_hallucination_risk": (
            100.0 * sum(1 for r in corrected if r.get("parametric_hallucination_risk")) / n
        ),
        "per_example_records": corrected,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    printable = {k: v for k, v in summary.items() if k != "per_example_records"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
