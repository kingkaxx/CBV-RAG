"""Recompute EM and F1 from a .records.jsonl file using evaluation.evaluate().

Usage
-----
    python scripts/recompute_metrics.py \
        --records logs/eval_il_hotpotqa.records.jsonl \
        --output  logs/eval_il_hotpotqa_fixed.json
"""
from __future__ import annotations

import argparse
import re
import json
import sys
from pathlib import Path

# evaluation.py lives in the repo root; add it to sys.path if needed.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evaluation import evaluate  # noqa: E402 (import after sys.path tweak)


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
        # Use the full prediction string — evaluate() / smart_match() handles
        # long predictions correctly; no 150-char truncation here.
        raw_pred = (rec.get("prediction") or rec.get("pred") or "").strip()
        # Extract answer after 'Answer:' label if present (structured prompt format)
        answer_match = re.search(r'Answer:\s*(.+?)(?:
Reasoning:|$)', raw_pred, re.IGNORECASE | re.DOTALL)
        if answer_match:
            pred = answer_match.group(1).strip().split('
')[0].strip()[:150]
        else:
            pred = raw_pred.split('\n')[0].strip()[:150]
        golds = rec.get("gold_answers") or rec.get("gold") or [""]
        if isinstance(golds, str):
            golds = [golds]
        question = rec.get("question", "")

        em, _ = evaluate(pred, golds, question)
        _, f1 = evaluate(pred.split("\n")[0].strip()[:150], golds, question)
        corrected.append({**rec, "em": float(em), "f1": float(f1)})

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
