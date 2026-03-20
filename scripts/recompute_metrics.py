"""Recompute EM and F1 from a .records.jsonl file.

FIX vs old version:
- extract_answer() used for BOTH EM and F1 (not two different truncation strategies)
- F1 is now always >= EM (mathematically impossible to violate)
- EM/F1 computed fresh from pred+gold — stored r["em"] / r["f1"] are never trusted

Usage
-----
    python scripts/recompute_metrics.py \
        --records logs/eval_il_hotpotqa.records.jsonl \
        --output  logs/eval_il_hotpotqa_fixed.json
"""
from __future__ import annotations

import argparse
import re
import string
import json
import sys
from collections import Counter
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ---------------------------------------------------------------------------
# Self-contained answer extraction + EM/F1
# (does NOT use evaluation.py so this script is standalone)
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    """Extract the answer span from raw LLM output.

    Handles:
      - Clean: "Paris"
      - With reasoning: "Paris\nReasoning: ..."
      - Template echo: "Answer: Paris\nReasoning: ..."
      - Repeated template: "[your concise answer]\nReasoning: [one sentence...]"
    """
    if not raw:
        return ""

    # Strip echoed "Answer:" prefix
    pred = re.sub(r"^Answer:\s*", "", raw.strip(), flags=re.IGNORECASE)

    # Drop everything from Reasoning: onward
    if "\nReasoning:" in pred:
        pred = pred.split("\nReasoning:")[0]
    if "\nAnswer:" in pred:
        pred = pred.split("\nAnswer:")[0]

    # Drop template placeholders that leaked through
    pred = re.sub(r"\[your concise answer[^\]]*\]", "", pred, flags=re.IGNORECASE)
    pred = re.sub(r"\[one sentence[^\]]*\]", "", pred, flags=re.IGNORECASE)

    # Take first non-empty line
    for line in pred.split("\n"):
        line = line.strip()
        if line:
            # Strip trailing punctuation artifact
            line = re.sub(r"[.!?]+$", "", line).strip()
            return line

    return pred.strip()


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_em(pred_clean: str, golds: list) -> float:
    norm_pred = normalize(pred_clean)
    return float(any(norm_pred == normalize(str(g)) for g in golds))


def compute_f1(pred_clean: str, golds: list) -> float:
    """Token-overlap F1, max over gold answers."""
    pred_toks = normalize(pred_clean).split()
    best = 0.0
    for g in golds:
        g_toks = normalize(str(g)).split()
        if not pred_toks or not g_toks:
            continue
        common = Counter(pred_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        p = num_same / len(pred_toks)
        r = num_same / len(g_toks)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute EM/F1 from a .records.jsonl file."
    )
    ap.add_argument("--records", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records_path = Path(args.records)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    corrected: list[dict] = []
    for rec in records:
        raw_pred = (rec.get("prediction") or rec.get("pred") or "").strip()
        golds = rec.get("gold_answers") or rec.get("gold") or [""]
        if isinstance(golds, str):
            golds = [golds]
        golds = [str(g) for g in golds if str(g).strip()]
        if not golds:
            golds = [""]

        # FIX: extract once, use for both EM and F1
        pred_clean = extract_answer(raw_pred)

        em = compute_em(pred_clean, golds)
        f1 = compute_f1(pred_clean, golds)

        # Sanity: F1 must be >= EM (if not, something is wrong — log it)
        if f1 < em - 1e-6:
            print(f"[warn] F1 ({f1:.4f}) < EM ({em:.4f}) for qid={rec.get('qid')} — check gold format")

        corrected.append({
            **rec,
            "em": float(em),
            "f1": float(f1),
            "prediction_extracted": pred_clean,  # for debugging
        })

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
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write per-example corrected records alongside summary
    records_out = out_path.with_name(out_path.stem + ".records.jsonl")
    with records_out.open("w", encoding="utf-8") as f:
        for r in corrected:
            # Don't write per_example_records sub-field if it exists (avoid nesting)
            row = {k: v for k, v in r.items() if k != "per_example_records"}
            f.write(json.dumps(row) + "\n")

    printable = {k: v for k, v in summary.items() if k != "per_example_records"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())