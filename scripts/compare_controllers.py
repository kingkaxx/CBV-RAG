from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: str):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows):
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "em": sum(1 for r in rows if r.get("correct")) / n,
        "f1": sum(float(r.get("f1", 0.0)) for r in rows) / n,
        "avg_total_tokens": sum(float(r.get("total_tokens", 0)) for r in rows) / n,
        "avg_retrieval_calls": sum(float(r.get("retrieval_calls", 0)) for r in rows) / n,
        "avg_steps": sum(float(r.get("steps", 0)) for r in rows) / n,
        "avg_branches": sum(float(r.get("branches", 1)) for r in rows) / n,
        "success_rate": sum(1 for r in rows if r.get("success", r.get("correct", False))) / n,
        "early_stop_rate": sum(1 for r in rows if r.get("early_exit", False)) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="controller_name=records.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    results = {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        results[name] = summarize(load_records(path))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
