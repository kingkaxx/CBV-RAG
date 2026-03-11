from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON from run_cbvrag_eval.py")
    ap.add_argument("--out", default="logs/frontier.png")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    labels, xs_tokens, xs_retrieval, ys = [], [], [], []
    for name, row in data.items():
        labels.append(name)
        ys.append(row["accuracy"])
        xs_tokens.append(row["avg_total_tokens"])
        xs_retrieval.append(row["avg_retrieval_calls"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(xs_tokens, ys)
    for i, lbl in enumerate(labels):
        ax[0].annotate(lbl, (xs_tokens[i], ys[i]))
    ax[0].set_xlabel("Avg Tokens")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy vs Tokens")

    ax[1].scatter(xs_retrieval, ys)
    for i, lbl in enumerate(labels):
        ax[1].annotate(lbl, (xs_retrieval[i], ys[i]))
    ax[1].set_xlabel("Avg Retrieval Calls")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy vs Retrieval")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
