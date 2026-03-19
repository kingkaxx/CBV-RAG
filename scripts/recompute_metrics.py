import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def extract_answer(pred: str) -> str:
    if not pred:
        return ""
    return pred.split("\n")[0].strip()[:150]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def compute_em(pred: str, golds: list) -> float:
    pred_n = normalize_answer(pred)
    for g in golds:
        if not g:
            continue
        g_n = normalize_answer(str(g))
        if g_n in pred_n or pred_n in g_n:
            return 1.0
    return 0.0


def compute_f1(pred: str, golds: list) -> float:
    return float(max(
        (token_f1(pred, str(g)) for g in golds if g),
        default=0.0
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.records) if l.strip()]

    em_list, f1_list = [], []
    tok_list, step_list, attr_list, ret_list, phr_list = [], [], [], [], []

    for rec in records:
        raw = (rec.get("prediction") or rec.get("pred") or "").strip()
        pred = extract_answer(raw)
        golds = rec.get("gold_answers") or rec.get("golds") or [""]
        if isinstance(golds, str):
            golds = [golds]

        em_list.append(compute_em(pred, golds))
        f1_list.append(compute_f1(pred, golds))
        tok_list.append(float(rec.get("total_tokens", rec.get("tokens", 0))))
        step_list.append(float(rec.get("steps", 0)))
        attr_list.append(float(rec.get("attr_score", 0)))
        ret_list.append(float(rec.get("retrieval_calls", 0)))
        phr_list.append(1.0 if rec.get("parametric_hallucination_risk") else 0.0)

    n = max(1, len(em_list))
    result = {
        "num_examples": n,
        "mean_em": round(sum(em_list) / n, 4),
        "mean_f1": round(sum(f1_list) / n, 4),
        "mean_tokens": round(sum(tok_list) / n, 2),
        "mean_steps": round(sum(step_list) / n, 3),
        "mean_attr_score": round(sum(attr_list) / n, 4),
        "mean_retrieval_calls": round(sum(ret_list) / n, 3),
        "pct_parametric_hallucination_risk": round(100 * sum(phr_list) / n, 1),
    }

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# Import CF-RAG's smart evaluate function
import sys
sys.path.insert(0, '.')
try:
    from evaluation import evaluate as cfrag_evaluate
    HAS_CFRAG_EVAL = True
except ImportError:
    HAS_CFRAG_EVAL = False


# Import CF-RAG's smart evaluate function
import sys
sys.path.insert(0, '.')
try:
    from evaluation import evaluate as cfrag_evaluate
    HAS_CFRAG_EVAL = True
except ImportError:
    HAS_CFRAG_EVAL = False
