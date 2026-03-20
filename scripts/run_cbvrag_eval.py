from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

from cbvrag.runner import run_episode
from data_loader import load_and_process_data
import evaluation


# ---------------------------------------------------------------------------
# Answer extraction — applied BEFORE EM/F1 scoring
# ---------------------------------------------------------------------------
def extract_answer(pred: str) -> str:
    """Extract a concise answer span from a raw LLM output.

    Handles common answer_prompt() outputs such as:
      1. "Paris"
      2. "Paris\nReasoning: ..."
      3. "Answer: Paris\nReasoning: ..."
      4. "Kevin Spacey. or None."
      5. "Ride a Wild Pony. (Note that there are two...)"

    Returns the shortest clean answer string for EM/F1 comparison.
    """
    if not pred:
        return ""

    # Remove common wrappers / prefixes
    pred = pred.strip()
    pred = re.sub(r"^answer\s*:\s*", "", pred, flags=re.IGNORECASE)
    pred = re.sub(r"^final answer\s*:\s*", "", pred, flags=re.IGNORECASE)
    pred = re.split(r"\n\s*(reasoning|explanation)\s*:", pred, maxsplit=1, flags=re.IGNORECASE)[0]
    pred = re.split(r"\n\s*answer\s*:", pred, maxsplit=1, flags=re.IGNORECASE)[0]

    # Take first non-empty line
    first = ""
    for line in pred.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^[\"'`\(\[]+", "", line)
            line = re.sub(r"[\"'`\)\]]+$", "", line)
            line = re.sub(r"[.!?]+$", "", line).strip()
            return line

    # Remove template placeholders / junk
    pred = re.sub(r"\[your concise answer[^\]]*\]", "", pred, flags=re.IGNORECASE)
    pred = re.sub(r"\[one sentence[^\]]*\]", "", pred, flags=re.IGNORECASE)

    # Trim parenthetical commentary
    pred = re.split(r"\s*\(", pred, maxsplit=1)[0]

    # Trim sentence continuations like:
    # "Kevin Spacey. or None." -> "Kevin Spacey"
    pred = re.split(
        r"\.\s+(?:or|and|but|however|note|this|that|it|in|the)\b",
        pred,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Trim non-period continuation tails
    pred = re.split(
        r"\s+\b(?:however|but|because|although|whereas|note that)\b",
        pred,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Clean trailing punctuation/quotes
    pred = pred.strip().strip("\"'`")
    pred = re.sub(r"[.!?]+$", "", pred).strip()

    return pred


# ---------------------------------------------------------------------------
# Standard HotpotQA / SQuAD normalization helpers
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
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(pred: str, golds: list[str], question: str) -> tuple[float, float]:
    """Return (EM, F1) using extracted answer.

    Key fixes:
    - extract_answer() is applied BEFORE scoring
    - smart_exact_match_score() is used for EM
    - the real question is passed into smart matching
    - the same extracted string is used for both EM and F1
    """
    from evaluation import smart_exact_match_score

    pred_clean = extract_answer(pred)
    em = max(
        float(evaluation.smart_exact_match_score(pred_clean, g, question))
        for g in golds
    ) if golds else 0.0
    f1 = max(token_f1(pred_clean, g) for g in golds) if golds else 0.0
    return em, f1


def _build_tools(models, llm_device: str) -> dict:
    from tools.llm import LLMEngine
    from tools.rerank import CrossEncoderReranker
    from tools.retrieve import RetrieverTool
    from retriever import KnowledgeBaseRetriever

    kb = KnowledgeBaseRetriever(models["embedding_model"])
    return {
        "llm": LLMEngine(
            getattr(
                models["llm_model"],
                "name_or_path",
                models["llm_model"].config._name_or_path,
            ),
            device=llm_device,
        ),
        "retrieve": RetrieverTool(kb),
        "rerank": CrossEncoderReranker(),
    }


def _build_controller(args):
    ct = args.controller_type
    if ct == "heuristic":
        from cbvrag.controller_trace_mixture import TraceMixtureController

        return TraceMixtureController()

    if not args.policy_ckpt:
        raise ValueError("--policy_ckpt is required for controller_type il/offline")

    from cbvrag.controller_learned import LearnedController

    return LearnedController(args.policy_ckpt, mode=args.policy_mode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate CBV-RAG with heuristic or learned controller."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--output", default="logs/cbvrag_eval.json")
    ap.add_argument("--baseline_jsonl", default=None)
    ap.add_argument(
        "--controller_type",
        choices=["heuristic", "il", "offline"],
        default="heuristic",
    )
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--policy_mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--llm_device", default="cuda:0")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument(
        "--compare_oracle_context",
        action="store_true",
        help="Run eval twice and print side-by-side summary: normal retrieval vs oracle context.",
    )
    ap.add_argument(
        "--use_oracle_context", action="store_true",
        help=(
            "Before each episode, build a temporary retrieval index from the "
            "example's gold 'context' field. Matches CF-RAG's eval setting."
        ),
    )
    args = ap.parse_args()

    import model_loader

    models = model_loader.load_all_models()
    tools = _build_tools(models, args.llm_device)
    retriever = tools["retrieve"].retriever
    data = load_and_process_data(args.dataset, args.cache_dir, args.num_samples)

    def _run_eval_once(use_oracle_context: bool):
        per_example_records = []
        for i, ex in enumerate(data):
            retriever.clear_temp_index()
            if use_oracle_context:
                context_docs = ex.get("context")
                if context_docs:
                    retriever.build_temp_index_from_docs(context_docs)

            controller = _build_controller(args)
            final_answer, log = run_episode(ex["question"], controller, tools, qid=str(i))
            pred = (final_answer or "").strip()

            golds = ex.get("answer") or [""]

            em, f1 = compute_metrics(pred, golds, ex["question"])

            state = log.get("state") or {}
            metrics = state.get("metrics") or {}
            steps_log = log.get("steps") or []
            null_branch = log.get("null_branch") or {}

            total_tokens = sum(
                int((s.get("costs") or {}).get("tokens_used_this_step", 0))
                for s in steps_log
            )
            attr_result = null_branch.get("attr_score") or {}
            attr_score = float(attr_result.get("attr", 0.0))
            parametric_risk = bool(null_branch.get("parametric_hallucination_risk", False))

            per_example_records.append({
                "qid": str(i),
                "question": ex["question"],
                "prediction": pred,
                "prediction_extracted": extract_answer(pred),
                "gold_answers": golds,
                "em": float(em),
                "f1": float(f1),
                "total_tokens": total_tokens,
                "steps": len(steps_log),
                "retrieval_calls": int(metrics.get("retrieval_calls", 0)),
                "attr_score": attr_score,
                "parametric_hallucination_risk": parametric_risk,
            })
        n = max(1, len(per_example_records))
        summary = {
            "dataset": args.dataset,
            "controller_type": args.controller_type,
            "num_samples": args.num_samples,
            "use_oracle_context": bool(use_oracle_context),
            "mean_em": sum(r["em"] for r in per_example_records) / n,
            "mean_f1": sum(r["f1"] for r in per_example_records) / n,
            "mean_tokens": sum(r["total_tokens"] for r in per_example_records) / n,
            "mean_steps": sum(r["steps"] for r in per_example_records) / n,
            "mean_attr_score": sum(r["attr_score"] for r in per_example_records) / n,
            "mean_retrieval_calls": sum(r["retrieval_calls"] for r in per_example_records) / n,
            "pct_parametric_hallucination_risk": (
                100.0 * sum(1 for r in per_example_records if r["parametric_hallucination_risk"]) / n
            ),
        }
        return summary, per_example_records

    summary, per_example_records = _run_eval_once(args.use_oracle_context)

    if args.baseline_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.baseline_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        nb = max(1, len(rows))
        summary["cfrag_baseline"] = {
            "accuracy": sum(1 for r in rows if r["correct"]) / nb,
            "avg_total_tokens": sum(r["total_tokens"] for r in rows) / nb,
            "avg_retrieval_calls": sum(r["retrieval_calls"] for r in rows) / nb,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_path = output_path.with_name(output_path.stem + ".records.jsonl")
    with records_path.open("w", encoding="utf-8") as f:
        for r in per_example_records:
            f.write(json.dumps(r) + "\n")

    full_out = {**summary, "per_example_records": per_example_records}
    if args.compare_oracle_context:
        normal_summary, _ = _run_eval_once(False)
        oracle_summary, _ = _run_eval_once(True)
        full_out["hotpotqa_retrieval_vs_oracle"] = {
            "normal_retrieval": normal_summary,
            "oracle_context": oracle_summary,
        }
        print(json.dumps({"normal_retrieval": normal_summary, "oracle_context": oracle_summary}, indent=2))
    output_path.write_text(json.dumps(full_out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
