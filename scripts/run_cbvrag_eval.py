from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

from cbvrag.runner import run_episode
from data_loader import load_and_process_data


# ---------------------------------------------------------------------------
# Answer extraction — applied BEFORE EM/F1 scoring
# ---------------------------------------------------------------------------

def extract_answer(pred: str) -> str:
    """Extract the answer span from a raw LLM output.

    Handles three output formats produced by answer_prompt():
      1. Clean: "Paris"
      2. With reasoning suffix: "Paris\nReasoning: ..."
      3. With Answer: prefix repeated: "Answer: Paris\nReasoning: ..."

    Returns the shortest clean answer string for EM comparison.
    """
    if not pred:
        return ""

    # Strip "Answer:" prefix if the model echoed it
    pred = re.sub(r"^Answer:\s*", "", pred.strip(), flags=re.IGNORECASE)

    # Take everything before the first Reasoning: block
    if "\nReasoning:" in pred:
        pred = pred.split("\nReasoning:")[0]

    # Take the first non-empty line
    for line in pred.split("\n"):
        line = line.strip()
        if line:
            # Strip any trailing punctuation artifact
            line = re.sub(r"[.!?]+$", "", line).strip()
            return line

    return pred.strip()


# ---------------------------------------------------------------------------
# Standard HotpotQA / SQuAD EM + F1
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
    """Return (EM, F1) using extracted answer.

    FIX vs old version:
    - extract_answer() is applied BEFORE normalization (not 150-char truncation)
    - same extracted string used for both EM and F1
    - F1 is now always >= EM (mathematically correct)
    """
    pred_clean = extract_answer(pred)
    em = max(
        float(normalize_answer(pred_clean) == normalize_answer(g))
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
    ap.add_argument("--controller_type", choices=["heuristic", "il", "offline"],
                    default="heuristic")
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--policy_mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--llm_device", default="cuda:0")
    ap.add_argument("--num_samples", type=int, default=None)
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

    per_example_records = []

    for i, ex in enumerate(data):
        if args.use_oracle_context:
            context_docs = ex.get("context")
            if context_docs:
                retriever.build_temp_index_from_docs(context_docs)

        controller = _build_controller(args)
        final_answer, log = run_episode(ex["question"], controller, tools, qid=str(i))
        pred = (final_answer or "").strip()

        golds = ex.get("answer") or [""]

        # FIX: extract_answer before scoring — same string for EM and F1
        em, f1 = compute_metrics(pred, golds)

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
            "prediction_extracted": extract_answer(pred),  # stored for debugging
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
    output_path.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())