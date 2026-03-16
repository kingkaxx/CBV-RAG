from __future__ import annotations

import argparse
import json
from pathlib import Path

from cbvrag.controller_heuristic import HeuristicController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from evaluation import smart_exact_match_score


def evaluate_records(records):
    n = max(1, len(records))
    return {
        "accuracy": sum(r["correct"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "avg_steps": sum(r["steps"] for r in records) / n,
        "avg_branches": sum(r["num_branches"] for r in records) / n,
        "pct_early": sum(1 for r in records if r["early_exit"]) / n,
        "success_rate": sum(1 for r in records if r["correct"]) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--baseline_jsonl", default=None)
    ap.add_argument("--output", default="logs/cbvrag_eval.json")
    args = ap.parse_args()

    data = load_and_process_data(args.dataset, args.cache_dir)

    from tools.llm import LLMEngine
    from tools.retrieve import RetrieverTool
    from tools.rerank import CrossEncoderReranker
    import model_loader
    from retriever import KnowledgeBaseRetriever

    models = model_loader.load_all_models()
    kb = KnowledgeBaseRetriever(models["embedding_model"])
    tools = {
        "llm": LLMEngine(
            getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path),
            device="cpu",
        ),
        "retrieve": RetrieverTool(kb),
        "rerank": CrossEncoderReranker(),
    }

    cbv_records = []
    prev_total_tokens = 0

    for i, ex in enumerate(data):
        controller = HeuristicController()
        pred, log = run_episode(ex["question"], controller, tools, qid=str(i))

        usage = tools["llm"].usage_tracker.summary()
        running_total = int(usage.get("total_tokens", 0))
        example_total_tokens = max(0, running_total - prev_total_tokens)
        prev_total_tokens = running_total

        retrieval_calls = int(log["state"]["metrics"]["retrieval_calls"])
        steps = len(log.get("steps", []))
        num_branches = len((log.get("state") or {}).get("branches", {}))
        correct = any(smart_exact_match_score(pred, g, ex["question"]) for g in ex["answer"])

        cbv_records.append(
            {
                "qid": str(i),
                "correct": bool(correct),
                "total_tokens": example_total_tokens,
                "retrieval_calls": retrieval_calls,
                "steps": steps,
                "num_branches": num_branches,
                "early_exit": retrieval_calls <= 1,
            }
        )

    out = {"cbvrag_heuristic": evaluate_records(cbv_records)}

        record = {
            "qid": str(i),
            "controller_type": args.controller_type,
            "policy_ckpt": args.policy_ckpt,
            "policy_mode": args.policy_mode,
            "correct": bool(correct),
            "f1": float(best_f1),
            "success": success,
            "total_tokens": total_tokens,
            "retrieval_calls": retrieval_calls,
            "steps": steps,
            "branches": branches,
            "early_exit": bool(log.get("explicit_terminal_action", -1) in {0, 10}),
            "support_hit": support_hit,
            "question": ex.get("question", ""),
            "prediction": pred,
            "final_answer": pred,
            "gold_answers": gold_answers,
            "support_titles": list(ex.get("support_titles") or []),
            "retrieved_titles": sorted(t for t in retrieved_titles if t),
            "selected_evidence_titles": selected_evidence_titles,
            "actions_taken": [s.get("action") for s in log.get("steps", []) if isinstance(s, dict)],
            "fallback_stop_was_used": bool(log.get("fallback_stop_was_used", False)),
            "explicit_stop_used": bool(log.get("explicit_stop_used", False)),
            "explicit_terminal_action": int(log.get("explicit_terminal_action", -1)),
            "forced_stop_used": bool(log.get("forced_stop_used", False)),
            "verification_status": log["state"].get("verification_status", "unknown"),
            "final_branch_count": len(log["state"].get("branches", {})),
            "selected_evidence_count": len(selected_evidence_ids),
            "unique_selected_title_count": len(set(selected_evidence_titles)),
            "top_rerank_scores": rerank_scores[:5],
        }
        cbv_records.append(record)

        if _debug_small_run(args):
            trace = getattr(controller, "trace", [])
            print(f"[pilot] qid={record['qid']} correct={record['correct']} f1={record['f1']:.3f}", flush=True)
            for t_idx, step_rec in enumerate(trace):
                info = step_rec.get("info", {}) if isinstance(step_rec, dict) else {}
                print(
                    "[pilot] "
                    f"qid={record['qid']} step={info.get('step', t_idx)} action={step_rec.get('action')} "
                    f"selected={info.get('selected_evidence_count', 'na')} unique_titles={info.get('unique_title_count', 'na')} "
                    f"retrieval_calls={info.get('retrieval_calls', retrieval_calls)} "
                    f"verification={info.get('verification_status', record['verification_status'])} "
                    f"rerank_gap={info.get('rerank_gap', 'na')}",
                    flush=True,
                )

    key = f"cbvrag_{args.controller_type}"
    out = {
        key: {
            **evaluate_records(cbv_records),
            "controller_type": args.controller_type,
            "policy_ckpt": args.policy_ckpt,
            "policy_mode": args.policy_mode,
            "llm_device": llm_device,
            "seed": args.seed,
        }
    }
    if args.baseline_jsonl:
        rows = [json.loads(l) for l in Path(args.baseline_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        out["cfrag_baseline"] = {
            "accuracy": sum(1 for r in rows if r["correct"]) / max(1, len(rows)),
            "avg_total_tokens": sum(r["total_tokens"] for r in rows) / max(1, len(rows)),
            "avg_retrieval_calls": sum(r["retrieval_calls"] for r in rows) / max(1, len(rows)),
            "pct_early": sum(1 for r in rows if r.get("retrieval_calls", 99) <= 1) / max(1, len(rows)),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
