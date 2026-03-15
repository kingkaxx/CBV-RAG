from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from statistics import mean

import torch
from sentence_transformers import SentenceTransformer

import config
import model_loader
from cbvrag.controller_heuristic import HeuristicController
from cbvrag.controller_learned import LearnedController
from cbvrag.runner import run_episode
from data_loader import load_and_process_data
from retrieval.global_index import GlobalChunkRetriever
from tools.llm import LLMEngine
from tools.rerank import CrossEncoderReranker
from tools.retrieve import RetrieverTool


def resolve_llm_device(requested: str | None) -> str:
    device = requested or getattr(config, "LLM_DEVICE", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested GPU device '{device}' but CUDA is unavailable.")
        if not model_loader.check_device_availability(device):
            raise RuntimeError(f"Requested GPU device '{device}' is unavailable on this host.")
    return device


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1

    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(pred: str, gold: str) -> bool:
    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)
    return bool(ngold) and ngold in npred


def evaluate_records(records):
    n = max(1, len(records))
    return {
        "em": sum(r["correct"] for r in records) / n,
        "f1": sum(r["f1"] for r in records) / n,
        "avg_total_tokens": sum(r["total_tokens"] for r in records) / n,
        "avg_retrieval_calls": sum(r["retrieval_calls"] for r in records) / n,
        "avg_steps": sum(r["steps"] for r in records) / n,
        "avg_branches": sum(r["branches"] for r in records) / n,
        "success_rate": sum(r["success"] for r in records) / n,
        "early_stop_rate": sum(r["early_exit"] for r in records) / n,
        "support_doc_recall": sum(r.get("support_hit", 0.0) for r in records) / n,
    }


def controller_from_args(args):
    if args.controller_type == "heuristic":
        return HeuristicController()
    if not args.policy_ckpt:
        raise ValueError("--policy_ckpt is required when controller_type is il/offline")
    return LearnedController(policy_ckpt=args.policy_ckpt, mode=args.policy_mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--baseline_jsonl", default=None)
    ap.add_argument("--output", default="logs/cbvrag_eval.json")
    ap.add_argument("--records_output", default=None)
    ap.add_argument("--controller_type", choices=["heuristic", "il", "offline"], default="heuristic")
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--policy_mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--llm_device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kb_jsonl", default="data/kb/hotpotqa_train_kb.jsonl")
    ap.add_argument("--index_dir", default="data/index/hotpotqa_train")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dataset_filter", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    llm_device = resolve_llm_device(args.llm_device)
    print(f"[run_cbvrag_eval] LLM device: {llm_device}", flush=True)

    data = load_and_process_data(args.dataset, args.cache_dir)

    models = model_loader.load_all_models()
    global_retriever_model = SentenceTransformer(args.embedding_model)
    kb = GlobalChunkRetriever(global_retriever_model)
    kb.load(args.index_dir)
    effective_filter = args.dataset_filter or args.dataset
    print(
        f"[run_cbvrag_eval] global_index_dir={args.index_dir} kb_rows={len(kb.rows)} dataset_filter={effective_filter}",
        flush=True,
    )
    tools = {
        "llm": LLMEngine(getattr(models["llm_model"], "name_or_path", models["llm_model"].config._name_or_path), device=llm_device),
        "retrieve": RetrieverTool(kb, dataset_filter=effective_filter),
        "rerank": CrossEncoderReranker(),
    }

    cbv_records = []
    prev_total_tokens = 0
    for i, ex in enumerate(data):
        controller = controller_from_args(args)
        pred, log = run_episode(ex["question"], controller, tools, qid=str(i))
        running = int(tools["llm"].usage_tracker.summary()["total_tokens"])
        total_tokens = max(0, running - prev_total_tokens)
        prev_total_tokens = running
        retrieval_calls = int(log["state"]["metrics"]["retrieval_calls"])
        steps = len(log.get("steps", []))
        branches = len(log["state"].get("branches", {}))
        correct = any(compute_em(pred, g) for g in ex["answer"])
        best_f1 = max([compute_f1(pred, g) for g in ex.get("answer", [""])], default=0.0)
        success = bool(correct)
        support_titles = set(ex.get("support_titles") or [])
        retrieved_titles = {
            (ev.get("title") or "").strip()
            for ev in log["state"].get("evidence_pool", {}).values()
            if isinstance(ev, dict)
        }
        support_hit = 0.0
        if support_titles:
            support_hit = 1.0 if (retrieved_titles & support_titles) else 0.0

        selected_evidence_ids = log["state"].get("selected_evidence_ids", [])
        evidence_pool = log["state"].get("evidence_pool", {})
        selected_evidence_titles = []
        for eid in selected_evidence_ids:
            ev = evidence_pool.get(eid, {}) if isinstance(evidence_pool, dict) else {}
            title = (ev.get("title") or "").strip() if isinstance(ev, dict) else ""
            if title:
                selected_evidence_titles.append(title)

        gold_answers = list(ex.get("answer", []))
        if correct and best_f1 == 0.0:
            print(
                f"[run_cbvrag_eval][warn] correct=True but f1=0.0 qid={i} pred={pred!r} gold={gold_answers!r}",
                flush=True,
            )

        cbv_records.append(
            {
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
                "early_exit": steps < log["state"]["budgets"].get("max_steps", steps),
                "support_hit": support_hit,
                "question": ex.get("question", ""),
                "prediction": pred,
                "gold_answers": gold_answers,
                "support_titles": list(ex.get("support_titles") or []),
                "retrieved_titles": sorted(t for t in retrieved_titles if t),
                "selected_evidence_titles": selected_evidence_titles,
                "actions_taken": [s.get("action") for s in log.get("steps", []) if isinstance(s, dict)],
                "verification_status": log["state"].get("verification_status", "unknown"),
                "final_branch_count": len(log["state"].get("branches", {})),
            }
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
            "em": sum(1 for r in rows if r.get("correct")) / max(1, len(rows)),
            "f1": mean([r.get("f1", 0.0) for r in rows]) if rows else 0.0,
            "avg_total_tokens": sum(r.get("total_tokens", 0) for r in rows) / max(1, len(rows)),
            "avg_retrieval_calls": sum(r.get("retrieval_calls", 0) for r in rows) / max(1, len(rows)),
            "avg_steps": mean([r.get("steps", 0) for r in rows]) if rows else 0.0,
            "avg_branches": mean([r.get("branches", 1) for r in rows]) if rows else 0.0,
            "success_rate": sum(1 for r in rows if r.get("success", r.get("correct", False))) / max(1, len(rows)),
            "early_stop_rate": sum(1 for r in rows if r.get("retrieval_calls", 99) <= 1) / max(1, len(rows)),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    records_output = Path(args.records_output) if args.records_output else output_path.with_suffix(".records.jsonl")
    records_output.write_text("\n".join(json.dumps(r) for r in cbv_records) + ("\n" if cbv_records else ""), encoding="utf-8")

    print(f"[run_cbvrag_eval] controller={args.controller_type} policy_ckpt={args.policy_ckpt}", flush=True)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
