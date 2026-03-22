from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from cbvrag.controller_trace_mixture import TraceMixtureController
from cbvrag.evidence_clusters import cluster_evidence_items, summarize_cluster_stats
from cbvrag.evidence_specificity import score_evidence_specificity
from cbvrag.runner import run_episode
from rl.train_offline import shape_reward_with_attr
from tools.llm import LLMEngine
from tools.rerank import CrossEncoderReranker
from tools.retrieve import RetrieverTool

import model_loader


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class GlobalIndexAdapter:
    def __init__(self, index_dir: str, kb_jsonl: str, embedding_model: Any | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index_dir = Path(index_dir)
        self.embedding_model = SentenceTransformer("/scratch/yl258/kp759/hf/model_cache/bge-large-en-v1.5-local")
        self.index = faiss.read_index(str(self.index_dir / "global.index"))
        self.rows = load_jsonl(kb_jsonl)

    def _embed(self, query: str) -> np.ndarray:
        emb = self.embedding_model.encode([query], convert_to_numpy=True)
        return np.asarray(emb, dtype="float32")

    def build_temp_index_from_docs(self, context_docs) -> None:
        """Build temporary oracle index from gold context docs."""
        import faiss, numpy as np
        texts = []
        if isinstance(context_docs, list):
            for item in context_docs:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # HotpotQA format: [title, [sent1, sent2, ...]]
                    sents = item[1] if isinstance(item[1], list) else [item[1]]
                    texts.append(" ".join(sents))
                elif isinstance(item, dict):
                    texts.append(item.get("text", str(item)))
        if not texts:
            return
        embs = self.embedding_model.encode(texts, convert_to_numpy=True).astype("float32")
        dim = embs.shape[1]
        self._temp_index = faiss.IndexFlatL2(dim)
        self._temp_index.add(embs)
        self._temp_docs = texts

    def clear_temp_index(self) -> None:
        self._temp_index = None
        self._temp_docs = []

    def search(self, query: str, top_k: int = 10, dataset_filter: str | None = None) -> List[Dict[str, Any]]:
        q = self._embed(query)
        if getattr(self, "_temp_index", None) is not None:
            scores, ids = self._temp_index.search(q, min(top_k, self._temp_index.ntotal))
            out = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self._temp_docs):
                    continue
                out.append({"doc_id": str(idx), "chunk_id": str(idx),
                            "text": self._temp_docs[int(idx)], "score": float(score),
                            "retriever_score": float(score), "title": "", "dataset": "", "meta": {}})
                if len(out) >= top_k:
                    break
            return out
        scores, ids = self.index.search(q, top_k * 5 if dataset_filter else top_k)

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.rows[int(idx)]
            if dataset_filter and row.get("dataset") != dataset_filter:
                continue
            out.append(
                {
                    "doc_id": str(row.get("doc_id", idx)),
                    "chunk_id": str(row.get("chunk_id", idx)),
                    "text": row.get("text", ""),
                    "score": float(score),
                    "retriever_score": float(score),
                    "title": str(row.get("title", "")),
                    "dataset": str(row.get("dataset", "")),
                    "meta": row.get("metadata", {}),
                }
            )
            if len(out) >= top_k:
                break
        return out


def _trajectory_score(log: dict, correct: bool) -> float:
    state = (log.get("state") or {})
    metrics = state.get("metrics") or {}
    steps = log.get("steps") or []
    retrieval_calls = int(metrics.get("retrieval_calls", 0))
    llm_calls = int(metrics.get("llm_calls", 0))
    no_progress = int(metrics.get("no_progress_streak", 0))
    explicit_stop = int(metrics.get("explicit_stop_used", 0))
    fallback_stop = int(metrics.get("fallback_stop_was_used", 0))
    forced_stop = int(metrics.get("forced_stop_used", 0))
    forced_actions = int(metrics.get("forced_action_count", 0))
    illegal_requested = int(metrics.get("illegal_action_requested", 0))

    score = 1.0 if correct else 0.0
    score += 0.15 * explicit_stop
    score -= 0.05 * retrieval_calls
    score -= 0.03 * llm_calls
    score -= 0.05 * no_progress
    score -= 0.10 * fallback_stop
    score -= 0.08 * forced_stop
    score -= 0.04 * forced_actions
    score -= 0.04 * illegal_requested
    score -= 0.02 * max(0, len(steps) - 4)
    return float(score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect traces using global index + QA manifest.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qa_jsonl", default="data/multidataset_qa.jsonl")
    ap.add_argument("--kb_jsonl", default="data/global_kb_chunks.jsonl")
    ap.add_argument("--index_dir", default="data/global_index")
    ap.add_argument("--output", default=None)
    ap.add_argument("--llm_device", default="cuda:0")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--use_attr_reward", action="store_true")
    ap.add_argument("--attr_lambda_token", type=float, default=0.1)
    ap.add_argument("--attr_lambda_step", type=float, default=0.05)
    ap.add_argument("--attr_bonus", type=float, default=0.2)
    ap.add_argument("--token_budget", type=int, default=4096)
    ap.add_argument("--use_oracle_context", action="store_true")
    args = ap.parse_args()

    output = Path(args.output or f"data/traces/{args.dataset}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(args.qa_jsonl)
    data = [r for r in all_rows if r.get("dataset") == args.dataset]
    if args.num_samples is not None:
        data = data[: args.num_samples]

    models = model_loader.load_all_models()
    retriever = GlobalIndexAdapter(
        index_dir=args.index_dir,
        kb_jsonl=args.kb_jsonl,
        embedding_model=models["embedding_model"],
    )

    tools = {
        "llm": LLMEngine(
            model_name_or_path=getattr(
                models["llm_model"],
                "name_or_path",
                models["llm_model"].config._name_or_path,
            ),
            device=args.llm_device,
            model=models["llm_model"],
            tokenizer=models["llm_tokenizer"],
        ),
        "retrieve": RetrieverTool(retriever, dataset_filter=args.dataset),
        "rerank": CrossEncoderReranker(),
    }

    summary_em: list = []
    summary_tokens: list = []
    summary_attr: list = []

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            qid = str(ex.get("qid", i))
            controller = TraceMixtureController(seed=1000 + i)
            if args.use_oracle_context:
                context_docs = ex.get("context")
                if context_docs:
                    retriever.build_temp_index_from_docs(context_docs)
            pred, log = run_episode(ex["question"], controller, tools, qid=qid)
            if args.use_oracle_context:
                retriever.clear_temp_index()

            golds = ex.get("answers") or [""]
            from cbvrag.env import _extract_answer, _any_em_correct
            correct = _any_em_correct(pred, golds)
            traj_score = _trajectory_score(log, correct)

            null_branch_record = log.get("null_branch") or {}
            attr_result = null_branch_record.get("attr_score") or {}
            episode_attr_score = float(attr_result.get("attr", 0.0))

            state_dict = (log.get("state") or {})
            evidence_pool_dict = state_dict.get("evidence_pool") or {}
            selected_ids = state_dict.get("selected_evidence_ids") or []
            step_logs = log.get("steps", [])
            episode_tokens = sum(
                int((s.get("costs") or {}).get("tokens_used_this_step", 0))
                for s in step_logs
            )

            class SimpleEvidence:
                def __init__(self, evidence_id: str, payload: dict):
                    self.evidence_id = evidence_id
                    self.doc_id = payload.get("doc_id", "")
                    self.title = payload.get("title", "")
                    self.rerank_score = float(payload.get("rerank_score", 0.0))
                    self.retriever_score = float(payload.get("retriever_score", 0.0))

            evs = [SimpleEvidence(eid, payload) for eid, payload in evidence_pool_dict.items()]
            cluster_stats = summarize_cluster_stats(cluster_evidence_items(evs, selected_ids=selected_ids))
            specificity_summary = score_evidence_specificity(ex["question"], evs, selected_ids=selected_ids)["summary"]

            for t, tr in enumerate(controller.trace):
                step_info = step_logs[t] if t < len(step_logs) else {}

                if args.use_attr_reward:
                    synthetic_row = {
                        "terminal_correct": bool(correct),
                        "step_info": step_info,
                        "attr_score": episode_attr_score,
                        "t": t,
                    }
                    row_reward = shape_reward_with_attr(
                        synthetic_row,
                        lambda_token=args.attr_lambda_token,
                        lambda_step=args.attr_lambda_step,
                        attr_bonus=args.attr_bonus,
                        token_budget=args.token_budget,
                    )
                else:
                    row_reward = float(step_info.get("costs", {}).get("new_branch_created", 0) * 0.01)

                row = {
                    "qid": qid,
                    "t": t,
                    "obs": tr["obs"],
                    "action": tr["action"],
                    "reward": row_reward,
                    "done": t == len(controller.trace) - 1,
                    "terminal_correct": bool(correct),
                    "trajectory_score": traj_score,
                    "attr_score": episode_attr_score,
                    "question": ex["question"],
                    "pred": pred,
                    "gold_answers": golds,
                    "step_info": step_info,
                    "trace_info": tr.get("info", {}),
                    "cluster_stats": cluster_stats,
                    "specificity_summary": specificity_summary,
                }
                f.write(json.dumps(row) + "\n")

            summary_em.append(1.0 if correct else 0.0)
            summary_tokens.append(episode_tokens)
            summary_attr.append(episode_attr_score)

    n = max(1, len(summary_em))
    summary = {
        "dataset": args.dataset,
        "num_episodes": n,
        "use_attr_reward": args.use_attr_reward,
        "mean_em": float(sum(summary_em) / n),
        "mean_token_use": float(sum(summary_tokens) / n),
        "mean_attr_score": float(sum(summary_attr) / n),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())