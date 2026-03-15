from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

from cbvrag.actions import Action
from cbvrag.features import build_features
from cbvrag.prompts import answer_prompt, counterfactual_prompt, verify_prompt
from cbvrag.state import Branch, EpisodeState, EvidenceItem
from tools.select import select_context


def default_budgets() -> Dict[str, int]:
    return {
        "max_steps": 8,
        "max_context_chunks": 8,
        "max_context_tokens": 1500,
        "max_branches": 3,
        "max_retrieval_calls": 5,
    }


def _make_state(question: str, qid: str, budgets: Dict) -> EpisodeState:
    root = Branch(branch_id="b0", parent_id=None, hypothesis="default")
    return EpisodeState(
        question=question,
        qid=qid,
        branches={"b0": root},
        active_branch_id="b0",
        budgets=budgets,
        metrics={"retrieval_calls": 0, "rerank_calls": 0, "verify_calls": 0, "llm_calls": 0},
    )


def execute_action(state: EpisodeState, action: Action, controller: Any, tools: Dict[str, Any]) -> Dict[str, Any]:
    step_costs = {"tokens_used_this_step": 0, "retrieval_calls_this_step": 0, "new_branch_created": 0}
    retriever = tools["retrieve"]
    reranker = tools["rerank"]
    llm = tools["llm"]

    if action in (Action.RETRIEVE_MORE_SMALL, Action.RETRIEVE_MORE_LARGE):
        pool_k = 10 if action == Action.RETRIEVE_MORE_SMALL else 40
        cands = retriever.retrieve(state.question, pool_k)
        state.metrics["retrieval_calls"] += 1
        step_costs["retrieval_calls_this_step"] = 1
        cands = reranker.rerank(state.question, cands)
        state.metrics["rerank_calls"] += 1
        for idx, c in enumerate(cands):
            eid = f"{c['doc_id']}::{c['chunk_id']}::{idx}"
            state.evidence_pool[eid] = EvidenceItem(
                evidence_id=eid,
                doc_id=str(c["doc_id"]),
                chunk_id=str(c["chunk_id"]),
                retriever_score=float(c.get("retriever_score", 0.0)),
                rerank_score=float(c.get("rerank_score", 0.0)),
                short_claim=c.get("text", "")[:180],
                branch_id=state.active_branch_id,
                title=str(c.get("title") or c.get("meta", {}).get("title", "")),
            )

    elif action == Action.SELECT_CONTEXT:
        pool = [
            {
                "doc_id": e.doc_id,
                "chunk_id": e.chunk_id,
                "text": e.short_claim,
                "retriever_score": e.retriever_score,
                "rerank_score": e.rerank_score,
                "evidence_id": e.evidence_id,
                "title": e.title,
            }
            for e in state.evidence_pool.values()
        ]
        selected = select_context(
            state.question,
            pool,
            tokenizer=llm.tokenizer,
            max_chunks=state.budgets["max_context_chunks"],
            max_tokens=state.budgets["max_context_tokens"],
        )
        state.selected_evidence_ids = [s["evidence_id"] for s in selected]

    elif action == Action.SPAWN_COUNTERFACTUAL and len(state.branches) < state.budgets["max_branches"]:
        prompt = counterfactual_prompt(state.question, "counter")
        hypo, usage = llm.generate(prompt, max_new_tokens=32, temperature=0.2, name="counterfactual")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        bid = f"b{len(state.branches)}"
        state.branches[bid] = Branch(branch_id=bid, parent_id=state.active_branch_id, hypothesis=hypo, step_created=state.step)
        state.add_edge(state.active_branch_id, bid, "derived_from")
        state.active_branch_id = bid
        step_costs["new_branch_created"] = 1

    elif action == Action.VERIFY_CHEAP:
        state.metrics["verify_calls"] += 1
        pool = sorted(state.evidence_pool.values(), key=lambda e: e.rerank_score, reverse=True)
        gap = (pool[0].rerank_score - pool[1].rerank_score) if len(pool) > 1 else 0.0
        state.verification_status = "supported" if gap > 0.15 else "unknown"

    elif action == Action.VERIFY_LLM:
        state.metrics["verify_calls"] += 1
        snippets = [state.evidence_pool[eid].short_claim for eid in state.selected_evidence_ids[:3] if eid in state.evidence_pool]
        claim = state.branches[state.active_branch_id].hypothesis or "candidate answer"
        prompt = verify_prompt(state.question, claim, snippets)
        verdict, usage = llm.generate(prompt, max_new_tokens=8, temperature=0.0, name="verify")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        verdict_lower = verdict.lower()
        if "supported" in verdict_lower:
            state.verification_status = "supported"
        elif "contradicted" in verdict_lower:
            state.verification_status = "contradicted"
        else:
            state.verification_status = "unknown"

    elif action == Action.SUMMARIZE_STATE:
        snippets = [state.evidence_pool[eid].short_claim for eid in state.selected_evidence_ids[:3] if eid in state.evidence_pool]
        prompt = "Summarize the current evidence in <=80 words:\n" + "\n".join(snippets)
        summary, usage = llm.generate(prompt, max_new_tokens=80, temperature=0.0, name="summarize")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        state.global_summary = summary[:1000]

    elif action in (Action.ANSWER_DIRECT, Action.STOP_AND_ANSWER):
        snippets = [state.evidence_pool[eid].short_claim for eid in state.selected_evidence_ids if eid in state.evidence_pool]
        prompt = answer_prompt(state.question, snippets, state.branches[state.active_branch_id].summary, state.global_summary)
        answer, usage = llm.generate(prompt, max_new_tokens=96, temperature=0.0, name="answer")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        state.final_answer = answer

    elif action == Action.PRUNE_BRANCH and len(state.branches) > 1:
        for bid in list(state.branches.keys()):
            if bid != state.active_branch_id:
                state.branches[bid].status = "pruned"
                break

    elif action == Action.MERGE_BRANCHES:
        state.global_summary = (state.global_summary + " merged").strip()

    state.step += 1
    return step_costs


def run_episode(question: str, controller: Any, tools: Dict[str, Any], budgets: Dict | None = None, qid: str = "") -> Tuple[str, Dict]:
    budgets = {**default_budgets(), **(budgets or {})}
    state = _make_state(question, qid=qid or "unknown", budgets=budgets)
    logs = []

    for _ in range(budgets["max_steps"]):
        obs = build_features(state)
        action_idx = controller.act(obs, state)
        action = Action(action_idx)

        # Early-stop safeguard: when controller tries to stop while still uncertain,
        # run one cheap verification first if there is remaining budget.
        if (
            action == Action.STOP_AND_ANSWER
            and state.verification_status == "unknown"
            and state.step < budgets["max_steps"] - 1
        ):
            action = Action.VERIFY_CHEAP

        costs = execute_action(state, action, controller, tools)
        logs.append({"step": state.step, "action": int(action), "costs": costs, "metrics": dict(state.metrics)})

        if action == Action.STOP_AND_ANSWER or state.metrics["retrieval_calls"] >= budgets["max_retrieval_calls"]:
            if not state.final_answer:
                execute_action(state, Action.STOP_AND_ANSWER, controller, tools)
            break

    return state.final_answer, {"state": asdict(state), "steps": logs}
