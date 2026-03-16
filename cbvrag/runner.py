from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from cbvrag.actions import Action
from cbvrag.evidence_clusters import cluster_evidence_items, summarize_cluster_stats
from cbvrag.evidence_specificity import score_evidence_specificity
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
        metrics={
            "retrieval_calls": 0,
            "rerank_calls": 0,
            "verify_calls": 0,
            "llm_calls": 0,
            "last_action": -1,
            "second_last_action": -1,
            "no_progress_streak": 0,
            "selected_evidence_changed": 0,
            "evidence_pool_changed": 0,
            "branch_count_changed": 0,
            "verification_status_changed": 0,
            "previous_selected_count": 0,
            "fallback_stop_was_used": 0,
            "explicit_stop_used": 0,
            "forced_stop_used": 0,
            "explicit_terminal_action": -1,
            "illegal_action_requested": 0,
            "forced_action_count": 0,
        },
    )




def _pool_as_dicts(state: EpisodeState) -> list[dict]:
    selected = set(state.selected_evidence_ids)
    return [
        {
            "evidence_id": e.evidence_id,
            "doc_id": e.doc_id,
            "chunk_id": e.chunk_id,
            "title": e.title,
            "rerank_score": float(e.rerank_score),
            "retriever_score": float(e.retriever_score),
            "text": e.short_claim,
            "is_selected": e.evidence_id in selected,
            "branch_id": e.branch_id,
        }
        for e in state.evidence_pool.values()
    ]


def _update_cluster_specificity_metrics(state: EpisodeState) -> None:
    items = _pool_as_dicts(state)
    clusters = cluster_evidence_items(items)
    cluster_stats = summarize_cluster_stats(clusters)
    cluster_map = {}
    for c in clusters:
        cid = c.get("cluster_id", "")
        for eid in c.get("member_ids", []) or []:
            cluster_map[eid] = cid

    retrieved_by_query = {
        "original": [dict(it, cluster_id=cluster_map.get(str(it.get("evidence_id", "")), "")) for it in items]
    }
    spec = score_evidence_specificity(state.question, {"original": state.question, "variants": []}, retrieved_by_query)
    selected = [eid for eid in state.selected_evidence_ids if eid in spec.get("chunk_scores", {})]
    selected_scores = [spec["chunk_scores"][eid] for eid in selected]
    best_spec = max([float(v.get("specificity", 0.0)) for v in spec.get("chunk_scores", {}).values()] or [0.0])
    best_resist = max([float(v.get("counterfactual_resistance", 0.0)) for v in spec.get("chunk_scores", {}).values()] or [0.0])
    mean_sel_spec = sum(float(v.get("specificity", 0.0)) for v in selected_scores) / max(1, len(selected_scores))

    sel_clusters = [cluster_map.get(eid, "") for eid in state.selected_evidence_ids if cluster_map.get(eid, "")]
    selected_cluster_count = len(set(sel_clusters))
    same_cluster_frac = 0.0
    if sel_clusters:
        same_cluster_frac = max(sel_clusters.count(c) for c in set(sel_clusters)) / max(1, len(sel_clusters))

    state.metrics.update(
        {
            **cluster_stats,
            "selected_cluster_count": float(selected_cluster_count),
            "selected_cluster_diversity": float(selected_cluster_count / max(1, int(cluster_stats.get("num_clusters", 0) or 1))),
            "selected_same_cluster_frac": float(same_cluster_frac),
            "evidence_redundancy_proxy": float(same_cluster_frac),
            "multi_cluster_support_flag": 1.0 if selected_cluster_count >= 2 else 0.0,
            "best_specificity_score": float(best_spec),
            "mean_specificity_selected": float(mean_sel_spec),
            "best_counterfactual_resistance": float(best_resist),
            "candidate_count_available": float(selected_cluster_count),
            "candidate_score_gap_top2": 0.0,
            "view_disagreement_score": 0.0,
            "original_specific_cluster_count": float(sum(1 for c in clusters if float(c.get("mean_rerank", 0.0)) >= 0.0)),
            "cluster_stats": cluster_stats,
            "specificity_stats": {
                "best_specificity_score": float(best_spec),
                "mean_specificity_selected": float(mean_sel_spec),
                "best_counterfactual_resistance": float(best_resist),
            },
        }
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
        _update_cluster_specificity_metrics(state)

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
        _update_cluster_specificity_metrics(state)

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
        snippets = _selected_snippets(state, max_items=3)
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
        snippets = _selected_snippets(state, max_items=3)
        prompt = "Summarize the current evidence in <=80 words:\n" + "\n".join(snippets)
        summary, usage = llm.generate(prompt, max_new_tokens=80, temperature=0.0, name="summarize")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        state.global_summary = summary[:1000]
        try:
            state.branches[state.active_branch_id].summary = summary[:1000]
        except Exception:
            pass

    elif action in (Action.ANSWER_DIRECT, Action.STOP_AND_ANSWER):
        snippets = _selected_snippets(state)
        branch_summary = getattr(state.branches[state.active_branch_id], "summary", "")
        prompt = answer_prompt(state.question, snippets, branch_summary, state.global_summary)
        answer, usage = llm.generate(prompt, max_new_tokens=96, temperature=0.0, name="answer")
        state.metrics["llm_calls"] += 1
        step_costs["tokens_used_this_step"] += usage["total_tokens"]
        state.final_answer = answer

    elif action == Action.PRUNE_BRANCH and len(state.branches) > 1:
        victim = None
        for bid, branch in state.branches.items():
            if bid != state.active_branch_id and getattr(branch, "status", "active") != "pruned":
                victim = bid
                break
        if victim is not None:
            state.branches[victim].status = "pruned"

    elif action == Action.MERGE_BRANCHES:
        state.global_summary = (state.global_summary + " merged").strip()

    state.step += 1
    return step_costs


def run_episode(question: str, controller: Any, tools: Dict[str, Any], budgets: Dict | None = None, qid: str = "") -> Tuple[str, Dict]:
    budgets = {**default_budgets(), **(budgets or {})}
    state = _make_state(question, qid=qid or "unknown", budgets=budgets)
    logs = []
    fallback_stop_was_used = False
    explicit_stop_used = False
    forced_stop_used = False
    explicit_terminal_action = None

    def _should_debug(qid_value: str) -> bool:
        try:
            return int(str(qid_value).split("-")[0]) < 3
        except Exception:
            return False

    def _append_log(
        action: Action,
        costs: Dict[str, Any],
        selected_evidence_changed: int,
        evidence_pool_changed: int,
        branch_count_changed: int,
        verification_status_changed: int,
        made_progress: bool,
        requested_action: int | None = None,
        action_was_forced: bool = False,
        action_mask: List[bool] | None = None,
    ) -> None:
        state.metrics["second_last_action"] = int(state.metrics.get("last_action", -1))
        state.metrics["last_action"] = int(action)
        state.metrics["selected_evidence_changed"] = selected_evidence_changed
        state.metrics["evidence_pool_changed"] = evidence_pool_changed
        state.metrics["branch_count_changed"] = branch_count_changed
        state.metrics["verification_status_changed"] = verification_status_changed
        state.metrics["no_progress_streak"] = 0 if made_progress else int(state.metrics.get("no_progress_streak", 0)) + 1

        logs.append(
            {
                "step": state.step,
                "action": int(action),
                "action_name": action.name,
                "requested_action": int(requested_action) if requested_action is not None else int(action),
                "action_was_forced": bool(action_was_forced),
                "action_mask": list(action_mask) if action_mask is not None else None,
                "costs": costs,
                "metrics": dict(state.metrics),
                "selected_evidence_changed": selected_evidence_changed,
                "evidence_pool_changed": evidence_pool_changed,
                "branch_count_changed": branch_count_changed,
                "verification_status_changed": verification_status_changed,
                "no_progress_streak": int(state.metrics.get("no_progress_streak", 0)),
            }
        )

    def _force_terminal_stop() -> None:
        nonlocal fallback_stop_was_used, forced_stop_used

        before_selected = set(state.selected_evidence_ids)
        before_pool = len(state.evidence_pool)
        before_branches = len(state.branches)
        before_ver_status = state.verification_status
        before_final = bool((state.final_answer or "").strip())

        costs = execute_action(state, Action.STOP_AND_ANSWER, controller, tools)

        after_selected = set(state.selected_evidence_ids)
        after_pool = len(state.evidence_pool)
        after_branches = len(state.branches)
        after_ver_status = state.verification_status
        after_final = bool((state.final_answer or "").strip())

        selected_evidence_changed = int(after_selected != before_selected)
        evidence_pool_changed = int(after_pool != before_pool)
        branch_count_changed = int(after_branches != before_branches)
        verification_status_changed = int(after_ver_status != before_ver_status)
        made_progress = any(
            [
                selected_evidence_changed,
                evidence_pool_changed,
                branch_count_changed,
                verification_status_changed,
                after_final != before_final,
            ]
        )

        _append_log(
            Action.STOP_AND_ANSWER,
            costs,
            selected_evidence_changed,
            evidence_pool_changed,
            branch_count_changed,
            verification_status_changed,
            made_progress,
            requested_action=int(Action.STOP_AND_ANSWER),
            action_was_forced=True,
            action_mask=[True] * len(Action),
        )

        fallback_stop_was_used = True
        forced_stop_used = True

    for _ in range(budgets["max_steps"]):
        obs = build_features(state)
        action_mask = compute_action_mask(state)

        try:
            action_idx = controller.act(obs, state, action_mask=action_mask)
        except TypeError:
            action_idx = controller.act(obs, state)
        controller_action = Action(action_idx)
        requested_action = controller_action
        action = controller_action
        action_was_forced = False

        action, action_was_forced, requested_action = choose_valid_action(action_idx, state, action_mask)

        if _should_debug(state.qid):
            print(
                f"[run_episode][debug] qid={state.qid} step={state.step} "
                f"requested={requested_action} resolved={action.name} forced={action_was_forced} "
                f"no_progress={int(state.metrics.get('no_progress_streak', 0))} "
                f"selected_nonempty={len(state.selected_evidence_ids) > 0}",
                flush=True,
            )

        if (
            action == Action.STOP_AND_ANSWER
            and state.verification_status == "unknown"
            and state.step < budgets["max_steps"] - 1
            and (
                len(state.selected_evidence_ids) == 0
                or int(state.metrics.get("retrieval_calls", 0)) == 0
                or state.step == 0
            )
        ):
            action = Action.VERIFY_CHEAP
            action_was_forced = True

        before_selected = set(state.selected_evidence_ids)
        before_pool = len(state.evidence_pool)
        before_branches = len(state.branches)
        before_ver_status = state.verification_status
        before_final = bool((state.final_answer or "").strip())

        costs = execute_action(state, action, controller, tools)

        after_selected = set(state.selected_evidence_ids)
        after_pool = len(state.evidence_pool)
        after_branches = len(state.branches)
        after_ver_status = state.verification_status
        after_final = bool((state.final_answer or "").strip())

        selected_evidence_changed = int(after_selected != before_selected)
        evidence_pool_changed = int(after_pool != before_pool)
        branch_count_changed = int(after_branches != before_branches)
        verification_status_changed = int(after_ver_status != before_ver_status)
        made_progress = any(
            [
                selected_evidence_changed,
                evidence_pool_changed,
                branch_count_changed,
                verification_status_changed,
                after_final != before_final,
            ]
        )

        state.metrics["previous_selected_count"] = len(before_selected)

        logs.append(
            {
                "step": state.step,
                "action": int(action),
                "action_name": action.name,
                "requested_action": int(requested_action),
                "requested_action_name": requested_action.name,
                "executed_action": int(action),
                "action_was_forced": bool(action_was_forced),
                "action_mask": list(action_mask),
                "costs": costs,
                "metrics": dict(state.metrics),
                "selected_evidence_changed": selected_evidence_changed,
                "evidence_pool_changed": evidence_pool_changed,
                "branch_count_changed": branch_count_changed,
                "verification_status_changed": verification_status_changed,
                "no_progress_streak": int(state.metrics.get("no_progress_streak", 0)),
            }
        )

        if (
            action in (Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT)
            or state.metrics["retrieval_calls"] >= budgets["max_retrieval_calls"]
        ):
            if action in (Action.STOP_AND_ANSWER, Action.ANSWER_DIRECT):
                if action_was_forced:
                    forced_stop_used = True
                else:
                    explicit_stop_used = True
                explicit_terminal_action = int(action)
            if not state.final_answer:
                _force_terminal_stop()
            break

        if (
            int(state.metrics.get("no_progress_streak", 0)) >= 1
            and int(state.metrics.get("retrieval_calls", 0)) >= 2
            and len(state.selected_evidence_ids) > 0
            and state.step < budgets["max_steps"]
        ):
            _force_terminal_stop()
            break

    if not state.final_answer:
        _force_terminal_stop()

    out = {
        "state": asdict(state),
        "steps": logs,
        "fallback_stop_was_used": fallback_stop_was_used,
        "explicit_stop_used": explicit_stop_used,
        "forced_stop_used": forced_stop_used,
        "explicit_terminal_action": explicit_terminal_action,
    }
    out["state"].setdefault("metrics", {})
    out["state"]["metrics"]["fallback_stop_was_used"] = int(fallback_stop_was_used)
    out["state"]["metrics"]["explicit_stop_used"] = int(explicit_stop_used)
    out["state"]["metrics"]["forced_stop_used"] = int(forced_stop_used)
    out["state"]["metrics"]["explicit_terminal_action"] = int(explicit_terminal_action) if explicit_terminal_action is not None else -1
    return state.final_answer, out
