from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from cbvrag.actions import Action
from cbvrag.attribution import compute_attr
from cbvrag.evidence_clusters import cluster_evidence_items, summarize_cluster_stats
from cbvrag.evidence_specificity import score_evidence_specificity
from cbvrag.features import build_features
from cbvrag.prompts import answer_prompt, counterfactual_prompt, verify_prompt
from cbvrag.state import Branch, EpisodeState, EvidenceItem
from tools.select import select_context

logger = logging.getLogger(__name__)


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


def _selected_snippets(state: EpisodeState, max_items: int | None = None) -> List[str]:
    ids = state.selected_evidence_ids if max_items is None else state.selected_evidence_ids[:max_items]
    return [state.evidence_pool[eid].short_claim for eid in ids if eid in state.evidence_pool]


def _can_stop_now(state: EpisodeState) -> bool:
    retrieval_calls = int(state.metrics.get("retrieval_calls", 0))
    selected_nonempty = len(state.selected_evidence_ids) > 0
    return selected_nonempty and (retrieval_calls >= 2 or state.verification_status == "supported")


def _phase1_stats(state: EpisodeState) -> Dict[str, Any]:
    pool = list(state.evidence_pool.values())
    clusters = cluster_evidence_items(pool, selected_ids=state.selected_evidence_ids)
    cluster_stats = summarize_cluster_stats(clusters)
    specificity = score_evidence_specificity(state.question, pool, selected_ids=state.selected_evidence_ids)
    return {
        "cluster_info": clusters,
        "cluster_stats": cluster_stats,
        "specificity": specificity,
        "specificity_summary": specificity["summary"],
    }


def compute_action_mask(state: EpisodeState) -> List[bool]:
    action_mask = [True] * len(Action)

    last_action = int(state.metrics.get("last_action", -1))
    second_last_action = int(state.metrics.get("second_last_action", -1))
    no_progress_streak = int(state.metrics.get("no_progress_streak", 0))
    selected_changed = int(state.metrics.get("selected_evidence_changed", 1))
    pool_changed = int(state.metrics.get("evidence_pool_changed", 1))
    retrieval_calls = int(state.metrics.get("retrieval_calls", 0))

    selected_nonempty = len(state.selected_evidence_ids) > 0
    pool_nonempty = len(state.evidence_pool) > 0
    branches_open = len(state.branches) < state.budgets["max_branches"]
    branches_prunable = len(state.branches) > 1

    if not pool_nonempty:
        action_mask[int(Action.SELECT_CONTEXT)] = False
        action_mask[int(Action.VERIFY_CHEAP)] = False
    if not selected_nonempty:
        action_mask[int(Action.VERIFY_LLM)] = False
        action_mask[int(Action.SUMMARIZE_STATE)] = False

    if last_action == int(Action.SELECT_CONTEXT) and selected_changed == 0:
        action_mask[int(Action.SELECT_CONTEXT)] = False
    if last_action == int(Action.SELECT_CONTEXT) and no_progress_streak >= 1:
        action_mask[int(Action.SELECT_CONTEXT)] = False
    if selected_nonempty and pool_changed == 0 and selected_changed == 0:
        action_mask[int(Action.SELECT_CONTEXT)] = False

    no_op_repeat_actions = {
        int(Action.SELECT_CONTEXT),
        int(Action.SUMMARIZE_STATE),
        int(Action.MERGE_BRANCHES),
        int(Action.PRUNE_BRANCH),
    }
    if no_progress_streak >= 1 and last_action == second_last_action and last_action in no_op_repeat_actions:
        action_mask[last_action] = False

    if not branches_open:
        action_mask[int(Action.SPAWN_COUNTERFACTUAL)] = False
    if not branches_prunable:
        action_mask[int(Action.PRUNE_BRANCH)] = False
        action_mask[int(Action.MERGE_BRANCHES)] = False

    if retrieval_calls >= state.budgets["max_retrieval_calls"]:
        action_mask[int(Action.RETRIEVE_MORE_SMALL)] = False
        action_mask[int(Action.RETRIEVE_MORE_LARGE)] = False

    can_stop_now = _can_stop_now(state)
    action_mask[int(Action.STOP_AND_ANSWER)] = can_stop_now
    action_mask[int(Action.ANSWER_DIRECT)] = can_stop_now

    if not any(action_mask):
        action_mask[int(Action.STOP_AND_ANSWER)] = True
    return action_mask


def choose_valid_action(
    proposed_action_idx: int,
    state: EpisodeState,
    action_mask: List[bool],
) -> tuple[Action, bool, int]:
    requested = int(proposed_action_idx)
    action = Action(requested)
    action_was_forced = False

    if action_mask[int(action)]:
        return action, action_was_forced, requested

    action_was_forced = True
    state.metrics["illegal_action_requested"] = int(state.metrics.get("illegal_action_requested", 0)) + 1
    state.metrics["forced_action_count"] = int(state.metrics.get("forced_action_count", 0)) + 1

    selected_nonempty = len(state.selected_evidence_ids) > 0
    pool_nonempty = len(state.evidence_pool) > 0
    retrieval_calls_now = int(state.metrics.get("retrieval_calls", 0))

    if retrieval_calls_now < 2 and action_mask[int(Action.RETRIEVE_MORE_SMALL)]:
        return Action.RETRIEVE_MORE_SMALL, True, requested
    if retrieval_calls_now < 2 and action_mask[int(Action.RETRIEVE_MORE_LARGE)]:
        return Action.RETRIEVE_MORE_LARGE, True, requested
    if action_mask[int(Action.VERIFY_CHEAP)] and pool_nonempty:
        return Action.VERIFY_CHEAP, True, requested
    if action_mask[int(Action.SELECT_CONTEXT)] and pool_nonempty:
        return Action.SELECT_CONTEXT, True, requested
    if action_mask[int(Action.STOP_AND_ANSWER)] and selected_nonempty:
        return Action.STOP_AND_ANSWER, True, requested
    if action_mask[int(Action.ANSWER_DIRECT)] and selected_nonempty:
        return Action.ANSWER_DIRECT, True, requested

    fallback_idx = next(i for i, ok in enumerate(action_mask) if ok)
    return Action(fallback_idx), True, requested


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
        phase1 = _phase1_stats(state)
        specificity_lookup = phase1["specificity"]["chunk_scores"]
        pool = []
        for e in state.evidence_pool.values():
            extra = specificity_lookup.get(e.evidence_id, {})
            pool.append(
                {
                    "doc_id": e.doc_id,
                    "chunk_id": e.chunk_id,
                    "text": e.short_claim,
                    "retriever_score": e.retriever_score,
                    "rerank_score": e.rerank_score,
                    "evidence_id": e.evidence_id,
                    "title": e.title,
                    "specificity": float(extra.get("specificity", 0.0)),
                    "genericity": float(extra.get("genericity", 0.0)),
                    "novelty": float(extra.get("novelty", 0.0)),
                }
            )

        selected = select_context(
            state.question,
            pool,
            tokenizer=llm.tokenizer,
            max_chunks=state.budgets["max_context_chunks"],
            max_tokens=state.budgets["max_context_tokens"],
            cluster_info=phase1["cluster_info"],
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


def generate_null_branch(
    question: str,
    llm: Any,
    null_branch_threshold: float = 0.3,
    attr_device: str = "cpu",
) -> Dict[str, Any]:
    """Generate a *null branch* answer by calling the LLM with **empty context**.

    The null branch implements M(q, ∅) — the model's purely parametric answer
    when no retrieved evidence is provided.  This captures what the LLM "already
    knows" from its pre-training weights, without any grounding.

    The null branch is used in arbitration: if the null branch wins (i.e. the
    Attr score of the evidence-grounded branches is below *null_branch_threshold*),
    the result is flagged as ``parametric_hallucination_risk=True`` because the
    model is relying on unverified parametric memory rather than retrieved facts.

    Parameters
    ----------
    question:
        The user query.
    llm:
        An ``LLMEngine`` instance (or any object with a ``.generate()`` method
        compatible with the LLMEngine interface).
    null_branch_threshold:
        Attr score threshold below which the null branch is considered to "win".
        Default is 0.3.
    attr_device:
        Torch device string used when computing the Attr score for comparison.
        Default is ``"cpu"``.

    Returns
    -------
    dict with keys:
        * ``"branch_type"`` (str): always ``"null"``.
        * ``"answer"`` (str): the model's raw parametric answer.
        * ``"usage"`` (dict): token usage record from the LLM call.
        * ``"null_branch_threshold"`` (float): threshold used for arbitration.
        * ``"parametric_hallucination_risk"`` (bool): set during arbitration
          (initially ``False``; updated by :func:`arbitrate_null_branch`).
    """
    # Call the LLM with an empty context — no snippets, no summaries.
    prompt = (
        "Answer the following question based solely on your own knowledge. "
        "Be concise.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    answer, usage = llm.generate(prompt, max_new_tokens=96, temperature=0.0, name="null_branch")
    logger.debug("generate_null_branch: question=%r  answer=%r", question[:80], answer[:80])

    return {
        "branch_type": "null",
        "answer": answer.strip(),
        "usage": usage,
        "null_branch_threshold": null_branch_threshold,
        "parametric_hallucination_risk": False,
    }


def arbitrate_null_branch(
    null_branch: Dict[str, Any],
    grounded_answer: str,
    docs: List[str],
    query: str,
    counterfactual_queries: Optional[List[str]] = None,
    alpha: float = 0.5,
    attr_device: str = "cpu",
    answer_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the null branch against evidence-grounded branches using Attr score.

    Arbitration logic
    -----------------
    1. Compute Attr(q, D) for the retrieved document set.
    2. If Attr < ``null_branch["null_branch_threshold"]``, the evidence-grounded
       branches are considered too weakly attributed — the null branch "wins" and
       ``parametric_hallucination_risk`` is set to ``True``.
    3. The winning answer and arbitration metadata are returned for inclusion in
       the episode record.

    Parameters
    ----------
    null_branch:
        The record returned by :func:`generate_null_branch`.
    grounded_answer:
        The answer produced by the evidence-grounded branch.
    docs:
        Retrieved document passages (plain text) used for Attr computation.
    query:
        The original user question.
    counterfactual_queries:
        Optional list of counterfactual query strings for the PS component.
    alpha:
        Mixing coefficient for :func:`~cbvrag.attribution.compute_attr`.
    attr_device:
        Torch device string passed to the NLI model.

    Returns
    -------
    dict — the updated *null_branch* record with additional keys:
        * ``"attr_score"`` (dict): full output of ``compute_attr``.
        * ``"grounded_answer"`` (str): the evidence-grounded answer.
        * ``"winning_branch"`` (str): ``"null"`` or ``"grounded"``.
        * ``"parametric_hallucination_risk"`` (bool): ``True`` if null wins.
    """
    counterfactual_queries = counterfactual_queries or []
    attr_result = compute_attr(
        query=query,
        docs=docs,
        counterfactual_queries=counterfactual_queries,
        alpha=alpha,
        device=attr_device,
        answer_text=answer_text if answer_text is not None else grounded_answer,
    )

    threshold = float(null_branch.get("null_branch_threshold", 0.3))
    attr_score = attr_result["attr"]
    null_wins = attr_score < threshold

    null_branch = dict(null_branch)  # shallow copy — do not mutate the original
    null_branch["attr_score"] = attr_result
    null_branch["grounded_answer"] = grounded_answer
    null_branch["winning_branch"] = "null" if null_wins else "grounded"
    null_branch["parametric_hallucination_risk"] = null_wins

    logger.info(
        "arbitrate_null_branch: attr=%.4f  threshold=%.2f  winner=%s  "
        "parametric_hallucination_risk=%s",
        attr_score,
        threshold,
        null_branch["winning_branch"],
        null_wins,
    )
    return null_branch


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

        phase1 = _phase1_stats(state)

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
                "num_clusters": float(phase1["cluster_stats"]["num_clusters"]),
                "largest_cluster_frac": float(phase1["cluster_stats"]["largest_cluster_frac"]),
                "best_specificity_score": float(phase1["specificity_summary"]["best_specificity_score"]),
                "mean_specificity_selected": float(phase1["specificity_summary"]["mean_specificity_selected"]),
                "cluster_stats": phase1["cluster_stats"],
                "specificity_summary": phase1["specificity_summary"],
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

        _append_log(
            action,
            costs,
            selected_evidence_changed,
            evidence_pool_changed,
            branch_count_changed,
            verification_status_changed,
            made_progress,
            requested_action=requested_action,
            action_was_forced=action_was_forced,
            action_mask=action_mask,
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

    # ---------- Null-branch arbitration ----------------------------------------
    # Generate a purely parametric answer (no context) and compare it against
    # the evidence-grounded answer using the Attr score.  The result is stored
    # in the episode record as an inspectable field so that downstream consumers
    # can decide how to handle parametric-hallucination-risk cases.
    null_branch_record: Dict[str, Any] = {}
    try:
        llm = tools["llm"]
        grounded_docs = [
            state.evidence_pool[eid].short_claim
            for eid in state.selected_evidence_ids
            if eid in state.evidence_pool
        ]
        # Derive counterfactual queries from spawned branches (if any).
        counterfactual_queries = [
            b.hypothesis
            for bid, b in state.branches.items()
            if bid != "b0" and b.hypothesis and getattr(b, "status", "active") != "pruned"
        ]
        null_branch_raw = generate_null_branch(question, llm)
        null_branch_record = arbitrate_null_branch(
            null_branch=null_branch_raw,
            grounded_answer=state.final_answer,
            docs=grounded_docs,
            query=question,
            counterfactual_queries=counterfactual_queries,
            answer_text=state.final_answer,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Null-branch arbitration failed (non-fatal): %s", exc)
        null_branch_record = {"branch_type": "null", "error": str(exc), "parametric_hallucination_risk": False}
    # --------------------------------------------------------------------------

    out = {
        "state": asdict(state),
        "steps": logs,
        "fallback_stop_was_used": fallback_stop_was_used,
        "explicit_stop_used": explicit_stop_used,
        "forced_stop_used": forced_stop_used,
        "explicit_terminal_action": explicit_terminal_action,
        "null_branch": null_branch_record,
    }
    out["state"].setdefault("metrics", {})
    out["state"]["metrics"]["fallback_stop_was_used"] = int(fallback_stop_was_used)
    out["state"]["metrics"]["explicit_stop_used"] = int(explicit_stop_used)
    out["state"]["metrics"]["forced_stop_used"] = int(forced_stop_used)
    out["state"]["metrics"]["explicit_terminal_action"] = (
        int(explicit_terminal_action) if explicit_terminal_action is not None else -1
    )
    out["state"]["metrics"]["parametric_hallucination_risk"] = int(
        bool(null_branch_record.get("parametric_hallucination_risk", False))
    )
    return state.final_answer, out