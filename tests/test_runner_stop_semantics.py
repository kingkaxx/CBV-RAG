from __future__ import annotations

from typing import Any, Dict, List

from cbvrag.actions import Action
from cbvrag.runner import run_episode


class DummyRetriever:
    def retrieve(self, question: str, k: int) -> List[Dict[str, Any]]:
        return [{"doc_id": "d1", "chunk_id": "c1", "text": "evidence", "rerank_score": 0.9, "retriever_score": 0.8, "title": "T1"}]


class DummyReranker:
    def rerank(self, question: str, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return cands


class DummyLLM:
    tokenizer = object()

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, name: str):
        return "answer", {"total_tokens": 1}


class SeqController:
    def __init__(self, seq: List[int]):
        self.seq = list(seq)
        self.i = 0
        self.trace = []

    def act(self, obs, state, action_mask=None):
        idx = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return idx


def _tools():
    return {"retrieve": DummyRetriever(), "rerank": DummyReranker(), "llm": DummyLLM()}


def test_runner_prefers_stop_after_selected_context(monkeypatch):
    monkeypatch.setattr("cbvrag.runner.select_context", lambda question, pool, tokenizer, max_chunks, max_tokens: [{"evidence_id": pool[0]["evidence_id"]}] if pool else [])
    ctrl = SeqController([int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.SELECT_CONTEXT), int(Action.SELECT_CONTEXT)])
    _, log = run_episode("q", ctrl, _tools(), budgets={"max_steps": 6})
    actions = [s["action"] for s in log["steps"]]
    assert log["forced_stop_used"] is True
    assert log["fallback_stop_was_used"] is True
    assert int(Action.ANSWER_DIRECT) not in actions


def test_explicit_vs_forced_stop_accounting(monkeypatch):
    monkeypatch.setattr("cbvrag.runner.select_context", lambda question, pool, tokenizer, max_chunks, max_tokens: [{"evidence_id": pool[0]["evidence_id"]}] if pool else [])

    explicit_ctrl = SeqController([int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.STOP_AND_ANSWER)])
    _, explicit_log = run_episode("q", explicit_ctrl, _tools(), budgets={"max_steps": 5})
    assert explicit_log["explicit_stop_used"] is True
    assert explicit_log["forced_stop_used"] is False

    forced_ctrl = SeqController([int(Action.RETRIEVE_MORE_SMALL), int(Action.SELECT_CONTEXT), int(Action.SELECT_CONTEXT)])
    _, forced_log = run_episode("q", forced_ctrl, _tools(), budgets={"max_steps": 5})
    assert forced_log["forced_stop_used"] is True
