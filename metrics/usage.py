from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Dict, List


@dataclass
class UsageRecord:
    name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ts: float


class UsageTracker:
    """Tracks LLM token usage and call volume for local HF generation calls."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.records: List[UsageRecord] = []
        self._lock = Lock()

    def track(self, name: str, prompt_tokens: int, completion_tokens: int) -> Dict:
        total_tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
        record = UsageRecord(
            name=name,
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
            total_tokens=total_tokens,
            ts=time.time(),
        )
        with self._lock:
            self.prompt_tokens += record.prompt_tokens
            self.completion_tokens += record.completion_tokens
            self.llm_calls += 1
            self.records.append(record)
        return asdict(record)

    def summary(self) -> Dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_calls": self.llm_calls,
                "records": [asdict(r) for r in self.records],
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.llm_calls = 0
            self.records = []
