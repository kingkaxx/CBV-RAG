from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional

import torch

from rl.policy import build_policy, policy_config_from_checkpoint


class LearnedController:
    def __init__(self, policy_ckpt: str, mode: str = "greedy") -> None:
        if mode not in {"greedy", "sample"}:
            raise ValueError("mode must be one of {'greedy', 'sample'}")
        self.mode = mode
        self.trace: List[Dict[str, Any]] = []
        self.ckpt = torch.load(policy_ckpt, map_location="cpu")
        cfg = policy_config_from_checkpoint(self.ckpt)
        self.history_len = max(1, cfg.history_len)
        self._history: deque[list[float]] = deque(maxlen=self.history_len)
        self.model = build_policy(cfg)
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.eval()

    def reset(self) -> None:
        self.trace.clear()
        self._history.clear()

    def _build_model_input(self, obs: Iterable[float]) -> torch.Tensor:
        obs_list = list(obs)
        self._history.append(obs_list)
        if len(self._history) < self.history_len:
            while len(self._history) < self.history_len:
                self._history.appendleft(obs_list)
        stacked = torch.tensor(list(self._history), dtype=torch.float32)
        if stacked.dim() == 2:
            return stacked.unsqueeze(0)
        return stacked

    def act(self, obs, state: Any, action_mask: Optional[List[bool]] = None) -> int:
        inp = self._build_model_input(obs)
        if inp.dim() == 3 and not hasattr(self.model, "gru"):
            logits = self.model(inp[:, -1, :])
        else:
            logits = self.model(inp)

        logits = logits.squeeze(0)
        if action_mask is not None:
            if len(action_mask) != logits.shape[-1]:
                raise ValueError("action_mask length does not match action dimension")
            mask = torch.tensor(action_mask, dtype=torch.bool)
            logits = logits.masked_fill(~mask, float("-inf"))

        if self.mode == "sample":
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        else:
            action = int(torch.argmax(logits).item())

        self.trace.append({"obs": list(obs), "action": action, "reward": 0.0, "done": False, "info": {}})
        return action
