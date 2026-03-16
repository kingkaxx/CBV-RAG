from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cbvrag.actions import ACTION_ENUM_VERSION, Action, action_names
from cbvrag.controller_learned import LearnedController
from cbvrag.features import FEATURE_SCHEMA_VERSION, build_features
from cbvrag.runner import _make_state
from rl.policy import PolicyConfig, build_policy


def _write_ckpt(path: Path, obs_dim: int, act_dim: int, *, action_names_value=None) -> None:
    cfg = PolicyConfig(obs_dim=obs_dim, act_dim=act_dim, policy_type="mlp")
    model = build_policy(cfg)
    ckpt = {
        "state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "arch": {
            "policy_type": "mlp",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.0,
            "history_len": 1,
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_enum_version": ACTION_ENUM_VERSION,
        "action_names": action_names() if action_names_value is None else action_names_value,
    }
    torch.save(ckpt, path)


def test_act_dim_synchronization_and_feature_dim_consistency() -> None:
    state = _make_state("q", "id", budgets={"max_steps": 8, "max_retrieval_calls": 5, "max_branches": 3, "max_context_chunks": 8})
    obs = build_features(state)
    assert len(obs) > 0
    cfg = PolicyConfig(obs_dim=len(obs), act_dim=len(Action), policy_type="mlp")
    model = build_policy(cfg)
    logits = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    assert logits.shape[-1] == len(Action)


def test_checkpoint_roundtrip_and_metadata_validation(tmp_path: Path) -> None:
    ckpt = tmp_path / "ok.pt"
    _write_ckpt(ckpt, obs_dim=46, act_dim=len(Action))
    ctl = LearnedController(str(ckpt), mode="greedy")
    assert ctl.expected_obs_dim == 46

    bad = tmp_path / "bad_names.pt"
    _write_ckpt(bad, obs_dim=46, act_dim=len(Action), action_names_value=list(reversed(action_names())))
    with pytest.raises(ValueError, match="Action-name order mismatch"):
        LearnedController(str(bad), mode="greedy")
