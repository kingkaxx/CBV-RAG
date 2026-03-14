from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class PolicyConfig:
    policy_type: str = "mlp"
    obs_dim: int = 0
    act_dim: int = 11
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    history_len: int = 1


class MLPPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        layers = []
        in_dim = cfg.obs_dim
        depth = max(1, cfg.num_layers)
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU()])
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.fc1(x))
        out = self.drop(out)
        out = self.fc2(out)
        return torch.relu(x + out)


class MLPResidualPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.input = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(max(1, cfg.num_layers))])
        self.head = nn.Linear(cfg.hidden_dim, cfg.act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class GRUPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.obs_dim = cfg.obs_dim
        self.history_len = max(1, cfg.history_len)
        self.gru = nn.GRU(
            input_size=cfg.obs_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=max(1, cfg.num_layers),
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_dim, cfg.act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            seq = x.unsqueeze(1).repeat(1, self.history_len, 1)
        else:
            seq = x
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])


def build_policy(cfg: PolicyConfig) -> nn.Module:
    if cfg.policy_type == "mlp":
        return MLPPolicy(cfg)
    if cfg.policy_type == "mlp_residual":
        return MLPResidualPolicy(cfg)
    if cfg.policy_type == "gru_policy":
        return GRUPolicy(cfg)
    raise ValueError(f"Unsupported policy_type={cfg.policy_type}")


def policy_config_from_checkpoint(ckpt: Dict[str, Any]) -> PolicyConfig:
    arch = ckpt.get("arch", {})
    return PolicyConfig(
        policy_type=arch.get("policy_type", ckpt.get("policy_type", "mlp")),
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt.get("act_dim", 11)),
        hidden_dim=int(arch.get("hidden_dim", 128)),
        num_layers=int(arch.get("num_layers", 2)),
        dropout=float(arch.get("dropout", 0.0)),
        history_len=int(arch.get("history_len", 1)),
    )
