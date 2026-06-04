from typing import Optional

import torch
import torch.nn as nn


class CorrectActionPrototypeBank(nn.Module):
    """Stage-wise prototypes for correct executions of each action."""

    def __init__(
        self,
        num_actions: int,
        num_phases: int,
        feature_dim: int,
        init_path: str = "",
        learnable: bool = False,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_phases = int(num_phases)
        self.feature_dim = int(feature_dim)

        prototypes = torch.zeros(self.num_actions, self.num_phases, self.feature_dim)
        if init_path:
            loaded = torch.load(init_path, map_location="cpu")
            if isinstance(loaded, dict):
                loaded = loaded.get("prototypes", loaded.get("state_dict", loaded))
            if not torch.is_tensor(loaded):
                raise ValueError(f"Prototype file must contain a tensor, got {type(loaded)}")
            if tuple(loaded.shape) != tuple(prototypes.shape):
                raise ValueError(
                    f"Prototype shape mismatch: expected {tuple(prototypes.shape)}, "
                    f"got {tuple(loaded.shape)}"
                )
            prototypes = loaded.float()

        if learnable:
            self.prototypes = nn.Parameter(prototypes)
        else:
            self.register_buffer("prototypes", prototypes)

    def forward(self, action_ids: torch.Tensor):
        if action_ids.ndim != 1:
            action_ids = action_ids.view(-1)
        return self.prototypes[action_ids.long()]

    @torch.no_grad()
    def update_from_features(
        self,
        phase_features: torch.Tensor,
        action_ids: torch.Tensor,
        momentum: Optional[float] = None,
    ):
        """Update prototypes from correct-sample phase features.

        Args:
            phase_features: [B, K, C]
            action_ids: [B]
            momentum: optional EMA momentum. None means arithmetic replacement
                by current mini-batch class mean.
        """
        if isinstance(self.prototypes, nn.Parameter):
            target = self.prototypes.data
        else:
            target = self.prototypes

        for action_id in action_ids.unique():
            mask = action_ids == action_id
            mean_feature = phase_features[mask].mean(dim=0)
            action_idx = int(action_id.item())
            if momentum is None:
                target[action_idx].copy_(mean_feature)
            else:
                target[action_idx].mul_(momentum).add_(mean_feature, alpha=1.0 - momentum)


class CorrectExecutionPrototypeComparator(nn.Module):
    """Compare phase-wise features against correct-execution prototypes."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 4),
            nn.Linear(self.feature_dim * 4, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor):
        if features.shape != prototypes.shape:
            raise ValueError(
                f"Feature/prototype shape mismatch: expected identical shapes, got "
                f"{tuple(features.shape)} vs {tuple(prototypes.shape)}"
            )

        residual = torch.abs(features - prototypes)
        alignment = features * prototypes
        fused = torch.cat([features, prototypes, residual, alignment], dim=-1)
        compared = self.project(fused)
        compared = self.gate(compared) * compared
        return {
            "prototype_features": prototypes,
            "prototype_residual": residual,
            "prototype_alignment": alignment,
            "prototype_compared": compared,
        }
