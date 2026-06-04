import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPhaseAssignment(nn.Module):
    """Boundary-based continuous soft phase segmentation over temporal features."""

    def __init__(
        self,
        feature_dim: int,
        num_phases: int,
        temperature: float = 0.08,
        min_duration_ratio: float = 0.08,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.temperature = float(temperature)
        self.min_duration_ratio = float(min_duration_ratio)

        hidden_dim = max(self.feature_dim, 64)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.duration_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_phases),
        )

    def forward(self, temporal_features: torch.Tensor):
        """Return phase features and continuous temporal phase masks.

        Args:
            temporal_features: [B, T, C]

        Returns:
            dict containing:
                phase_features: [B, K, C]
                phase_weights: [B, K, T]      # continuous phase masks
                phase_pool_weights: [B, K, T] # normalized for temporal pooling
                phase_boundaries: [B, K-1]
                phase_durations: [B, K]
        """
        if temporal_features.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(temporal_features.shape)}")

        batch_size, num_frames, _ = temporal_features.shape
        summary = self.summary_proj(temporal_features).mean(dim=1)
        duration_logits = self.duration_head(summary)
        phase_durations = self._predict_durations(duration_logits)
        phase_boundaries = torch.cumsum(phase_durations[:, :-1], dim=-1) if self.num_phases > 1 else phase_durations.new_zeros(batch_size, 0)

        phase_masks = self._build_phase_masks(num_frames, phase_boundaries, temporal_features.dtype, temporal_features.device)
        phase_pool_weights = phase_masks / phase_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_features = torch.einsum("bkt,btc->bkc", phase_pool_weights, temporal_features)

        return {
            "phase_features": phase_features,
            "phase_weights": phase_masks,
            "phase_pool_weights": phase_pool_weights,
            "phase_boundaries": phase_boundaries,
            "phase_durations": phase_durations,
        }

    def _predict_durations(self, duration_logits: torch.Tensor) -> torch.Tensor:
        if self.num_phases <= 0:
            raise ValueError("num_phases must be positive")

        safe_min = min(max(self.min_duration_ratio, 0.0), 0.95 / max(self.num_phases, 1))
        remaining_mass = max(1.0 - self.num_phases * safe_min, 1e-4)
        duration_probs = F.softmax(duration_logits, dim=-1)
        return safe_min + remaining_mass * duration_probs

    def _build_phase_masks(
        self,
        num_frames: int,
        boundaries: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.num_phases == 1:
            return torch.ones(boundaries.shape[0], 1, num_frames, dtype=dtype, device=device)

        time_steps = (torch.arange(num_frames, device=device, dtype=dtype) + 0.5) / max(num_frames, 1)
        time_steps = time_steps.view(1, 1, num_frames)
        tau = max(self.temperature, 1e-4)

        masks = []
        first_boundary = boundaries[:, 0:1, None]
        masks.append(1.0 - torch.sigmoid((time_steps - first_boundary) / tau))

        for phase_idx in range(1, self.num_phases - 1):
            left = boundaries[:, phase_idx - 1:phase_idx, None]
            right = boundaries[:, phase_idx:phase_idx + 1, None]
            masks.append(torch.sigmoid((time_steps - left) / tau) - torch.sigmoid((time_steps - right) / tau))

        last_boundary = boundaries[:, -1:, None]
        masks.append(torch.sigmoid((time_steps - last_boundary) / tau))

        phase_masks = torch.cat(masks, dim=1).clamp_min(0.0)
        phase_masks = phase_masks / phase_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return phase_masks


def phase_duration_regularization(phase_durations: torch.Tensor) -> torch.Tensor:
    if phase_durations.ndim != 2:
        raise ValueError(f"Expected [B, K], got {tuple(phase_durations.shape)}")
    if phase_durations.shape[1] <= 1:
        return phase_durations.sum() * 0.0

    entropy = -(phase_durations * phase_durations.clamp_min(1e-8).log()).sum(dim=-1)
    normalizer = math.log(phase_durations.shape[1])
    normalized_entropy = entropy / max(normalizer, 1e-6)
    return (1.0 - normalized_entropy).mean()
