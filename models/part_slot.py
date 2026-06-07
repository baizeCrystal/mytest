from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_3d_grid(resolution: Sequence[int]) -> torch.Tensor:
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed3D(nn.Module):
    def __init__(self, hidden_size: int, resolution: Sequence[int]):
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pos = self.embedding(self.grid.to(device=inputs.device, dtype=inputs.dtype))
        return inputs + pos


BODY_PART_PRESETS: Dict[str, List[Dict[str, float]]] = {
    "lower_body_6": [
        {"name": "torso", "x": 0.50, "y": 0.32, "sigma": 0.18},
        {"name": "pelvis", "x": 0.50, "y": 0.54, "sigma": 0.16},
        {"name": "left_arm", "x": 0.26, "y": 0.32, "sigma": 0.15},
        {"name": "right_arm", "x": 0.74, "y": 0.32, "sigma": 0.15},
        {"name": "left_leg", "x": 0.38, "y": 0.76, "sigma": 0.18},
        {"name": "right_leg", "x": 0.62, "y": 0.76, "sigma": 0.18},
    ],
    "full_body_7": [
        {"name": "head", "x": 0.50, "y": 0.12, "sigma": 0.11},
        {"name": "torso", "x": 0.50, "y": 0.34, "sigma": 0.18},
        {"name": "left_arm", "x": 0.24, "y": 0.32, "sigma": 0.15},
        {"name": "right_arm", "x": 0.76, "y": 0.32, "sigma": 0.15},
        {"name": "left_leg", "x": 0.39, "y": 0.72, "sigma": 0.18},
        {"name": "right_leg", "x": 0.61, "y": 0.72, "sigma": 0.18},
        {"name": "feet_contact", "x": 0.50, "y": 0.93, "sigma": 0.12},
    ],
}


def available_part_slot_presets() -> List[str]:
    return sorted(BODY_PART_PRESETS)


def _generic_slot_layout(num_slots: int) -> List[Dict[str, float]]:
    preset = [
        {"name": "slot_00", "x": 0.50, "y": 0.42, "sigma": 0.18},
        {"name": "slot_01", "x": 0.28, "y": 0.32, "sigma": 0.18},
        {"name": "slot_02", "x": 0.72, "y": 0.32, "sigma": 0.18},
        {"name": "slot_03", "x": 0.38, "y": 0.72, "sigma": 0.18},
        {"name": "slot_04", "x": 0.62, "y": 0.72, "sigma": 0.18},
        {"name": "slot_05", "x": 0.50, "y": 0.92, "sigma": 0.18},
    ]
    if num_slots <= len(preset):
        return preset[:num_slots]

    centers = list(preset)
    extra = num_slots - len(preset)
    for idx in range(extra):
        x = (idx + 1) / (extra + 1)
        y = 0.55 if idx % 2 == 0 else 0.80
        centers.append(
            {
                "name": f"slot_{len(centers):02d}",
                "x": x,
                "y": y,
                "sigma": 0.18,
            }
        )
    return centers


def resolve_part_slot_layout(slot_preset: str, num_slots: int) -> List[Dict[str, float]]:
    if slot_preset == "custom":
        return _generic_slot_layout(num_slots)
    if slot_preset not in BODY_PART_PRESETS:
        raise ValueError(
            f"Unknown part-slot preset '{slot_preset}'. Available presets: "
            f"{', '.join(available_part_slot_presets())}, custom"
        )
    return [dict(part) for part in BODY_PART_PRESETS[slot_preset]]


def build_part_priors(
    slot_layout: Sequence[Dict[str, float]],
    resolution: Sequence[int],
    background_slot: bool = False,
) -> torch.Tensor:
    _, height, width = resolution
    ys = torch.linspace(0.0, 1.0, steps=height)
    xs = torch.linspace(0.0, 1.0, steps=width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    priors = []
    for part in slot_layout:
        cx, cy = float(part["x"]), float(part["y"])
        sigma = float(part["sigma"])
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        priors.append(-dist2 / (2.0 * sigma * sigma))
    priors = torch.stack(priors, dim=0).unsqueeze(1).repeat(1, resolution[0], 1, 1)
    if background_slot:
        priors = torch.cat([priors, torch.zeros(1, *resolution)], dim=0)
    return priors


class BodyPartPrototypeDiscovery(nn.Module):
    """PAT-style body-part discovery with fixed semantic part prototypes."""

    def __init__(
        self,
        num_parts: int,
        dim: int,
        resolution: Sequence[int],
        slot_preset: str = "full_body_7",
        prior_strength: float = 2.5,
    ):
        super().__init__()
        slot_layout = resolve_part_slot_layout(slot_preset, int(num_parts))
        self.slot_layout = slot_layout
        self.slot_names = [str(part["name"]) for part in slot_layout]
        self.slot_preset = str(slot_preset)
        self.num_parts = len(slot_layout)
        self.dim = int(dim)
        self.scale = self.dim ** -0.5
        self.prior_strength = float(prior_strength)

        self.part_prototypes = nn.Parameter(torch.randn(1, self.num_parts, self.dim) * 0.02)
        self.pe = SoftPositionEmbed3D(self.dim, resolution)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.norm_parts = nn.LayerNorm(self.dim)
        self.norm_inputs = nn.LayerNorm(self.dim)
        self.to_q = nn.Linear(self.dim, self.dim)
        self.to_k = nn.Linear(self.dim, self.dim)
        self.to_v = nn.Linear(self.dim, self.dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

        priors = build_part_priors(
            slot_layout=self.slot_layout,
            resolution=resolution,
            background_slot=False,
        )
        self.register_buffer("part_priors", priors, persistent=False)

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, T, H, W, C], got {tuple(inputs.shape)}")

        batch_size, num_frames, height, width, channels = inputs.shape
        tokens = self.pe(inputs)
        tokens = tokens.reshape(batch_size, -1, channels)
        tokens = self.input_proj(tokens)

        q = self.to_q(self.norm_parts(self.part_prototypes.expand(batch_size, -1, -1)))
        k = self.to_k(self.norm_inputs(tokens))
        v = self.to_v(tokens)

        logits = torch.einsum("bpd,bnd->bpn", q, k) * self.scale
        prior_logits = self.part_priors.view(1, self.num_parts, -1).to(device=logits.device, dtype=logits.dtype)
        logits = logits + self.prior_strength * prior_logits

        part_weights = F.softmax(logits, dim=-1)
        part_tokens = torch.einsum("bpn,bnd->bpd", part_weights, v)
        part_tokens = part_tokens + self.ffn(part_tokens)

        part_maps = part_weights.view(batch_size, self.num_parts, num_frames, height, width)
        return part_tokens, part_maps


class PhaseAwarePartPrototypeAggregator(nn.Module):
    """Apply soft phase masks before PAT-style body-part discovery."""

    def __init__(
        self,
        feature_dim: int,
        num_phases: int,
        num_part_slots: int = 7,
        slot_preset: str = "full_body_7",
        spatial_resolution: Optional[Sequence[int]] = None,
        prior_strength: float = 2.5,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.slot_preset = str(slot_preset)
        self.slot_layout = resolve_part_slot_layout(self.slot_preset, int(num_part_slots))
        self.part_slot_names = [str(part["name"]) for part in self.slot_layout]
        self.num_part_slots = len(self.slot_layout)
        self.spatial_resolution = spatial_resolution
        self.prior_strength = float(prior_strength)

        self.part_discovery = None
        if spatial_resolution is not None:
            self.part_discovery = self._build_part_discovery(spatial_resolution)

        self.part_proj = nn.Linear(self.num_part_slots * self.feature_dim, self.feature_dim)
        self.part_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Sigmoid(),
        )

    def _build_part_discovery(self, resolution: Sequence[int]) -> BodyPartPrototypeDiscovery:
        return BodyPartPrototypeDiscovery(
            num_parts=self.num_part_slots,
            dim=self.feature_dim,
            resolution=resolution,
            slot_preset=self.slot_preset,
            prior_strength=self.prior_strength,
        )

    def _maybe_init_part_discovery(self, spatial_features: torch.Tensor):
        if self.part_discovery is not None:
            return
        _, num_frames, height, width, _ = spatial_features.shape
        resolution = (num_frames, height, width)
        self.part_discovery = self._build_part_discovery(resolution).to(spatial_features.device)

    def forward(self, spatial_features: torch.Tensor, phase_weights: torch.Tensor):
        if spatial_features.ndim != 5:
            raise ValueError(f"Expected [B, T, H, W, C], got {tuple(spatial_features.shape)}")
        if phase_weights.ndim != 3:
            raise ValueError(f"Expected [B, K, T], got {tuple(phase_weights.shape)}")

        self._maybe_init_part_discovery(spatial_features)
        batch_size, num_frames, height, width, channels = spatial_features.shape
        if phase_weights.shape[2] != num_frames:
            raise ValueError(
                f"Phase weights time dim {phase_weights.shape[2]} does not match spatial feature time dim {num_frames}"
            )

        weighted_maps = phase_weights[:, :, :, None, None, None] * spatial_features[:, None]
        weighted_maps = weighted_maps.reshape(batch_size * self.num_phases, num_frames, height, width, channels)

        part_tokens, part_attn = self.part_discovery(weighted_maps)
        part_tokens = part_tokens.view(batch_size, self.num_phases, self.num_part_slots, channels)
        part_attn = part_attn.view(batch_size, self.num_phases, self.num_part_slots, num_frames, height, width)

        pooled_tokens = part_tokens.mean(dim=2)
        projected_tokens = self.part_proj(part_tokens.flatten(2))
        gated_tokens = self.part_gate(projected_tokens) * projected_tokens
        phase_part_features = pooled_tokens + gated_tokens

        return {
            "part_tokens": part_tokens,
            "part_attn": part_attn,
            "phase_part_features": phase_part_features,
        }

    def get_part_slot_names(self) -> List[str]:
        return list(self.part_slot_names)
