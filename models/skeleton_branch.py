from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


SKELETON_PART_GROUPS: Dict[str, Dict[str, List[List[int]]]] = {
    "coco17": {
        "full_body_7": [
            [0, 1, 2, 3, 4],
            [5, 6, 11, 12],
            [5, 7, 9],
            [6, 8, 10],
            [11, 13, 15],
            [12, 14, 16],
            [15, 16],
        ],
        "lower_body_6": [
            [5, 6, 11, 12],
            [11, 12],
            [5, 7, 9],
            [6, 8, 10],
            [11, 13, 15],
            [12, 14, 16],
        ],
    }
}


def build_skeleton_adjacency(num_joints: int, layout: str = "coco17") -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    if layout == "coco17" and num_joints >= 17:
        edges = COCO17_EDGES
    else:
        edges = tuple((idx, idx + 1) for idx in range(max(num_joints - 1, 0)))

    for src, dst in edges:
        if 0 <= src < num_joints and 0 <= dst < num_joints:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0

    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def build_part_pool_map(
    num_joints: int,
    num_parts: int,
    slot_preset: str = "full_body_7",
    skeleton_layout: str = "coco17",
) -> torch.Tensor:
    groups = SKELETON_PART_GROUPS.get(skeleton_layout, {}).get(slot_preset)
    if groups is None or len(groups) != num_parts:
        groups = []
        chunk = max(1, num_joints // max(num_parts, 1))
        start = 0
        for part_idx in range(num_parts):
            end = num_joints if part_idx == num_parts - 1 else min(num_joints, start + chunk)
            groups.append(list(range(start, max(end, start + 1))))
            start = end

    pool = torch.zeros(num_parts, num_joints, dtype=torch.float32)
    for part_idx, joint_ids in enumerate(groups):
        valid_ids = [joint_id for joint_id in joint_ids if 0 <= joint_id < num_joints]
        if not valid_ids:
            valid_ids = [min(part_idx, num_joints - 1)]
        weight = 1.0 / float(len(valid_ids))
        for joint_id in valid_ids:
            pool[part_idx, joint_id] = weight
    return pool


class KinematicGraphBlock(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_proj = nn.Linear(feature_dim, feature_dim)
        self.neigh_proj = nn.Linear(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, joint_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_context = torch.einsum("ij,btjc->btic", adjacency, joint_features)
        joint_features = joint_features + self.self_proj(joint_features) + self.neigh_proj(neighbor_context)
        joint_features = self.norm1(joint_features)
        joint_features = joint_features + self.ffn(joint_features)
        return joint_features


class SkeletonKinematicEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_joints: int = 17,
        input_dim: int = 3,
        skeleton_layout: str = "coco17",
        num_graph_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_joints = int(num_joints)
        self.input_dim = int(input_dim)
        self.skeleton_layout = str(skeleton_layout)

        fused_input_dim = self.input_dim * 2 + 1
        self.input_proj = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.graph_blocks = nn.ModuleList(
            [KinematicGraphBlock(self.feature_dim) for _ in range(max(1, int(num_graph_layers)))]
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
        )

        adjacency = build_skeleton_adjacency(self.num_joints, layout=self.skeleton_layout)
        self.register_buffer("adjacency", adjacency, persistent=False)

    def forward(self, skeleton: torch.Tensor, has_skeleton: Optional[torch.Tensor] = None):
        if skeleton.ndim != 4:
            raise ValueError(f"Expected skeleton [B, T, J, C], got {tuple(skeleton.shape)}")

        skeleton = torch.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0)
        if skeleton.shape[2] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} skeleton joints, got {skeleton.shape[2]}. "
                "Keep dataset and model skeleton_num_joints aligned."
            )
        if skeleton.shape[3] != self.input_dim:
            raise ValueError(
                f"Expected skeleton input dim {self.input_dim}, got {skeleton.shape[3]}. "
                "Keep dataset and model skeleton_input_dim aligned."
            )

        skeleton = self._normalize_skeleton(skeleton)
        velocity = torch.diff(skeleton, dim=1, prepend=skeleton[:, :1])
        valid = skeleton.abs().sum(dim=-1) > 1e-6
        if has_skeleton is not None:
            has_skeleton = has_skeleton.view(-1, 1, 1).bool()
            valid = valid & has_skeleton

        encoder_input = torch.cat(
            [skeleton, velocity, valid.unsqueeze(-1).to(dtype=skeleton.dtype)],
            dim=-1,
        )
        joint_features = self.input_proj(encoder_input)
        joint_features = joint_features * valid.unsqueeze(-1).to(dtype=joint_features.dtype)

        adjacency = self.adjacency.to(device=joint_features.device, dtype=joint_features.dtype)
        for block in self.graph_blocks:
            joint_features = block(joint_features, adjacency)
            joint_features = joint_features * valid.unsqueeze(-1).to(dtype=joint_features.dtype)

        temporal_features = joint_features.mean(dim=2)
        temporal_update = self.temporal_conv(temporal_features.transpose(1, 2)).transpose(1, 2)
        temporal_features = temporal_features + temporal_update

        return {
            "joint_features": joint_features,
            "temporal_features": temporal_features,
            "joint_velocity": velocity,
            "joint_valid": valid,
            "joint_coords": skeleton,
        }

    def _normalize_skeleton(self, skeleton: torch.Tensor) -> torch.Tensor:
        coord_dims = min(2, self.input_dim)
        if coord_dims <= 0:
            return skeleton

        coords = skeleton[..., :coord_dims]
        valid = coords.abs().sum(dim=-1, keepdim=True) > 1e-6
        root = self._compute_root_center(coords, valid)
        coords = coords - root

        flattened = coords.norm(dim=-1)
        flattened = torch.where(valid.squeeze(-1), flattened, torch.zeros_like(flattened))
        scale = flattened.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        coords = coords / scale[:, None]
        coords = coords * valid.to(dtype=coords.dtype)

        if coord_dims == self.input_dim:
            return coords
        return torch.cat([coords, skeleton[..., coord_dims:]], dim=-1)

    def _compute_root_center(self, coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.skeleton_layout == "coco17" and self.num_joints >= 13:
            root_indices = [11, 12]
        else:
            root_indices = [0]

        root_coords = coords[:, :, root_indices]
        root_valid = valid[:, :, root_indices]
        root_count = root_valid.sum(dim=2, keepdim=False).clamp_min(1.0)
        root_center = (root_coords * root_valid.to(dtype=coords.dtype)).sum(dim=2) / root_count

        missing_root = root_valid.any(dim=2).logical_not()
        if missing_root.any():
            all_valid = valid.sum(dim=2, keepdim=False).clamp_min(1.0)
            fallback_center = (coords * valid.to(dtype=coords.dtype)).sum(dim=2) / all_valid
            root_center = torch.where(missing_root, fallback_center, root_center)

        return root_center.unsqueeze(2)


class PhaseAwareSkeletonAggregator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_phases: int,
        num_joints: int = 17,
        input_dim: int = 3,
        num_part_slots: int = 7,
        slot_preset: str = "full_body_7",
        skeleton_layout: str = "coco17",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.num_joints = int(num_joints)
        self.input_dim = int(input_dim)
        self.num_part_slots = int(num_part_slots)
        self.slot_preset = str(slot_preset)
        self.skeleton_layout = str(skeleton_layout)

        self.encoder = SkeletonKinematicEncoder(
            feature_dim=self.feature_dim,
            num_joints=self.num_joints,
            input_dim=self.input_dim,
            skeleton_layout=self.skeleton_layout,
        )
        part_pool = build_part_pool_map(
            num_joints=self.num_joints,
            num_parts=self.num_part_slots,
            slot_preset=self.slot_preset,
            skeleton_layout=self.skeleton_layout,
        )
        self.register_buffer("part_pool", part_pool, persistent=False)

        self.part_proj = nn.Linear(self.num_part_slots * self.feature_dim, self.feature_dim)
        self.part_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Sigmoid(),
        )
        motion_input_dim = self.num_part_slots * self.input_dim * 2
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(motion_input_dim),
            nn.Linear(motion_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

    def forward(
        self,
        skeleton: torch.Tensor,
        phase_weights: torch.Tensor,
        has_skeleton: Optional[torch.Tensor] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if phase_weights.ndim != 3:
            raise ValueError(f"Expected phase weights [B, K, T], got {tuple(phase_weights.shape)}")

        encoded = self.encoder(skeleton, has_skeleton=has_skeleton) if encoded is None else encoded
        part_features, part_coords, part_velocity = self._pool_parts(
            joint_features=encoded["joint_features"],
            joint_coords=encoded["joint_coords"],
            joint_velocity=encoded["joint_velocity"],
            joint_valid=encoded["joint_valid"],
        )

        phase_pool = phase_weights / phase_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_temporal_features = torch.einsum("bkt,btc->bkc", phase_pool, encoded["temporal_features"])
        phase_part_tokens = torch.einsum("bkt,btpc->bkpc", phase_pool, part_features)
        phase_part_coords = torch.einsum("bkt,btpd->bkpd", phase_pool, part_coords)
        phase_part_velocity = torch.einsum("bkt,btpd->bkpd", phase_pool, part_velocity)

        projected_parts = self.part_proj(phase_part_tokens.flatten(2))
        gated_parts = self.part_gate(projected_parts) * projected_parts
        phase_part_features = phase_part_tokens.mean(dim=2) + gated_parts

        motion_input = torch.cat([phase_part_coords, phase_part_velocity], dim=-1).flatten(2)
        phase_motion_features = self.motion_proj(motion_input)
        phase_skeleton_features = phase_temporal_features + phase_part_features + phase_motion_features

        if has_skeleton is not None:
            mask = has_skeleton.view(-1, 1, 1).to(dtype=phase_skeleton_features.dtype)
            phase_temporal_features = phase_temporal_features * mask
            phase_part_tokens = phase_part_tokens * mask.unsqueeze(2)
            phase_part_features = phase_part_features * mask
            phase_motion_features = phase_motion_features * mask
            phase_skeleton_features = phase_skeleton_features * mask
            phase_part_coords = phase_part_coords * mask.unsqueeze(-1)
            phase_part_velocity = phase_part_velocity * mask.unsqueeze(-1)

        outputs = {
            "skeleton_joint_features": encoded["joint_features"],
            "skeleton_temporal_features": encoded["temporal_features"],
            "skeleton_joint_velocity": encoded["joint_velocity"],
            "skeleton_joint_valid": encoded["joint_valid"],
            "skeleton_part_features": part_features,
            "skeleton_part_coords": part_coords,
            "skeleton_part_velocity": part_velocity,
            "phase_skeleton_temporal_features": phase_temporal_features,
            "phase_skeleton_part_tokens": phase_part_tokens,
            "phase_skeleton_part_features": phase_part_features,
            "phase_skeleton_part_coords": phase_part_coords,
            "phase_skeleton_part_velocity": phase_part_velocity,
            "phase_skeleton_motion_features": phase_motion_features,
            "phase_skeleton_features": phase_skeleton_features,
        }
        return outputs

    def _pool_parts(
        self,
        joint_features: torch.Tensor,
        joint_coords: torch.Tensor,
        joint_velocity: torch.Tensor,
        joint_valid: torch.Tensor,
    ):
        part_pool = self.part_pool.to(device=joint_features.device, dtype=joint_features.dtype)
        weights = part_pool.view(1, 1, self.num_part_slots, self.num_joints)
        valid_weights = weights * joint_valid.unsqueeze(2).to(dtype=joint_features.dtype)
        valid_weights = valid_weights / valid_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        part_features = torch.einsum("btpj,btjc->btpc", valid_weights, joint_features)
        part_coords = torch.einsum("btpj,btjd->btpd", valid_weights, joint_coords)
        part_velocity = torch.einsum("btpj,btjd->btpd", valid_weights, joint_velocity)
        return part_features, part_coords, part_velocity


def skeleton_temporal_smoothness_loss(
    joint_velocity: torch.Tensor,
    has_skeleton: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if joint_velocity.ndim != 4:
        raise ValueError(f"Expected joint velocity [B, T, J, C], got {tuple(joint_velocity.shape)}")
    acceleration = torch.diff(joint_velocity, dim=1)
    if acceleration.numel() == 0:
        return joint_velocity.sum() * 0.0

    per_sample = acceleration.abs().mean(dim=(1, 2, 3))
    if has_skeleton is None:
        return per_sample.mean()

    mask = has_skeleton.view(-1).to(dtype=per_sample.dtype)
    if float(mask.sum()) <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def cross_modal_alignment_loss(
    rgb_phase_features: torch.Tensor,
    skeleton_phase_features: torch.Tensor,
    has_skeleton: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if rgb_phase_features.shape != skeleton_phase_features.shape:
        raise ValueError(
            "Cross-modal alignment expects the same tensor shape, got "
            f"{tuple(rgb_phase_features.shape)} and {tuple(skeleton_phase_features.shape)}"
        )

    similarity = F.cosine_similarity(rgb_phase_features, skeleton_phase_features, dim=-1)
    per_sample = (1.0 - similarity).flatten(1).mean(dim=1)
    if has_skeleton is None:
        return per_sample.mean()

    mask = has_skeleton.view(-1).to(dtype=per_sample.dtype, device=per_sample.device)
    if float(mask.sum()) <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
