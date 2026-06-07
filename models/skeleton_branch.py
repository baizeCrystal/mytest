from typing import Optional, Tuple

import torch
import torch.nn as nn

from .body_topology import build_part_pool_map


class SkeletonKinematicEncoder(nn.Module):
    """Pure kinematic encoder that returns only normalized coordinates and velocity."""

    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 3,
        skeleton_layout: str = "coco17",
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.input_dim = int(input_dim)
        self.skeleton_layout = str(skeleton_layout)

    def forward(self, skeleton: torch.Tensor, has_skeleton: Optional[torch.Tensor] = None):
        if skeleton.ndim != 4:
            raise ValueError(f"Expected skeleton [B, T, J, C], got {tuple(skeleton.shape)}")
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

        skeleton = torch.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0)
        skeleton = self._normalize_skeleton(skeleton)
        velocity = torch.diff(skeleton, dim=1, prepend=skeleton[:, :1])
        valid = skeleton.abs().sum(dim=-1) > 1e-6
        if has_skeleton is not None:
            valid = valid & has_skeleton.view(-1, 1, 1).bool()

        return {
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
    """Pool raw kinematic quantities into RGB-defined temporal phases and body parts."""

    def __init__(
        self,
        num_phases: int,
        num_joints: int = 17,
        input_dim: int = 3,
        num_part_slots: int = 7,
        slot_preset: str = "full_body_7",
        skeleton_layout: str = "coco17",
    ):
        super().__init__()
        self.num_phases = int(num_phases)
        self.num_joints = int(num_joints)
        self.input_dim = int(input_dim)
        self.num_part_slots = int(num_part_slots)
        self.slot_preset = str(slot_preset)
        self.skeleton_layout = str(skeleton_layout)

        self.encoder = SkeletonKinematicEncoder(
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

    def forward(
        self,
        skeleton: torch.Tensor,
        phase_weights: torch.Tensor,
        has_skeleton: Optional[torch.Tensor] = None,
    ):
        if phase_weights.ndim != 3:
            raise ValueError(f"Expected phase weights [B, K, T], got {tuple(phase_weights.shape)}")
        if skeleton.ndim != 4:
            raise ValueError(f"Expected skeleton [B, T, J, C], got {tuple(skeleton.shape)}")
        if skeleton.shape[0] != phase_weights.shape[0]:
            raise ValueError(
                f"Batch mismatch between skeleton {tuple(skeleton.shape)} and phase weights {tuple(phase_weights.shape)}"
            )
        if skeleton.shape[1] != phase_weights.shape[2]:
            raise ValueError(
                f"Time mismatch between skeleton {tuple(skeleton.shape)} and phase weights {tuple(phase_weights.shape)}"
            )

        encoded = self.encoder(skeleton, has_skeleton=has_skeleton)
        part_coords, part_velocity = self._pool_parts(
            joint_coords=encoded["joint_coords"],
            joint_velocity=encoded["joint_velocity"],
            joint_valid=encoded["joint_valid"],
        )

        phase_pool = phase_weights / phase_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_part_coords = torch.einsum("bkt,btpd->bkpd", phase_pool, part_coords)
        phase_part_velocity = torch.einsum("bkt,btpd->bkpd", phase_pool, part_velocity)

        if has_skeleton is not None:
            mask = has_skeleton.view(-1, 1, 1).to(dtype=phase_part_coords.dtype)
            phase_part_coords = phase_part_coords * mask.unsqueeze(-1)
            phase_part_velocity = phase_part_velocity * mask.unsqueeze(-1)

        return {
            "phase_skeleton_part_coords": phase_part_coords,
            "phase_skeleton_part_velocity": phase_part_velocity,
        }

    def _pool_parts(
        self,
        joint_coords: torch.Tensor,
        joint_velocity: torch.Tensor,
        joint_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        part_pool = self.part_pool.to(device=joint_coords.device, dtype=joint_coords.dtype)
        weights = part_pool.view(1, 1, self.num_part_slots, self.num_joints)
        valid_weights = weights * joint_valid.unsqueeze(2).to(dtype=joint_coords.dtype)
        valid_weights = valid_weights / valid_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        part_coords = torch.einsum("btpj,btjd->btpd", valid_weights, joint_coords)
        part_velocity = torch.einsum("btpj,btjd->btpd", valid_weights, joint_velocity)
        return part_coords, part_velocity
