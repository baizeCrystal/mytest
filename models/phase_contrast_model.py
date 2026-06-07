from typing import Optional

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ActionSlotTemporalBackbone, SimpleCNNTemporalBackbone, X3DTemporalBackbone, freeze_module
from .kinematic_chain import (
    KinematicChainReasoner,
    kinematic_chain_length_loss,
)
from .part_slot import PhaseAwarePartPrototypeAggregator
from .skeleton_branch import PhaseAwareSkeletonAggregator
from .soft_phase import SoftPhaseAssignment, phase_duration_regularization


class PhaseContrastActionErrorModel(nn.Module):
    """Phase-aware student action error model with explicit body-part kinematic reasoning."""

    def __init__(
        self,
        num_actions: int,
        num_error_classes: int,
        num_phases: int = 4,
        feature_dim: int = 256,
        backbone: str = "x3d",
        action_slot_repo: str = "",
        freeze_backbone: bool = False,
        train_last_blocks: int = 2,
        phase_temperature: float = 0.08,
        phase_min_duration_ratio: float = 0.08,
        pretrained_backbone: bool = True,
        action_slot_args=None,
        model_variant: str = "kinematic_chain",
        num_part_slots: int = 7,
        part_slot_preset: str = "full_body_7",
        part_slot_prior_strength: float = 2.5,
        use_skeleton: bool = False,
        skeleton_num_joints: int = 17,
        skeleton_input_dim: int = 3,
        skeleton_layout: str = "coco17",
    ):
        super().__init__()

        self.num_actions = int(num_actions)
        self.num_error_classes = int(num_error_classes)
        self.num_phases = int(num_phases)
        self.feature_dim = int(feature_dim)
        self.backbone_name = str(backbone)
        self.model_variant = str(model_variant)
        if self.model_variant not in {"kinematic_chain", "part_slot"}:
            raise ValueError(f"Unsupported model_variant: {self.model_variant}")
        self.num_part_slots = int(num_part_slots)
        self.part_slot_preset = str(part_slot_preset)
        requested_skeleton = bool(use_skeleton)
        self.use_kinematic_chain = bool(requested_skeleton and self.model_variant == "kinematic_chain")
        self.use_skeleton = bool(self.use_kinematic_chain)
        self.skeleton_num_joints = int(skeleton_num_joints)
        self.skeleton_input_dim = int(skeleton_input_dim)
        self.skeleton_layout = str(skeleton_layout)
        self.part_aggregator = None
        self.skeleton_aggregator = None
        self.kinematic_reasoner = None

        if backbone == "x3d":
            self.backbone = X3DTemporalBackbone(out_dim=feature_dim, pretrained=pretrained_backbone)
        elif backbone == "simple_cnn":
            self.backbone = SimpleCNNTemporalBackbone(out_dim=feature_dim)
        elif backbone == "action_slot":
            if not action_slot_repo:
                raise ValueError("action_slot_repo is required when backbone='action_slot'")
            args = action_slot_args or self._default_action_slot_args(feature_dim, num_phases)
            self.backbone = ActionSlotTemporalBackbone(
                args=args,
                action_slot_repo=action_slot_repo,
                num_actor_class=max(num_error_classes, num_phases),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if self.backbone.out_dim != self.feature_dim:
            self.feature_adapter = nn.Linear(self.backbone.out_dim, self.feature_dim)
        else:
            self.feature_adapter = nn.Identity()

        if freeze_backbone:
            freeze_module(self.backbone, train_last_blocks=train_last_blocks)

        self.phase_assign = SoftPhaseAssignment(
            feature_dim=self.feature_dim,
            num_phases=self.num_phases,
            temperature=phase_temperature,
            min_duration_ratio=phase_min_duration_ratio,
        )

        self.part_aggregator = PhaseAwarePartPrototypeAggregator(
            feature_dim=self.feature_dim,
            num_phases=self.num_phases,
            num_part_slots=self.num_part_slots,
            slot_preset=self.part_slot_preset,
            prior_strength=part_slot_prior_strength,
        )
        self.num_part_slots = self.part_aggregator.num_part_slots

        if self.use_skeleton:
            self.skeleton_aggregator = PhaseAwareSkeletonAggregator(
                num_phases=self.num_phases,
                num_joints=self.skeleton_num_joints,
                input_dim=self.skeleton_input_dim,
                num_part_slots=self.num_part_slots,
                slot_preset=self.part_slot_preset,
                skeleton_layout=self.skeleton_layout,
            )
            if self.use_kinematic_chain:
                self.kinematic_reasoner = KinematicChainReasoner(
                    feature_dim=self.feature_dim,
                    num_parts=self.num_part_slots,
                    coord_dim=self.skeleton_input_dim,
                    slot_preset=self.part_slot_preset,
                )

        if self.use_kinematic_chain:
            phase_context_dim = self.feature_dim * 2
        else:
            phase_context_dim = self.feature_dim * 3
        self.phase_error_head = nn.Sequential(
            nn.LayerNorm(phase_context_dim),
            nn.Linear(phase_context_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, self.num_error_classes),
        )

    @staticmethod
    def _default_action_slot_args(feature_dim: int, num_phases: int):
        return SimpleNamespace(
            dataset="charades",
            backbone="x3d",
            pretrain="",
            cp="",
            num_slots=max(64, num_phases),
            allocated_slot=True,
            bg_slot=False,
            box=False,
            channel=feature_dim,
            seq_len=16,
        )

    def _extract_backbone_features(self, videos):
        backbone_outputs = self.backbone(videos, return_spatial=True)
        if isinstance(backbone_outputs, dict):
            temporal_features = backbone_outputs["temporal_features"]
            spatial_features = backbone_outputs.get("spatial_features")
        else:
            temporal_features = backbone_outputs
            spatial_features = None

        temporal_features = self.feature_adapter(temporal_features)
        if spatial_features is not None:
            spatial_features = self.feature_adapter(spatial_features)
        return temporal_features, spatial_features

    def _build_missing_skeleton(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            num_frames,
            self.skeleton_num_joints,
            self.skeleton_input_dim,
            device=device,
            dtype=dtype,
        )

    def _forward_part_branch(
        self,
        spatial_features: torch.Tensor,
        phase_features: torch.Tensor,
        phase_weights: torch.Tensor,
        skeleton_outputs: Optional[dict] = None,
    ):
        if spatial_features is None:
            raise RuntimeError("Kinematic-chain variant requires spatial backbone features")

        part_outputs = self.part_aggregator(spatial_features, phase_weights)
        phase_part_features = part_outputs["phase_part_features"]
        outputs = {
            "part_tokens": part_outputs["part_tokens"],
            "part_attn": part_outputs["part_attn"],
            "phase_part_features": phase_part_features,
        }

        if skeleton_outputs is None:
            phase_context = torch.cat(
                [
                    phase_features,
                    phase_part_features,
                    torch.abs(phase_features - phase_part_features),
                ],
                dim=-1,
            )
            outputs["phase_context"] = phase_context
            return outputs

        if not self.use_kinematic_chain or self.kinematic_reasoner is None:
            phase_context = torch.cat(
                [
                    phase_features,
                    phase_part_features,
                    torch.abs(phase_features - phase_part_features),
                ],
                dim=-1,
            )
            outputs["phase_context"] = phase_context
            return outputs

        kinematic_outputs = self.kinematic_reasoner(
            rgb_part_tokens=part_outputs["part_tokens"],
            part_coords=skeleton_outputs["phase_skeleton_part_coords"],
            part_velocity=skeleton_outputs["phase_skeleton_part_velocity"],
            has_skeleton=skeleton_outputs.get("has_skeleton"),
        )
        phase_context = torch.cat(
            [
                phase_part_features,
                kinematic_outputs["phase_kinematic_features"],
            ],
            dim=-1,
        )

        outputs.update(skeleton_outputs)
        outputs.update(kinematic_outputs)
        outputs["phase_context"] = phase_context
        return outputs

    def forward(self, videos, action_id, skeleton=None, has_skeleton: Optional[torch.Tensor] = None):
        del action_id

        temporal_features, spatial_features = self._extract_backbone_features(videos)
        skeleton_outputs = None

        if self.use_skeleton:
            batch_size, num_frames, _ = temporal_features.shape
            if skeleton is None:
                skeleton = self._build_missing_skeleton(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    device=temporal_features.device,
                    dtype=temporal_features.dtype,
                )
                has_skeleton = torch.zeros(batch_size, device=temporal_features.device, dtype=torch.bool)
            else:
                skeleton = skeleton.to(device=temporal_features.device, dtype=temporal_features.dtype)
                if has_skeleton is None:
                    has_skeleton = torch.ones(batch_size, device=temporal_features.device, dtype=torch.bool)
                else:
                    has_skeleton = has_skeleton.to(device=temporal_features.device, dtype=torch.bool)

        phase_outputs = self.phase_assign(temporal_features)
        phase_features = phase_outputs["phase_features"]
        phase_weights = phase_outputs["phase_weights"]

        if self.use_skeleton:
            skeleton_outputs = self.skeleton_aggregator(
                skeleton=skeleton,
                phase_weights=phase_weights,
                has_skeleton=has_skeleton,
            )
            skeleton_outputs["has_skeleton"] = has_skeleton

        branch_outputs = self._forward_part_branch(
            spatial_features=spatial_features,
            phase_features=phase_features,
            phase_weights=phase_weights,
            skeleton_outputs=skeleton_outputs,
        )

        phase_context = branch_outputs["phase_context"]
        phase_logits = self.phase_error_head(phase_context)
        video_logits = phase_logits.max(dim=1).values

        outputs = {
            "logits": video_logits,
            "phase_logits": phase_logits,
            "phase_features": phase_features,
            "temporal_features": temporal_features,
            "fused_temporal_features": temporal_features,
            "spatial_features": spatial_features,
        }
        outputs.update(phase_outputs)
        outputs.update(branch_outputs)
        return outputs

    def compute_losses(
        self,
        outputs,
        error_targets,
        phase_duration_weight: float = 0.0,
        kinematic_length_weight: float = 0.0,
    ):
        losses = {}
        losses["error"] = F.binary_cross_entropy_with_logits(outputs["logits"], error_targets)
        total = losses["error"]

        if phase_duration_weight > 0:
            duration_reg = phase_duration_regularization(outputs["phase_durations"])
            losses["phase_duration"] = duration_reg
            total = total + phase_duration_weight * duration_reg

        if self.use_kinematic_chain and kinematic_length_weight > 0 and "phase_skeleton_part_coords" in outputs:
            length_reg = kinematic_chain_length_loss(
                outputs["phase_skeleton_part_coords"],
                self.kinematic_reasoner.get_edge_index().to(outputs["phase_skeleton_part_coords"].device),
                has_skeleton=outputs.get("has_skeleton"),
            )
            losses["kinematic_length"] = length_reg
            total = total + kinematic_length_weight * length_reg

        losses["total"] = total
        return losses
