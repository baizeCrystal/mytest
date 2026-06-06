from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PART_CHAIN_EDGES: Dict[str, List[Tuple[int, int]]] = {
    "full_body_7": [
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (4, 6),
        (5, 6),
    ],
    "lower_body_6": [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
    ],
}

PART_SYMMETRY_PAIRS: Dict[str, List[Tuple[int, int]]] = {
    "full_body_7": [(2, 3), (4, 5)],
    "lower_body_6": [(2, 3), (4, 5)],
}


def resolve_chain_edges(num_parts: int, slot_preset: str) -> List[Tuple[int, int]]:
    if slot_preset in PART_CHAIN_EDGES:
        edges = [edge for edge in PART_CHAIN_EDGES[slot_preset] if edge[0] < num_parts and edge[1] < num_parts]
        if edges:
            return edges

    edges = []
    for part_idx in range(max(num_parts - 1, 0)):
        edges.append((part_idx, part_idx + 1))
    return edges


def resolve_symmetry_pairs(num_parts: int, slot_preset: str) -> List[Tuple[int, int]]:
    pairs = PART_SYMMETRY_PAIRS.get(slot_preset, [])
    return [pair for pair in pairs if pair[0] < num_parts and pair[1] < num_parts]


def build_chain_adjacency(num_parts: int, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    adjacency = torch.eye(num_parts, dtype=torch.float32)
    for src, dst in edges:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def build_edge_incidence(num_parts: int, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    incidence = torch.zeros(num_parts, len(edges), dtype=torch.float32)
    for edge_idx, (src, dst) in enumerate(edges):
        incidence[src, edge_idx] = 1.0
        incidence[dst, edge_idx] = 1.0
    return incidence


class KinematicChainLayer(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_self = nn.Linear(feature_dim, feature_dim)
        self.node_neigh = nn.Linear(feature_dim, feature_dim)
        self.node_edge = nn.Linear(feature_dim, feature_dim)
        self.node_norm = nn.LayerNorm(feature_dim)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.edge_update = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.edge_norm = nn.LayerNorm(feature_dim)
        self.edge_ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        incidence: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]
        src_nodes = node_features[:, :, src_idx]
        dst_nodes = node_features[:, :, dst_idx]

        edge_update = self.edge_update(torch.cat([edge_features, src_nodes, dst_nodes], dim=-1))
        edge_features = self.edge_norm(edge_features + edge_update)
        edge_features = edge_features + self.edge_ffn(edge_features)

        neigh_context = torch.einsum("pq,bkqc->bkpc", adjacency, node_features)
        edge_context = torch.einsum("pe,bkec->bkpc", incidence, edge_features)

        node_update = self.node_self(node_features) + self.node_neigh(neigh_context) + self.node_edge(edge_context)
        node_features = self.node_norm(node_features + node_update)
        node_features = node_features + self.node_ffn(node_features)
        return node_features, edge_features


class KinematicChainReasoner(nn.Module):
    """Reason over phase-aligned body parts with an explicit kinematic chain graph."""

    def __init__(
        self,
        feature_dim: int,
        num_parts: int,
        coord_dim: int = 3,
        slot_preset: str = "full_body_7",
        num_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_parts = int(num_parts)
        self.coord_dim = int(coord_dim)
        self.slot_preset = str(slot_preset)

        edges = resolve_chain_edges(self.num_parts, self.slot_preset)
        if not edges:
            edges = [(0, 0)]
        self.num_edges = len(edges)
        self.symmetry_pairs = resolve_symmetry_pairs(self.num_parts, self.slot_preset)

        adjacency = build_chain_adjacency(self.num_parts, edges)
        incidence = build_edge_incidence(self.num_parts, edges)
        edge_index = torch.tensor(edges, dtype=torch.long)

        self.register_buffer("chain_adjacency", adjacency, persistent=False)
        self.register_buffer("chain_incidence", incidence, persistent=False)
        self.register_buffer("chain_edge_index", edge_index, persistent=False)

        motion_input_dim = self.coord_dim * 2 + 1
        edge_input_dim = self.coord_dim * 2 + 2

        self.motion_proj = nn.Sequential(
            nn.LayerNorm(motion_input_dim),
            nn.Linear(motion_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(edge_input_dim),
            nn.Linear(edge_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.layers = nn.ModuleList([KinematicChainLayer(self.feature_dim) for _ in range(max(1, int(num_layers)))])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 3),
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

    def forward(
        self,
        rgb_part_tokens: torch.Tensor,
        part_coords: torch.Tensor,
        part_velocity: torch.Tensor,
        has_skeleton: Optional[torch.Tensor] = None,
    ):
        if rgb_part_tokens.ndim != 4:
            raise ValueError(f"Expected rgb part tokens [B, K, P, C], got {tuple(rgb_part_tokens.shape)}")

        speed = torch.linalg.norm(part_velocity, dim=-1, keepdim=True)
        node_features = rgb_part_tokens + self.motion_proj(torch.cat([part_coords, part_velocity, speed], dim=-1))

        src_idx = self.chain_edge_index[:, 0]
        dst_idx = self.chain_edge_index[:, 1]
        coord_delta = part_coords[:, :, src_idx] - part_coords[:, :, dst_idx]
        velocity_delta = part_velocity[:, :, src_idx] - part_velocity[:, :, dst_idx]
        edge_length = torch.linalg.norm(coord_delta, dim=-1, keepdim=True)
        edge_speed = torch.linalg.norm(velocity_delta, dim=-1, keepdim=True)
        edge_features = self.edge_proj(torch.cat([coord_delta, velocity_delta, edge_length, edge_speed], dim=-1))

        adjacency = self.chain_adjacency.to(device=node_features.device, dtype=node_features.dtype)
        incidence = self.chain_incidence.to(device=node_features.device, dtype=node_features.dtype)
        edge_index = self.chain_edge_index.to(device=node_features.device)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, adjacency, incidence, edge_index)

        pooled_mean = node_features.mean(dim=2)
        pooled_max = node_features.max(dim=2).values
        edge_pool = edge_features.mean(dim=2)
        phase_kinematic_features = self.output_proj(torch.cat([pooled_mean, pooled_max, edge_pool], dim=-1))

        if has_skeleton is not None:
            mask = has_skeleton.view(-1, 1, 1).to(dtype=phase_kinematic_features.dtype, device=phase_kinematic_features.device)
            node_features = node_features * mask.unsqueeze(2)
            edge_features = edge_features * mask.unsqueeze(2)
            phase_kinematic_features = phase_kinematic_features * mask

        return {
            "phase_kinematic_features": phase_kinematic_features,
            "kinematic_node_features": node_features,
            "kinematic_edge_features": edge_features,
            "kinematic_edge_lengths": edge_length,
            "kinematic_edge_speed": edge_speed,
        }

    def get_edge_index(self) -> torch.Tensor:
        return self.chain_edge_index.detach().clone()

    def get_symmetry_pairs(self) -> List[Tuple[int, int]]:
        return list(self.symmetry_pairs)


def _masked_average(per_sample: torch.Tensor, has_skeleton: Optional[torch.Tensor]) -> torch.Tensor:
    if has_skeleton is None:
        return per_sample.mean()
    mask = has_skeleton.view(-1).to(dtype=per_sample.dtype, device=per_sample.device)
    if float(mask.sum()) <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def kinematic_chain_length_loss(
    part_coords: torch.Tensor,
    edge_index: torch.Tensor,
    has_skeleton: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if part_coords.ndim != 4:
        raise ValueError(f"Expected phase part coords [B, K, P, D], got {tuple(part_coords.shape)}")
    if part_coords.shape[1] <= 1:
        return part_coords.sum() * 0.0

    src_idx = edge_index[:, 0]
    dst_idx = edge_index[:, 1]
    edge_vectors = part_coords[:, :, src_idx] - part_coords[:, :, dst_idx]
    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
    per_sample = edge_lengths.var(dim=1, unbiased=False).mean(dim=-1)
    return _masked_average(per_sample, has_skeleton)


def kinematic_chain_symmetry_loss(
    part_velocity: torch.Tensor,
    symmetry_pairs: Sequence[Tuple[int, int]],
    has_skeleton: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if part_velocity.ndim != 4:
        raise ValueError(f"Expected phase part velocity [B, K, P, D], got {tuple(part_velocity.shape)}")
    if not symmetry_pairs:
        return part_velocity.sum() * 0.0

    left_idx = torch.tensor([pair[0] for pair in symmetry_pairs], device=part_velocity.device, dtype=torch.long)
    right_idx = torch.tensor([pair[1] for pair in symmetry_pairs], device=part_velocity.device, dtype=torch.long)
    left_speed = torch.linalg.norm(part_velocity[:, :, left_idx], dim=-1)
    right_speed = torch.linalg.norm(part_velocity[:, :, right_idx], dim=-1)
    per_sample = (left_speed - right_speed).abs().mean(dim=(1, 2))
    return _masked_average(per_sample, has_skeleton)
