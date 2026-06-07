from typing import Dict, List, Tuple

import torch


# Joint-to-part grouping used for skeleton pooling.
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


# Part-to-part topology used by the kinematic graph after part pooling.
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


def resolve_chain_edges(num_parts: int, slot_preset: str) -> List[Tuple[int, int]]:
    if slot_preset in PART_CHAIN_EDGES:
        edges = [edge for edge in PART_CHAIN_EDGES[slot_preset] if edge[0] < num_parts and edge[1] < num_parts]
        if edges:
            return edges

    edges = []
    for part_idx in range(max(num_parts - 1, 0)):
        edges.append((part_idx, part_idx + 1))
    return edges
