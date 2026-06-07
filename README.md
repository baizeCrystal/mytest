# Student Action Error Recognition

This project implements a human-centric action error recognizer for student sports videos.
The current codebase is centered on a single main architecture:

1. RGB backbone for semantic video understanding
2. boundary-based soft phase segmentation
3. phase-aware body-part discovery
4. late skeleton kinematic pooling
5. explicit kinematic-chain reasoning
6. phase-wise error prediction with max-MIL video aggregation

The old correct-action prototype bank route has been removed from the main codebase.

## Manifest Format

Use JSON or JSONL. Each sample points to a folder of extracted frames.

```json
{
  "id": "jump_0002_knee_valgus",
  "video_dir": "jump/0002_error",
  "skeleton_path": "skeletons/jump_0002_knee_valgus.npy",
  "action_id": 0,
  "error_labels": [1],
  "is_correct": false,
  "phase_labels": [0, 0, 1, 1]
}
```

Required fields:

- `video_dir`: frame folder. Relative paths are resolved against `--data_root`.
- `action_id`: integer action type, for example jump=0, squat=1.
- `error_labels`: multi-label error ids. Correct samples use an empty list.

Optional fields:

- `id`: sample id.
- `skeleton_path`: path to `.npy`, `.npz`, `.pt`, `.pth`, or `.json` keypoint file.
- `is_correct`: whether this is a correct demonstration.
- `phase_labels`: optional per-frame or per-sampled-clip phase labels. Missing labels are supported.

Frame folders can contain `.jpg`, `.jpeg`, `.png`, or `.bmp` files. Numeric filename suffixes are used for temporal sorting.
Skeleton files are expected to contain a single-person sequence in a common `[T, J, C]` style layout. The loader also supports common 4D multi-person arrays and keeps the primary person only.

## Train Error Recognizer

### Part-slot variant

This is the RGB-only baseline path. It uses soft phase masks and body-part aggregation, without skeleton-driven kinematic reasoning.

```bash
python scripts/train_student_error.py \
  --train_manifest data/train.jsonl \
  --val_manifest data/val.jsonl \
  --data_root /path/to/student_frames \
  --num_actions 4 \
  --num_error_classes 8 \
  --num_phases 4 \
  --model_variant part_slot \
  --num_part_slots 6 \
  --batch_size 8 \
  --epochs 50 \
  --amp \
  --logdir outputs/student_error_x3d
```

For the part-slot variant, horizontal flip augmentation is disabled by default because left/right body-part priors would otherwise be inconsistent.

### RGB + skeleton variant

```bash
python scripts/train_student_error.py \
  --train_manifest data/train.jsonl \
  --val_manifest data/val.jsonl \
  --data_root /path/to/student_frames \
  --skeleton_root /path/to/student_skeletons \
  --use_skeleton \
  --skeleton_layout coco17 \
  --skeleton_num_joints 17 \
  --skeleton_input_dim 3 \
  --num_actions 4 \
  --num_error_classes 8 \
  --num_phases 4 \
  --model_variant kinematic_chain \
  --num_part_slots 7 \
  --skeleton_smoothness_weight 0.01 \
  --kinematic_length_weight 0.05 \
  --kinematic_symmetry_weight 0.05 \
  --batch_size 8 \
  --epochs 50 \
  --amp \
  --logdir outputs/student_error_rgb_skeleton
```

When `--use_skeleton` is enabled together with `--model_variant kinematic_chain`, the trainer reads `skeleton_path` from the manifest. If a sample has no skeleton file and `--skeleton_required` is not set, the model falls back to a zero skeleton tensor for that sample so the RGB path can still run. In the current design, skeletons do not participate in early phase encoding; they are injected only at the late kinematic-chain reasoning stage as motion evidence.

## Backbones

Default backbone is `x3d`, loaded from PyTorchVideo. A dependency-free `simple_cnn`
backbone is also available for smoke tests, but it is not intended as the final
experimental backbone.

To reuse the original Action-Slot feature extractor:

```bash
python scripts/train_student_error.py \
  --backbone action_slot \
  --action_slot_repo D:/action_slot/Action-slot \
  ...
```

The Action-Slot wrapper reuses the original backbone and `conv3d` projection, then exposes temporal features before the original classification head.

## Patent-Oriented Technical Points

The current implementation is structured around four main technical features:

- `SoftPhaseAssignment`: learnable soft phase queries for fuzzy movement-stage assignment.
- `PhaseAwarePartPrototypeAggregator`: phase-conditioned human-part discovery with fixed semantic part priors.
- `KinematicChainReasoner`: explicit body-part graph reasoning using phase-aligned skeleton kinematics.
- `PhaseContrastActionErrorModel`: the unified training scaffold for RGB-only and RGB+skeleton kinematic-chain variants.
