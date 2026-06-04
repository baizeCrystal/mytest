# Student Action Error Recognition

This is an independent extension project built on top of `D:\action_slot\Action-slot`.
It now supports RGB-only and RGB+skeleton phase-aware error recognizers for student sports videos:

1. Extract temporal video features with X3D or the original Action-Slot backbone.
2. Optionally encode single-person skeleton sequences with a lightweight kinematic graph branch.
3. Assign frames to learnable soft action phases.
4. Either compare each student phase feature with a correct-action prototype bank, or aggregate phase-specific body-part slots with Action-Slot style attention.
5. Fuse phase-aware RGB and skeleton cues before predicting video-level error classes.

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

## Build Correct Prototypes

Use correct samples only, or a mixed manifest where correct samples are marked by `is_correct=true`.

```bash
python scripts/build_correct_prototypes.py \
  --manifest data/correct_train.jsonl \
  --data_root /path/to/student_frames \
  --output outputs/correct_prototypes.pth \
  --num_actions 4 \
  --num_error_classes 8 \
  --num_phases 4 \
  --batch_size 8 \
  --amp
```

If you already have a trained checkpoint, pass `--checkpoint` so prototypes are built with the trained feature space.

## Train Error Recognizer

### Part-slot variant

This is the default training path. It keeps the soft phase masks, then replaces prototype contrast with body-part slot aggregation.

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
  --model_variant part_slot \
  --num_part_slots 7 \
  --skeleton_smoothness_weight 0.01 \
  --cross_modal_alignment_weight 0.05 \
  --batch_size 8 \
  --epochs 50 \
  --amp \
  --logdir outputs/student_error_rgb_skeleton
```

When `--use_skeleton` is enabled, the trainer reads `skeleton_path` from the manifest. If a sample has no skeleton file and `--skeleton_required` is not set, the model falls back to a zero skeleton tensor for that sample so the RGB path can still run.

### Prototype variant

```bash
python scripts/train_student_error.py \
  --train_manifest data/train.jsonl \
  --val_manifest data/val.jsonl \
  --data_root /path/to/student_frames \
  --num_actions 4 \
  --num_error_classes 8 \
  --num_phases 4 \
  --model_variant prototype \
  --prototype_path outputs/correct_prototypes.pth \
  --batch_size 8 \
  --epochs 50 \
  --amp \
  --logdir outputs/student_error_proto
```

If no prototype file is passed, the prototype bank is initialized as learnable zero prototypes. That is useful for debugging, but for the intended prototype method you should build or load correct-action prototypes.

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

The code is structured around four protectable technical features:

- `SoftPhaseAssignment`: learnable soft phase queries for fuzzy movement-stage assignment.
- `CorrectActionPrototypeBank`: correct-action phase prototypes indexed by action type and phase.
- `PhaseAwarePartSlotAggregator`: phase-conditioned body-part slot aggregation derived from Action-Slot attention.
- `PhaseContrastActionErrorModel`: a shared training scaffold that supports both prototype contrast and part-slot aggregation.
