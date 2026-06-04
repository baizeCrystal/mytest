import csv
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def _read_manifest(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.lower().endswith(".csv"):
        return _read_csv_manifest(path)

    if path.lower().endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list, JSONL file, or {'samples': [...]} object")
    return data


def _read_csv_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        if {"id", "actions", "length"}.issubset(fieldnames):
            return [row for row in reader if row.get("actions", "").strip()]

        if "video_dir" in fieldnames:
            return list(reader)

    raise ValueError(
        "CSV manifest must either contain Charades fields "
        "('id', 'actions', 'length') or generic manifest field 'video_dir'"
    )


def _frame_sort_key(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return 0, filename
    return int(numbers[-1]), filename


SKELETON_RECORD_KEYS = (
    "skeleton_path",
    "pose_path",
    "keypoints_path",
    "keypoint_path",
)
SKELETON_FILE_EXTENSIONS = (".npy", ".npz", ".pt", ".pth", ".json")


class StudentActionDataset(Dataset):
    """Generic frame-folder dataset for student movement error recognition.

    Manifest format, JSON or JSONL:
        {
          "id": "sample_001",
          "video_dir": "relative/or/absolute/frame_folder",
          "action_id": 0,
          "error_labels": [2, 5],
          "is_correct": false,
          "phase_labels": [0, 0, 1, ...]   # optional per-frame or sampled phase labels
        }

    The loader returns one clip per sample. Labels are video-level multi-label
    error targets; correct samples simply have an all-zero error vector.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str = "",
        split: str = "train",
        num_error_classes: int = 1,
        seq_len: int = 16,
        sampling_rate: int = 2,
        image_size: int = 224,
        resize_size: int = 256,
        random_sample: Optional[bool] = None,
        normalize: str = "pytorchvideo",
        allow_empty_errors: bool = True,
        enable_hflip: bool = True,
        use_skeleton: bool = False,
        skeleton_root: str = "",
        skeleton_layout: str = "coco17",
        skeleton_num_joints: int = 17,
        skeleton_input_dim: int = 3,
        skeleton_required: bool = False,
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.data_root = os.path.abspath(data_root) if data_root else ""
        self.split = split
        self.num_error_classes = int(num_error_classes)
        self.seq_len = int(seq_len)
        self.sampling_rate = max(1, int(sampling_rate))
        self.image_size = int(image_size)
        self.resize_size = int(resize_size)
        self.random_sample = (split == "train") if random_sample is None else bool(random_sample)
        self.normalize = normalize
        self.allow_empty_errors = allow_empty_errors
        self.enable_hflip = bool(enable_hflip)
        self.use_skeleton = bool(use_skeleton)
        self.skeleton_root = os.path.abspath(skeleton_root) if skeleton_root else ""
        self.skeleton_layout = str(skeleton_layout)
        self.skeleton_num_joints = max(1, int(skeleton_num_joints))
        self.skeleton_input_dim = max(1, int(skeleton_input_dim))
        self.skeleton_required = bool(skeleton_required)

        raw_records = _read_manifest(self.manifest_path)
        self.samples = [self._normalize_record(record) for record in raw_records]
        if not self.samples:
            raise ValueError(f"No valid samples found in manifest: {manifest_path}")

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        video_dir = self._resolve_video_dir(record)
        if not os.path.isabs(video_dir):
            base = self.data_root or os.path.dirname(self.manifest_path)
            video_dir = os.path.join(base, video_dir)
        video_dir = os.path.abspath(video_dir)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Frame folder not found: {video_dir}")

        frames = self._list_frames(video_dir)
        if not frames:
            raise ValueError(f"No image frames found in: {video_dir}")

        charades_meta = self._build_charades_metadata(record, frames)
        if charades_meta is not None:
            frames = charades_meta["frames"]
            if not frames:
                raise ValueError(f"No valid active frames found in Charades sample: {record.get('id', video_dir)}")

        error_labels = record.get("error_labels", [])
        if error_labels is None:
            error_labels = []
        if isinstance(error_labels, int):
            error_labels = [error_labels]
        if charades_meta is not None and not error_labels:
            error_labels = sorted({label for label, _, _ in charades_meta["actions"]})
        if not error_labels and not record.get("is_correct", False) and not self.allow_empty_errors:
            raise ValueError(f"Sample has no error labels and is not marked correct: {record}")

        action_id = int(record.get("action_id", 0))
        sample_id = str(record.get("id", os.path.basename(video_dir)))

        phase_labels = record.get("phase_labels")
        if phase_labels is not None:
            phase_labels = [int(x) for x in phase_labels]

        skeleton_path = None
        if self.use_skeleton:
            skeleton_path = self._resolve_skeleton_path(
                record=record,
                video_dir=video_dir,
                sample_id=sample_id,
            )
            if self.skeleton_required and skeleton_path is None:
                raise FileNotFoundError(
                    f"No skeleton file found for sample '{sample_id}'. "
                    "Provide one of the manifest keys "
                    f"{SKELETON_RECORD_KEYS} or place a file under skeleton_root."
                )

        sample = {
            "id": sample_id,
            "video_dir": video_dir,
            "frames": frames,
            "action_id": action_id,
            "error_labels": [int(x) for x in error_labels],
            "is_correct": bool(record.get("is_correct", len(error_labels) == 0)),
            "phase_labels": phase_labels,
            "skeleton_path": skeleton_path,
            "meta": record,
        }
        if charades_meta is not None:
            sample["charades_actions"] = charades_meta["actions"]
            sample["frame_timestamps"] = charades_meta["frame_timestamps"]
        return sample

    def _resolve_video_dir(self, record: Dict[str, Any]) -> str:
        if "video_dir" in record and record["video_dir"]:
            return str(record["video_dir"])
        if self._is_charades_record(record):
            return str(record["id"])
        raise KeyError("Each sample must contain 'video_dir', or Charades CSV fields {'id', 'actions', 'length'}")

    @staticmethod
    def _is_charades_record(record: Dict[str, Any]) -> bool:
        return all(key in record for key in ["id", "actions", "length"]) and "video_dir" not in record

    def _build_charades_metadata(self, record: Dict[str, Any], frames: List[str]) -> Optional[Dict[str, Any]]:
        if not self._is_charades_record(record):
            return None

        actions = self._parse_charades_actions(record.get("actions", ""))
        duration = float(record["length"])
        if duration <= 0:
            raise ValueError(f"Charades sample has non-positive duration: {record}")

        num_frames = len(frames)
        fps = num_frames / duration
        timestamps = np.arange(num_frames, dtype=np.float32) / fps
        active_mask = np.zeros(num_frames, dtype=bool)
        for label_id, start_time, end_time in actions:
            if label_id >= self.num_error_classes:
                raise ValueError(
                    f"Charades label c{label_id:03d} exceeds num_error_classes={self.num_error_classes}"
                )
            active_mask |= (timestamps > start_time) & (timestamps < end_time)

        if active_mask.any():
            selected_indices = np.flatnonzero(active_mask)
            selected_frames = [frames[int(idx)] for idx in selected_indices]
            selected_timestamps = timestamps[selected_indices]
        else:
            selected_frames = frames
            selected_timestamps = timestamps

        return {
            "actions": actions,
            "frames": selected_frames,
            "frame_timestamps": selected_timestamps,
        }

    @staticmethod
    def _parse_charades_actions(actions: str) -> List[tuple[int, float, float]]:
        parsed = []
        actions = actions.strip()
        if not actions:
            return parsed

        for item in actions.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid Charades action annotation: {item}")
            label, start_time, end_time = parts
            if not re.fullmatch(r"[cC]\d+", label):
                raise ValueError(f"Invalid Charades class token: {label}")
            parsed.append((int(label[1:]), float(start_time), float(end_time)))
        return parsed

    @staticmethod
    def _list_frames(video_dir: str) -> List[str]:
        frames = [
            name for name in os.listdir(video_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        return sorted(frames, key=_frame_sort_key)

    def _resolve_skeleton_path(
        self,
        record: Dict[str, Any],
        video_dir: str,
        sample_id: str,
    ) -> Optional[str]:
        for key in SKELETON_RECORD_KEYS:
            raw_path = record.get(key)
            if raw_path:
                resolved = self._resolve_optional_path(str(raw_path), video_dir=video_dir)
                if resolved is not None:
                    return resolved

        candidates = self._infer_skeleton_candidates(record=record, video_dir=video_dir, sample_id=sample_id)
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    def _resolve_optional_path(self, path: str, video_dir: str) -> Optional[str]:
        if os.path.isabs(path):
            return os.path.abspath(path) if os.path.isfile(path) else None

        bases = []
        if self.skeleton_root:
            bases.append(self.skeleton_root)
        if self.data_root:
            bases.append(self.data_root)
        bases.extend([os.path.dirname(self.manifest_path), video_dir])

        for base in bases:
            candidate = os.path.abspath(os.path.join(base, path))
            if os.path.isfile(candidate):
                return candidate
        return None

    def _infer_skeleton_candidates(
        self,
        record: Dict[str, Any],
        video_dir: str,
        sample_id: str,
    ) -> List[str]:
        candidates = []
        bases = []
        if self.skeleton_root:
            bases.append(self.skeleton_root)
        bases.extend([os.path.dirname(video_dir), self.data_root, os.path.dirname(self.manifest_path)])

        relative_video_dir = record.get("video_dir")
        stems = []
        if relative_video_dir:
            relative_video_dir = os.path.splitext(str(relative_video_dir).rstrip("/"))[0]
            stems.append(relative_video_dir)
            stems.append(os.path.basename(relative_video_dir))
        stems.extend(
            [
                os.path.splitext(video_dir.rstrip("/"))[0],
                video_dir.rstrip("/"),
                sample_id,
            ]
        )

        dedup = []
        for stem in stems:
            stem = str(stem)
            if stem and stem not in dedup:
                dedup.append(stem)

        for base in bases:
            if not base:
                continue
            for stem in dedup:
                for ext in SKELETON_FILE_EXTENSIONS:
                    candidates.append(os.path.abspath(os.path.join(base, f"{stem}{ext}")))
        return candidates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        indices = self.sample_indices(len(sample["frames"]), random_sample=self.random_sample)
        frames = self._load_frames(sample, indices)

        error_target = self._build_error_target(sample, indices)

        phase_target = self._sample_phase_labels(sample.get("phase_labels"), indices)

        batch = {
            "videos": frames,
            "error": error_target,
            "action_id": torch.tensor(sample["action_id"], dtype=torch.long),
            "is_correct": torch.tensor(sample["is_correct"], dtype=torch.bool),
            "phase": phase_target,
            "id": sample["id"],
        }
        if self.use_skeleton:
            skeleton, has_skeleton = self._load_skeleton(sample, indices)
            batch["skeleton"] = skeleton
            batch["has_skeleton"] = torch.tensor(has_skeleton, dtype=torch.bool)
        return batch

    def _build_error_target(self, sample: Dict[str, Any], indices: np.ndarray) -> torch.Tensor:
        if "charades_actions" in sample:
            timestamps = sample["frame_timestamps"][indices]
            error_target = torch.zeros(self.num_error_classes, dtype=torch.float32)
            for label_id, start_time, end_time in sample["charades_actions"]:
                active = (timestamps > start_time) & (timestamps < end_time)
                if np.any(active):
                    error_target[label_id] = 1.0
            return error_target

        error_target = torch.zeros(self.num_error_classes, dtype=torch.float32)
        for label in sample["error_labels"]:
            if 0 <= label < self.num_error_classes:
                error_target[label] = 1.0
        return error_target

    def sample_indices(self, total_frames: int, random_sample: bool = False) -> np.ndarray:
        if total_frames <= 0:
            raise ValueError("Cannot sample from an empty frame sequence")

        clip_span = (self.seq_len - 1) * self.sampling_rate + 1
        if total_frames < clip_span:
            return np.linspace(0, total_frames - 1, self.seq_len).round().astype(np.int64)

        max_start = total_frames - clip_span
        if random_sample:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2
        return start + np.arange(self.seq_len, dtype=np.int64) * self.sampling_rate

    def _sample_phase_labels(self, phase_labels: Optional[Iterable[int]], indices: np.ndarray):
        if phase_labels is None:
            return torch.full((self.seq_len,), -1, dtype=torch.long)

        labels = list(phase_labels)
        if len(labels) == self.seq_len:
            return torch.tensor(labels, dtype=torch.long)

        if len(labels) < max(indices) + 1:
            return torch.full((self.seq_len,), -1, dtype=torch.long)
        return torch.tensor([labels[int(idx)] for idx in indices], dtype=torch.long)

    def _load_frames(self, sample: Dict[str, Any], indices: np.ndarray) -> List[torch.Tensor]:
        pil_frames = []
        for idx in indices:
            frame_name = sample["frames"][int(idx)]
            path = os.path.join(sample["video_dir"], frame_name)
            pil_frames.append(Image.open(path).convert("RGB"))
        return self._transform_frames(pil_frames)

    def _load_skeleton(self, sample: Dict[str, Any], indices: np.ndarray):
        skeleton_path = sample.get("skeleton_path")
        if not skeleton_path:
            return self._empty_skeleton(), False

        skeleton_array = self._read_skeleton_file(skeleton_path)
        skeleton_array = self._sample_skeleton_sequence(
            skeleton_array,
            indices=indices,
            reference_length=len(sample["frames"]),
        )
        skeleton_array = self._normalize_skeleton_sequence(skeleton_array)
        return torch.from_numpy(skeleton_array), True

    def _empty_skeleton(self) -> torch.Tensor:
        return torch.zeros(self.seq_len, self.skeleton_num_joints, self.skeleton_input_dim, dtype=torch.float32)

    def _read_skeleton_file(self, path: str) -> np.ndarray:
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".npy":
            data = np.load(path, allow_pickle=True)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=True) as data_file:
                data = self._extract_array_from_mapping(dict(data_file.items()))
        elif suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported skeleton file format: {path}")
        return self._coerce_skeleton_array(data)

    def _extract_array_from_mapping(self, data: Dict[str, Any]) -> Any:
        for key in ("keypoints", "pose", "skeleton", "joints", "data", "arr_0"):
            if key in data:
                return data[key]
        if data:
            return next(iter(data.values()))
        raise ValueError("Skeleton file does not contain any arrays")

    def _coerce_skeleton_array(self, data: Any) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if isinstance(data, dict):
            data = self._extract_array_from_mapping(data)
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            data = data.item()
            if isinstance(data, dict):
                data = self._extract_array_from_mapping(data)
        array = np.asarray(data, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        if array.ndim == 4:
            array = self._collapse_person_dimension(array)

        if array.ndim != 3:
            raise ValueError(
                "Skeleton array must be [T, J, C] or a common 4D single/multi-person layout, "
                f"got shape {tuple(array.shape)}"
            )

        if array.shape[-1] in (2, 3, 4):
            sequence = array
        elif array.shape[0] in (2, 3, 4):
            sequence = np.transpose(array, (1, 2, 0))
        elif array.shape[1] in (2, 3, 4):
            sequence = np.transpose(array, (2, 0, 1))
        else:
            raise ValueError(f"Cannot infer skeleton coordinate axis from shape {tuple(array.shape)}")
        return sequence.astype(np.float32, copy=False)

    def _collapse_person_dimension(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-1] not in (2, 3, 4):
            raise ValueError(f"Unsupported 4D skeleton shape {tuple(array.shape)}")

        if array.shape[1] <= 4 and array.shape[2] >= 5:
            return self._select_primary_person(array, person_axis=1)
        if array.shape[0] <= 4 and array.shape[2] >= 5:
            return self._select_primary_person(array, person_axis=0)
        raise ValueError(f"Cannot infer person axis from skeleton shape {tuple(array.shape)}")

    def _select_primary_person(self, array: np.ndarray, person_axis: int) -> np.ndarray:
        moved = np.moveaxis(array, person_axis, 0)
        coord_dims = min(2, moved.shape[-1])
        scores = np.abs(moved[..., :coord_dims]).sum(axis=tuple(range(1, moved.ndim)))
        best_idx = int(np.argmax(scores))
        return moved[best_idx]

    def _sample_skeleton_sequence(
        self,
        skeleton: np.ndarray,
        indices: np.ndarray,
        reference_length: int,
    ) -> np.ndarray:
        num_steps = skeleton.shape[0]
        if num_steps <= 0:
            return self._empty_skeleton().numpy()
        if num_steps == len(indices):
            sampled = skeleton
        elif num_steps >= max(indices) + 1:
            sampled = skeleton[indices]
        elif reference_length > 1:
            scaled = np.round(indices.astype(np.float32) / max(reference_length - 1, 1) * (num_steps - 1))
            sampled = skeleton[scaled.astype(np.int64)]
        else:
            sampled = np.repeat(skeleton[:1], repeats=self.seq_len, axis=0)
        if sampled.shape[0] != self.seq_len:
            timeline = np.linspace(0, sampled.shape[0] - 1, self.seq_len).round().astype(np.int64)
            sampled = sampled[timeline]
        return sampled.astype(np.float32, copy=False)

    def _normalize_skeleton_sequence(self, skeleton: np.ndarray) -> np.ndarray:
        skeleton = np.nan_to_num(skeleton.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        skeleton = self._match_joint_count(skeleton)
        skeleton = self._match_feature_dim(skeleton)

        coord_dims = min(2, skeleton.shape[-1])
        if coord_dims > 0:
            coords = skeleton[..., :coord_dims]
            valid = np.any(np.abs(coords) > 1e-6, axis=-1)
            if valid.any():
                root = self._compute_root_centers(coords, valid)
                coords = coords - root[:, None, :]

                norms = np.linalg.norm(coords, axis=-1)
                norms = np.where(valid, norms, 0.0)
                scale = norms.max(axis=1, keepdims=True)
                scale = np.clip(scale, a_min=1e-6, a_max=None)
                coords = coords / scale[:, :, None]
                coords *= valid[..., None].astype(np.float32)
                skeleton[..., :coord_dims] = coords
        return skeleton

    def _match_joint_count(self, skeleton: np.ndarray) -> np.ndarray:
        num_joints = skeleton.shape[1]
        if num_joints == self.skeleton_num_joints:
            return skeleton
        if num_joints > self.skeleton_num_joints:
            return skeleton[:, :self.skeleton_num_joints]

        pad_shape = (skeleton.shape[0], self.skeleton_num_joints - num_joints, skeleton.shape[2])
        padding = np.zeros(pad_shape, dtype=skeleton.dtype)
        return np.concatenate([skeleton, padding], axis=1)

    def _match_feature_dim(self, skeleton: np.ndarray) -> np.ndarray:
        feature_dim = skeleton.shape[2]
        if feature_dim == self.skeleton_input_dim:
            return skeleton
        if feature_dim > self.skeleton_input_dim:
            return skeleton[:, :, :self.skeleton_input_dim]

        pad_shape = (skeleton.shape[0], skeleton.shape[1], self.skeleton_input_dim - feature_dim)
        padding = np.zeros(pad_shape, dtype=skeleton.dtype)
        return np.concatenate([skeleton, padding], axis=2)

    def _compute_root_centers(self, coords: np.ndarray, valid: np.ndarray) -> np.ndarray:
        root_indices = self._skeleton_root_indices()
        available_roots = [idx for idx in root_indices if 0 <= idx < coords.shape[1]]
        centers = np.zeros((coords.shape[0], coords.shape[2]), dtype=coords.dtype)

        if available_roots:
            root_coords = coords[:, available_roots]
            root_valid = valid[:, available_roots]
            denom = root_valid.sum(axis=1, keepdims=True).clip(min=1).astype(coords.dtype)
            centers = (root_coords * root_valid[..., None].astype(coords.dtype)).sum(axis=1) / denom
            fallback = ~root_valid.any(axis=1)
        else:
            fallback = np.ones(coords.shape[0], dtype=bool)

        if np.any(fallback):
            weighted = coords * valid[..., None].astype(coords.dtype)
            denom = valid.sum(axis=1, keepdims=True).clip(min=1).astype(coords.dtype)
            centers[fallback] = (weighted.sum(axis=1) / denom)[fallback]
        return centers

    def _skeleton_root_indices(self) -> Sequence[int]:
        if self.skeleton_layout == "coco17":
            return (11, 12)
        return (0,)

    def _transform_frames(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        frames = [
            TF.resize(frame, self.resize_size, interpolation=InterpolationMode.BILINEAR)
            for frame in frames
        ]

        if self.split == "train":
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                frames[0],
                scale=(0.75, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            frames = [
                TF.resized_crop(
                    frame,
                    i,
                    j,
                    h,
                    w,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                for frame in frames
            ]
            if self.enable_hflip and random.random() < 0.5:
                frames = [TF.hflip(frame) for frame in frames]
        else:
            frames = [TF.center_crop(frame, [self.image_size, self.image_size]) for frame in frames]

        mean, std = self._normalization_stats()
        return [TF.normalize(TF.to_tensor(frame), mean=mean, std=std) for frame in frames]

    def _normalization_stats(self):
        if self.normalize == "imagenet":
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.normalize == "none":
            return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        return [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
