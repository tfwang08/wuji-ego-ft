#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic calibrated MINT fine-tuning dataset.

Expected layout::

    <root>/splits/train.jsonl
    <root>/episodes/.../video_<view>_virtual.mp4
    <root>/episodes/.../labels_<view>.npz

Each manifest row describes one fixed-length, single-view, single-hand window.
The loader preserves the upstream MINT batch keys and also returns calibrated
2D/QC fields for validation and later 2D losses.
"""

from collections import OrderedDict
import json
import math
import os
from typing import Dict, Iterable, List

import numpy as np
import torch

from core.registry import DATASETS
from data.base_dataset import BaseClipDataset


_REQUIRED_META = {
    "sample_id",
    "view",
    "hand_side_index",
    "frame_start",
    "clip_len",
    "video_path",
    "label_path",
    "label_row_start",
}

_REQUIRED_LABELS = {
    "frame_index",
    "timestamp",
    "hand_kept",
    "hand_gt",
    "kpt21_3d",
    "kpt21_3d_valid",
    "kpt21_2d",
    "kpt21_2d_valid",
    "quality_mask",
    "quality_score",
    "K_virtual",
    "image_size_wh",
}


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{lineno}: {error}") from error
            if not isinstance(item, dict):
                raise RuntimeError(f"manifest row must be an object at {path}:{lineno}")
            yield item


def _finite_or_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


@DATASETS.register("my_data")
class MyDataDataset(BaseClipDataset):
    """Read normalized virtual-camera clips for MINT hand fine-tuning."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.root = os.path.abspath(os.path.expanduser(str(cfg["root"])))
        self.split = str(cfg.get("split", "train"))
        self.target_hand_only = bool(cfg.get("target_hand_only", True))
        self.strict_target_window = bool(cfg.get("strict_target_window", True))
        self.author_projector_compatible = bool(
            cfg.get("author_projector_compatible", True)
        )
        self.principal_point_tolerance_px = float(
            cfg.get("principal_point_tolerance_px", 0.5)
        )
        self.intrinsics_tolerance = float(cfg.get("intrinsics_tolerance", 1.0e-6))
        if self.principal_point_tolerance_px < 0:
            raise ValueError("principal_point_tolerance_px must be non-negative")
        if self.intrinsics_tolerance < 0:
            raise ValueError("intrinsics_tolerance must be non-negative")

        quality_tiers = cfg.get("quality_tiers", ["gold"])
        self.quality_tiers = None if quality_tiers is None else {
            str(value) for value in quality_tiers
        }
        self.label_cache_size = max(0, int(cfg.get("label_cache_size", 2)))
        self.video_cache_size = max(0, int(cfg.get("video_cache_size", 2)))
        self._label_cache = OrderedDict()
        self._video_cache = OrderedDict()

        manifest_cfg = cfg.get("manifest")
        if manifest_cfg:
            self.manifest_path = self._resolve_path(str(manifest_cfg))
        else:
            split_path = os.path.join(self.root, "splits", f"{self.split}.jsonl")
            self.manifest_path = (
                split_path
                if os.path.isfile(split_path)
                else os.path.join(self.root, "manifest.jsonl")
            )
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"manifest does not exist: {self.manifest_path}")

        self.samples: List[dict] = []
        for item in _read_jsonl(self.manifest_path):
            if self.quality_tiers is not None:
                tier = str(item.get("quality_tier", ""))
                if tier not in self.quality_tiers:
                    continue
            self._validate_manifest_item(item)
            self.samples.append(item)

        if not self.samples:
            tiers = sorted(self.quality_tiers) if self.quality_tiers else None
            raise RuntimeError(
                f"no usable samples in {self.manifest_path}; quality_tiers={tiers}"
            )

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def _validate_manifest_item(self, item: dict) -> None:
        missing = sorted(_REQUIRED_META - set(item))
        if missing:
            raise RuntimeError(
                f"manifest sample {item.get('sample_id', '<unknown>')} missing fields: {missing}"
            )
        if int(item["clip_len"]) != self.clip_len:
            raise RuntimeError(
                f"sample {item['sample_id']} clip_len={item['clip_len']} "
                f"does not match data.clip_len={self.clip_len}"
            )
        if item["view"] not in {"left", "right"}:
            raise RuntimeError(
                f"sample {item['sample_id']} has invalid view={item['view']!r}"
            )
        side = int(item["hand_side_index"])
        if side not in (0, 1):
            raise RuntimeError(
                f"sample {item['sample_id']} has invalid hand_side_index={side}"
            )
        if int(item["frame_start"]) < 0 or int(item["label_row_start"]) < 0:
            raise RuntimeError(f"sample {item['sample_id']} has negative frame/label start")
        image_hw = item.get("image_hw")
        if image_hw is not None and tuple(int(v) for v in image_hw) != self.size_hw:
            raise RuntimeError(
                f"sample {item['sample_id']} image_hw={image_hw} does not match "
                f"data.size_hw={self.size_hw}"
            )

    def _load_labels(self, path: str) -> Dict[str, np.ndarray]:
        cached = self._label_cache.pop(path, None)
        if cached is not None:
            self._label_cache[path] = cached
            return cached

        if not os.path.isfile(path):
            raise FileNotFoundError(f"label file does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(_REQUIRED_LABELS - set(archive.files))
            if missing:
                raise RuntimeError(f"label file {path} missing fields: {missing}")
            labels = {name: archive[name] for name in archive.files}

        if self.label_cache_size > 0:
            self._label_cache[path] = labels
            while len(self._label_cache) > self.label_cache_size:
                self._label_cache.popitem(last=False)
        return labels

    def _video_reader(self, path: str):
        cached = self._video_cache.pop(path, None)
        if cached is not None:
            self._video_cache[path] = cached
            return cached

        if not os.path.isfile(path):
            raise FileNotFoundError(f"video file does not exist: {path}")
        from decord import VideoReader

        reader = VideoReader(path, num_threads=1)
        if self.video_cache_size > 0:
            self._video_cache[path] = reader
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return reader

    def __len__(self):
        return len(self.samples)

    def _camera_fields(self, labels: Dict[str, np.ndarray], sample_id: str):
        height, width = self.size_hw
        image_size_wh = np.asarray(labels["image_size_wh"]).reshape(-1)
        if image_size_wh.shape != (2,):
            raise RuntimeError(
                f"sample {sample_id}: image_size_wh must have shape [2], "
                f"got {labels['image_size_wh'].shape}"
            )
        label_width, label_height = (int(v) for v in image_size_wh)
        if (label_height, label_width) != (height, width):
            raise RuntimeError(
                f"sample {sample_id}: labels describe {(label_height, label_width)} but "
                f"data.size_hw={(height, width)}"
            )

        K = np.asarray(labels["K_virtual"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise RuntimeError(
                f"sample {sample_id}: K_virtual must be finite [3,3], got {K.shape}"
            )
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(f"sample {sample_id}: K_virtual has non-positive focal length")

        if self.author_projector_compatible:
            matrix_tol = self.intrinsics_tolerance
            if (
                abs(float(K[0, 1])) > matrix_tol
                or abs(float(K[1, 0])) > matrix_tol
                or abs(float(K[2, 0])) > matrix_tol
                or abs(float(K[2, 1])) > matrix_tol
                or abs(float(K[2, 2]) - 1.0) > matrix_tol
            ):
                raise RuntimeError(
                    f"sample {sample_id}: K_virtual contains skew/non-canonical "
                    "homogeneous terms that upstream MINT FoV projection cannot represent"
                )
            expected_cx, expected_cy = width / 2.0, height / 2.0
            tol = self.principal_point_tolerance_px
            if abs(cx - expected_cx) > tol or abs(cy - expected_cy) > tol:
                raise RuntimeError(
                    f"sample {sample_id}: K_virtual principal point ({cx:.4f}, {cy:.4f}) "
                    f"is incompatible with upstream MINT projector ({expected_cx:.4f}, "
                    f"{expected_cy:.4f}); tolerance={tol}px"
                )

        vertical_fov = 2.0 * math.atan(height / (2.0 * fy))
        horizontal_fov = 2.0 * math.atan(width / (2.0 * fx))
        cam_fov = torch.tensor([vertical_fov, horizontal_fov], dtype=torch.float32)

        # Compatibility field for upstream losses/tools. Camera trajectory itself
        # is not supervised by this dataset; only calibrated FoV is meaningful.
        gt_pose_enc = torch.zeros(self.clip_len, 9, dtype=torch.float32)
        gt_pose_enc[:, 6] = 1.0  # identity quaternion in xyzw convention
        gt_pose_enc[:, 7:9] = cam_fov
        return torch.from_numpy(K.astype(np.float32)), cam_fov, gt_pose_enc

    def __getitem__(self, idx):
        meta = self.samples[idx]
        sample_id = str(meta["sample_id"])
        side = int(meta["hand_side_index"])
        frame_start = int(meta["frame_start"])
        label_start = int(meta["label_row_start"])
        label_stop = label_start + self.clip_len

        label_path = self._resolve_path(str(meta["label_path"]))
        video_path = self._resolve_path(str(meta["video_path"]))
        labels = self._load_labels(label_path)

        frame_index_all = np.asarray(labels["frame_index"])
        if label_stop > len(frame_index_all):
            raise RuntimeError(
                f"sample {sample_id}: label rows [{label_start},{label_stop}) exceed "
                f"{len(frame_index_all)} rows"
            )
        rows = slice(label_start, label_stop)
        frame_index = frame_index_all[rows].astype(np.int64, copy=False)
        expected_frames = np.arange(
            frame_start, frame_start + self.clip_len, dtype=np.int64
        )
        if not np.array_equal(frame_index, expected_frames):
            raise RuntimeError(
                f"sample {sample_id}: label frame_index does not match video frame range; "
                f"labels={frame_index[[0, -1]].tolist()} expected="
                f"{expected_frames[[0, -1]].tolist()}"
            )

        timestamp = np.asarray(labels["timestamp"][rows], dtype=np.float64)
        if timestamp.shape != (self.clip_len,):
            raise RuntimeError(
                f"sample {sample_id}: timestamp must have shape [{self.clip_len}], "
                f"got {timestamp.shape}"
            )
        if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0.0):
            raise RuntimeError(f"sample {sample_id}: timestamp must be finite and increasing")

        hand_gt = np.asarray(labels["hand_gt"][rows], dtype=np.float32)
        hand_kept_raw = np.asarray(labels["hand_kept"][rows], dtype=bool)
        quality_mask = np.asarray(labels["quality_mask"][rows], dtype=bool)
        quality_score = np.asarray(labels["quality_score"][rows], dtype=np.float32)
        kpt21_3d = np.asarray(labels["kpt21_3d"][rows], dtype=np.float32)
        kpt21_3d_valid = np.asarray(labels["kpt21_3d_valid"][rows], dtype=bool)
        kpt21_2d = np.asarray(labels["kpt21_2d"][rows], dtype=np.float32)
        kpt21_2d_valid = np.asarray(labels["kpt21_2d_valid"][rows], dtype=bool)

        expected_shapes = {
            "hand_gt": (self.clip_len, 218),
            "hand_kept": (self.clip_len, 2),
            "quality_mask": (self.clip_len, 2),
            "quality_score": (self.clip_len, 2),
            "kpt21_3d": (self.clip_len, 2, 21, 3),
            "kpt21_3d_valid": (self.clip_len, 2, 21),
            "kpt21_2d": (self.clip_len, 2, 21, 2),
            "kpt21_2d_valid": (self.clip_len, 2, 21),
        }
        actual_shapes = {
            "hand_gt": hand_gt.shape,
            "hand_kept": hand_kept_raw.shape,
            "quality_mask": quality_mask.shape,
            "quality_score": quality_score.shape,
            "kpt21_3d": kpt21_3d.shape,
            "kpt21_3d_valid": kpt21_3d_valid.shape,
            "kpt21_2d": kpt21_2d.shape,
            "kpt21_2d_valid": kpt21_2d_valid.shape,
        }
        bad = {
            name: (actual_shapes[name], shape)
            for name, shape in expected_shapes.items()
            if actual_shapes[name] != shape
        }
        if bad:
            raise RuntimeError(f"sample {sample_id}: invalid label shapes: {bad}")

        hand_kept = hand_kept_raw & quality_mask
        if self.target_hand_only:
            side_mask = np.zeros((1, 2), dtype=bool)
            side_mask[0, side] = True
            hand_kept = hand_kept & side_mask
        if self.strict_target_window and not hand_kept[:, side].all():
            raise RuntimeError(
                f"sample {sample_id}: target hand {side} is not valid for every frame"
            )

        hand_gt_view = hand_gt.reshape(self.clip_len, 2, 109)
        finite_hand = np.isfinite(hand_gt_view).all(axis=-1)
        if np.any(hand_kept & ~finite_hand):
            raise RuntimeError(
                f"sample {sample_id}: supervised hand_gt contains non-finite values"
            )
        hand_gt = _finite_or_zero(hand_gt)

        finite_3d = np.isfinite(kpt21_3d).all(axis=-1)
        finite_2d = np.isfinite(kpt21_2d).all(axis=-1)
        kpt21_3d_valid = kpt21_3d_valid & finite_3d
        kpt21_2d_valid = kpt21_2d_valid & finite_2d
        kpt21_3d = _finite_or_zero(kpt21_3d)
        kpt21_2d = _finite_or_zero(kpt21_2d)
        quality_score = _finite_or_zero(quality_score)

        reader = self._video_reader(video_path)
        frame_stop = frame_start + self.clip_len
        if frame_stop > len(reader):
            raise RuntimeError(
                f"sample {sample_id}: video frames [{frame_start},{frame_stop}) exceed "
                f"video length {len(reader)}; frame clamping is intentionally disabled"
            )
        frames = reader.get_batch(list(range(frame_start, frame_stop))).asnumpy()
        expected_image_shape = (self.clip_len, self.size_hw[0], self.size_hw[1], 3)
        if tuple(frames.shape) != expected_image_shape:
            raise RuntimeError(
                f"sample {sample_id}: decoded frames have shape {tuple(frames.shape)}, "
                f"expected {expected_image_shape}; resize/remap must happen offline"
            )
        images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        images = images.contiguous()

        K_virtual, cam_fov, gt_pose_enc = self._camera_fields(labels, sample_id)

        required_joint_mask = np.broadcast_to(
            hand_kept[..., None], kpt21_3d_valid.shape
        )
        supervised_joint_mask = kpt21_3d_valid & required_joint_mask
        kpt21_gt_valid = bool(
            not required_joint_mask.any()
            or supervised_joint_mask[required_joint_mask].all()
        )
        mano_gt_valid = bool(hand_kept.any())

        fps = float(meta.get("fps", 30.0))
        if not math.isfinite(fps) or fps <= 0.0:
            raise RuntimeError(f"sample {sample_id}: invalid fps={fps}")

        return {
            # Upstream MINT-compatible fields.
            "images": images,
            "gt_pose_enc": gt_pose_enc,
            "state_mask": torch.zeros(2, dtype=torch.bool),
            "hand_gt": torch.from_numpy(hand_gt),
            "hand_kept": torch.from_numpy(hand_kept.copy()),
            "hand_valid": torch.tensor(mano_gt_valid, dtype=torch.bool),
            "mano_gt_valid": torch.tensor(mano_gt_valid, dtype=torch.bool),
            "kpt21_gt": torch.from_numpy(kpt21_3d),
            "kpt21_gt_valid": torch.tensor(kpt21_gt_valid, dtype=torch.bool),

            # Calibrated fields retained for exact 2D validation/losses.
            "K_virtual": K_virtual,
            "cam_fov": cam_fov,
            "hand_kept_raw": torch.from_numpy(hand_kept_raw.copy()),
            "quality_mask": torch.from_numpy(quality_mask.copy()),
            "quality_score": torch.from_numpy(quality_score),
            "kpt21_3d_valid": torch.from_numpy(kpt21_3d_valid.copy()),
            "kpt21_2d": torch.from_numpy(kpt21_2d),
            "kpt21_2d_valid": torch.from_numpy(kpt21_2d_valid.copy()),
            "frame_index": torch.from_numpy(frame_index.copy()),
            "timestamp": torch.from_numpy(timestamp.copy()),
            "fps": torch.tensor(fps, dtype=torch.float32),
            "sample_id": sample_id,
            "view": str(meta["view"]),
            "hand_side_index": side,
        }
