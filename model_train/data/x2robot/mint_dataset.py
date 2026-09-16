#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic calibrated MINT fine-tuning dataset.

Expected layout::

    <root>/splits/train.jsonl
    <root>/episodes/.../video_<view>_virtual.mp4
    <root>/episodes/.../labels_<view>.npz
    <root>/episodes/.../episode_camera_pose.npz

Each manifest row describes one fixed-length, single-view, single-hand window.
The loader preserves the upstream MINT batch keys while adapting canonical
SLAM/VIO C2W poses into the current virtual-camera pose expected by MINT.
"""

from collections import OrderedDict
import json
import math
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch

from core.registry import DATASETS
from data.base_dataset import BaseClipDataset


_MODEL_TRAIN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDOR = os.path.join(_MODEL_TRAIN, "_vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat  # noqa: E402


_REQUIRED_META = {
    "sample_id",
    "view",
    "hand_side_index",
    "frame_start",
    "clip_len",
    "video_path",
    "label_path",
    "label_row_start",
    "slam_pose_path",
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
    "rectify_R",
}

_REQUIRED_POSE = {
    "T_world_wnl",
    "valid",
    "frame_ids",
    "frame_ts_ns",
    "query_ts_ns",
    "source",
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


def _scalar_text(value) -> str:
    array = np.asarray(value)
    if array.shape not in [(), (1,)]:
        raise RuntimeError(f"expected scalar string, got shape {array.shape}")
    item = array.reshape(-1)[0] if array.shape else array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _check_rigid_transforms(transforms: np.ndarray, *, name: str, atol: float) -> None:
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
        raise RuntimeError(f"{name} must have shape [F,4,4], got {transforms.shape}")
    if not np.isfinite(transforms).all():
        raise RuntimeError(f"{name} contains non-finite values")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(transforms[:, 3, :], expected_bottom, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} has non-canonical homogeneous rows")
    rotation = transforms[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    ortho = np.swapaxes(rotation, -1, -2) @ rotation
    if not np.allclose(ortho, identity, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} contains non-orthogonal rotations")
    determinant = np.linalg.det(rotation)
    if not np.allclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} contains rotations with determinant != +1")


@DATASETS.register("my_data")
class MyDataDataset(BaseClipDataset):
    """Read normalized virtual-camera clips for MINT fine-tuning."""

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
        self.pose_rigid_tolerance = float(cfg.get("pose_rigid_tolerance", 1.0e-5))
        self.pose_roundtrip_tolerance = float(
            cfg.get("pose_roundtrip_tolerance", 1.0e-5)
        )
        if self.principal_point_tolerance_px < 0:
            raise ValueError("principal_point_tolerance_px must be non-negative")
        if self.intrinsics_tolerance < 0:
            raise ValueError("intrinsics_tolerance must be non-negative")
        if self.pose_rigid_tolerance <= 0:
            raise ValueError("pose_rigid_tolerance must be positive")
        if self.pose_roundtrip_tolerance <= 0:
            raise ValueError("pose_roundtrip_tolerance must be positive")

        quality_tiers = cfg.get("quality_tiers", ["gold"])
        self.quality_tiers = None if quality_tiers is None else {
            str(value) for value in quality_tiers
        }
        self.label_cache_size = max(0, int(cfg.get("label_cache_size", 2)))
        self.video_cache_size = max(0, int(cfg.get("video_cache_size", 2)))
        self.pose_cache_size = max(0, int(cfg.get("pose_cache_size", 2)))
        self._label_cache = OrderedDict()
        self._video_cache = OrderedDict()
        self._pose_cache = OrderedDict()

        manifest_cfg = cfg.get("manifest")
        if manifest_cfg:
            self.manifest_path = self._resolve_path(str(manifest_cfg))
        else:
            split_path = os.path.join(self.root, "splits", f"{self.split}.jsonl")
            self.manifest_path = (
                split_path if os.path.isfile(split_path)
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
            raise RuntimeError(
                f"no usable samples in {self.manifest_path}; "
                f"quality_tiers={sorted(self.quality_tiers) if self.quality_tiers else None}"
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
        pose_scale = float(item.get("pose_translation_scale_to_m", 1.0))
        if not math.isfinite(pose_scale) or pose_scale <= 0.0:
            raise RuntimeError(
                f"sample {item['sample_id']} has invalid pose_translation_scale_to_m="
                f"{pose_scale}"
            )

    def _load_npz_cached(
        self,
        path: str,
        *,
        cache: OrderedDict,
        cache_size: int,
        required: set,
        kind: str,
    ) -> Dict[str, np.ndarray]:
        cached = cache.pop(path, None)
        if cached is not None:
            cache[path] = cached
            return cached
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{kind} file does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"{kind} file {path} missing fields: {missing}")
            result = {name: archive[name] for name in archive.files}
        if cache_size > 0:
            cache[path] = result
            while len(cache) > cache_size:
                cache.popitem(last=False)
        return result

    def _load_labels(self, path: str) -> Dict[str, np.ndarray]:
        return self._load_npz_cached(
            path,
            cache=self._label_cache,
            cache_size=self.label_cache_size,
            required=_REQUIRED_LABELS,
            kind="label",
        )

    def _load_pose(self, path: str) -> Dict[str, np.ndarray]:
        return self._load_npz_cached(
            path,
            cache=self._pose_cache,
            cache_size=self.pose_cache_size,
            required=_REQUIRED_POSE,
            kind="camera pose",
        )

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
        cam_fov = torch.tensor(
            [vertical_fov, horizontal_fov], dtype=torch.float32
        )
        return torch.from_numpy(K.astype(np.float32)), cam_fov

    def _pose_window(
        self,
        meta: dict,
        labels: Dict[str, np.ndarray],
        frame_start: int,
        sample_id: str,
    ):
        pose_path = self._resolve_path(str(meta["slam_pose_path"]))
        pose = self._load_pose(pose_path)
        pose_stop = frame_start + self.clip_len

        frame_ids_all = np.asarray(pose["frame_ids"], dtype=np.int64)
        if pose_stop > len(frame_ids_all):
            raise RuntimeError(
                f"sample {sample_id}: pose rows [{frame_start},{pose_stop}) exceed "
                f"{len(frame_ids_all)} rows"
            )
        rows = slice(frame_start, pose_stop)
        expected_frames = np.arange(frame_start, pose_stop, dtype=np.int64)
        frame_ids = frame_ids_all[rows]
        if not np.array_equal(frame_ids, expected_frames):
            raise RuntimeError(
                f"sample {sample_id}: pose frame_ids do not match video frame range"
            )

        T_world_wnl = np.asarray(pose["T_world_wnl"][rows], dtype=np.float64).copy()
        slam_valid = np.asarray(pose["valid"][rows], dtype=bool)
        frame_ts_ns = np.asarray(pose["frame_ts_ns"][rows], dtype=np.int64)
        query_ts_ns = np.asarray(pose["query_ts_ns"][rows], dtype=np.int64)
        source = _scalar_text(pose["source"])

        if T_world_wnl.shape != (self.clip_len, 4, 4):
            raise RuntimeError(
                f"sample {sample_id}: T_world_wnl must be "
                f"[{self.clip_len},4,4], got {T_world_wnl.shape}"
            )
        if slam_valid.shape != (self.clip_len,):
            raise RuntimeError(
                f"sample {sample_id}: slam valid must be [{self.clip_len}], "
                f"got {slam_valid.shape}"
            )
        if frame_ts_ns.shape != (self.clip_len,) or query_ts_ns.shape != (self.clip_len,):
            raise RuntimeError(
                f"sample {sample_id}: SLAM timestamps must be [{self.clip_len}]"
            )
        if np.any(np.diff(frame_ts_ns) <= 0):
            raise RuntimeError(f"sample {sample_id}: frame_ts_ns must be strictly increasing")

        pose_offset_ns = int(meta.get("pose_offset_ns", 0))
        if not np.array_equal(query_ts_ns, frame_ts_ns + pose_offset_ns):
            raise RuntimeError(
                f"sample {sample_id}: query_ts_ns != frame_ts_ns + pose_offset_ns"
            )

        scale_to_m = float(meta.get("pose_translation_scale_to_m", 1.0))
        T_world_wnl[:, :3, 3] *= scale_to_m
        _check_rigid_transforms(
            T_world_wnl,
            name=f"sample {sample_id}: T_world_wnl",
            atol=self.pose_rigid_tolerance,
        )

        if source == "fixed_camera":
            identity = np.eye(4, dtype=np.float64)
            if not np.all(slam_valid):
                raise RuntimeError(
                    f"sample {sample_id}: fixed_camera pose contains invalid frames"
                )
            if not np.allclose(
                T_world_wnl, identity, atol=self.pose_rigid_tolerance, rtol=0.0
            ):
                raise RuntimeError(
                    f"sample {sample_id}: fixed_camera pose is not identity"
                )

        if not np.all(slam_valid):
            bad = np.flatnonzero(~slam_valid).tolist()
            raise RuntimeError(
                f"sample {sample_id}: SLAM invalid inside training window at offsets {bad}; "
                "the upstream camera loss has no per-frame mask, so invalid-pose windows "
                "must be excluded by the manifest"
            )

        right_offset_ns = int(meta.get("right_offset_ns", 0))
        precomputed = labels.get("T_world_view_virtual")
        if precomputed is not None:
            label_start = int(meta["label_row_start"])
            label_stop = label_start + self.clip_len
            T_world_view_virtual_saved = np.asarray(
                precomputed[label_start:label_stop], dtype=np.float64
            )
            _check_rigid_transforms(
                T_world_view_virtual_saved,
                name=f"sample {sample_id}: stored T_world_view_virtual",
                atol=self.pose_rigid_tolerance,
            )
        else:
            T_world_view_virtual_saved = None

        view = str(meta["view"])
        rectify_R = np.asarray(labels["rectify_R"], dtype=np.float64)
        if rectify_R.shape != (3, 3) or not np.isfinite(rectify_R).all():
            raise RuntimeError(
                f"sample {sample_id}: rectify_R must be finite [3,3], got "
                f"{rectify_R.shape}"
            )
        if not np.allclose(
            rectify_R.T @ rectify_R,
            np.eye(3),
            atol=self.pose_rigid_tolerance,
            rtol=0.0,
        ) or not math.isclose(
            float(np.linalg.det(rectify_R)),
            1.0,
            abs_tol=self.pose_rigid_tolerance,
        ):
            raise RuntimeError(f"sample {sample_id}: rectify_R is not a proper rotation")

        A_virtual = np.eye(4, dtype=np.float64)
        A_virtual[:3, :3] = rectify_R.T

        if view == "left":
            view_raw_from_left = np.eye(4, dtype=np.float64)
        else:
            if right_offset_ns != 0 and T_world_view_virtual_saved is None:
                raise RuntimeError(
                    f"sample {sample_id}: right_offset_ns={right_offset_ns} requires "
                    "precomputed T_world_view_virtual; the DataLoader will not guess "
                    "right-eye pose timing"
                )
            if "T_right_from_left" not in labels:
                raise RuntimeError(
                    f"sample {sample_id}: right-view labels require T_right_from_left"
                )
            T_right_from_left = np.asarray(
                labels["T_right_from_left"], dtype=np.float64
            )
            if T_right_from_left.shape != (4, 4):
                raise RuntimeError(
                    f"sample {sample_id}: T_right_from_left must be [4,4]"
                )
            _check_rigid_transforms(
                T_right_from_left[None],
                name=f"sample {sample_id}: T_right_from_left",
                atol=self.pose_rigid_tolerance,
            )
            view_raw_from_left = np.linalg.inv(T_right_from_left)

        T_world_view_virtual_computed = (
            T_world_wnl @ view_raw_from_left[None] @ A_virtual[None]
        )
        _check_rigid_transforms(
            T_world_view_virtual_computed,
            name=f"sample {sample_id}: computed T_world_view_virtual",
            atol=self.pose_rigid_tolerance,
        )

        if T_world_view_virtual_saved is not None:
            if right_offset_ns == 0 and not np.allclose(
                T_world_view_virtual_saved,
                T_world_view_virtual_computed,
                atol=5.0 * self.pose_rigid_tolerance,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"sample {sample_id}: stored T_world_view_virtual disagrees with "
                    "SLAM + rig extrinsics + rectify_R"
                )
            T_world_view_virtual = T_world_view_virtual_saved
        else:
            T_world_view_virtual = T_world_view_virtual_computed

        return {
            "T_world_wnl": T_world_wnl,
            "T_world_view_virtual": T_world_view_virtual,
            "slam_valid": slam_valid,
            "frame_ts_ns": frame_ts_ns,
            "query_ts_ns": query_ts_ns,
            "source": source,
            "view_raw_from_left": view_raw_from_left,
        }

    def _encode_pose(
        self,
        T_world_view_virtual: np.ndarray,
        cam_fov: torch.Tensor,
        sample_id: str,
    ) -> torch.Tensor:
        T0_inv = np.linalg.inv(T_world_view_virtual[0])
        T_rel = T0_inv[None] @ T_world_view_virtual
        _check_rigid_transforms(
            T_rel,
            name=f"sample {sample_id}: rebased T_rel",
            atol=self.pose_rigid_tolerance,
        )

        rotation = torch.from_numpy(T_rel[:, :3, :3].astype(np.float32))
        with torch.no_grad():
            quaternion = mat_to_quat(rotation)
            decoded_rotation = quat_to_mat(quaternion)
        if not torch.allclose(
            decoded_rotation,
            rotation,
            atol=self.pose_roundtrip_tolerance,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"sample {sample_id}: pose quaternion round-trip failed"
            )

        translation = torch.from_numpy(T_rel[:, :3, 3].astype(np.float32))
        fov = cam_fov.reshape(1, 2).expand(self.clip_len, 2)
        gt_pose_enc = torch.cat((translation, quaternion.float(), fov), dim=-1)
        if gt_pose_enc.shape != (self.clip_len, 9):
            raise RuntimeError(
                f"sample {sample_id}: gt_pose_enc has invalid shape "
                f"{tuple(gt_pose_enc.shape)}"
            )
        return gt_pose_enc.contiguous()

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
        actual = {
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
            key: (actual[key], shape)
            for key, shape in expected_shapes.items()
            if actual[key] != shape
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
        valid_hand_values = np.isfinite(hand_gt_view).all(axis=-1)
        if np.any(hand_kept & ~valid_hand_values):
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

        K_virtual, cam_fov = self._camera_fields(labels, sample_id)
        pose_window = self._pose_window(meta, labels, frame_start, sample_id)
        gt_pose_enc = self._encode_pose(
            pose_window["T_world_view_virtual"], cam_fov, sample_id
        )

        supervised_joint_mask = kpt21_3d_valid & hand_kept[..., None]
        required_joint_mask = np.broadcast_to(
            hand_kept[..., None], kpt21_3d_valid.shape
        )
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

            # Calibrated/QC fields retained for validation and future tooling.
            "K_virtual": K_virtual,
            "view_K_virtual": K_virtual.clone(),
            "cam_fov": cam_fov,
            "hand_kept_raw": torch.from_numpy(hand_kept_raw.copy()),
            "quality_mask": torch.from_numpy(quality_mask.copy()),
            "quality_score": torch.from_numpy(quality_score),
            "kpt21_3d_valid": torch.from_numpy(kpt21_3d_valid.copy()),
            "kpt21_2d": torch.from_numpy(kpt21_2d),
            "kpt21_2d_valid": torch.from_numpy(kpt21_2d_valid.copy()),
            "frame_index": torch.from_numpy(frame_index.copy()),
            "timestamp": torch.from_numpy(timestamp.copy()),
            "slam_T_world_from_wnl": torch.from_numpy(
                pose_window["T_world_wnl"].astype(np.float32)
            ),
            "T_world_view_virtual": torch.from_numpy(
                pose_window["T_world_view_virtual"].astype(np.float32)
            ),
            "slam_valid": torch.from_numpy(pose_window["slam_valid"].copy()),
            "camera_loss_mask": torch.from_numpy(pose_window["slam_valid"].copy()),
            "frame_ts_ns": torch.from_numpy(pose_window["frame_ts_ns"].copy()),
            "query_ts_ns": torch.from_numpy(pose_window["query_ts_ns"].copy()),
            "view_raw_from_left": torch.from_numpy(
                pose_window["view_raw_from_left"].astype(np.float32)
            ),
            "camera_pose_source": pose_window["source"],
            "fps": torch.tensor(fps, dtype=torch.float32),
            "sample_id": sample_id,
            "view": str(meta["view"]),
            "hand_side_index": side,
        }
