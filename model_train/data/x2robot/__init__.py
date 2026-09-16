"""Generic normalized MINT fine-tuning dataset exposed from the x2robot path.

The canonical X2Robot camera poses are C2W, while the upstream MINT camera
loss consumes first-frame-rebased W2C extrinsics. Keep that conversion in the
data adapter so the model, forward path, and losses stay untouched.
"""

import numpy as np
import torch

from data.x2robot.mint_dataset import MyDataDataset, _check_rigid_transforms
from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat


_BASE_MY_DATA_INIT = MyDataDataset.__init__


def _init_with_bimanual_default(self, cfg: dict):
    """Keep both hands unless a caller explicitly requests target-hand-only mode."""
    cfg = dict(cfg)
    cfg.setdefault("target_hand_only", False)
    _BASE_MY_DATA_INIT(self, cfg)


def _encode_pose_as_upstream_w2c(
    self,
    T_world_view_virtual: np.ndarray,
    cam_fov: torch.Tensor,
    sample_id: str,
) -> torch.Tensor:
    """Encode canonical virtual-camera C2W with the upstream MINT W2C convention."""
    c2w = np.asarray(T_world_view_virtual, dtype=np.float64)
    if c2w.shape != (self.clip_len, 4, 4):
        raise RuntimeError(
            f"sample {sample_id}: T_world_view_virtual must be "
            f"[{self.clip_len},4,4], got {c2w.shape}"
        )

    # Official MINT starts from W2C extrinsics E_t=[R_t|T_t] and rebases with
    # R'_t = R_t R_0^T, T'_t = T_t - R'_t T_0. Because C_t = inv(E_t), the
    # exact same transform is E_t @ inv(E_0) = inv(C_t) @ C_0.
    w2c = np.linalg.inv(c2w)
    T_rel_w2c = w2c @ c2w[0][None]

    # Cross-check against the author's explicit algebra to prevent another
    # C2W/W2C direction regression.
    rotation_w2c = w2c[:, :3, :3]
    translation_w2c = w2c[:, :3, 3]
    rebased_rotation = rotation_w2c @ rotation_w2c[0].T
    rebased_translation = translation_w2c - np.einsum(
        "sij,j->si", rebased_rotation, translation_w2c[0]
    )
    author_rel = np.zeros_like(T_rel_w2c)
    author_rel[:, 3, 3] = 1.0
    author_rel[:, :3, :3] = rebased_rotation
    author_rel[:, :3, 3] = rebased_translation
    if not np.allclose(T_rel_w2c, author_rel, atol=1.0e-9, rtol=0.0):
        raise RuntimeError(
            f"sample {sample_id}: C2W-to-upstream-W2C rebase equivalence failed"
        )

    _check_rigid_transforms(
        T_rel_w2c,
        name=f"sample {sample_id}: rebased W2C T_rel",
        atol=self.pose_rigid_tolerance,
    )

    rotation = torch.from_numpy(T_rel_w2c[:, :3, :3].astype(np.float32))
    with torch.no_grad():
        quaternion = mat_to_quat(rotation)
        decoded_rotation = quat_to_mat(quaternion)
    if not torch.allclose(
        decoded_rotation,
        rotation,
        atol=self.pose_roundtrip_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(f"sample {sample_id}: pose quaternion round-trip failed")

    translation = torch.from_numpy(T_rel_w2c[:, :3, 3].astype(np.float32))
    fov = cam_fov.reshape(1, 2).expand(self.clip_len, 2)
    gt_pose_enc = torch.cat((translation, quaternion.float(), fov), dim=-1)
    if gt_pose_enc.shape != (self.clip_len, 9):
        raise RuntimeError(
            f"sample {sample_id}: gt_pose_enc has invalid shape {tuple(gt_pose_enc.shape)}"
        )

    if not torch.allclose(
        gt_pose_enc[0, :3], torch.zeros(3), atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError(f"sample {sample_id}: first rebased translation is not zero")
    if not torch.allclose(
        quat_to_mat(gt_pose_enc[0:1, 3:7])[0],
        torch.eye(3),
        atol=self.pose_roundtrip_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(f"sample {sample_id}: first rebased rotation is not identity")
    return gt_pose_enc.contiguous()


# Patch the registered dataset class itself. Importing data.x2robot.mint_dataset
# initializes this package first, so the registry and direct imports both observe
# the same corrected class object.
MyDataDataset.__init__ = _init_with_bimanual_default
MyDataDataset._encode_pose = _encode_pose_as_upstream_w2c

__all__ = ["MyDataDataset"]
