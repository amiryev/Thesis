from typing import Tuple

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _skew(v: Tensor) -> Tensor:
    """
    Construct a skew-symmetric matrix from a batch of vectors.

    Parameters
    ----------
    v : Tensor
        Shape (B, 3)

    Returns
    -------
    Tensor
        Shape (B, 3, 3)

            [  0  -z   y ]
        K = [  z   0  -x ]
            [ -y   x   0 ]
    """
    if v.ndim != 2 or v.shape[-1] != 3:
        raise ValueError(f"Expected (B,3), got {tuple(v.shape)}")

    B = v.shape[0]

    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    K = torch.zeros((B, 3, 3), dtype=v.dtype, device=v.device)

    K[:, 0, 1] = -z
    K[:, 0, 2] = y

    K[:, 1, 0] = z
    K[:, 1, 2] = -x

    K[:, 2, 0] = -y
    K[:, 2, 1] = x

    return K


# -----------------------------------------------------------------------------
# SO(3) exponential map
# -----------------------------------------------------------------------------

def so3_exp(v: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Convert an axis-angle vector to a rotation matrix using
    Rodrigues' exponential map.

    Parameters
    ----------
    v : Tensor
        (B, 3) axis-angle vectors (radians)

    eps : float
        Threshold below which the identity matrix is returned to
        avoid numerical instability.

    Returns
    -------
    Tensor
        (B, 3, 3) rotation matrices

    Notes
    -----
    Rodrigues formula:

        theta = ||v||

        k = v / theta

        R = I
            + sin(theta) * K
            + (1 - cos(theta)) * K²

    where K is the skew-symmetric matrix of k.
    """

    if v.ndim != 2 or v.shape[-1] != 3:
        raise ValueError(f"Expected (B,3), got {tuple(v.shape)}")

    B = v.shape[0]
    device = v.device
    dtype = v.dtype

    theta = torch.linalg.norm(v, dim=1, keepdim=True)

    R = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

    valid = (theta.squeeze(1) > eps)

    if valid.any():
        theta_valid = theta[valid]
        axis = v[valid] / theta_valid
        K = _skew(axis)
        K2 = K @ K

        sin_theta = torch.sin(theta_valid).view(-1, 1, 1)
        cos_theta = torch.cos(theta_valid).view(-1, 1, 1)

        I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)

        R_valid = (I + sin_theta * K + (1.0 - cos_theta) * K2)
        R[valid] = R_valid

    return R


# -----------------------------------------------------------------------------
# Pose update
# -----------------------------------------------------------------------------

def apply_delta(R_base: Tensor, t_base: Tensor, delta_rot: Tensor, delta_trans: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Apply a learnable optimization update to a fixed base pose.

    Parameters
    ----------
    R_base : Tensor
        (B,3,3) base rotation matrix

    t_base : Tensor
        (B,3) base translation (mm)

    delta_rot : Tensor
        (B,3) axis-angle correction (radians)

    delta_trans : Tensor
        (B,3) translation correction (mm)

    Returns
    -------
    R_current : Tensor
        (B,3,3)

    t_current : Tensor
        (B,3)

    Convention
    ----------
    Left multiplication:

        R_current = Exp(delta_rot) @ R_base

        t_current = t_base + delta_trans
    """

    if R_base.ndim != 3 or R_base.shape[-2:] != (3, 3):
        raise ValueError(f"Expected R_base of shape (B,3,3), got {tuple(R_base.shape)}")

    if t_base.ndim != 2 or t_base.shape[-1] != 3:
        raise ValueError(f"Expected t_base of shape (B,3), got {tuple(t_base.shape)}")

    R_delta = so3_exp(delta_rot)

    R_current = R_delta @ R_base
    t_current = t_base + delta_trans

    return R_current, t_current

# -----------------------------------------------------------------------------
# Pose composition
# -----------------------------------------------------------------------------

def compose_poses(R1: Tensor, t1: Tensor, R_delta: Tensor, t_delta: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compose a reference pose with a relative transform.

    This function is used during multi-view optimization to derive the pose of
    a dependent view from the optimized reference view.

    Convention
    ----------
    Left-multiplication in the world frame.

        R_composed = R_delta @ R1
        t_composed = R_delta @ t1 + t_delta

    Parameters
    ----------
    R1 : Tensor
        (B,3,3) reference rotation.

    t1 : Tensor
        (B,3) reference translation in mm.

    R_delta : Tensor
        (B,3,3) relative rotation.

    t_delta : Tensor
        (B,3) relative translation in mm.

    Returns
    -------
    Tuple[Tensor, Tensor]
        (R_composed, t_composed)
    """

    if R1.ndim != 3 or R1.shape[-2:] != (3, 3):
        raise ValueError(f"Expected R1 with shape (B,3,3), got {tuple(R1.shape)}")

    if R_delta.ndim != 3 or R_delta.shape[-2:] != (3, 3):
        raise ValueError(f"Expected R_delta with shape (B,3,3), got {tuple(R_delta.shape)}")

    if t1.ndim != 2 or t1.shape[-1] != 3:
        raise ValueError(f"Expected t1 with shape (B,3), got {tuple(t1.shape)}")

    if t_delta.ndim != 2 or t_delta.shape[-1] != 3:
        raise ValueError(f"Expected t_delta with shape (B,3), got {tuple(t_delta.shape)}")

    R_composed = R_delta @ R1
    t_composed = (torch.matmul(R_delta, t1.unsqueeze(-1)).squeeze(-1) + t_delta)

    return R_composed, t_composed


# -----------------------------------------------------------------------------
# Relative transform
# -----------------------------------------------------------------------------

def compute_relative_transform(R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the rigid transform that maps pose 1 to pose 2.

    The returned transform is the exact inverse of compose_poses():

        delta = compute_relative_transform(R1, t1, R2, t2)

        compose_poses(R1, t1, *delta)
            == (R2, t2)

    (up to floating-point precision).

    Parameters
    ----------
    R1 : Tensor
        (B,3,3) rotation of pose 1.

    t1 : Tensor
        (B,3) translation of pose 1.

    R2 : Tensor
        (B,3,3) rotation of pose 2.

    t2 : Tensor
        (B,3) translation of pose 2.

    Returns
    -------
    Tuple[Tensor, Tensor]
        (R_delta, t_delta)
    """

    if R1.ndim != 3 or R2.ndim != 3:
        raise ValueError("Rotation tensors must have shape (B,3,3).")

    if t1.ndim != 2 or t2.ndim != 2:
        raise ValueError("Translation tensors must have shape (B,3).")

    # Relative rotation
    R_delta = R2 @ R1.transpose(-1, -2)

    # Relative translation
    rotated_t1 = torch.matmul(R_delta, t1.unsqueeze(-1)).squeeze(-1)
    t_delta = t2 - rotated_t1

    return R_delta, t_delta

def clone_pose(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Clone a pose while preserving autograd.
    """
    return R.clone(), t.clone()

def rt_to_homogeneous(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of rotation matrices and translation vectors into homogeneous transformation matrices.
    
    Args:
        R (Tensor): Rotation matrices of shape (B, 3, 3)
        t (Tensor): Translation vectors of shape (B, 3)
        
    Returns:
        Tensor: Homogeneous transformation matrices of shape (B, 4, 4)
    """
    B = R.shape[0]
    
    # Create a batch of (4, 4) identity matrices
    extrinsic = torch.eye(4, dtype=R.dtype, device=R.device).unsqueeze(0).repeat(B, 1, 1)
    
    # Insert the rotation (B, 3, 3) and translation (B, 3, 1)
    extrinsic[:, :3, :3] = R
    extrinsic[:, :3, 3] = t
    
    return extrinsic
