"""Grasp pose generation from an OBB-based object pose in the base frame.

Conventions
-----------
We emit a right-handed grasp frame ``G`` such that:
  - ``z_G`` (approach axis) points in the approach direction (default world -z,
    i.e. top-down).
  - ``y_G`` (closing axis) is aligned with the OBB shortest in-plane axis so
    the fingers cross the narrowest dimension of the object.
  - ``x_G = y_G x z_G`` completes the frame.

The final 4x4 ``T_B_grasp`` is then sent to the arm as the target end-effector
pose at the pick point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class GraspCandidate:
    """Parameters describing a single top-down parallel-jaw grasp.

    Two distinct jaw openings are carried here on purpose:

    * ``opening_m`` is the pre-grasp clearance: wider than the object by the
      configured ``opening_margin_m`` so the fingers can descend past the
      object without clipping it.
    * ``close_opening_m`` is the final commanded opening when closing on the
      object. It must be *smaller* than the object's short axis so the jaws
      actually make contact. Setting it equal to ``opening_m`` (minus a
      small delta) leaves the fingers in mid-air and produces no grip.
    """

    T_B_grasp: np.ndarray           # 4x4 transform: grasp frame in base frame
    opening_m: float                # pre-grasp jaw opening (clearance wider than object)
    close_opening_m: float          # target opening when closing to grip the object
    object_width_m: float           # object short-axis width used for closing
    approach_m: float               # pre-grasp clearance distance
    lift_m: float                   # post-grasp lift height
    score: float = 1.0              # higher is better (pluggable)
    feasible: bool = True
    reason: str = "ok"

    @property
    def pre_grasp(self) -> np.ndarray:
        """Pose offset from the grasp along -z_grasp by ``approach_m``."""
        T = np.eye(4)
        T[2, 3] = -self.approach_m
        return self.T_B_grasp @ T

    @property
    def lift(self) -> np.ndarray:
        """Pose offset from the grasp along +world_z by ``lift_m``."""
        T = self.T_B_grasp.copy()
        T[2, 3] += self.lift_m
        return T


def _safe_cross_basis(y_raw: np.ndarray, z_grasp: np.ndarray) -> np.ndarray:
    """Return a right-handed (x, y, z) triple given desired y_raw and z_grasp.

    ``y_raw`` is projected to be orthogonal to ``z_grasp`` before building the
    frame. Raises ``ValueError`` if the two are near-parallel.
    """
    z = z_grasp / (np.linalg.norm(z_grasp) + 1e-12)
    y = y_raw - (y_raw @ z) * z
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        raise ValueError(
            "closing axis is nearly parallel to approach axis; "
            "pick a different candidate axis"
        )
    y /= ny
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-12
    return np.column_stack([x, y, z])


def plan_topdown_grasp(
    T_B_O: np.ndarray,
    extent: Sequence[float],
    approach_world: Sequence[float] = (0.0, 0.0, -1.0),
    opening_margin_m: float = 0.005,
    close_overclose_m: float = 0.002,
    approach_m: float = 0.10,
    lift_m: float = 0.12,
    max_opening_m: float = 0.070,
    center_height_offset_m: float = 0.0,
) -> GraspCandidate:
    """Build a top-down grasp from an OBB-derived object pose.

    Parameters
    ----------
    T_B_O:
        Object pose in the base frame. Columns of its rotation are the OBB
        axes, ordered longest to shortest (as emitted by
        :func:`perception.pose_estimator.estimate_pose_from_obb`).
    extent:
        OBB side lengths matching the rotation columns ordering (meters).
    approach_world:
        Desired approach direction in the base frame. Default ``(0, 0, -1)``
        is top-down.
    opening_margin_m:
        Extra width added to the object short axis when pre-opening the
        jaws; keeps the fingers from clipping the object during descent.
    close_overclose_m:
        How much *narrower* than the object short axis the jaws should
        close to when gripping. Positive values produce a squeeze; 0 just
        touches the surface. Must be less than the object width or the
        commanded opening would be negative (and will be clamped to 0).
    center_height_offset_m:
        Raise or lower the pick point along world z relative to the OBB center.
        Positive values grasp higher on tall objects.
    """
    T_B_O = np.asarray(T_B_O, dtype=np.float64)
    R_O = T_B_O[:3, :3]
    center = T_B_O[:3, 3].copy()
    ext = np.asarray(extent, dtype=np.float64).reshape(3)

    # Column 0 = longest axis, column 1 = middle, column 2 = shortest.
    # Pick whichever axis is most horizontal as the closing direction:
    # ideal for a parallel jaw gripper, and robust to the OBB short axis
    # pointing vertically on thin objects (e.g. a lying-down cylinder).
    z_grasp = np.asarray(approach_world, dtype=np.float64)
    z_grasp /= np.linalg.norm(z_grasp) + 1e-12

    # Score each OBB axis by horizontality (1 - |axis . approach|) * (1 / extent)
    # so we prefer short, horizontal axes as the closing direction.
    best_idx = 0
    best_score = -np.inf
    for i in range(3):
        axis = R_O[:, i]
        horizontal = 1.0 - abs(axis @ z_grasp)
        tightness = 1.0 / max(ext[i], 1e-4)
        score = horizontal * tightness
        if score > best_score:
            best_score = score
            best_idx = i

    y_raw = R_O[:, best_idx]
    try:
        R_grasp = _safe_cross_basis(y_raw, z_grasp)
    except ValueError as exc:
        # Fall back to using the second-best axis.
        alt_idx = int(np.argsort(ext)[1])
        if alt_idx == best_idx:
            alt_idx = (best_idx + 1) % 3
        R_grasp = _safe_cross_basis(R_O[:, alt_idx], z_grasp)
        best_idx = alt_idx

    pick_point = center + np.array([0.0, 0.0, center_height_offset_m])

    T_B_grasp = np.eye(4)
    T_B_grasp[:3, :3] = R_grasp
    T_B_grasp[:3, 3] = pick_point

    opening = float(ext[best_idx]) + float(opening_margin_m)
    object_width = float(ext[best_idx])
    # Close target = squeeze past the surface of the object by
    # ``close_overclose_m``. Without this, the jaws stop a few mm short
    # of contact and the pick fails silently (no force, no grip).
    close_opening = max(0.0, object_width - float(close_overclose_m))
    feasible = True
    reason = "ok"
    if opening > float(max_opening_m):
        feasible = False
        reason = (
            f"object short axis {ext[best_idx]*1000:.1f} mm (+margin "
            f"{opening_margin_m*1000:.1f} mm) exceeds max gripper opening "
            f"{max_opening_m*1000:.1f} mm"
        )
        # Keep the *real* requested opening in the candidate so the caller
        # can see the magnitude of the violation. Only clamp it when the
        # plan is still feasible, to ensure a safe command.
    else:
        opening = min(opening, float(max_opening_m))

    return GraspCandidate(
        T_B_grasp=T_B_grasp,
        opening_m=opening,
        close_opening_m=close_opening,
        object_width_m=object_width,
        approach_m=float(approach_m),
        lift_m=float(lift_m),
        feasible=feasible,
        reason=reason,
    )


def align_obb_axes(
    R_ref: np.ndarray,
    extent_ref: Sequence[float],
    R_src: np.ndarray,
    extent_src: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Permute + sign-flip the columns of ``R_src`` to align with ``R_ref``.

    OBB axes from PCA are only defined up to a signed permutation. If you
    blend a rough and a close-up OBB without first matching which axis is
    which (and which direction), you end up with a rotation that no longer
    corresponds to the side-length vector. The planner then scores axes
    against the wrong extents and picks a bad closing direction.

    We greedily match each column of ``R_ref`` to the column of ``R_src``
    with the largest absolute dot product; the sign is chosen to make the
    dot product positive, and the corresponding entry of ``extent_src`` is
    moved along with its column. A final sign flip on column 2 restores
    right-handedness if needed.

    Returns
    -------
    (R_aligned, extent_aligned)
        ``R_aligned`` is a proper rotation (det = +1) whose column *i*
        points in roughly the same direction as ``R_ref[:, i]``.
        ``extent_aligned`` is the permuted side lengths matching those
        columns.
    """
    R_ref = np.asarray(R_ref, dtype=np.float64)
    R_src = np.asarray(R_src, dtype=np.float64)
    ext_src = np.asarray(extent_src, dtype=np.float64).reshape(3)

    dots = R_ref.T @ R_src  # (3, 3): row i = ref axis i dotted with each src axis
    remaining = [0, 1, 2]
    perm = [0, 0, 0]
    signs = [1.0, 1.0, 1.0]
    for i in range(3):
        # best src axis for ref axis i, among those not yet assigned
        best_j = max(remaining, key=lambda j: abs(dots[i, j]))
        perm[i] = best_j
        signs[i] = float(np.sign(dots[i, best_j])) or 1.0
        remaining.remove(best_j)

    R_out = R_src[:, perm] * np.asarray(signs).reshape(1, 3)
    if np.linalg.det(R_out) < 0:
        # Flip the axis with the weakest agreement to restore right-handedness.
        weakest = int(np.argmin([abs(dots[i, perm[i]]) for i in range(3)]))
        R_out[:, weakest] = -R_out[:, weakest]
    ext_out = ext_src[perm]
    return R_out, ext_out


def fuse_poses(
    T_rough: np.ndarray,
    T_fine: np.ndarray,
    weight_rough: float = 0.3,
    weight_fine: float = 0.7,
    rotation_disagreement_deg: float = 25.0,
) -> np.ndarray:
    """Weighted average of two 4x4 poses.

    Translation is linearly blended; rotation is blended via quaternion slerp
    with the fine pose weight as interpolation ``t``.

    OBB-derived rotations are ambiguous: when the object has two close
    extents the PCA axes from the rough and close-up views may differ by
    ~90 deg, in which case slerp produces a meaningless frame. If the
    angular distance between the two rotations exceeds
    ``rotation_disagreement_deg`` we drop the rough rotation and keep the
    fine one verbatim (the close-up view is assumed to be more reliable).
    """
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    w = float(weight_fine) / float(max(weight_rough + weight_fine, 1e-9))

    t_rough = T_rough[:3, 3]
    t_fine = T_fine[:3, 3]
    t_fused = (1.0 - w) * t_rough + w * t_fine

    r0 = R.from_matrix(T_rough[:3, :3])
    r1 = R.from_matrix(T_fine[:3, :3])

    # Angular distance between the two rotations, in degrees.
    rel = r1 * r0.inv()
    angle_deg = float(np.rad2deg(np.linalg.norm(rel.as_rotvec())))

    if angle_deg > rotation_disagreement_deg:
        r_fused = r1
    else:
        slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
        r_fused = slerp([w])[0]

    T = np.eye(4)
    T[:3, :3] = r_fused.as_matrix()
    T[:3, 3] = t_fused
    return T
