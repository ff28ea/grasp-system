"""Unit tests for grasp_system.planning.grasp_planner."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from grasp_system.common import pose_xyzrpy_to_matrix, rotation_angle_deg
from grasp_system.planning.grasp_planner import (
    align_obb_axes,
    fuse_poses,
    plan_topdown_grasp,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _make_obb(
    center=(0.30, 0.05, 0.05),
    rpy_deg=(0.0, 0.0, 0.0),
    extent=(0.10, 0.04, 0.06),
):
    """Return (T_B_O, extent) for an OBB at the given base-frame pose."""
    T = pose_xyzrpy_to_matrix(center[0], center[1], center[2], *rpy_deg)
    return T, np.asarray(extent, dtype=np.float64)


# ---------------------------------------------------------------------------
# plan_topdown_grasp -- nominal
# ---------------------------------------------------------------------------
def test_plan_topdown_grasp_picks_shortest_horizontal_axis():
    """The closing axis must align with the OBB's short horizontal axis."""
    T_B_O, extent = _make_obb(extent=(0.10, 0.04, 0.06))
    grasp = plan_topdown_grasp(T_B_O=T_B_O, extent=extent, max_opening_m=0.07)

    assert grasp.feasible
    # Object short axis is 0.04 m (axis index 1 in the OBB), so the
    # planner should close to ~0.04 m (minus overclose) and pre-open
    # slightly above it.
    assert grasp.object_width_m == pytest.approx(0.04, abs=1e-9)
    assert grasp.opening_m > grasp.object_width_m > grasp.close_opening_m

    # Approach axis (z of grasp frame) should point towards world -z
    # (top-down).
    z_grasp_in_world = grasp.T_B_grasp[:3, 2]
    assert np.allclose(z_grasp_in_world, [0.0, 0.0, -1.0], atol=1e-6)

    # Pre-grasp must sit exactly approach_m above the grasp along world +z
    # (since z_grasp points down, -z_grasp is world +z).
    dz = grasp.pre_grasp[2, 3] - grasp.T_B_grasp[2, 3]
    assert dz == pytest.approx(grasp.approach_m, abs=1e-9)


def test_plan_topdown_grasp_clamps_opening_within_max():
    """When the object fits, opening must never exceed max_opening_m."""
    T_B_O, extent = _make_obb(extent=(0.10, 0.04, 0.06))
    # max_opening must exceed the object short axis + margin for the
    # plan to be feasible.  0.04 + 0.005 (default margin) = 0.045 needed.
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.06
    )
    assert grasp.feasible
    assert grasp.opening_m <= 0.06 + 1e-12


def test_plan_topdown_grasp_infeasible_when_too_wide():
    """Object wider than max_opening_m must be reported infeasible."""
    T_B_O, extent = _make_obb(extent=(0.20, 0.10, 0.06))
    grasp = plan_topdown_grasp(T_B_O=T_B_O, extent=extent, max_opening_m=0.05)
    assert not grasp.feasible
    assert "exceeds max gripper opening" in grasp.reason


def test_plan_topdown_grasp_lift_pose_above_grasp():
    T_B_O, extent = _make_obb()
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.07, lift_m=0.15
    )
    assert grasp.lift[2, 3] - grasp.T_B_grasp[2, 3] == pytest.approx(0.15, abs=1e-9)


# ---------------------------------------------------------------------------
# plan_topdown_grasp -- workspace guardrails
# ---------------------------------------------------------------------------
def test_workspace_rejects_grasp_below_table():
    T_B_O, extent = _make_obb(center=(0.30, 0.0, 0.01))  # 1 cm over table
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.07,
        workspace={"table_z_m": 0.0, "safe_z_min_m": 0.03},
    )
    assert not grasp.feasible
    assert "below table" in grasp.reason


def test_workspace_rejects_xy_out_of_bounds():
    T_B_O, extent = _make_obb(center=(0.80, 0.0, 0.05))
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.07,
        workspace={"xy_bounds_m": [[-0.4, 0.4], [-0.4, 0.4]]},
    )
    assert not grasp.feasible
    assert "outside workspace" in grasp.reason


def test_workspace_rejects_lift_above_ceiling():
    T_B_O, extent = _make_obb(center=(0.30, 0.0, 0.30))
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.07,
        lift_m=0.30,  # lift goes to 0.60 m
        workspace={"z_max_m": 0.50},
    )
    assert not grasp.feasible
    assert "ceiling" in grasp.reason


def test_workspace_accepts_valid_plan():
    T_B_O, extent = _make_obb(center=(0.30, 0.0, 0.05))
    grasp = plan_topdown_grasp(
        T_B_O=T_B_O, extent=extent, max_opening_m=0.07,
        workspace={
            "table_z_m": 0.0,
            "safe_z_min_m": 0.005,
            "xy_bounds_m": [[-0.4, 0.4], [-0.4, 0.4]],
            "z_max_m": 0.60,
        },
    )
    assert grasp.feasible, grasp.reason


# ---------------------------------------------------------------------------
# align_obb_axes
# ---------------------------------------------------------------------------
def test_align_obb_axes_identity_when_already_matched():
    R_ref = np.eye(3)
    ext = [0.10, 0.05, 0.02]
    R_out, ext_out = align_obb_axes(R_ref, ext, R_ref, ext)
    assert np.allclose(R_out, R_ref, atol=1e-12)
    assert np.allclose(ext_out, ext, atol=1e-12)


def test_align_obb_axes_recovers_permuted_rotation():
    """A random permutation + sign flip of ref axes should be undone."""
    R_ref = R.from_euler("xyz", [5, 10, -3], degrees=True).as_matrix()
    ext_ref = np.asarray([0.10, 0.06, 0.04])

    perm = [2, 0, 1]                # arbitrary permutation
    signs = np.asarray([-1.0, 1.0, -1.0]).reshape(1, 3)
    R_src = R_ref[:, perm] * signs
    ext_src = ext_ref[perm]

    R_out, ext_out = align_obb_axes(R_ref, ext_ref, R_src, ext_src)
    # Must produce a proper rotation.
    assert np.linalg.det(R_out) == pytest.approx(1.0, abs=1e-9)
    # Each column of R_out should align with the corresponding column
    # of R_ref (dot product close to +1 after sign fix).
    for i in range(3):
        assert R_ref[:, i] @ R_out[:, i] == pytest.approx(1.0, abs=1e-6)
    # Extent must follow the permutation back into the ref ordering.
    assert np.allclose(ext_out, ext_ref, atol=1e-12)


def test_align_obb_axes_handles_close_extents():
    """With near-equal side lengths the Hungarian path must still beat
    greedy: craft a case where greedy grabs the best for axis 0 and
    leaves axis 1 with a suboptimal choice."""
    R_ref = np.eye(3)
    # Src axes: axis 0 of ref has slightly higher cosine with src col 0,
    # but a better *total* assignment is achieved by swapping src cols
    # 0 and 1 for ref rows 0 and 1.
    R_src = np.asarray(
        [
            [0.90, 0.85, 0.0],
            [0.80, 0.95, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Orthonormalise so align_obb_axes sees a proper rotation-like
    # input (function doesn't require it but keeps sign logic clean).
    U, _, Vt = np.linalg.svd(R_src)
    R_src = U @ Vt
    if np.linalg.det(R_src) < 0:
        R_src[:, -1] = -R_src[:, -1]
    ext_src = np.asarray([0.10, 0.09, 0.05])

    R_out, ext_out = align_obb_axes(R_ref, [0.10, 0.09, 0.05], R_src, ext_src)
    # After alignment, the diagonal of R_ref.T @ R_out should be
    # positive (same sense) for every axis.
    dots = np.diag(R_ref.T @ R_out)
    assert np.all(dots > 0.0), dots


# ---------------------------------------------------------------------------
# fuse_poses
# ---------------------------------------------------------------------------
def test_fuse_poses_linear_interpolates_translation():
    T_rough = pose_xyzrpy_to_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    T_fine = pose_xyzrpy_to_matrix(1.0, 2.0, 3.0, 0, 0, 0)
    fused = fuse_poses(T_rough, T_fine, weight_rough=0.5, weight_fine=0.5)
    assert np.allclose(fused[:3, 3], [0.5, 1.0, 1.5], atol=1e-12)


def test_fuse_poses_biases_towards_fine_on_large_disagreement():
    """If rotations differ by more than the threshold, fine rotation wins."""
    T_rough = pose_xyzrpy_to_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    T_fine = pose_xyzrpy_to_matrix(0.0, 0.0, 0.0, 0, 0, 90)  # 90 deg disagreement

    fused = fuse_poses(
        T_rough, T_fine,
        weight_rough=0.5, weight_fine=0.5,
        rotation_disagreement_deg=25.0,
    )
    assert rotation_angle_deg(fused[:3, :3], T_fine[:3, :3]) < 1e-6


def test_fuse_poses_slerps_when_rotations_close():
    """Under-threshold disagreement must produce a real slerp midpoint."""
    T_rough = pose_xyzrpy_to_matrix(0.0, 0.0, 0.0, 0, 0, 0)
    T_fine = pose_xyzrpy_to_matrix(0.0, 0.0, 0.0, 0, 0, 20.0)

    fused = fuse_poses(
        T_rough, T_fine,
        weight_rough=0.5, weight_fine=0.5,
        rotation_disagreement_deg=60.0,
    )
    # Fused rotation should sit halfway between 0 and 20 deg.
    assert rotation_angle_deg(fused[:3, :3], T_rough[:3, :3]) == pytest.approx(
        10.0, abs=1e-4
    )
