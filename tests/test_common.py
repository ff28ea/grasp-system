"""Unit tests for grasp_system.common transform helpers."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from grasp_system.common import (
    camera_look_down_rotation,
    euler_xyz_deg_to_matrix,
    invert_transform,
    make_transform,
    matrix_to_euler_xyz_deg,
    matrix_to_xyzrpy,
    pose_xyzrpy_to_matrix,
    rotation_angle_deg,
    validate_transform,
)


# ---------------------------------------------------------------------------
# invert_transform
# ---------------------------------------------------------------------------
def test_invert_transform_round_trip():
    """T @ inv(T) should be identity for arbitrary rigid transforms."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        rvec = rng.uniform(-np.pi, np.pi, size=3)
        t = rng.uniform(-1.0, 1.0, size=3)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(rvec).as_matrix()
        T[:3, 3] = t
        inv = invert_transform(T)
        assert np.allclose(T @ inv, np.eye(4), atol=1e-10)
        assert np.allclose(inv @ T, np.eye(4), atol=1e-10)


def test_invert_transform_is_transpose_for_rotation():
    """For a rigid transform, the inverse rotation is the transpose."""
    T = make_transform(R.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(),
                       [0.1, -0.2, 0.3])
    inv = invert_transform(T)
    assert np.allclose(inv[:3, :3], T[:3, :3].T, atol=1e-12)
    # Translation part: -R^T @ t
    assert np.allclose(inv[:3, 3], -T[:3, :3].T @ T[:3, 3], atol=1e-12)


# ---------------------------------------------------------------------------
# Euler <-> matrix round-trip (PiPER convention)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rpy_deg",
    [
        (0.0, 0.0, 0.0),
        (10.0, 20.0, 30.0),
        (-45.0, 30.0, 90.0),
        (179.0, -5.0, -170.0),
    ],
)
def test_euler_matrix_round_trip(rpy_deg):
    """Extrinsic xyz Euler -> matrix -> Euler returns the original angles
    (within a 360 deg ambiguity handled by recomposing the rotation)."""
    rx, ry, rz = rpy_deg
    M = euler_xyz_deg_to_matrix(rx, ry, rz)
    back = matrix_to_euler_xyz_deg(M)
    M_back = euler_xyz_deg_to_matrix(*back)
    # Compare rotations, not component-wise angles: scipy may pick an
    # equivalent but different triple near gimbal lock.
    assert rotation_angle_deg(M, M_back) < 1e-5


def test_pose_xyzrpy_matrix_round_trip():
    xyzrpy = np.array([0.1, -0.2, 0.3, 20.0, -15.0, 45.0])
    T = pose_xyzrpy_to_matrix(*xyzrpy.tolist())
    back = matrix_to_xyzrpy(T)
    assert np.allclose(back[:3], xyzrpy[:3], atol=1e-12)
    # Same convention re-encode: round-trip through scipy should match
    # to within float precision.
    assert np.allclose(
        euler_xyz_deg_to_matrix(*back[3:].tolist()),
        euler_xyz_deg_to_matrix(*xyzrpy[3:].tolist()),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# rotation_angle_deg
# ---------------------------------------------------------------------------
def test_rotation_angle_zero_for_identity():
    I = np.eye(3)
    assert rotation_angle_deg(I, I) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("angle_deg", [1.0, 10.0, 90.0, 179.0])
def test_rotation_angle_matches_scalar_rotation(axis, angle_deg):
    """For a single-axis rotation, the geodesic angle equals |angle|."""
    R_a = np.eye(3)
    R_b = R.from_euler(axis, angle_deg, degrees=True).as_matrix()
    measured = rotation_angle_deg(R_a, R_b)
    assert measured == pytest.approx(angle_deg, abs=1e-6)


def test_rotation_angle_is_symmetric():
    rng = np.random.default_rng(42)
    for _ in range(10):
        a = R.from_rotvec(rng.uniform(-1.5, 1.5, size=3)).as_matrix()
        b = R.from_rotvec(rng.uniform(-1.5, 1.5, size=3)).as_matrix()
        assert rotation_angle_deg(a, b) == pytest.approx(
            rotation_angle_deg(b, a), abs=1e-9
        )


# ---------------------------------------------------------------------------
# validate_transform
# ---------------------------------------------------------------------------
def test_validate_transform_accepts_clean_rigid():
    T = pose_xyzrpy_to_matrix(0.1, 0.2, 0.3, 5.0, -10.0, 20.0)
    assert validate_transform(T, name="T") is True


def test_validate_transform_rejects_shape():
    assert validate_transform(np.eye(3), name="T") is False


def test_validate_transform_flags_non_orthonormal():
    T = np.eye(4)
    T[:3, :3] *= 1.1                # scale -> not orthonormal
    assert validate_transform(T, name="T", rot_tol=1e-4) is False


def test_validate_transform_flags_reflection():
    T = np.eye(4)
    T[0, 0] = -1.0                  # det = -1
    assert validate_transform(T, name="T") is False


def test_validate_transform_flags_translation_bound():
    T = pose_xyzrpy_to_matrix(10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert validate_transform(T, name="T", max_translation_m=0.5) is False


# ---------------------------------------------------------------------------
# camera_look_down_rotation
# ---------------------------------------------------------------------------
def test_camera_look_down_maps_z_cam_to_minus_z_world():
    """With no tilt/yaw, the camera's +z axis should point towards world -z.

    The rotation matrix columns express the camera basis in world
    coordinates, so the third column is the direction of +z_cam in
    world space.
    """
    R_mat = camera_look_down_rotation(tilt_deg=0.0, yaw_deg=0.0)
    z_cam_in_world = R_mat[:, 2]
    assert np.allclose(z_cam_in_world, [0.0, 0.0, -1.0], atol=1e-12)
