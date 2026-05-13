"""Unit tests for grasp_system.common.load_intrinsics validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grasp_system.common import load_intrinsics


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
def _write_valid(path: Path) -> None:
    K = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros(5, dtype=np.float64)
    np.savez(
        path,
        K=K,
        dist=dist,
        width=np.int32(640),
        height=np.int32(480),
        rms=np.float32(0.3),
    )


def test_load_intrinsics_reads_valid_file(tmp_path):
    p = tmp_path / "intr.npz"
    _write_valid(p)
    out = load_intrinsics(p)
    assert out["K"].shape == (3, 3)
    assert out["dist"].shape == (5,)
    assert out["width"] == 640
    assert out["height"] == 480
    assert out["rms"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------
def test_load_intrinsics_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, K=np.eye(3))                    # missing dist, width, height
    with pytest.raises(ValueError, match="missing required keys"):
        load_intrinsics(p)


def test_load_intrinsics_rejects_wrong_K_shape(tmp_path):
    p = tmp_path / "bad_K.npz"
    np.savez(
        p,
        K=np.eye(4),                            # not 3x3
        dist=np.zeros(5),
        width=640,
        height=480,
    )
    with pytest.raises(ValueError, match="K must be 3x3"):
        load_intrinsics(p)


def test_load_intrinsics_rejects_weird_dist_length(tmp_path):
    p = tmp_path / "bad_dist.npz"
    np.savez(
        p,
        K=np.eye(3) * 500,
        dist=np.zeros(3),                       # 3 coeffs is not OpenCV-legal
        width=640,
        height=480,
    )
    with pytest.raises(ValueError, match="coefficients"):
        load_intrinsics(p)


def test_load_intrinsics_rejects_zero_focal_length(tmp_path):
    p = tmp_path / "bad_fx.npz"
    K = np.eye(3)
    K[0, 0] = 0.0                               # fx = 0
    np.savez(p, K=K, dist=np.zeros(5), width=640, height=480)
    with pytest.raises(ValueError, match="focal length"):
        load_intrinsics(p)


def test_load_intrinsics_rejects_negative_size(tmp_path):
    p = tmp_path / "bad_size.npz"
    np.savez(
        p,
        K=np.eye(3) * 500,
        dist=np.zeros(5),
        width=-1,
        height=480,
    )
    with pytest.raises(ValueError, match="non-positive"):
        load_intrinsics(p)


def test_load_intrinsics_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_intrinsics(tmp_path / "does-not-exist.npz")
