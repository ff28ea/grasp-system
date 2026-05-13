"""Unit tests for grasp_system.common.rescale_intrinsics."""
from __future__ import annotations

import numpy as np
import pytest

from grasp_system.common import rescale_intrinsics


def _K(fx=600.0, fy=600.0, cx=320.0, cy=240.0, skew=0.0) -> np.ndarray:
    return np.asarray(
        [[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def test_rescale_intrinsics_identity_when_sizes_match():
    """No-op when src == dst (but still returns a fresh copy)."""
    K = _K()
    out = rescale_intrinsics(K, (640, 480), (640, 480))
    assert np.allclose(out, K, atol=1e-12)
    # Must be a copy, not the same object, so callers can mutate safely.
    assert out is not K


def test_rescale_intrinsics_halves_all_components_on_2x_downsample():
    """1280x720 calibration -> 640x360 runtime halves fx, fy, cx, cy."""
    K = _K(fx=1200.0, fy=1200.0, cx=640.0, cy=360.0)
    out = rescale_intrinsics(K, (1280, 720), (640, 360))
    assert out[0, 0] == pytest.approx(600.0, abs=1e-9)
    assert out[1, 1] == pytest.approx(600.0, abs=1e-9)
    assert out[0, 2] == pytest.approx(320.0, abs=1e-9)
    assert out[1, 2] == pytest.approx(180.0, abs=1e-9)


def test_rescale_intrinsics_anisotropic_scaling():
    """When aspect ratio changes, fx/cx and fy/cy scale independently."""
    K = _K(fx=1000.0, fy=1000.0, cx=640.0, cy=480.0)
    # Horizontally shrink 2x, vertically stay the same: 1280x960 -> 640x960.
    out = rescale_intrinsics(K, (1280, 960), (640, 960))
    assert out[0, 0] == pytest.approx(500.0, abs=1e-9)       # fx halves
    assert out[1, 1] == pytest.approx(1000.0, abs=1e-9)      # fy unchanged
    assert out[0, 2] == pytest.approx(320.0, abs=1e-9)       # cx halves
    assert out[1, 2] == pytest.approx(480.0, abs=1e-9)       # cy unchanged


def test_rescale_intrinsics_preserves_backprojection_ratio():
    """Back-projecting the same physical pixel under original and
    rescaled K should produce the same 3D ray direction."""
    K = _K(fx=900.0, fy=900.0, cx=640.0, cy=360.0)
    src = (1280, 720)
    dst = (640, 360)
    K2 = rescale_intrinsics(K, src, dst)

    # A pixel at the top-right corner (1280, 720) in the original
    # resolution corresponds to (640, 360) in the downsampled one.
    # The unprojected ray (x/z, y/z, 1) should be identical.
    u1, v1 = 1280.0, 720.0
    x1 = (u1 - K[0, 2]) / K[0, 0]
    y1 = (v1 - K[1, 2]) / K[1, 1]

    u2, v2 = 640.0, 360.0
    x2 = (u2 - K2[0, 2]) / K2[0, 0]
    y2 = (v2 - K2[1, 2]) / K2[1, 1]

    assert x1 == pytest.approx(x2, abs=1e-12)
    assert y1 == pytest.approx(y2, abs=1e-12)


def test_rescale_intrinsics_rejects_bad_shape():
    with pytest.raises(ValueError, match="K must be 3x3"):
        rescale_intrinsics(np.eye(4), (1, 1), (1, 1))


def test_rescale_intrinsics_rejects_non_positive_size():
    K = _K()
    with pytest.raises(ValueError, match="positive"):
        rescale_intrinsics(K, (0, 480), (640, 480))
    with pytest.raises(ValueError, match="positive"):
        rescale_intrinsics(K, (640, 480), (640, -1))


def test_rescale_intrinsics_does_not_mutate_input():
    K = _K()
    K_before = K.copy()
    _ = rescale_intrinsics(K, (640, 480), (320, 240))
    assert np.allclose(K, K_before, atol=1e-12)
