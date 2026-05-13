"""Common utilities: config loading, transform helpers, logging."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent


def project_path(rel: str | Path) -> Path:
    """Resolve a path relative to the grasp_system package root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return PKG_ROOT / p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "grasp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "configs/system.yaml") -> Dict[str, Any]:
    p = project_path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(path: str | Path = "configs/classes.yaml") -> Dict[int, Dict[str, Any]]:
    p = project_path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("classes", {}) or {}
    # Normalise keys to int.
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        out[int(k)] = v
    return out


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------
def make_transform(R_mat: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_mat, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (cheap, avoids np.linalg.inv)."""
    T = np.asarray(T, dtype=np.float64)
    R_m = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_m.T
    T_inv[:3, 3] = -R_m.T @ t
    return T_inv


def validate_transform(
    T: np.ndarray,
    name: str = "T",
    rot_tol: float = 1e-3,
    max_translation_m: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Sanity-check a 4x4 rigid transform.

    Checks:
      * shape is (4, 4);
      * bottom row is (0, 0, 0, 1) within tolerance;
      * rotation block is orthonormal (``R^T R = I`` within ``rot_tol``);
      * rotation determinant is +1 (not -1 / reflection);
      * optional translation magnitude upper bound (meters).

    Returns True if the transform looks clean, False otherwise. Issues
    are emitted as warnings via the given logger so a suspicious
    calibration is flagged loud at load time rather than producing
    silently wrong poses downstream.
    """
    log = logger or get_logger("transform")
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        log.error("%s must be 4x4, got %s", name, T.shape)
        return False
    if not np.all(np.isfinite(T)):
        log.error("%s contains NaN/Inf", name)
        return False
    bottom = T[3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        log.warning("%s bottom row is %s, not [0 0 0 1]", name, bottom)

    R_m = T[:3, :3]
    orth_err = float(np.linalg.norm(R_m.T @ R_m - np.eye(3)))
    det = float(np.linalg.det(R_m))
    ok = True
    if orth_err > rot_tol:
        log.warning(
            "%s rotation not orthonormal: |R^T R - I|=%.2e (> %.2e)",
            name, orth_err, rot_tol,
        )
        ok = False
    if abs(det - 1.0) > max(rot_tol, 1e-3):
        log.warning("%s rotation det=%.4f, expected +1", name, det)
        ok = False

    if max_translation_m is not None:
        t_norm = float(np.linalg.norm(T[:3, 3]))
        if t_norm > max_translation_m:
            log.warning(
                "%s translation norm %.3f m exceeds %.3f m",
                name, t_norm, max_translation_m,
            )
            ok = False
    return ok


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """PiPER convention: extrinsic (fixed) xyz, degrees -> 3x3 rotation matrix."""
    return R.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


def matrix_to_euler_xyz_deg(R_mat: np.ndarray) -> np.ndarray:
    """3x3 rotation -> extrinsic xyz Euler in degrees, matching PiPER convention."""
    return R.from_matrix(np.asarray(R_mat)).as_euler("xyz", degrees=True)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angular distance between two 3x3 rotations, in degrees.

    Use this instead of comparing Euler angles component-wise: Euler
    representation is discontinuous near gimbal lock and not a metric, so
    per-axis absolute differences can both over- and under-estimate the
    real rotational error.
    """
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    R_rel = R_a.T @ R_b
    # trace is in [-1, 3]; clamp against numerical noise before acos.
    cos_theta = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def pose_xyzrpy_to_matrix(
    x_m: float, y_m: float, z_m: float,
    rx_deg: float, ry_deg: float, rz_deg: float,
) -> np.ndarray:
    """Position (m) + Euler xyz (deg) -> 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    T[:3, 3] = [x_m, y_m, z_m]
    return T


def matrix_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """4x4 -> [x_m, y_m, z_m, rx_deg, ry_deg, rz_deg]."""
    t = T[:3, 3]
    e = matrix_to_euler_xyz_deg(T[:3, :3])
    return np.asarray([t[0], t[1], t[2], e[0], e[1], e[2]], dtype=np.float64)


def translate(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def camera_look_down_rotation(tilt_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation making the camera optical axis (+z_cam) point towards -z_world.

    Equivalent to a 180 deg rotation about world x, with optional additional
    tilt about world x (tilt_deg) and yaw about world z (yaw_deg).
    """
    base = R.from_euler("x", 180.0, degrees=True)
    extra = R.from_euler("zx", [yaw_deg, tilt_deg], degrees=True)
    return (extra * base).as_matrix()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_npy(path: str | Path, arr: np.ndarray) -> Path:
    p = ensure_parent(path)
    np.save(p, np.asarray(arr))
    return p


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def load_intrinsics(path: str | Path) -> Dict[str, Any]:
    """Load camera intrinsics saved by calibrate_intrinsics.py.

    Validates shape, dtype, and basic plausibility (fx > 0, image size
    positive). A corrupt or out-of-date calibration file produces a
    clear ``ValueError`` at load time rather than a mysterious
    "back-projected nothing" downstream.
    """
    path = Path(path)
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise FileNotFoundError(
            f"failed to read intrinsics from {path}: {exc}"
        ) from exc

    missing = [k for k in ("K", "dist", "width", "height") if k not in data.files]
    if missing:
        raise ValueError(
            f"intrinsics file {path} is missing required keys: {missing}. "
            "Re-run `python -m grasp_system.calibration.calibrate_intrinsics`."
        )

    K = np.asarray(data["K"], dtype=np.float64)
    dist = np.asarray(data["dist"], dtype=np.float64)
    width = int(data["width"])
    height = int(data["height"])

    if K.shape != (3, 3):
        raise ValueError(f"intrinsics K must be 3x3, got {K.shape} from {path}")
    if dist.ndim != 1 and dist.ndim != 2:
        raise ValueError(f"intrinsics dist must be 1D or 2D, got shape {dist.shape}")
    # OpenCV supports 4/5/8/12/14 distortion coefficients; anything else
    # is almost certainly a file authored by hand with a wrong layout.
    dist_flat = dist.reshape(-1)
    if dist_flat.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"intrinsics dist has {dist_flat.size} coefficients; expected "
            "4/5/8/12/14 (OpenCV convention)"
        )
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError(
            f"intrinsics K has non-positive focal length ({K[0, 0]}, {K[1, 1]}); "
            "calibration likely failed"
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"intrinsics image size {width}x{height} is non-positive"
        )

    out: Dict[str, Any] = {"K": K, "dist": dist, "width": width, "height": height}
    if "rms" in data.files:
        out["rms"] = float(data["rms"])
    return out


def fx_fy_cx_cy(K: np.ndarray) -> tuple[float, float, float, float]:
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def rescale_intrinsics(
    K: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Scale a pinhole intrinsic matrix to a different image resolution.

    Calibration runs at one resolution (e.g. 1280x720 for better corner
    localisation) but the runtime stream may be lower (640x480). The
    distortion coefficients are dimensionless and need no change, but
    K is in pixel units and must be linearly rescaled, or every
    back-projected 3D point will be off by the ratio.

    Uses the simple proportional form
        fx' = fx * Wdst/Wsrc,  fy' = fy * Hdst/Hsrc
        cx' = cx * Wdst/Wsrc,  cy' = cy * Hdst/Hsrc
    without a half-pixel correction. This matches OpenCV's
    ``cv2.resize`` convention and the RealSense SDK's own resolution
    downscaling, and is accurate to well within 1 pixel for integer
    resolution ratios.

    Parameters
    ----------
    K:
        Source 3x3 intrinsic matrix.
    src_size:
        ``(width, height)`` of the image K was calibrated at.
    dst_size:
        ``(width, height)`` of the image stream K should now apply to.

    Returns
    -------
    A *new* 3x3 matrix; the input is not modified. Returns the original
    matrix unchanged (up to a copy) when ``src_size == dst_size``.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    sw, sh = int(src_size[0]), int(src_size[1])
    dw, dh = int(dst_size[0]), int(dst_size[1])
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        raise ValueError(
            f"src/dst sizes must be positive, got src={src_size} dst={dst_size}"
        )
    if (sw, sh) == (dw, dh):
        return K.copy()
    sx = dw / float(sw)
    sy = dh / float(sh)
    K_out = K.copy()
    K_out[0, 0] *= sx       # fx
    K_out[1, 1] *= sy       # fy
    K_out[0, 2] *= sx       # cx
    K_out[1, 2] *= sy       # cy
    # Off-diagonal skew (K[0, 1]) stays the same *in pixel units* under
    # anisotropic scaling only when sx == sy; for safety scale it too.
    K_out[0, 1] *= sx
    return K_out
