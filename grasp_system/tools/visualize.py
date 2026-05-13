"""Visualization helpers for the grasp pipeline.

Two tiers of helpers live here:

* **Per-frame overlays** (``draw_detections``, ``overlay_obb``) return BGR
  images you can ``cv2.imshow`` or write to disk. They don't touch Open3D.

* **Scene builders** (``build_scene_geometries``, ``show_scene``,
  ``save_grasp_artifacts``) work in 3D. They assemble the scene cloud,
  the object OBB, and the grasp / pre-grasp / lift frames into a single
  Open3D geometry list, expressed in whatever frame the caller passes
  in. The main pipeline uses them with everything expressed in the
  robot base frame so the preview matches what the arm will execute.

Open3D is imported lazily: if it's not installed, the 2D helpers still
work, and the 3D functions raise a clear ImportError only when called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]

from ..perception.detector import Detection


# ---------------------------------------------------------------------------
# 2D overlays (image space)
# ---------------------------------------------------------------------------
def draw_detections(
    image_bgr: np.ndarray,
    detections: Iterable[Detection],
    alpha: float = 0.4,
) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = vis.copy()
    H_img, W_img = image_bgr.shape[:2]
    rng = np.random.default_rng(42)
    for det in detections:
        color = tuple(int(c) for c in rng.integers(64, 255, size=3))
        # Detections may have been produced at a different target_hw
        # (e.g. depth image shape) than the visualization image. Resize
        # the mask to the image grid before fancy-indexing, otherwise
        # ``overlay[det.mask] = color`` raises IndexError/ValueError.
        if det.mask.shape[:2] != (H_img, W_img):
            mask_vis = cv2.resize(
                det.mask.astype(np.uint8),
                (W_img, H_img),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_vis = det.mask
        overlay[mask_vis] = color
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int).tolist()
        # Clamp to valid image bounds so cv2.rectangle/putText never receive
        # out-of-range coordinates (negative or beyond width/height), which
        # would either raise or silently draw garbage outside the image.
        x1 = max(0, min(x1, W_img - 1))
        y1 = max(0, min(y1, H_img - 1))
        x2 = max(0, min(x2, W_img - 1))
        y2 = max(0, min(y2, H_img - 1))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        txt = f"{det.label or det.class_id} {det.confidence:.2f}"
        cv2.putText(
            vis,
            txt,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)
    return vis


def overlay_obb(
    image_bgr: np.ndarray,
    center_cam: np.ndarray,
    R_cam: np.ndarray,
    extent: Sequence[float],
    K: np.ndarray,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Project an OBB into the image for quick sanity checking."""
    half = np.asarray(extent, dtype=np.float64) * 0.5
    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    corners_obj = signs * half
    corners_cam = (R_cam @ corners_obj.T).T + center_cam
    # Perspective projection. Clip z to a small positive epsilon so
    # corners at or behind the camera plane don't explode to inf/NaN
    # when cast to int -- that would feed garbage pixels to cv2.line.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zs = np.maximum(corners_cam[:, 2], 1e-3)
    us = fx * corners_cam[:, 0] / zs + cx
    vs = fy * corners_cam[:, 1] / zs + cy
    pts = np.stack([us, vs], axis=1).astype(int)
    # Flag corners that are not actually in front of the camera so we
    # can skip edges that would otherwise draw meaningless lines.
    in_front = corners_cam[:, 2] > 1e-3

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    vis = image_bgr.copy()
    for a, b in edges:
        if not (in_front[a] and in_front[b]):
            continue
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), color, thickness)
    return vis


def make_detection_overlay(
    image_bgr: np.ndarray,
    detection: Detection,
    center_cam: np.ndarray,
    R_cam: np.ndarray,
    extent: Sequence[float],
    K: np.ndarray,
    title: Optional[str] = None,
) -> np.ndarray:
    """Combined overlay: YOLO mask + OBB projection + caption.

    Used to produce the ``detection_*.jpg`` artifacts saved by the main
    pipeline. Everything is self-contained so the caller only needs
    Python's standard numeric stack and OpenCV to render it.
    """
    vis = draw_detections(image_bgr, [detection])
    vis = overlay_obb(vis, center_cam, R_cam, extent, K)
    if title:
        cv2.putText(
            vis,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return vis


# ---------------------------------------------------------------------------
# 3D scene builders (base frame)
# ---------------------------------------------------------------------------
def _require_o3d() -> None:
    if o3d is None:
        raise ImportError(
            "open3d is not installed. `pip install open3d` to use 3D visualization."
        )


def _transform_pointcloud(
    pcd: "o3d.geometry.PointCloud", T: np.ndarray
) -> "o3d.geometry.PointCloud":
    """Return a copy of ``pcd`` with its points transformed by ``T``."""
    _require_o3d()
    # Open3D's transform() mutates in place; deep-copy so the caller's
    # object keeps its camera-frame coordinates. This matters when the
    # same cloud is reused to fit an OBB after drawing.
    out = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out


def _obb_lineset(
    center: np.ndarray,
    R_mat: np.ndarray,
    extent: Sequence[float],
    color: tuple = (1.0, 0.85, 0.1),
) -> "o3d.geometry.LineSet":
    """Build a wireframe box at ``center`` with given rotation and side lengths."""
    _require_o3d()
    half = np.asarray(extent, dtype=np.float64) * 0.5
    signs = np.array(
        [
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    corners = (np.asarray(R_mat) @ (signs * half).T).T + np.asarray(center)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def _frame(T: np.ndarray, size: float = 0.05) -> "o3d.geometry.TriangleMesh":
    _require_o3d()
    m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    m.transform(np.asarray(T, dtype=np.float64))
    return m


def build_scene_geometries(
    scene_pcd_base: Optional["o3d.geometry.PointCloud"] = None,
    T_B_O: Optional[np.ndarray] = None,
    extent: Optional[Sequence[float]] = None,
    T_B_grasp: Optional[np.ndarray] = None,
    T_B_pregrasp: Optional[np.ndarray] = None,
    T_B_lift: Optional[np.ndarray] = None,
    T_B_place: Optional[np.ndarray] = None,
    frame_size: float = 0.05,
) -> list:
    """Assemble the full grasp scene in the base frame.

    All inputs are optional so the function can be used for partial
    previews (e.g. only the rough pose before active perception has
    run).

    Geometry legend:
      * point cloud   -- scene in base frame
      * yellow box    -- fitted OBB (when T_B_O + extent are given)
      * larger axes   -- world base frame
      * axes at T_B_O -- object frame (what perception produced)
      * axes at grasp -- commanded end-effector pose
      * axes at pre   -- pose from which we begin the linear approach
      * axes at lift  -- pose after the upward lift
      * axes at place -- drop-off pose (if configured)
    """
    _require_o3d()
    geoms: list = []

    # Base / world frame first, slightly bigger than per-pose frames so
    # the user can tell it apart when the scene is zoomed out.
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 2.0)
    )

    if scene_pcd_base is not None and len(scene_pcd_base.points) > 0:
        geoms.append(scene_pcd_base)

    if T_B_O is not None:
        geoms.append(_frame(T_B_O, size=frame_size))
        if extent is not None:
            geoms.append(_obb_lineset(T_B_O[:3, 3], T_B_O[:3, :3], extent))

    if T_B_pregrasp is not None:
        geoms.append(_frame(T_B_pregrasp, size=frame_size * 0.6))
    if T_B_grasp is not None:
        geoms.append(_frame(T_B_grasp, size=frame_size))
    if T_B_lift is not None:
        geoms.append(_frame(T_B_lift, size=frame_size * 0.6))
    if T_B_place is not None:
        geoms.append(_frame(T_B_place, size=frame_size * 0.8))

    return geoms


def show_scene(
    geoms: list,
    window_name: str = "grasp preview (base frame)",
) -> None:
    """Blocking Open3D viewer. Close the window to continue."""
    _require_o3d()
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------
def save_image(path: str | Path, image_bgr: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), image_bgr)
    return p


def save_pointcloud(
    path: str | Path, pcd: "o3d.geometry.PointCloud"
) -> Optional[Path]:
    """Write an Open3D cloud to PLY. Returns None if the cloud is empty or
    open3d is unavailable."""
    if o3d is None or pcd is None or len(pcd.points) == 0:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)
    return p


def save_poses_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Save a dict of named arrays (typically transforms) to a single npz."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    np.savez(p, **clean)
    return p


# ---------------------------------------------------------------------------
# Legacy entry point (kept for back-compat with existing callers)
# ---------------------------------------------------------------------------
def draw_grasp_frame(
    scene_pcd: "o3d.geometry.PointCloud",
    T_grasp: np.ndarray,
    T_obj: Optional[np.ndarray] = None,
    frame_size: float = 0.05,
    window_name: str = "grasp",
) -> None:
    """Backwards-compatible single-frame preview (no OBB, no pre/lift)."""
    geoms: List = build_scene_geometries(
        scene_pcd_base=scene_pcd,
        T_B_O=T_obj,
        T_B_grasp=T_grasp,
        frame_size=frame_size,
    )
    show_scene(geoms, window_name=window_name)
