"""Small visualization helpers.

``draw_detections`` overlays YOLOv8-seg masks and bboxes on a BGR image.
``draw_grasp_frame`` renders the grasp frame in an Open3D window next to
the scene cloud for debugging. Kept lightweight so the rest of the
pipeline has no hard dependency on these functions.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]

from ..perception.detector import Detection


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


def draw_grasp_frame(
    scene_pcd: "o3d.geometry.PointCloud",
    T_grasp: np.ndarray,
    T_obj: Optional[np.ndarray] = None,
    frame_size: float = 0.05,
    window_name: str = "grasp",
) -> None:
    if o3d is None:
        raise ImportError("open3d is not installed.")

    geoms = [scene_pcd]
    if T_obj is not None:
        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        obj_frame.transform(T_obj)
        geoms.append(obj_frame)
    g_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    g_frame.transform(T_grasp)
    geoms.append(g_frame)
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 1.5)
    geoms.append(world)
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def overlay_obb(
    image_bgr: np.ndarray,
    center_cam: np.ndarray,
    R_cam: np.ndarray,
    extent: Sequence[float],
    K: np.ndarray,
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
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), (0, 255, 255), 2)
    return vis
