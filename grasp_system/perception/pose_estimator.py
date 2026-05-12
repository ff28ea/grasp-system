"""Point-cloud based 6-DoF pose estimation.

The pipeline is:

    mask + depth
       -> backproject pixels -> object point cloud in camera frame
       -> statistical outlier removal
       -> voxel downsample
       -> (optional) RANSAC plane removal to drop the supporting table
       -> oriented bounding box (OBB)
       -> T_C_O assembled from OBB.R and OBB.center

An optional FPFH + point-to-plane ICP refinement step is available when
a target template point cloud is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..common import get_logger

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]


_log = get_logger("pose_estimator")


@dataclass
class ObjectPose:
    """Result of pose estimation for a single object."""

    T_C_O: np.ndarray               # 4x4, object frame in camera frame
    center: np.ndarray              # 3, OBB center in camera frame
    extent: np.ndarray              # 3, OBB side lengths (meters)
    R: np.ndarray                   # 3x3, OBB orientation
    num_points: int                 # points used after filtering
    icp_fitness: Optional[float] = None
    icp_inlier_rmse: Optional[float] = None


def backproject_mask_to_pointcloud(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth_trunc_m: float = 1.2,
    dist: Optional[np.ndarray] = None,
) -> "o3d.geometry.PointCloud":
    """Build an Open3D point cloud from masked pixels.

    ``depth_m`` must be a float array in meters. ``mask`` must be the same
    shape as ``depth_m`` (bool or uint8). ``dist`` are the OpenCV distortion
    coefficients; when provided, pixels are undistorted before
    back-projection so the 3D reconstruction actually uses the lens
    calibration from ``calibrate_intrinsics.py``.
    """
    if o3d is None:
        raise ImportError("open3d is not installed.")
    if color_bgr.shape[:2] != depth_m.shape[:2]:
        raise ValueError("color and depth must share height/width")
    if mask.shape[:2] != depth_m.shape[:2]:
        raise ValueError("mask must match depth image shape")

    mask_bool = mask.astype(bool)
    depth = depth_m.astype(np.float32).copy()
    depth[~mask_bool] = 0.0
    depth[depth > float(depth_trunc_m)] = 0.0

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    zs = depth
    valid = zs > 0
    if not np.any(valid):
        return o3d.geometry.PointCloud()

    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    z = zs[valid].astype(np.float32)

    # Undistort pixel coordinates using the calibrated distortion model
    # before pinhole back-projection. If ``dist`` is omitted or effectively
    # zero we short-circuit to the fast path (no cv2 dependency).
    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    use_undistort = dist_arr is not None and np.any(np.abs(dist_arr) > 1e-9)

    if use_undistort:
        if cv2 is None:
            raise ImportError(
                "cv2 is required for distortion-aware back-projection."
            )
        pts_in = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.float32)
        # undistortPoints returns normalised camera-plane coords (x/z, y/z)
        # when P is omitted, which is exactly what we need next.
        norm = cv2.undistortPoints(
            pts_in, K.astype(np.float64), dist_arr.astype(np.float64)
        ).reshape(-1, 2)
        x = norm[:, 0] * z
        y = norm[:, 1] * z
    else:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)

    # Colors: convert BGR uint8 -> RGB float in [0, 1]
    rgb = color_bgr[..., ::-1].astype(np.float32) / 255.0
    colors = rgb[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def clean_pointcloud(
    pcd: "o3d.geometry.PointCloud",
    voxel_size_m: float = 0.002,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    remove_plane: bool = True,
    plane_distance_threshold_m: float = 0.006,
    plane_ransac_n: int = 3,
    plane_num_iterations: int = 1000,
    min_plane_points: int = 300,
) -> "o3d.geometry.PointCloud":
    """Statistical outlier removal + voxel downsample + optional plane removal."""
    if o3d is None:
        raise ImportError("open3d is not installed.")
    if len(pcd.points) == 0:
        return pcd

    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(outlier_nb_neighbors),
        std_ratio=float(outlier_std_ratio),
    )
    if len(cleaned.points) == 0:
        return cleaned

    if voxel_size_m and voxel_size_m > 0 and len(cleaned.points) > 0:
        downsampled = cleaned.voxel_down_sample(float(voxel_size_m))
        # voxel_down_sample can return an empty cloud when all points fall
        # inside a single voxel (very dense small object or huge voxel).
        # Fall back to the pre-downsample cloud in that case so the pipeline
        # keeps running rather than raising "not enough points" later.
        if len(downsampled.points) > 0:
            cleaned = downsampled

    if remove_plane and len(cleaned.points) >= min_plane_points:
        try:
            _, inliers = cleaned.segment_plane(
                distance_threshold=float(plane_distance_threshold_m),
                ransac_n=int(plane_ransac_n),
                num_iterations=int(plane_num_iterations),
            )
            # Only drop the plane if it is a meaningful fraction of the cloud.
            if 0.1 * len(cleaned.points) < len(inliers) < 0.95 * len(cleaned.points):
                cleaned = cleaned.select_by_index(inliers, invert=True)
        except Exception as exc:
            # RANSAC can fail on degenerate clouds (<3 non-collinear
            # points, all-zero normals, etc). Keep the un-segmented cloud
            # rather than aborting, but tell the user so they can tune
            # voxel_size / outlier params.
            _log.warning("plane segmentation failed: %s", exc)

    return cleaned


def estimate_pose_from_obb(
    pcd: "o3d.geometry.PointCloud",
    up_world_in_cam: Optional[np.ndarray] = None,
) -> ObjectPose:
    """Fit an oriented bounding box and assemble a right-handed T_C_O.

    Parameters
    ----------
    up_world_in_cam:
        If provided, the longest in-plane axis of the OBB is disambiguated so
        that its dot product with this vector is >= 0. Useful to kill the
        sign ambiguity of PCA.
    """
    if o3d is None:
        raise ImportError("open3d is not installed.")
    if len(pcd.points) < 10:
        raise ValueError("not enough points to fit an OBB")

    obb = pcd.get_oriented_bounding_box()
    R_obb = np.asarray(obb.R, dtype=np.float64).copy()
    extent = np.asarray(obb.extent, dtype=np.float64).copy()
    center = np.asarray(obb.center, dtype=np.float64).copy()

    # Order axes by extent descending so axis 0 is longest.
    order = np.argsort(-extent)
    R_obb = R_obb[:, order]
    extent = extent[order]

    # Disambiguate signs.
    if up_world_in_cam is not None:
        up = np.asarray(up_world_in_cam, dtype=np.float64).reshape(3)
        up /= np.linalg.norm(up) + 1e-12
        # Flip the longest axis so it aligns with the reference direction in xy.
        if R_obb[:, 0] @ up < 0:
            R_obb[:, 0] = -R_obb[:, 0]
            R_obb[:, 1] = -R_obb[:, 1]  # preserve right-handedness

    # Ensure right-handed.
    if np.linalg.det(R_obb) < 0:
        R_obb[:, 2] = -R_obb[:, 2]

    T_C_O = np.eye(4)
    T_C_O[:3, :3] = R_obb
    T_C_O[:3, 3] = center

    return ObjectPose(
        T_C_O=T_C_O,
        center=center,
        extent=extent,
        R=R_obb,
        num_points=int(len(pcd.points)),
    )


def refine_with_icp(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    voxel_size_m: float = 0.003,
    max_correspondence_m: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """FPFH + point-to-plane ICP refinement.

    Returns
    -------
    (T_target_source, fitness, inlier_rmse)
        ``T_target_source`` transforms points from ``source`` into ``target``.
    """
    if o3d is None:
        raise ImportError("open3d is not installed.")

    src_d = source.voxel_down_sample(voxel_size_m)
    tgt_d = target.voxel_down_sample(voxel_size_m)

    radius_normal = voxel_size_m * 2.0
    src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30))
    tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30))

    radius_feature = voxel_size_m * 5.0
    src_f = o3d.pipelines.registration.compute_fpfh_feature(
        src_d, o3d.geometry.KDTreeSearchParamHybrid(radius_feature, 100)
    )
    tgt_f = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius_feature, 100)
    )

    distance_threshold = voxel_size_m * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d,
        tgt_d,
        src_f,
        tgt_f,
        True,  # mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    icp_threshold = (
        max_correspondence_m
        if max_correspondence_m is not None
        else voxel_size_m * 0.4
    )
    # Point-to-plane ICP needs normals on the target. FPFH above only
    # estimated normals on the voxel-downsampled copies (src_d/tgt_d), so
    # we must estimate them on the full-resolution clouds before the
    # refinement ICP -- otherwise Open3D either raises or silently
    # produces a degenerate transform.
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30)
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30)
        )
    icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_threshold,
        ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return (
        np.asarray(icp.transformation, dtype=np.float64),
        float(icp.fitness),
        float(icp.inlier_rmse),
    )
