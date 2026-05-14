"""Preview D435i frames and save an Open3D OBB pose-estimation screenshot."""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..common import (
    get_logger,
    load_classes,
    load_config,
    load_intrinsics,
    project_path,
    rescale_intrinsics,
)
from ..perception.camera import CameraIntrinsics, RealSenseCamera, depth_to_meters
from ..perception.detector import Detection, SegmentationDetector
from ..perception.pose_estimator import (
    backproject_mask_to_pointcloud,
    clean_pointcloud,
    estimate_pose_from_obb,
)
from .visualize import draw_detections, save_image, save_pointcloud

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _resolve_target_class(
    target: Optional[str], classes: dict[int, dict[str, object]]
) -> Optional[int]:
    if target is None:
        return None
    try:
        return int(target)
    except ValueError:
        pass
    target_norm = target.strip().lower()
    for class_id, meta in classes.items():
        if str(meta.get("name", "")).strip().lower() == target_norm:
            return class_id
    raise ValueError(f"unknown target class: {target}")


def _select_detection(
    detections: list[Detection], target_class: Optional[int]
) -> Detection:
    if target_class is not None:
        detections = [d for d in detections if d.class_id == target_class]
        if not detections:
            raise RuntimeError(f"class {target_class} not detected")
    if not detections:
        raise RuntimeError("no instances detected")
    return max(detections, key=lambda d: d.confidence)


def _load_runtime_intrinsics(
    color_bgr: np.ndarray,
    cam_intrinsics: CameraIntrinsics,
    intrinsics_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    K = cam_intrinsics.K
    dist = cam_intrinsics.dist.reshape(-1)
    if intrinsics_path.exists():
        intr = load_intrinsics(intrinsics_path)
        K = rescale_intrinsics(
            intr["K"],
            (int(intr["width"]), int(intr["height"])),
            (int(color_bgr.shape[1]), int(color_bgr.shape[0])),
        )
        dist = intr["dist"]
    return K, dist


def _obb_lineset(
    center: np.ndarray,
    R_mat: np.ndarray,
    extent: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.55, 0.0),
) -> "o3d.geometry.LineSet":
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
    corners = (np.asarray(R_mat, dtype=np.float64) @ (signs * half).T).T + center
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return lines


def _pose_frame(T: np.ndarray, size: float) -> "o3d.geometry.TriangleMesh":
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    frame.transform(np.asarray(T, dtype=np.float64))
    return frame


def _paint_cloud(
    pcd: "o3d.geometry.PointCloud",
    color: tuple[float, float, float],
) -> "o3d.geometry.PointCloud":
    out = o3d.geometry.PointCloud(pcd)
    out.paint_uniform_color(color)
    return out


def _render_open3d_screenshot(
    geometries: list,
    path: Path,
    lookat: np.ndarray,
    front: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    point_size: float,
    visible: bool,
) -> None:
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(
        window_name=path.stem,
        width=int(width),
        height=int(height),
        visible=bool(visible),
    ):
        raise RuntimeError("failed to create Open3D visualization window")
    try:
        for geom in geometries:
            vis.add_geometry(geom)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        opt.point_size = float(point_size)
        opt.line_width = 5.0
        opt.light_on = True

        ctr = vis.get_view_control()
        ctr.set_lookat([float(v) for v in lookat])
        ctr.set_front([float(v) for v in front])
        ctr.set_up([float(v) for v in up])
        ctr.set_zoom(0.52)
        for _ in range(10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
        vis.capture_screen_image(str(path), do_render=True)
    finally:
        vis.destroy_window()


def _label_image(
    image_path: Path,
    out_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read screenshot: {image_path}")
    cv2.rectangle(image, (0, 0), (image.shape[1], 74), (255, 255, 255), -1)
    cv2.putText(
        image,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        subtitle,
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return save_image(out_path, image)


def _process_snapshot(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    K: np.ndarray,
    dist: np.ndarray,
    detector: SegmentationDetector,
    p_cfg: dict,
    target_class: Optional[int],
    up_in_cam: Optional[np.ndarray],
    out_dir: Path,
    width: int,
    height: int,
    point_size: float,
    visible: bool,
    save_ply: bool,
    log: logging.Logger,
) -> Path:
    depth_m = depth_to_meters(depth_raw, depth_scale)
    detections = detector.predict(
        color_bgr,
        conf=float(p_cfg.get("conf_threshold", 0.5)),
        min_pixels=int(p_cfg.get("min_mask_pixels", 0)),
        mask_erode_px=int(p_cfg.get("mask_erode_px", 0)),
        target_hw=depth_m.shape,
    )
    det = _select_detection(detections, target_class)
    log.info(
        "selected detection: class=%d label=%s conf=%.3f mask_pixels=%d",
        det.class_id,
        det.label,
        det.confidence,
        det.num_pixels,
    )

    raw = backproject_mask_to_pointcloud(
        color_bgr,
        depth_m,
        det.mask,
        K,
        depth_trunc_m=float(p_cfg.get("depth_trunc_m", 1.2)),
        dist=dist,
    )
    if len(raw.points) < 50:
        raise RuntimeError(f"mask produced too few 3D points: {len(raw.points)}")

    so = p_cfg["statistical_outlier"]
    plane = p_cfg["plane_segmentation"]
    ro_cfg = p_cfg.get("radius_outlier", {}) or {}
    clean = clean_pointcloud(
        raw,
        voxel_size_m=float(p_cfg["voxel_size_m"]),
        outlier_nb_neighbors=int(so["nb_neighbors"]),
        outlier_std_ratio=float(so["std_ratio"]),
        remove_plane=True,
        plane_distance_threshold_m=float(plane["distance_threshold_m"]),
        plane_ransac_n=int(plane["ransac_n"]),
        plane_num_iterations=int(plane["num_iterations"]),
        min_plane_points=int(plane.get("min_points", 300)),
        up_in_cam=up_in_cam,
        plane_normal_tol_deg=float(plane.get("normal_tol_deg", 25.0)),
        radius_outlier_nb_points=int(ro_cfg.get("nb_points", 0)),
        radius_outlier_radius_m=float(ro_cfg.get("radius_m", 0.01)),
    )
    if len(clean.points) < 20:
        raise RuntimeError(f"cleaned point cloud too sparse: {len(clean.points)}")

    pose = estimate_pose_from_obb(clean, up_world_in_cam=up_in_cam)
    stamp = _timestamp()
    detection_path = out_dir / f"detection_{stamp}.jpg"
    save_image(detection_path, draw_detections(color_bgr, [det]))
    log.info("wrote %s", detection_path.resolve())

    if save_ply:
        raw_path = save_pointcloud(out_dir / f"raw_cloud_{stamp}.ply", raw)
        clean_path = save_pointcloud(out_dir / f"clean_cloud_{stamp}.ply", clean)
        if raw_path is not None:
            log.info("wrote %s", raw_path.resolve())
        if clean_path is not None:
            log.info("wrote %s", clean_path.resolve())

    T_C_O = pose.T_C_O
    frame_size = max(float(np.max(pose.extent)) * 0.45, 0.025)
    geoms = [
        _paint_cloud(clean, (0.0, 0.45, 0.36)),
        _obb_lineset(pose.center, pose.R, pose.extent),
        _pose_frame(T_C_O, frame_size),
    ]

    lookat = np.asarray(pose.center, dtype=np.float64)
    front = np.asarray([0.35, -0.35, -1.0], dtype=np.float64)
    front /= np.linalg.norm(front)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    raw_screenshot = out_dir / f"obb_pose_raw_{stamp}.jpg"
    _render_open3d_screenshot(
        geoms,
        raw_screenshot,
        lookat=lookat,
        front=front,
        up=up,
        width=int(width),
        height=int(height),
        point_size=float(point_size),
        visible=bool(visible),
    )
    title = f"OBB pose: {det.label or det.class_id}  conf={det.confidence:.2f}"
    subtitle = (
        f"points={len(clean.points)}  center=({pose.center[0]:.3f}, "
        f"{pose.center[1]:.3f}, {pose.center[2]:.3f}) m  "
        f"extent=({pose.extent[0]:.3f}, {pose.extent[1]:.3f}, {pose.extent[2]:.3f}) m"
    )
    out_path = out_dir / f"obb_pose_{stamp}.jpg"
    labeled = _label_image(raw_screenshot, out_path, title, subtitle)
    log.info("wrote OBB pose screenshot: %s", labeled.resolve())
    return labeled


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview D435i frames; press SPACE to save an Open3D OBB pose screenshot."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("grasp_system/runs/obb_pose"))
    ap.add_argument("--target-class", type=str, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--warmup-frames", type=int, default=None)
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--image-width", type=int, default=1200)
    ap.add_argument("--image-height", type=int, default=900)
    ap.add_argument("--point-size", type=float, default=7.0)
    ap.add_argument(
        "--up-in-cam",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="optional support-plane / OBB sign hint in camera frame",
    )
    ap.add_argument("--visible", action="store_true", help="show Open3D window while capturing")
    ap.add_argument("--save-ply", action="store_true", help="also save raw and cleaned PLY clouds")
    ap.add_argument("--window-name", type=str, default="D435i OBB pose capture")
    args = ap.parse_args()

    if o3d is None:
        raise ImportError("open3d is not installed.")

    cfg = load_config(args.config)
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log = get_logger("obb_pose_snapshot", level=getattr(logging, level_name, logging.INFO))

    camera_cfg = cfg.get("camera", {}) or {}
    p_cfg = cfg.get("perception", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(paths_cfg.get("classes", "configs/classes.yaml"))
    target_class = _resolve_target_class(args.target_class, classes)
    class_names = {k: v.get("name", str(k)) for k, v in classes.items()}
    detector = SegmentationDetector(
        model_path=project_path(paths_cfg.get("model", "models/best.pt")),
        conf=float(args.conf if args.conf is not None else p_cfg.get("conf_threshold", 0.5)),
        device=args.device if args.device is not None else p_cfg.get("device"),
        class_names=class_names,
    )
    if args.conf is not None:
        p_cfg = dict(p_cfg)
        p_cfg["conf_threshold"] = float(args.conf)

    intrinsics_path = project_path(paths_cfg.get("intrinsics", "configs/camera_intrinsics.npz"))
    up_in_cam = None if args.up_in_cam is None else np.asarray(args.up_in_cam, dtype=np.float64)

    log.info("starting D435i stream")
    try:
        with RealSenseCamera(
            width=int(camera_cfg.get("width", 640)),
            height=int(camera_cfg.get("height", 480)),
            color_fps=int(camera_cfg.get("color_fps", 30)),
            depth_fps=int(camera_cfg.get("depth_fps", 30)),
            align_to=str(camera_cfg.get("align_to", "color")),
            depth_scale=float(camera_cfg.get("depth_scale", 0.001)),
            warmup_frames=int(
                args.warmup_frames
                if args.warmup_frames is not None
                else camera_cfg.get("warmup_frames", 30)
            ),
            spatial_filter=bool(camera_cfg.get("spatial_filter", False)),
            temporal_filter=bool(camera_cfg.get("temporal_filter", False)),
            hole_filling_filter=bool(camera_cfg.get("hole_filling_filter", False)),
        ) as cam:
            depth_scale = float(cam.depth_scale)
            last_capture_time = 0.0
            while True:
                color_bgr, depth_raw = cam.grab_aligned(timeout_ms=int(args.timeout_ms))
                preview = color_bgr.copy()
                cv2.putText(
                    preview,
                    "SPACE: capture OBB pose  q/ESC: quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if target_class is not None:
                    cv2.putText(
                        preview,
                        f"target class: {target_class}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(args.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key != 32:
                    continue
                now = time.monotonic()
                if now - last_capture_time < 0.5:
                    continue
                last_capture_time = now

                K, dist = _load_runtime_intrinsics(color_bgr, cam.intrinsics, intrinsics_path)
                try:
                    screenshot_path = _process_snapshot(
                        color_bgr=color_bgr,
                        depth_raw=depth_raw,
                        depth_scale=depth_scale,
                        K=K,
                        dist=dist,
                        detector=detector,
                        p_cfg=p_cfg,
                        target_class=target_class,
                        up_in_cam=up_in_cam,
                        out_dir=args.out_dir,
                        width=int(args.image_width),
                        height=int(args.image_height),
                        point_size=float(args.point_size),
                        visible=bool(args.visible),
                        save_ply=bool(args.save_ply),
                        log=log,
                    )
                    image = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
                    if image is not None:
                        h, w = preview.shape[:2]
                        cv2.imshow(args.window_name, cv2.resize(image, (w, h)))
                        cv2.waitKey(1)
                except Exception as exc:
                    log.error("capture failed: %s", exc)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
