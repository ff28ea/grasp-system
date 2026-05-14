"""Preview D435i frames and save Open3D point-cloud cleaning comparisons."""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
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
from ..perception.camera import RealSenseCamera, depth_to_meters
from ..perception.camera import CameraIntrinsics
from ..perception.detector import Detection, SegmentationDetector
from ..perception.pose_estimator import backproject_mask_to_pointcloud
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


@dataclass
class CleaningStage:
    label: str
    pcd: "o3d.geometry.PointCloud"
    geometries: list
    note: str = ""


def _cloud_points(pcd: "o3d.geometry.PointCloud") -> np.ndarray:
    if len(pcd.points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(pcd.points, dtype=np.float64)


def _make_voxel_blocks(
    pcd: "o3d.geometry.PointCloud",
    color: tuple[float, float, float],
    size_m: float,
) -> list:
    pts = _cloud_points(pcd)
    if len(pts) == 0:
        return []

    # Keep block counts bounded so Open3D screenshots stay responsive.
    max_blocks = 700
    if len(pts) > max_blocks:
        step = int(np.ceil(len(pts) / max_blocks))
        pts = pts[::step]

    geoms = []
    half = float(size_m) * 0.5
    for pt in pts:
        box = o3d.geometry.TriangleMesh.create_box(
            width=float(size_m),
            height=float(size_m),
            depth=float(size_m),
        )
        box.translate(np.asarray(pt, dtype=np.float64) - half)
        box.paint_uniform_color(color)
        geoms.append(box)
    return geoms


def _painted_copy(
    pcd: "o3d.geometry.PointCloud",
    color: tuple[float, float, float],
) -> "o3d.geometry.PointCloud":
    out = o3d.geometry.PointCloud(pcd)
    out.paint_uniform_color(color)
    return out


def _stage_geometries(
    *clouds: "o3d.geometry.PointCloud" | list,
) -> list:
    geoms = []
    for cloud in clouds:
        if isinstance(cloud, list):
            geoms.extend(cloud)
        elif len(cloud.points) > 0:
            geoms.append(cloud)
    return geoms


def _cleaning_stages(
    pcd_raw: "o3d.geometry.PointCloud",
    p_cfg: dict,
    up_in_cam: Optional[np.ndarray],
    log: logging.Logger,
    render_mode: str = "blocks",
    block_size_m: float = 0.004,
) -> list[CleaningStage]:
    so = p_cfg["statistical_outlier"]
    plane = p_cfg["plane_segmentation"]

    stages: list[CleaningStage] = [
        CleaningStage(
            label="Raw points",
            pcd=pcd_raw,
            geometries=_stage_geometries(
                _stage_cloud(pcd_raw, (0.0, 0.45, 0.36), render_mode, block_size_m),
            ),
            note="mask + depth",
        )
    ]

    cleaned, sor_indices = pcd_raw.remove_statistical_outlier(
        nb_neighbors=int(so["nb_neighbors"]),
        std_ratio=float(so["std_ratio"]),
    )
    sor_removed = pcd_raw.select_by_index(sor_indices, invert=True)
    stages.append(
        CleaningStage(
            label="1 Outliers removed",
            pcd=cleaned,
            geometries=_stage_geometries(
                _stage_cloud(cleaned, (0.0, 0.45, 0.36), render_mode, block_size_m),
                _stage_cloud(sor_removed, (0.9, 0.05, 0.05), render_mode, block_size_m * 1.4),
            ),
            note=f"red removed: {len(sor_removed.points)}",
        )
    )

    voxel = cleaned
    voxel_size_m = float(p_cfg["voxel_size_m"])
    if voxel_size_m > 0 and len(voxel.points) > 0:
        downsampled = voxel.voxel_down_sample(voxel_size_m)
        if len(downsampled.points) > 0:
            voxel = downsampled
    stages.append(
        CleaningStage(
            label="2 Voxel downsample",
            pcd=voxel,
            geometries=_stage_geometries(
                _stage_cloud(cleaned, (0.78, 0.78, 0.78), "points", block_size_m),
                _stage_cloud(voxel, (0.02, 0.24, 0.95), render_mode, block_size_m * 1.4),
            ),
            note=f"gray source: {len(cleaned.points)}",
        )
    )

    plane_removed = voxel
    plane_inlier_cloud = o3d.geometry.PointCloud()
    min_plane_points = int(plane.get("min_points", 300))
    plane_note = "not enough points"
    if len(plane_removed.points) >= min_plane_points:
        try:
            plane_model, inliers = plane_removed.segment_plane(
                distance_threshold=float(plane["distance_threshold_m"]),
                ransac_n=int(plane["ransac_n"]),
                num_iterations=int(plane["num_iterations"]),
            )
            is_support = True
            if up_in_cam is not None:
                up = np.asarray(up_in_cam, dtype=np.float64).reshape(3)
                up_n = np.linalg.norm(up)
                normal = np.asarray(plane_model[:3], dtype=np.float64)
                n_n = np.linalg.norm(normal)
                if up_n > 1e-9 and n_n > 1e-9:
                    cos_ang = abs(float(np.dot(normal, up) / (n_n * up_n)))
                    cos_tol = float(
                        np.cos(np.deg2rad(float(plane.get("normal_tol_deg", 25.0))))
                    )
                    is_support = cos_ang >= cos_tol
                else:
                    is_support = False
            if (
                is_support
                and 0.1 * len(plane_removed.points)
                < len(inliers)
                < 0.95 * len(plane_removed.points)
            ):
                plane_inlier_cloud = plane_removed.select_by_index(inliers)
                plane_removed = plane_removed.select_by_index(inliers, invert=True)
                plane_note = f"red plane: {len(plane_inlier_cloud.points)}"
            else:
                plane_note = f"plane kept: {len(inliers)} inliers"
                log.info(
                    "plane kept: support=%s inliers=%d total=%d",
                    is_support,
                    len(inliers),
                    len(voxel.points),
                )
        except Exception as exc:
            plane_note = "plane failed"
            log.warning("plane segmentation failed: %s", exc)
    else:
        log.info(
            "plane skipped: %d points < min_points=%d",
            len(plane_removed.points),
            min_plane_points,
        )
    stages.append(
        CleaningStage(
            label="3 Plane removed",
            pcd=plane_removed,
            geometries=_stage_geometries(
                _stage_cloud(plane_removed, (0.0, 0.45, 0.36), render_mode, block_size_m * 1.8),
                _stage_cloud(plane_inlier_cloud, (0.9, 0.05, 0.05), render_mode, block_size_m * 1.4),
            ),
            note=plane_note,
        )
    )
    return stages


def _stage_cloud(
    pcd: "o3d.geometry.PointCloud",
    color: tuple[float, float, float],
    render_mode: str,
    block_size_m: float,
) -> "o3d.geometry.PointCloud" | list:
    if render_mode == "blocks":
        return _make_voxel_blocks(pcd, color, block_size_m)
    return _painted_copy(pcd, color)


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
        opt.light_on = True

        ctr = vis.get_view_control()
        ctr.set_lookat([float(v) for v in lookat])
        ctr.set_front([float(v) for v in front])
        ctr.set_up([float(v) for v in up])
        ctr.set_zoom(0.36)
        for _ in range(8):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
        vis.capture_screen_image(str(path), do_render=True)
    finally:
        vis.destroy_window()


def _crop_to_content(image: np.ndarray, margin: int = 40) -> np.ndarray:
    # Open3D screenshots use a white background. Crop to saturated geometry
    # colors instead of every non-white pixel; antialiasing shadows otherwise
    # leave nearly invisible specks that keep huge blank borders.
    b, g, r = cv2.split(image)
    red = (r > 150) & (g < 120) & (b < 120)
    green = (g > 80) & (r < 80) & (b < 120)
    blue = (b > 130) & (r < 100)
    gray = (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 12) & (
        np.abs(g.astype(np.int16) - b.astype(np.int16)) < 12
    ) & (r < 220)
    mask = red | green | blue | gray
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(image.shape[0], int(ys.max()) + margin + 1)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y1:y2, x1:x2]


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.full((height, width, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    out[y : y + new_h, x : x + new_w] = resized
    return out


def _label_panel(image: np.ndarray, label: str, points: int, note: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 68), (255, 255, 255), -1)
    cv2.putText(
        out,
        label,
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"{points} points",
        (14, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    if note:
        cv2.putText(
            out,
            note,
            (14, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (120, 80, 40),
            1,
            cv2.LINE_AA,
        )
    return out


def _compose_comparison(
    stage_paths: list[Path],
    stage_labels: list[str],
    stage_counts: list[int],
    stage_notes: list[str],
    out_path: Path,
) -> Path:
    panels = []
    for path, label, count, note in zip(stage_paths, stage_labels, stage_counts, stage_notes):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read screenshot: {path}")
        cropped = _crop_to_content(image)
        panel = _fit_panel(cropped, 900, 700)
        panels.append(_label_panel(panel, label, count, note))

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    comparison = np.vstack([top, bottom])
    return save_image(out_path, comparison)


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
    image_width: int,
    image_height: int,
    point_size: float,
    render_mode: str,
    block_size_m: float,
    visible: bool,
    save_ply: bool,
    log: logging.Logger,
) -> Path:
    depth_m = depth_to_meters(depth_raw, depth_scale)
    conf = float(p_cfg.get("conf_threshold", 0.5))
    detections = detector.predict(
        color_bgr,
        conf=conf,
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

    pcd_raw = backproject_mask_to_pointcloud(
        color_bgr,
        depth_m,
        det.mask,
        K,
        depth_trunc_m=float(p_cfg.get("depth_trunc_m", 1.2)),
        dist=dist,
    )
    if len(pcd_raw.points) < 50:
        raise RuntimeError(f"mask produced too few 3D points: {len(pcd_raw.points)}")

    stages = _cleaning_stages(
        pcd_raw,
        p_cfg,
        up_in_cam,
        log,
        render_mode=render_mode,
        block_size_m=block_size_m,
    )

    stamp = _timestamp()
    detection_path = out_dir / f"detection_{stamp}.jpg"
    save_image(detection_path, draw_detections(color_bgr, [det]))
    log.info("wrote %s", detection_path.resolve())

    raw_bbox = pcd_raw.get_axis_aligned_bounding_box()
    lookat = np.asarray(raw_bbox.get_center(), dtype=np.float64)
    front = np.asarray([0.35, -0.35, -1.0], dtype=np.float64)
    front /= np.linalg.norm(front)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    stage_paths: list[Path] = []
    stage_labels: list[str] = []
    stage_counts: list[int] = []
    stage_notes: list[str] = []
    for idx, stage in enumerate(stages):
        safe_label = stage.label.lower().replace(" ", "_")
        screenshot_path = out_dir / f"{idx}_{safe_label}_{stamp}.jpg"
        _render_open3d_screenshot(
            stage.geometries,
            screenshot_path,
            lookat=lookat,
            front=front,
            up=up,
            width=int(image_width),
            height=int(image_height),
            point_size=float(point_size),
            visible=bool(visible),
        )
        stage_paths.append(screenshot_path)
        stage_labels.append(stage.label)
        stage_counts.append(len(stage.pcd.points))
        stage_notes.append(stage.note)
        log.info("wrote %s (%d points)", screenshot_path.resolve(), len(stage.pcd.points))
        if save_ply:
            ply_path = save_pointcloud(
                out_dir / f"{idx}_{safe_label}_{stamp}.ply",
                stage.pcd,
            )
            if ply_path is not None:
                log.info("wrote %s", ply_path.resolve())

    comparison_path = _compose_comparison(
        stage_paths,
        stage_labels,
        stage_counts,
        stage_notes,
        out_dir / f"cleaning_comparison_{stamp}.jpg",
    )
    log.info("wrote comparison: %s", comparison_path.resolve())
    return comparison_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview D435i frames; press SPACE to save Open3D cleaning-stage screenshots."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("grasp_system/runs/pointcloud_cleaning"),
    )
    ap.add_argument("--target-class", type=str, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--warmup-frames", type=int, default=None)
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--image-width", type=int, default=900)
    ap.add_argument("--image-height", type=int, default=700)
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument(
        "--render-mode",
        choices=("blocks", "points"),
        default="blocks",
        help="Open3D screenshot style; blocks is clearer for presentation images",
    )
    ap.add_argument(
        "--block-size-m",
        type=float,
        default=0.004,
        help="cube size used by --render-mode blocks",
    )
    ap.add_argument(
        "--up-in-cam",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="optional support-plane normal hint in camera frame",
    )
    ap.add_argument("--visible", action="store_true", help="show Open3D windows while capturing")
    ap.add_argument("--save-ply", action="store_true", help="also save each stage as PLY")
    ap.add_argument("--window-name", type=str, default="D435i point-cloud cleaning capture")
    args = ap.parse_args()

    if o3d is None:
        raise ImportError("open3d is not installed.")

    cfg = load_config(args.config)
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log = get_logger(
        "pointcloud_cleaning_compare",
        level=getattr(logging, level_name, logging.INFO),
    )

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
                    "SPACE: capture cleaning comparison  q/ESC: quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if target_class is not None:
                    cv2.putText(
                        preview,
                        f"target class: {target_class}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
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

                K, dist = _load_runtime_intrinsics(
                    color_bgr,
                    cam.intrinsics,
                    intrinsics_path,
                )
                try:
                    comparison_path = _process_snapshot(
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
                        image_width=int(args.image_width),
                        image_height=int(args.image_height),
                        point_size=float(args.point_size),
                        render_mode=str(args.render_mode),
                        block_size_m=float(args.block_size_m),
                        visible=bool(args.visible),
                        save_ply=bool(args.save_ply),
                        log=log,
                    )
                    comparison = cv2.imread(str(comparison_path), cv2.IMREAD_COLOR)
                    if comparison is not None:
                        h, w = preview.shape[:2]
                        comparison_preview = cv2.resize(comparison, (w, h))
                        cv2.imshow(args.window_name, comparison_preview)
                        cv2.waitKey(1)
                except Exception as exc:
                    log.error("capture failed: %s", exc)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
