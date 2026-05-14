"""Preview D435i frames and save YOLOv8-seg snapshots on SPACE."""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

from ..common import ensure_parent, get_logger, load_classes, load_config, project_path
from ..perception.camera import RealSenseCamera
from ..perception.detector import SegmentationDetector
from .visualize import draw_detections


def _timestamped_path(out_dir: Path, prefix: str, suffix: str = ".jpg") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{stamp}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview RealSense D435i color frames; press SPACE to save YOLO overlay."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("grasp_system/runs/yolo_snapshot"),
        help="directory for saved screenshots",
    )
    ap.add_argument("--save-raw", action="store_true", help="also save raw color frames")
    ap.add_argument("--conf", type=float, default=None, help="override YOLO confidence")
    ap.add_argument("--device", type=str, default=None, help='override inference device, e.g. "cpu" or "cuda:0"')
    ap.add_argument("--warmup-frames", type=int, default=None, help="override camera warmup frames")
    ap.add_argument("--timeout-ms", type=int, default=5000, help="frame wait timeout")
    ap.add_argument("--window-name", type=str, default="D435i YOLO snapshot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log = get_logger("snapshot_yolo", level=getattr(logging, level_name, logging.INFO))

    camera_cfg = cfg.get("camera", {}) or {}
    perception_cfg = cfg.get("perception", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    classes = load_classes(paths_cfg.get("classes", "configs/classes.yaml"))
    class_names = {k: v.get("name", str(k)) for k, v in classes.items()}
    detector = SegmentationDetector(
        model_path=project_path(paths_cfg.get("model", "models/best.pt")),
        conf=float(args.conf if args.conf is not None else perception_cfg.get("conf_threshold", 0.5)),
        device=args.device if args.device is not None else perception_cfg.get("device"),
        class_names=class_names,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
            last_save_time = 0.0
            while True:
                color_bgr = cam.grab_color(timeout_ms=int(args.timeout_ms))
                preview = color_bgr.copy()
                cv2.putText(
                    preview,
                    "SPACE: save YOLO  q/ESC: quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(args.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key != 32:
                    continue
                now = time.monotonic()
                if now - last_save_time < 0.5:
                    continue
                last_save_time = now

                detections = detector.predict(
                    color_bgr,
                    min_pixels=int(perception_cfg.get("min_mask_pixels", 0)),
                    mask_erode_px=0,
                )
                vis = draw_detections(color_bgr, detections)
                out_path = _timestamped_path(args.out_dir, "yolo_snapshot")
                ensure_parent(out_path)
                if not cv2.imwrite(str(out_path), vis):
                    raise RuntimeError(f"failed to write visualization image: {out_path}")
                if args.save_raw:
                    raw_path = _timestamped_path(args.out_dir, "raw_color")
                    if not cv2.imwrite(str(raw_path), color_bgr):
                        raise RuntimeError(f"failed to write raw image: {raw_path}")
                    log.info("saved raw frame: %s", raw_path.resolve())

                log.info("detections: %d", len(detections))
                for det in detections:
                    log.info(
                        "class=%s id=%d conf=%.3f bbox=%s mask_pixels=%d",
                        det.label or det.class_id,
                        det.class_id,
                        det.confidence,
                        det.bbox_xyxy.tolist(),
                        det.num_pixels,
                    )
                log.info("saved visualization: %s", out_path.resolve())
                cv2.imshow(args.window_name, vis)
                cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
