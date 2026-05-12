"""Chessboard-based intrinsics calibration for the RealSense D435i color stream.

Usage::

    python -m grasp_system.calibration.calibrate_intrinsics \\
        --num-views 20 --pattern 11 8 --square 0.025

The script is interactive: it opens a live preview, you move the arm around
so the chessboard is visible from varied viewpoints, then press SPACE to
capture each view and ESC to finish. Saves ``configs/camera_intrinsics.npz``
containing ``K``, ``dist``, ``width``, ``height`` and the RMS reprojection
error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..common import ensure_parent, get_logger, load_config, project_path
from ..perception.camera import RealSenseCamera


def _build_object_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square
    return objp


def run(
    num_views: int,
    pattern: tuple[int, int],
    square: float,
    out_path: Path,
    cam_cfg: Optional[dict] = None,
) -> None:
    log = get_logger("calib-intr")
    objp = _build_object_points(pattern, square)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # Use the same resolution / fps that will be used at inference time so
    # the calibrated K and dist match the live stream exactly.
    cam_cfg = cam_cfg or {}
    with RealSenseCamera(
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
        color_fps=int(cam_cfg.get("color_fps", 30)),
        depth_fps=int(cam_cfg.get("depth_fps", 30)),
        warmup_frames=int(cam_cfg.get("warmup_frames", 30)),
    ) as cam:
        width, height = cam.intrinsics.width, cam.intrinsics.height
        log.info("streaming %dx%d; SPACE=capture, ESC=finish", width, height)

        while len(obj_points) < num_views:
            color = cam.grab_color()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            vis = color.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_crit)
                cv2.drawChessboardCorners(vis, pattern, corners, True)

            cv2.putText(
                vis,
                f"captured {len(obj_points)}/{num_views}  {'OK' if found else 'no board'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("intrinsics calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32 and found:  # SPACE
                obj_points.append(objp.copy())
                img_points.append(corners.copy())
                log.info("captured view %d", len(obj_points))
                time.sleep(0.2)

        cv2.destroyAllWindows()

    if len(obj_points) < 10:
        log.error("need at least ~10 views, got %d; aborting", len(obj_points))
        sys.exit(1)

    log.info("calibrating with %d views...", len(obj_points))
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        (width, height),
        None,
        None,
    )
    log.info("reprojection RMS: %.4f px (target < 0.5)", rms)
    log.info("K=\n%s", np.array2string(K, precision=4, suppress_small=True))
    log.info("dist=%s", np.array2string(dist.ravel(), precision=6, suppress_small=True))

    ensure_parent(out_path)
    np.savez(
        out_path,
        K=K,
        dist=dist,
        width=np.int32(width),
        height=np.int32(height),
        rms=np.float32(rms),
    )
    log.info("saved %s", out_path)


def main() -> None:
    cfg = load_config()
    chess = cfg.get("chessboard", {})
    default_pattern = list(chess.get("pattern_size", [11, 8]))
    default_square = float(chess.get("square_size_m", 0.025))
    default_out = project_path(cfg["paths"]["intrinsics"])

    ap = argparse.ArgumentParser(description="RealSense color-stream intrinsics calibration.")
    ap.add_argument("--num-views", type=int, default=20)
    ap.add_argument(
        "--pattern",
        nargs=2,
        type=int,
        default=default_pattern,
        metavar=("COLS", "ROWS"),
        help="inner-corner count (cols rows)",
    )
    ap.add_argument("--square", type=float, default=default_square, help="square edge in meters")
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()

    run(
        num_views=int(args.num_views),
        pattern=(int(args.pattern[0]), int(args.pattern[1])),
        square=float(args.square),
        out_path=Path(args.out),
        cam_cfg=cfg.get("camera", {}),
    )


if __name__ == "__main__":
    main()
