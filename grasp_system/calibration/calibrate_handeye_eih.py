"""Eye-in-hand hand-eye calibration (PiPER + RealSense D435i).

Solves ``T_E_C`` from multiple poses where a chessboard is fixed on the
table and the camera is rigidly mounted on the arm end-effector. Uses
``cv2.calibrateHandEye`` in its native eye-in-hand convention:

    inputs  : R/t_gripper2base (=T_B_E)  +  R/t_target2cam (=T_C_T)
    output  : R/t_cam2gripper  (=T_E_C)

No input inversion is required in this configuration.

Usage::

    python -m grasp_system.calibration.calibrate_handeye_eih \\
        --num-poses 20

Interactive loop:
    1. Teach/jog the arm to a pose where the chessboard is fully visible.
    2. Press SPACE or left-click the preview window to record. The script
       captures the image, solves PnP, and reads the current end-pose from
       the arm.
    3. Press ESC to finish; need at least ``--min-poses`` good pairs.

Saves::

    configs/T_eef_cam.npy      # 4x4 transform meters/radians
    configs/handeye_report.npz # raw pairs + stats for QA
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..common import (
    ensure_parent,
    get_logger,
    load_config,
    load_intrinsics,
    make_transform,
    project_path,
)
from ..control.piper_controller import PiperController
from ..perception.camera import RealSenseCamera


_HANDEYE_METHODS = {
    "CALIB_HAND_EYE_PARK": cv2.CALIB_HAND_EYE_PARK,
    "CALIB_HAND_EYE_TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "CALIB_HAND_EYE_HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "CALIB_HAND_EYE_ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "CALIB_HAND_EYE_DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _build_object_points(pattern: Tuple[int, int], square: float) -> np.ndarray:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square
    return objp


def _detect_chessboard(
    color: np.ndarray,
    pattern: Tuple[int, int],
    objp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None, None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, None, None
    R_CT, _ = cv2.Rodrigues(rvec)
    return True, R_CT, tvec.reshape(3)


def _collect_poses(
    cfg: dict,
    num_poses: int,
    pattern: Tuple[int, int],
    square: float,
    intrinsics_path: Path,
    settle_s: float,
) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    log = get_logger("handeye")
    intr = load_intrinsics(intrinsics_path)
    K = intr["K"]
    dist = intr["dist"]
    objp = _build_object_points(pattern, square)

    R_g2b: List[np.ndarray] = []
    t_g2b: List[np.ndarray] = []
    R_t2c: List[np.ndarray] = []
    t_t2c: List[np.ndarray] = []
    window_name = "handeye eye-in-hand"
    mouse_capture_requested = False

    def _request_capture_on_left_click(
        event: int,
        _x: int,
        _y: int,
        _flags: int,
        _param: object,
    ) -> None:
        nonlocal mouse_capture_requested
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_capture_requested = True

    piper_port = cfg.get("piper", {}).get("can_port", "can0")
    with RealSenseCamera() as cam, PiperController(can_port=piper_port) as piper:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _request_capture_on_left_click)
        log.info(
            "move arm to a new calibration pose; SPACE/left-click to capture, ESC to finish"
        )
        while len(R_g2b) < num_poses:
            color, _ = cam.grab_aligned()
            ok, R_CT, t_CT = _detect_chessboard(color, pattern, objp, K, dist)

            vis = color.copy()
            status = "OK" if ok else "no board"
            color_txt = (0, 255, 0) if ok else (0, 0, 255)
            if ok:
                rvec, _ = cv2.Rodrigues(R_CT)
                cv2.drawFrameAxes(vis, K, dist, rvec, t_CT, square * 3.0)
            cv2.putText(
                vis,
                f"captured {len(R_g2b)}/{num_poses}  {status}  SPACE/click=capture",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color_txt,
                2,
            )
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            clicked = mouse_capture_requested
            mouse_capture_requested = False
            if key == 27:
                break
            wants_capture = key == 32 or clicked
            if not wants_capture:
                continue

            if not ok:
                log.warning("no chessboard detected; skipping")
                continue

            log.info("settling for %.2fs before reading end-pose", settle_s)
            time.sleep(settle_s)

            # Re-capture after settle so image matches logged pose.
            color, _ = cam.grab_aligned()
            ok2, R_CT2, t_CT2 = _detect_chessboard(color, pattern, objp, K, dist)
            if not ok2:
                log.warning("lost chessboard after settle; skipping")
                continue

            T_BE = piper.get_end_pose_matrix()
            R_g2b.append(T_BE[:3, :3].copy())
            t_g2b.append(T_BE[:3, 3].copy())
            R_t2c.append(R_CT2)
            t_t2c.append(t_CT2)
            log.info("recorded pose %d", len(R_g2b))

        cv2.destroyAllWindows()

    return R_g2b, t_g2b, R_t2c, t_t2c


def _solve(
    R_g2b: List[np.ndarray],
    t_g2b: List[np.ndarray],
    R_t2c: List[np.ndarray],
    t_t2c: List[np.ndarray],
    method: int,
) -> np.ndarray:
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=method,
    )
    return make_transform(R_c2g, np.asarray(t_c2g).reshape(3))


def _consistency_report(
    T_EC: np.ndarray,
    R_g2b: List[np.ndarray],
    t_g2b: List[np.ndarray],
    R_t2c: List[np.ndarray],
    t_t2c: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the position of the board origin in the base frame for each
    pose; a perfect calibration would put them all at the same point."""
    positions = []
    for R_BE, t_BE, R_CT, t_CT in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        T_BE = make_transform(R_BE, t_BE)
        T_CT = make_transform(R_CT, t_CT)
        T_BT = T_BE @ T_EC @ T_CT
        positions.append(T_BT[:3, 3])
    positions = np.asarray(positions)
    std_mm = np.std(positions, axis=0) * 1000.0
    mean = positions.mean(axis=0)
    return std_mm, mean


def main() -> None:
    cfg = load_config()
    chess = cfg.get("chessboard", {})
    he = cfg.get("handeye", {})
    default_pattern = list(chess.get("pattern_size", [11, 8]))
    default_square = float(chess.get("square_size_m", 0.025))
    default_method = str(he.get("method", "CALIB_HAND_EYE_PARK"))
    default_min = int(he.get("min_poses", 15))
    default_settle = float(he.get("settle_time_s", 1.0))

    ap = argparse.ArgumentParser(description="Eye-in-hand calibration.")
    ap.add_argument("--num-poses", type=int, default=default_min + 5)
    ap.add_argument("--min-poses", type=int, default=default_min)
    ap.add_argument("--pattern", nargs=2, type=int, default=default_pattern)
    ap.add_argument("--square", type=float, default=default_square)
    ap.add_argument(
        "--method",
        choices=sorted(_HANDEYE_METHODS.keys()),
        default=default_method,
    )
    ap.add_argument(
        "--intrinsics",
        type=Path,
        default=project_path(cfg["paths"]["intrinsics"]),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=project_path(cfg["paths"]["t_eef_cam"]),
    )
    ap.add_argument("--settle", type=float, default=default_settle)
    args = ap.parse_args()

    log = get_logger("handeye")

    R_g2b, t_g2b, R_t2c, t_t2c = _collect_poses(
        cfg,
        num_poses=int(args.num_poses),
        pattern=(int(args.pattern[0]), int(args.pattern[1])),
        square=float(args.square),
        intrinsics_path=Path(args.intrinsics),
        settle_s=float(args.settle),
    )
    if len(R_g2b) < args.min_poses:
        log.error(
            "only %d poses captured; need at least %d",
            len(R_g2b),
            args.min_poses,
        )
        sys.exit(1)

    T_EC = _solve(R_g2b, t_g2b, R_t2c, t_t2c, _HANDEYE_METHODS[args.method])

    log.info("T_E_C =\n%s", np.array2string(T_EC, precision=6, suppress_small=True))

    std_mm, mean_pos = _consistency_report(T_EC, R_g2b, t_g2b, R_t2c, t_t2c)
    log.info(
        "board origin in base: mean=%s (m), std=%s (mm)",
        np.array2string(mean_pos, precision=4),
        np.array2string(std_mm, precision=3),
    )
    if np.max(std_mm) > 3.0:
        log.warning(
            "position std > 3 mm; consider adding more varied poses or checking "
            "the camera mount rigidity"
        )

    ensure_parent(args.out)
    np.save(args.out, T_EC)
    log.info("saved %s", args.out)

    report_path = args.out.with_name("handeye_report.npz")
    np.savez(
        report_path,
        T_E_C=T_EC,
        R_gripper2base=np.stack(R_g2b, 0),
        t_gripper2base=np.stack(t_g2b, 0),
        R_target2cam=np.stack(R_t2c, 0),
        t_target2cam=np.stack(t_t2c, 0),
        position_std_mm=std_mm,
        method=np.bytes_(args.method),
    )
    log.info("saved QA report %s", report_path)


if __name__ == "__main__":
    main()
