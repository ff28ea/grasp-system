"""Checkerboard-based physical contact verification.

Workflow:
  1. Move the arm to the saved observe pose.
  2. Detect the tabletop checkerboard and solve its pose in the camera frame.
  3. Transform selected board points into the robot base frame.
  4. Move the current end-effector orientation above each point, then descend
     slowly for a supervised physical contact check.

This is intentionally separate from calibration. It validates the complete
runtime chain ``T_B_T = T_B_E * T_E_C * T_C_T`` by touching known points on a
checkerboard placed on the table.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..common import (
    ensure_parent,
    get_logger,
    load_config,
    load_intrinsics,
    load_npy,
    make_transform,
    project_path,
    rescale_intrinsics,
    validate_transform,
)
from ..control.piper_controller import PiperController
from ..perception.camera import RealSenseCamera


@dataclass(frozen=True)
class ContactPoint:
    name: str
    p_board: np.ndarray


@dataclass(frozen=True)
class ContactPlan:
    point: ContactPoint
    p_base: np.ndarray
    T_above: np.ndarray
    T_near: np.ndarray
    T_touch: np.ndarray


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _build_object_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_m)
    return objp


def _detect_chessboard_pose(
    color_bgr: np.ndarray,
    pattern: tuple[int, int],
    objp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=sb_flags)

    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
            )

    if not found:
        return False, None, None, None

    corners = np.asarray(corners, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, None, None, None
    R_CT, _ = cv2.Rodrigues(rvec)
    return True, R_CT, tvec.reshape(3), corners


def _contact_points(
    pattern: tuple[int, int],
    square_m: float,
    mode: str,
) -> list[ContactPoint]:
    cols, rows = pattern
    if cols < 2 or rows < 2:
        raise ValueError(f"checkerboard pattern must be at least 2x2, got {pattern}")

    max_x = float(cols - 1) * float(square_m)
    max_y = float(rows - 1) * float(square_m)
    center = ContactPoint("center", np.asarray([max_x * 0.5, max_y * 0.5, 0.0]))
    corners = [
        ContactPoint("corner_00", np.asarray([0.0, 0.0, 0.0])),
        ContactPoint("corner_x0", np.asarray([max_x, 0.0, 0.0])),
        ContactPoint("corner_xy", np.asarray([max_x, max_y, 0.0])),
        ContactPoint("corner_0y", np.asarray([0.0, max_y, 0.0])),
    ]
    if mode == "center":
        return [center]
    if mode == "corners":
        return corners
    if mode == "center-corners":
        return [center, *corners]
    raise ValueError(f"unsupported contact point mode: {mode}")


def _point_board_to_base(T_B_T: np.ndarray, p_board: np.ndarray) -> np.ndarray:
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = np.asarray(p_board, dtype=np.float64).reshape(3)
    return (T_B_T @ p_h)[:3]


def _eef_pose_for_contact_point(
    p_contact_base: np.ndarray,
    R_B_E: np.ndarray,
    tool_offset_eef_m: np.ndarray,
    height_m: float,
) -> np.ndarray:
    """Return ``T_B_E`` whose tool contact point is above/touching p_contact.

    ``tool_offset_eef_m`` is the vector from the end-effector origin to the
    physical touch point, expressed in the end-effector frame.
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_B_E, dtype=np.float64)
    p_contact = np.asarray(p_contact_base, dtype=np.float64).reshape(3).copy()
    p_contact[2] += float(height_m)
    T[:3, 3] = p_contact - T[:3, :3] @ np.asarray(tool_offset_eef_m, dtype=np.float64)
    return T


def _build_contact_plans(
    *,
    points: Iterable[ContactPoint],
    T_B_T: np.ndarray,
    T_B_E_reference: np.ndarray,
    tool_offset_eef_m: np.ndarray,
    approach_height_m: float,
    near_height_m: float,
    touch_z_offset_m: float,
) -> list[ContactPlan]:
    plans: list[ContactPlan] = []
    R_B_E = T_B_E_reference[:3, :3]
    for point in points:
        p_base = _point_board_to_base(T_B_T, point.p_board)
        T_above = _eef_pose_for_contact_point(
            p_base, R_B_E, tool_offset_eef_m, approach_height_m
        )
        T_near = _eef_pose_for_contact_point(
            p_base, R_B_E, tool_offset_eef_m, near_height_m
        )
        T_touch = _eef_pose_for_contact_point(
            p_base, R_B_E, tool_offset_eef_m, touch_z_offset_m
        )
        plans.append(ContactPlan(point, p_base, T_above, T_near, T_touch))
    return plans


def _validate_plans(plans: list[ContactPlan], cfg: dict, max_below_table_m: float) -> None:
    workspace = cfg.get("workspace", {}) or {}
    xy_bounds = workspace.get("xy_bounds_m")
    z_max = workspace.get("z_max_m")
    table_z = workspace.get("table_z_m")

    if xy_bounds is not None:
        xmin, xmax = map(float, xy_bounds[0])
        ymin, ymax = map(float, xy_bounds[1])
        for plan in plans:
            for label, xyz in (
                (f"{plan.point.name} contact", plan.p_base),
                (f"{plan.point.name} above", plan.T_above[:3, 3]),
                (f"{plan.point.name} near", plan.T_near[:3, 3]),
                (f"{plan.point.name} touch", plan.T_touch[:3, 3]),
            ):
                if not (xmin <= float(xyz[0]) <= xmax and ymin <= float(xyz[1]) <= ymax):
                    raise RuntimeError(
                        f"{label} xy=({xyz[0]:.3f}, {xyz[1]:.3f}) is outside "
                        f"workspace x=[{xmin:.3f},{xmax:.3f}] y=[{ymin:.3f},{ymax:.3f}]"
                    )

    if table_z is not None:
        min_allowed = float(table_z) - float(max_below_table_m)
        for plan in plans:
            if float(plan.p_base[2]) < min_allowed:
                raise RuntimeError(
                    f"{plan.point.name} board z={plan.p_base[2]:.4f} m is below "
                    f"table_z_m={float(table_z):.4f} m by more than "
                    f"{max_below_table_m * 1000.0:.1f} mm; refusing contact"
                )

    if z_max is not None:
        max_allowed = float(z_max)
        for plan in plans:
            for label, pose in (
                (f"{plan.point.name} above", plan.T_above),
                (f"{plan.point.name} near", plan.T_near),
                (f"{plan.point.name} touch", plan.T_touch),
            ):
                if float(pose[2, 3]) > max_allowed:
                    raise RuntimeError(
                        f"{label} z={pose[2, 3]:.3f} m exceeds z_max_m={max_allowed:.3f} m"
                    )


def _draw_preview(
    color_bgr: np.ndarray,
    pattern: tuple[int, int],
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R_CT: np.ndarray,
    t_CT: np.ndarray,
    points: list[ContactPoint],
    square_m: float,
) -> np.ndarray:
    vis = color_bgr.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, True)
    rvec, _ = cv2.Rodrigues(R_CT)
    cv2.drawFrameAxes(vis, K, dist, rvec, t_CT, float(square_m) * 3.0)
    pts = np.asarray([p.p_board for p in points], dtype=np.float32).reshape(-1, 1, 3)
    image_pts, _ = cv2.projectPoints(pts, rvec, t_CT.reshape(3, 1), K, dist)
    for point, uv in zip(points, image_pts.reshape(-1, 2)):
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            point.name,
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        "checkerboard contact verification",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return vis


def _save_report(
    *,
    out_dir: Path,
    preview: np.ndarray,
    T_C_T: np.ndarray,
    T_B_T: np.ndarray,
    T_B_E_capture: np.ndarray,
    plans: list[ContactPlan],
    tool_offset_eef_m: np.ndarray,
    args: argparse.Namespace,
    log,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    image_path = out_dir / f"checkerboard_contact_{stamp}.jpg"
    npz_path = out_dir / f"checkerboard_contact_{stamp}.npz"
    json_path = out_dir / f"checkerboard_contact_{stamp}.json"

    if not cv2.imwrite(str(image_path), preview):
        raise RuntimeError(f"failed to write preview image: {image_path}")

    names = np.asarray([p.point.name for p in plans])
    p_board = np.stack([p.point.p_board for p in plans], axis=0)
    p_base = np.stack([p.p_base for p in plans], axis=0)
    T_above = np.stack([p.T_above for p in plans], axis=0)
    T_near = np.stack([p.T_near for p in plans], axis=0)
    T_touch = np.stack([p.T_touch for p in plans], axis=0)
    np.savez(
        npz_path,
        T_C_T=T_C_T,
        T_B_T=T_B_T,
        T_B_E_capture=T_B_E_capture,
        point_names=names,
        points_board=p_board,
        points_base=p_base,
        T_B_E_above=T_above,
        T_B_E_near=T_near,
        T_B_E_touch=T_touch,
        tool_offset_eef_m=np.asarray(tool_offset_eef_m, dtype=np.float64),
    )

    summary = {
        "preview_image": str(image_path.resolve()),
        "npz": str(npz_path.resolve()),
        "points": [
            {
                "name": p.point.name,
                "board_m": p.point.p_board.tolist(),
                "base_m": p.p_base.tolist(),
                "touch_eef_xyz_m": p.T_touch[:3, 3].tolist(),
            }
            for p in plans
        ],
        "tool_offset_eef_m": np.asarray(tool_offset_eef_m, dtype=float).tolist(),
        "approach_height_m": float(args.approach_height),
        "near_height_m": float(args.near_height),
        "touch_z_offset_m": float(args.touch_z_offset),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("saved preview: %s", image_path.resolve())
    log.info("saved contact report: %s", npz_path.resolve())
    log.info("saved summary: %s", json_path.resolve())


def _check_can_health(piper: PiperController, cfg: dict) -> None:
    min_fps = float(cfg.get("motion", {}).get("min_can_fps", 50.0))
    if not piper.is_ok():
        raise RuntimeError("CAN reader is not receiving frames; check power, CAN, and e-stop")
    fps = piper.can_fps()
    if fps < min_fps:
        raise RuntimeError(f"CAN fps {fps:.0f} Hz below minimum {min_fps:.0f} Hz")


def _move_or_raise(
    piper: PiperController,
    pose: np.ndarray,
    *,
    tag: str,
    speed_pct: int,
    pos_tol_m: float,
    ang_tol_deg: float,
    timeout_s: float,
    linear: bool,
    log,
) -> None:
    log.info("moving to %s: xyz=%s", tag, np.array2string(pose[:3, 3], precision=4))
    ok = piper.move_to_pose(
        pose,
        linear=linear,
        speed_pct=speed_pct,
        pos_tol_m=pos_tol_m,
        ang_tol_deg=ang_tol_deg,
        timeout_s=timeout_s,
    )
    if not ok:
        cur = piper.get_end_pose_matrix()
        err = cur[:3, 3] - pose[:3, 3]
        raise RuntimeError(
            f"timeout reaching {tag}; residual xyz error "
            f"({err[0] * 1000.0:.1f}, {err[1] * 1000.0:.1f}, {err[2] * 1000.0:.1f}) mm"
        )


def _prompt(message: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    input(message)


def main() -> None:
    cfg0 = load_config()
    chess = cfg0.get("chessboard", {}) or {}
    observe_cfg = cfg0.get("observe", {}) or {}

    ap = argparse.ArgumentParser(description="Checkerboard physical contact verification.")
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument("--pattern", nargs=2, type=int, default=list(chess.get("pattern_size", [11, 8])))
    ap.add_argument("--square", type=float, default=float(chess.get("square_size_m", 0.025)))
    ap.add_argument(
        "--points",
        choices=["center", "corners", "center-corners"],
        default="center",
        help="checkerboard points to touch; default is only the center point",
    )
    ap.add_argument(
        "--tool-offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "EEF-frame vector from end-effector origin to physical touch point, "
            "meters; defaults to grasp.tool_offset_eef_m from the config"
        ),
    )
    ap.add_argument("--approach-height", type=float, default=0.080)
    ap.add_argument("--near-height", type=float, default=0.015)
    ap.add_argument(
        "--touch-z-offset",
        type=float,
        default=0.0,
        help="base-z offset applied to detected board surface for contact; positive stops above it",
    )
    ap.add_argument(
        "--max-below-table",
        type=float,
        default=0.002,
        help="refuse contact if detected board surface is this far below workspace.table_z_m",
    )
    ap.add_argument("--observe-speed", type=int, default=int(observe_cfg.get("move_speed_pct", 20)))
    ap.add_argument("--contact-speed", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--settle", type=float, default=None)
    ap.add_argument("--touch-dwell", type=float, default=0.5)
    ap.add_argument("--can-port", type=str, default=None)
    ap.add_argument("--skip-observe", action="store_true", help="do not move to observe first")
    ap.add_argument("--plan-only", action="store_true", help="detect and save target poses, but do not touch")
    ap.add_argument("--yes", action="store_true", help="skip interactive prompts")
    ap.add_argument("--no-gui", action="store_true", help="save preview image without displaying it")
    ap.add_argument("--timeout-ms", type=int, default=5000, help="camera frame wait timeout")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("grasp_system/runs/contact_verify"),
        help="directory for preview and contact report artifacts",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log = get_logger("contact-verify", level=getattr(logging, level_name, logging.INFO))

    pattern = (int(args.pattern[0]), int(args.pattern[1]))
    square_m = float(args.square)
    if square_m <= 0.0:
        raise ValueError("--square must be positive")
    if args.approach_height <= args.near_height:
        raise ValueError("--approach-height must be greater than --near-height")
    if args.near_height < max(0.0, args.touch_z_offset):
        raise ValueError("--near-height must be above --touch-z-offset")

    T_E_C = load_npy(project_path(cfg["paths"]["t_eef_cam"]))
    if T_E_C.shape != (4, 4):
        raise RuntimeError("T_E_C must be a 4x4 matrix")
    validate_transform(T_E_C, name="T_E_C", max_translation_m=0.5, logger=log)
    intr = load_intrinsics(project_path(cfg["paths"]["intrinsics"]))
    objp = _build_object_points(pattern, square_m)
    points = _contact_points(pattern, square_m, str(args.points))
    if args.tool_offset is None:
        tool_offset = np.asarray(
            cfg.get("grasp", {}).get("tool_offset_eef_m", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
    else:
        tool_offset = np.asarray(args.tool_offset, dtype=np.float64).reshape(3)
    log.info(
        "using contact tool offset in EEF frame: %s m",
        np.array2string(tool_offset, precision=3),
    )

    if np.allclose(tool_offset, 0.0):
        log.warning(
            "tool offset is zero: contact target is the EEF origin. "
            "Set grasp.tool_offset_eef_m or pass --tool-offset X Y Z."
        )

    cam_cfg = cfg.get("camera", {}) or {}
    piper_cfg = cfg.get("piper", {}) or {}
    motion_cfg = cfg.get("motion", {}) or {}
    can_port = args.can_port if args.can_port is not None else piper_cfg.get("can_port", "can0")
    install_pos = piper_cfg.get("installation_pos")
    cart_timeout = float(
        args.timeout
        if args.timeout is not None
        else motion_cfg.get("cartesian_timeout_s", 8.0)
    )
    settle_s = float(
        args.settle
        if args.settle is not None
        else cfg.get("timing", {}).get("observe_settle_s", 0.5)
    )
    pos_tol = float(motion_cfg.get("pos_tol_m", 0.003))
    ang_tol = float(motion_cfg.get("ang_tol_deg", 1.0))

    with RealSenseCamera(
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
        color_fps=int(cam_cfg.get("color_fps", 30)),
        depth_fps=int(cam_cfg.get("depth_fps", 30)),
        align_to=str(cam_cfg.get("align_to", "color")),
        depth_scale=float(cam_cfg.get("depth_scale", 0.001)),
        warmup_frames=int(cam_cfg.get("warmup_frames", 30)),
        spatial_filter=bool(cam_cfg.get("spatial_filter", False)),
        temporal_filter=bool(cam_cfg.get("temporal_filter", False)),
        hole_filling_filter=bool(cam_cfg.get("hole_filling_filter", False)),
    ) as cam, PiperController(
        can_port=str(can_port),
        installation_pos=install_pos,
    ) as piper:
        piper.set_joint_limits_from_config(cfg)
        piper.disable_on_disconnect = False
        _check_can_health(piper, cfg)

        observe_pose = load_npy(project_path(cfg["paths"]["observe_joints"]))
        if observe_pose.shape != (6,):
            raise RuntimeError("observe_joints.npy must contain 6 joint angles")
        log.info("observe joints target (deg): %s", np.array2string(np.rad2deg(observe_pose), precision=3))

        if not args.skip_observe:
            _prompt(
                "Release the physical teach/drag button, keep clear of the arm, "
                "then press ENTER to move to observe pose...",
                bool(args.yes),
            )
            log.info("moving to observe pose at speed %d%%", int(args.observe_speed))
            ok = piper.move_joints_rad(
                observe_pose,
                speed_pct=int(args.observe_speed),
                tol_deg=1.0,
                timeout_s=float(observe_cfg.get("arrive_timeout_s", 20.0)),
            )
            if not ok:
                current = piper.get_joints_rad()
                raise RuntimeError(
                    "observe pose was not reached; remaining joint error (deg): "
                    + np.array2string(np.rad2deg(observe_pose - current), precision=3)
                )
            time.sleep(settle_s)

        color, _ = cam.grab_aligned(timeout_ms=int(args.timeout_ms))
        K = np.asarray(intr["K"], dtype=np.float64)
        dist = np.asarray(intr["dist"], dtype=np.float64)
        if (cam.intrinsics.width, cam.intrinsics.height) != (intr["width"], intr["height"]):
            log.warning(
                "calibration resolution %dx%d differs from live stream %dx%d; rescaling K",
                intr["width"],
                intr["height"],
                cam.intrinsics.width,
                cam.intrinsics.height,
            )
            K = rescale_intrinsics(
                K,
                src_size=(intr["width"], intr["height"]),
                dst_size=(cam.intrinsics.width, cam.intrinsics.height),
            )

        ok, R_CT, t_CT, corners = _detect_chessboard_pose(color, pattern, objp, K, dist)
        if not ok or R_CT is None or t_CT is None or corners is None:
            raw_path = args.out_dir / f"checkerboard_not_found_{_timestamp()}.jpg"
            ensure_parent(raw_path)
            cv2.imwrite(str(raw_path), color)
            raise RuntimeError(f"checkerboard was not detected; saved frame: {raw_path.resolve()}")

        T_C_T = make_transform(R_CT, t_CT)
        T_B_E_capture = piper.get_end_pose_matrix()
        T_B_T = T_B_E_capture @ T_E_C @ T_C_T
        validate_transform(T_B_T, name="T_B_T", logger=log)

        plans = _build_contact_plans(
            points=points,
            T_B_T=T_B_T,
            T_B_E_reference=T_B_E_capture,
            tool_offset_eef_m=tool_offset,
            approach_height_m=float(args.approach_height),
            near_height_m=float(args.near_height),
            touch_z_offset_m=float(args.touch_z_offset),
        )
        _validate_plans(plans, cfg, max_below_table_m=float(args.max_below_table))

        log.info("detected board origin in base: %s", np.array2string(T_B_T[:3, 3], precision=4))
        for plan in plans:
            log.info(
                "%s contact point base xyz=%s; touch EEF xyz=%s",
                plan.point.name,
                np.array2string(plan.p_base, precision=4),
                np.array2string(plan.T_touch[:3, 3], precision=4),
            )

        preview = _draw_preview(
            color,
            pattern,
            corners,
            K,
            dist,
            R_CT,
            t_CT,
            points,
            square_m,
        )
        _save_report(
            out_dir=args.out_dir,
            preview=preview,
            T_C_T=T_C_T,
            T_B_T=T_B_T,
            T_B_E_capture=T_B_E_capture,
            plans=plans,
            tool_offset_eef_m=tool_offset,
            args=args,
            log=log,
        )

        if not args.no_gui:
            cv2.namedWindow("checkerboard contact verification", cv2.WINDOW_NORMAL)
            cv2.imshow("checkerboard contact verification", preview)
            cv2.waitKey(1)

        if args.plan_only:
            log.info("plan-only requested; skipping physical contact")
            return

        _prompt(
            "Physical contact will start now. Keep one hand near e-stop; "
            "press ENTER to continue...",
            bool(args.yes),
        )
        for plan in plans:
            _prompt(
                f"Next point: {plan.point.name}. Press ENTER to move above it...",
                bool(args.yes),
            )
            _move_or_raise(
                piper,
                plan.T_above,
                tag=f"{plan.point.name} above",
                speed_pct=int(args.contact_speed),
                pos_tol_m=pos_tol,
                ang_tol_deg=ang_tol,
                timeout_s=cart_timeout,
                linear=False,
                log=log,
            )
            _move_or_raise(
                piper,
                plan.T_near,
                tag=f"{plan.point.name} near",
                speed_pct=max(1, int(args.contact_speed)),
                pos_tol_m=pos_tol,
                ang_tol_deg=ang_tol,
                timeout_s=cart_timeout,
                linear=True,
                log=log,
            )
            _prompt(
                f"Ready to descend to {plan.point.name} contact. Press ENTER...",
                bool(args.yes),
            )
            _move_or_raise(
                piper,
                plan.T_touch,
                tag=f"{plan.point.name} touch",
                speed_pct=max(1, min(5, int(args.contact_speed))),
                pos_tol_m=pos_tol,
                ang_tol_deg=ang_tol,
                timeout_s=cart_timeout,
                linear=True,
                log=log,
            )
            time.sleep(float(args.touch_dwell))
            cur = piper.get_end_pose_matrix()
            actual_contact = cur[:3, 3] + cur[:3, :3] @ tool_offset
            log.info(
                "%s actual contact estimate base xyz=%s; error to target=%s mm",
                plan.point.name,
                np.array2string(actual_contact, precision=4),
                np.array2string((actual_contact - plan.p_base) * 1000.0, precision=2),
            )
            _move_or_raise(
                piper,
                plan.T_above,
                tag=f"{plan.point.name} retreat",
                speed_pct=max(1, int(args.contact_speed)),
                pos_tol_m=pos_tol,
                ang_tol_deg=ang_tol,
                timeout_s=cart_timeout,
                linear=True,
                log=log,
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
