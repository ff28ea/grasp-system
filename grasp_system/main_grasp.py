"""End-to-end pick loop for the eye-in-hand system.

Steps:
    1. Connect to PiPER + D435i, load calibration files.
    2. Move to the saved observe pose.
    3. Capture an aligned RGB-D frame.
    4. Run YOLOv8-seg and pick the top detection (or the requested class).
    5. Back-project the mask, clean the point cloud, fit an OBB -> T_C_O.
    6. Compose T_B_O = T_B_E * T_E_C * T_C_O, using the arm's current
       end-pose read just before the shot.
    7. Optionally repeat from a close-up pose (active perception) and
       fuse the two estimates.
    8. Plan a top-down grasp from the OBB, execute:
       pre-grasp -> approach (MOVE_L) -> close gripper -> lift -> place
       -> home.

This is a reference integration; you will likely tune thresholds, speeds
and the place pose for your exact workspace.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .common import (
    camera_look_down_rotation,
    get_logger,
    invert_transform,
    load_classes,
    load_config,
    load_intrinsics,
    load_npy,
    project_path,
    validate_transform,
)
from .control.piper_controller import PiperController
from .perception.camera import RealSenseCamera, depth_to_meters
from .perception.detector import Detection, SegmentationDetector
from .perception.pose_estimator import (
    backproject_mask_to_pointcloud,
    clean_pointcloud,
    estimate_pose_from_obb,
)
from .planning.grasp_planner import (
    GraspCandidate,
    align_obb_axes,
    fuse_poses,
    plan_topdown_grasp,
)


# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------
def _perceive_object_in_camera(
    cam: RealSenseCamera,
    detector: SegmentationDetector,
    cfg: dict,
    target_class: Optional[int],
    log,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
):
    p_cfg = cfg["perception"]
    color, depth_raw = cam.grab_aligned()
    depth_m = depth_to_meters(depth_raw, cam.depth_scale)

    detections = detector.predict(
        color,
        conf=p_cfg["conf_threshold"],
        min_pixels=int(p_cfg["min_mask_pixels"]),
        mask_erode_px=int(p_cfg["mask_erode_px"]),
        target_hw=depth_m.shape,
    )
    if not detections:
        raise RuntimeError("no instances detected")

    if target_class is not None:
        detections = [d for d in detections if d.class_id == target_class]
        if not detections:
            raise RuntimeError(f"class {target_class} not detected")

    det = max(detections, key=lambda d: d.confidence)
    log.info(
        "detection: class=%d label=%s conf=%.2f pixels=%d",
        det.class_id,
        det.label,
        det.confidence,
        det.num_pixels,
    )

    # Prefer the calibrated intrinsics + distortion from configs/camera_intrinsics.npz
    # over the RealSense factory values. The two agree on fx/fy/cx/cy within a
    # couple of pixels but disagree on lens distortion, which is the bit that
    # actually matters for 3D back-projection at the image corners.
    if K is None:
        K = cam.intrinsics.K
    pcd_raw = backproject_mask_to_pointcloud(
        color,
        depth_m,
        det.mask,
        K,
        depth_trunc_m=float(p_cfg["depth_trunc_m"]),
        dist=dist,
    )
    if len(pcd_raw.points) < 50:
        raise RuntimeError("mask produced too few 3D points")

    so = p_cfg["statistical_outlier"]
    plane = p_cfg["plane_segmentation"]
    pcd = clean_pointcloud(
        pcd_raw,
        voxel_size_m=float(p_cfg["voxel_size_m"]),
        outlier_nb_neighbors=int(so["nb_neighbors"]),
        outlier_std_ratio=float(so["std_ratio"]),
        remove_plane=True,
        plane_distance_threshold_m=float(plane["distance_threshold_m"]),
        plane_ransac_n=int(plane["ransac_n"]),
        plane_num_iterations=int(plane["num_iterations"]),
    )
    if len(pcd.points) < 20:
        raise RuntimeError("cleaned point cloud too sparse")

    pose = estimate_pose_from_obb(pcd)
    return det, pose


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def _go_to_observe(piper: PiperController, cfg: dict, log) -> None:
    joints_path = project_path(cfg["paths"]["observe_joints"])
    if not Path(joints_path).exists():
        raise FileNotFoundError(
            f"missing {joints_path}; run tools.teach_observe first"
        )
    joints = load_npy(joints_path)
    speed = int(cfg["observe"]["move_speed_pct"])
    log.info("moving to observe pose (joints rad=%s)", np.array2string(joints, precision=3))
    piper.joint_ctrl_rad(joints, speed_pct=speed)
    timeout = float(cfg["observe"]["arrive_timeout_s"])
    if not piper.wait_joints_arrive(joints, tol_deg=1.0, timeout_s=timeout):
        log.warning("observe-pose arrival timeout; continuing anyway")
    time.sleep(0.5)


def _snapshot_T_B_O(
    piper: PiperController,
    cam: RealSenseCamera,
    detector: SegmentationDetector,
    cfg: dict,
    target_class: Optional[int],
    T_E_C: np.ndarray,
    log,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
):
    # Ensure arm is fully settled so the end-pose feedback matches the frame.
    time.sleep(float(cfg["handeye"].get("settle_time_s", 1.0)) * 0.5)
    T_B_E = piper.get_end_pose_matrix()
    det, pose_cam = _perceive_object_in_camera(
        cam, detector, cfg, target_class, log, K=K, dist=dist
    )
    T_B_O = T_B_E @ T_E_C @ pose_cam.T_C_O
    return det, pose_cam, T_B_E, T_B_O


def _active_perception_close_up(
    piper: PiperController,
    cam: RealSenseCamera,
    detector: SegmentationDetector,
    cfg: dict,
    T_E_C: np.ndarray,
    T_B_O_rough: np.ndarray,
    extent_rough: np.ndarray,
    target_class: Optional[int],
    log,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
):
    """Move the camera above ``T_B_O_rough`` and re-capture a close-up frame."""
    ap_cfg = cfg["active_perception"]
    near_h = float(ap_cfg["near_height_m"])
    tilt = float(ap_cfg.get("look_down_tilt_deg", 0.0))

    # Sanity: the camera must stay above the tallest OBB axis plus some
    # margin, otherwise we risk driving the EEF into the object while
    # aiming for the close-up. ``extent_rough`` is the full side length
    # of the OBB, so half the max dimension is the worst-case vertical
    # clearance; add a 5 cm safety margin on top of that.
    min_clearance = 0.5 * float(np.max(extent_rough)) + 0.05
    if near_h < min_clearance:
        log.warning(
            "near_height_m=%.3f m is below object clearance %.3f m; raising",
            near_h, min_clearance,
        )
        near_h = min_clearance

    # Camera target pose in base frame: above the object, looking down.
    T_B_C_near = np.eye(4)
    T_B_C_near[:3, :3] = camera_look_down_rotation(tilt_deg=tilt, yaw_deg=0.0)
    T_B_C_near[:3, 3] = T_B_O_rough[:3, 3] + np.array([0.0, 0.0, near_h])

    # T_B_C = T_B_E * T_E_C  =>  T_B_E = T_B_C * inv(T_E_C)
    T_B_E_near = T_B_C_near @ invert_transform(T_E_C)

    # Safety: limit how far we move on z so we don't crash through the object.
    cur_pose = piper.get_end_pose_matrix()
    dz = float(T_B_E_near[2, 3] - cur_pose[2, 3])
    if dz < -near_h:
        log.warning("close-up z delta %.3f m too negative; clipping", dz)
        T_B_E_near[2, 3] = cur_pose[2, 3] - near_h

    log.info(
        "moving to close-up pose at camera z=%.3f m above object",
        T_B_C_near[2, 3],
    )
    speed = int(cfg["piper"]["cartesian_move_speed_pct"])
    m_cfg = cfg.get("motion", {})
    pos_tol = float(m_cfg.get("pos_tol_m", 0.003))
    ang_tol = float(m_cfg.get("ang_tol_deg", 1.0))
    cart_timeout = float(m_cfg.get("cartesian_timeout_s", 8.0))
    piper.end_pose_ctrl(T_B_E_near, linear=False, speed_pct=speed)
    piper.wait_cartesian_arrive(
        T_B_E_near, pos_tol_m=pos_tol, ang_tol_deg=ang_tol, timeout_s=cart_timeout
    )
    time.sleep(0.4)

    return _snapshot_T_B_O(
        piper, cam, detector, cfg, target_class, T_E_C, log, K=K, dist=dist
    )


def _execute_grasp(
    piper: PiperController,
    cfg: dict,
    grasp: GraspCandidate,
    place_pose_base: Optional[np.ndarray],
    log,
) -> None:
    g_cfg = cfg["grasp"]
    speed_cart = int(cfg["piper"]["cartesian_move_speed_pct"])
    effort_mNm = float(cfg["piper"]["default_effort_mNm"])

    # Read cartesian arrival tolerances from [motion] so they are tunable
    # from system.yaml rather than scattered hard-coded values.
    m_cfg = cfg.get("motion", {})
    pos_tol = float(m_cfg.get("pos_tol_m", 0.003))
    ang_tol = float(m_cfg.get("ang_tol_deg", 1.0))
    cart_timeout = float(m_cfg.get("cartesian_timeout_s", 8.0))

    log.info("opening gripper to %.3f m", float(g_cfg["max_opening_m"]))
    piper.open_gripper(
        opening_m=float(g_cfg["max_opening_m"]),
        effort_mNm=effort_mNm,
    )
    time.sleep(0.3)

    pre = grasp.pre_grasp
    log.info(
        "moving to pre-grasp: xyz=%s",
        np.array2string(pre[:3, 3], precision=3),
    )
    piper.end_pose_ctrl(pre, linear=False, speed_pct=speed_cart)
    piper.wait_cartesian_arrive(
        pre, pos_tol_m=pos_tol, ang_tol_deg=ang_tol, timeout_s=cart_timeout
    )
    time.sleep(0.2)

    log.info("linear approach to grasp")
    piper.end_pose_ctrl(grasp.T_B_grasp, linear=True, speed_pct=max(5, speed_cart // 2))
    piper.wait_cartesian_arrive(
        grasp.T_B_grasp, pos_tol_m=pos_tol, ang_tol_deg=ang_tol, timeout_s=cart_timeout
    )
    time.sleep(0.2)

    # Close to ``close_opening_m`` (narrower than the object) so the jaws
    # squeeze the object. Using open_gripper() keeps the unit conversion
    # (m -> mm) in one place (PiperController), rather than repeating the
    # *1000 factor here.
    log.info(
        "closing gripper to %.1f mm (object width %.1f mm)",
        grasp.close_opening_m * 1000.0,
        grasp.object_width_m * 1000.0,
    )
    piper.open_gripper(
        opening_m=grasp.close_opening_m,
        effort_mNm=effort_mNm,
    )
    time.sleep(0.6)

    log.info("lifting")
    lift_pose = grasp.lift
    piper.end_pose_ctrl(lift_pose, linear=True, speed_pct=max(5, speed_cart // 2))
    piper.wait_cartesian_arrive(
        lift_pose, pos_tol_m=pos_tol, ang_tol_deg=ang_tol, timeout_s=cart_timeout
    )
    time.sleep(0.2)

    if place_pose_base is not None:
        log.info("moving to place pose")
        piper.end_pose_ctrl(place_pose_base, linear=False, speed_pct=speed_cart)
        piper.wait_cartesian_arrive(
            place_pose_base, pos_tol_m=pos_tol, ang_tol_deg=ang_tol, timeout_s=cart_timeout
        )
        piper.open_gripper(
            opening_m=float(g_cfg["max_opening_m"]),
            effort_mNm=effort_mNm,
        )
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Eye-in-hand grasp loop.")
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="only grasp instances of this class id",
    )
    ap.add_argument(
        "--no-active-perception",
        action="store_true",
        help="skip the close-up re-estimation step",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="perceive + plan but do not move the arm during the grasp",
    )
    args = ap.parse_args()

    log = get_logger("main")
    cfg = load_config(args.config)

    # -- load calibration artefacts ------------------------------------
    T_E_C = load_npy(project_path(cfg["paths"]["t_eef_cam"]))
    if T_E_C.shape != (4, 4):
        raise RuntimeError("T_E_C must be a 4x4 matrix")
    validate_transform(
        T_E_C,
        name="T_E_C",
        max_translation_m=0.5,   # camera should be within 50 cm of the EEF
        logger=log,
    )
    intr = load_intrinsics(project_path(cfg["paths"]["intrinsics"]))
    classes = load_classes(cfg["paths"]["classes"])
    log.info(
        "loaded T_E_C, intrinsics (%dx%d, rms=%s), classes=%s",
        intr["width"],
        intr["height"],
        intr.get("rms", "?"),
        list(classes.keys()),
    )

    class_names = {k: v.get("name", str(k)) for k, v in classes.items()}
    detector = SegmentationDetector(
        model_path=project_path(cfg["paths"]["model"]),
        conf=float(cfg["perception"]["conf_threshold"]),
        class_names=class_names,
    )

    # -- hardware ------------------------------------------------------
    can_port = cfg["piper"]["can_port"]
    K_cal = intr["K"]
    dist_cal = intr["dist"]
    with RealSenseCamera(
        width=int(cfg["camera"]["width"]),
        height=int(cfg["camera"]["height"]),
        color_fps=int(cfg["camera"]["color_fps"]),
        depth_fps=int(cfg["camera"]["depth_fps"]),
        align_to=str(cfg["camera"].get("align_to", "color")),
        depth_scale=float(cfg["camera"]["depth_scale"]),
        warmup_frames=int(cfg["camera"].get("warmup_frames", 30)),
    ) as cam, PiperController(can_port=can_port) as piper:

        # Sanity-check calibration vs. runtime stream.
        rs_intr = cam.intrinsics
        if (rs_intr.width, rs_intr.height) != (intr["width"], intr["height"]):
            log.warning(
                "calibration resolution %dx%d differs from live stream %dx%d; "
                "consider re-running calibrate_intrinsics at the current settings",
                intr["width"], intr["height"], rs_intr.width, rs_intr.height,
            )

        # 1) observe pose
        _go_to_observe(piper, cfg, log)

        # 2) rough perception
        det, pose_cam_rough, _, T_B_O_rough = _snapshot_T_B_O(
            piper, cam, detector, cfg, args.target_class, T_E_C, log,
            K=K_cal, dist=dist_cal,
        )
        log.info(
            "T_B_O (rough) translation: %s",
            np.array2string(T_B_O_rough[:3, 3], precision=4),
        )

        # 3) optional close-up active perception
        T_B_O_final = T_B_O_rough
        extent_final = pose_cam_rough.extent
        if cfg["active_perception"].get("enable", True) and not args.no_active_perception:
            try:
                det_f, pose_cam_fine, _, T_B_O_fine = _active_perception_close_up(
                    piper,
                    cam,
                    detector,
                    cfg,
                    T_E_C,
                    T_B_O_rough,
                    pose_cam_rough.extent,
                    args.target_class,
                    log,
                    K=K_cal,
                    dist=dist_cal,
                )
                ap_cfg = cfg["active_perception"]
                # Align the fine OBB axes to the rough ones *before* fusing,
                # so that fuse_poses blends rotations that mean the same
                # thing physically. Without this, a 90 deg axis swap would
                # slerp through a garbage mid-pose, and extent_final would
                # describe a different axis than R in the fused result.
                R_fine_aligned, extent_fine_aligned = align_obb_axes(
                    T_B_O_rough[:3, :3],
                    pose_cam_rough.extent,
                    T_B_O_fine[:3, :3],
                    pose_cam_fine.extent,
                )
                T_B_O_fine_aligned = T_B_O_fine.copy()
                T_B_O_fine_aligned[:3, :3] = R_fine_aligned
                T_B_O_final = fuse_poses(
                    T_B_O_rough,
                    T_B_O_fine_aligned,
                    weight_rough=float(ap_cfg.get("fuse_rough_weight", 0.3)),
                    weight_fine=float(ap_cfg.get("fuse_fine_weight", 0.7)),
                )
                extent_final = extent_fine_aligned
                log.info(
                    "T_B_O (fused) translation: %s",
                    np.array2string(T_B_O_final[:3, 3], precision=4),
                )
            except Exception as exc:
                log.warning("active perception failed (%s); using rough estimate", exc)

        # 4) class-specific planner tweaks (approach, gripper max)
        cinfo = classes.get(det.class_id, {}) if classes else {}
        max_opening_m = float(cinfo.get("max_opening_mm", cfg["grasp"]["max_opening_m"] * 1000.0)) / 1000.0
        lift_m = float(cinfo.get("safe_height_mm", cfg["grasp"]["lift_height_m"] * 1000.0)) / 1000.0

        grasp = plan_topdown_grasp(
            T_B_O=T_B_O_final,
            extent=extent_final,
            approach_world=tuple(cfg["grasp"]["approach_direction_world"]),
            opening_margin_m=float(cfg["grasp"]["opening_margin_m"]),
            close_overclose_m=float(cfg["grasp"].get("close_overclose_m", 0.002)),
            approach_m=float(cfg["grasp"]["pre_grasp_offset_m"]),
            lift_m=lift_m,
            max_opening_m=max_opening_m,
        )
        log.info(
            "grasp plan: opening=%.1f mm, feasible=%s, reason=%s",
            grasp.opening_m * 1000.0,
            grasp.feasible,
            grasp.reason,
        )
        if not grasp.feasible:
            log.error("aborting: %s", grasp.reason)
            return

        if args.dry_run:
            log.info("dry-run; skipping motion")
            return

        # 5) Build place pose from config. Position comes from ``place.xyz_m``;
        # the orientation is an independent top-down frame (not copied from the
        # grasp R, since a tilted grasp would then drop the object sideways).
        place_cfg = cfg.get("place", {}) or {}
        place_xyz = place_cfg.get("xyz_m", [0.30, -0.15, 0.18])
        place_R = camera_look_down_rotation(
            tilt_deg=float(place_cfg.get("tilt_deg", 0.0)),
            yaw_deg=float(place_cfg.get("yaw_deg", 0.0)),
        )
        place = np.eye(4)
        place[:3, :3] = place_R
        place[:3, 3] = [float(x) for x in place_xyz]

        _execute_grasp(piper, cfg, grasp, place_pose_base=place, log=log)

        # 6) return home by going back to observe pose.
        _go_to_observe(piper, cfg, log)
        log.info("done")


if __name__ == "__main__":
    main()
