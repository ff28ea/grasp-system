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
import dataclasses
import logging
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
    rescale_intrinsics,
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
from .perception.pose_estimator import refine_with_icp  # noqa: F401 (lazy; used in _refine_pose_with_icp_if_enabled)
from .tools.visualize import (
    build_scene_geometries,
    make_detection_overlay,
    save_image,
    save_pointcloud,
    save_poses_npz,
    show_scene,
)

try:  # open3d is optional at import time; only needed if visualization is requested
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _up_direction_in_cam(T_B_E: np.ndarray, T_E_C: np.ndarray) -> np.ndarray:
    """Express the base-frame +z axis in camera coordinates.

    Used to disambiguate OBB axis signs and to orient the table-plane
    removal RANSAC. Rotating world +z into the camera frame requires
    only the rotation block of ``T_B_C^{-1}``.
    """
    T_B_C = T_B_E @ T_E_C
    R_C_B = T_B_C[:3, :3].T     # cam <- base rotation (orthonormal)
    return R_C_B @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _refine_pose_with_icp_if_enabled(
    pcd_cam,
    pose_cam,
    cfg: dict,
    classes: dict,
    class_id: int,
    log,
):
    """Optionally refine ``pose_cam.T_C_O`` by ICP against a class template.

    Returns the (possibly-updated) ``pose_cam`` or the original when
    ICP is disabled, the template is missing, or the fit is poor.

    Activation requires *all* of:
      * ``perception.icp.enable: true`` in ``system.yaml``
      * a per-class ``template_ply`` path in ``classes.yaml`` that
        resolves to a non-empty point cloud
      * a successful ICP run with ``fitness >= fitness_threshold`` and
        ``inlier_rmse <= inlier_rmse_threshold_m``.

    This makes the refinement strictly opt-in: the default behaviour
    after enabling ``icp`` is unchanged for classes without a template.
    """
    icp_cfg = cfg.get("perception", {}).get("icp", {}) or {}
    if not bool(icp_cfg.get("enable", False)):
        return pose_cam
    if o3d is None:
        log.warning("ICP requested but open3d is not installed; skipping")
        return pose_cam

    cinfo = classes.get(class_id, {}) or {}
    template_rel = cinfo.get("template_ply")
    if not template_rel:
        # No template for this class -> nothing to align against. The
        # OBB pose is still perfectly usable; this branch exists so
        # users can mix templated and non-templated classes in one
        # config.
        return pose_cam

    template_path = project_path(template_rel)
    if not template_path.exists():
        log.warning(
            "class %d template %s missing; skipping ICP", class_id, template_path,
        )
        return pose_cam

    try:
        template = o3d.io.read_point_cloud(str(template_path))
    except Exception as exc:
        log.warning("failed to read template %s: %s; skipping ICP", template_path, exc)
        return pose_cam
    if len(template.points) == 0:
        log.warning("template %s has no points; skipping ICP", template_path)
        return pose_cam

    # The template is assumed to be expressed in the *object* frame
    # (origin at the object center, axes aligned with the OBB). We
    # align the observed scene cloud to the template-in-camera-frame by
    # transforming the template with the current pose guess.
    template_in_cam = o3d.geometry.PointCloud(template)
    template_in_cam.transform(pose_cam.T_C_O)

    try:
        T_delta, fitness, rmse = refine_with_icp(
            template_in_cam,
            pcd_cam,
            voxel_size_m=float(icp_cfg.get("voxel_size_m", 0.003)),
        )
    except Exception as exc:
        log.warning("ICP raised %s; keeping OBB pose", exc)
        return pose_cam

    min_fit = float(icp_cfg.get("fitness_threshold", 0.6))
    max_rmse = float(icp_cfg.get("inlier_rmse_threshold_m", 0.005))
    log.info("ICP: fitness=%.3f rmse=%.4f m", fitness, rmse)
    if fitness < min_fit or rmse > max_rmse:
        log.warning(
            "ICP quality below thresholds (fit>=%.2f, rmse<=%.3f m); "
            "falling back to OBB pose", min_fit, max_rmse,
        )
        return pose_cam

    # ``T_delta`` takes the template-in-cam (pose guess) to the observed
    # cloud in cam. Compose with the OBB pose to get the refined T_C_O.
    T_C_O_refined = T_delta @ pose_cam.T_C_O
    # Return a shallow copy so we don't mutate the estimator's output.
    return dataclasses.replace(
        pose_cam,
        T_C_O=T_C_O_refined,
        icp_fitness=float(fitness),
        icp_inlier_rmse=float(rmse),
    )


def _resolve_target_class(
    arg: Optional[str], classes: dict[int, dict]
) -> Optional[int]:
    """Turn a CLI ``--target-class`` value into a class id.

    Accepts either a numeric string (``"2"``) or the class ``name``
    field as declared in ``configs/classes.yaml``. Case-insensitive for
    names. Returns None when the caller passed no filter.
    """
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return int(s)
    lookup = {str(v.get("name", "")).lower(): int(k) for k, v in classes.items()}
    key = s.lower()
    if key not in lookup:
        available = ", ".join(sorted(lookup.keys()))
        raise ValueError(
            f"unknown --target-class {arg!r}; "
            f"classes.yaml defines: {available or '<none>'}"
        )
    return lookup[key]


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
    up_in_cam: Optional[np.ndarray] = None,
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
        raise RuntimeError(
            f"mask produced too few 3D points ({len(pcd_raw.points)}); "
            "check lighting / depth_trunc_m"
        )

    so = p_cfg["statistical_outlier"]
    plane = p_cfg["plane_segmentation"]
    ro_cfg = p_cfg.get("radius_outlier", {}) or {}
    pcd = clean_pointcloud(
        pcd_raw,
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
    if len(pcd.points) < 20:
        raise RuntimeError(
            f"cleaned point cloud too sparse ({len(pcd.points)}); "
            "consider lowering mask_erode_px or disabling plane removal"
        )

    pose = estimate_pose_from_obb(pcd, up_world_in_cam=up_in_cam)
    # Return the raw color image and the cleaned point cloud alongside the
    # pose so downstream visualization has access to the same data the
    # estimator used. Cheap: both are already in memory.
    return det, pose, color, pcd


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def _go_to_observe(
    piper: PiperController,
    cfg: dict,
    log,
) -> None:
    joints_path = project_path(cfg["paths"]["observe_joints"])
    if not Path(joints_path).exists():
        raise FileNotFoundError(
            f"missing {joints_path}; run tools.teach_observe first"
        )
    joints = load_npy(joints_path)
    speed = int(cfg["observe"]["move_speed_pct"])
    log.info("moving to observe pose (joints rad=%s)", np.array2string(joints, precision=3))
    timeout = float(cfg["observe"]["arrive_timeout_s"])
    if not piper.move_joints_rad(
        joints,
        speed_pct=speed,
        tol_deg=1.0,
        timeout_s=timeout,
    ):
        current = piper.get_joints_rad()
        log.error(
            "observe-pose arrival timeout; current joints (deg): %s",
            np.array2string(np.rad2deg(current), precision=3),
        )
        log.error(
            "remaining joint error (deg): %s",
            np.array2string(np.rad2deg(joints - current), precision=3),
        )
        raise RuntimeError(
            "observe pose was not reached; aborting before perception. "
            "Use `python -m grasp_system.tools.test_observe_pose` to diagnose motion."
        )
    time.sleep(0.5)


def _safe_retreat(
    piper: PiperController,
    cfg: dict,
    log,
) -> None:
    """Open the gripper and lift straight up before homing.

    Called in the ``finally`` clause so that, regardless of where the
    pipeline failed, the arm ends up in a safe state: no object
    clamped in the jaws, and the tool tip well above the table before
    we start a joint-space motion to the observe pose (joint paths
    don't respect cartesian obstacles).
    """
    # Use .get() with safe defaults so this function never raises a
    # KeyError — it runs inside exception handlers where cfg may be
    # partially initialised.
    grasp_cfg = cfg.get("grasp", {}) or {}
    piper_cfg = cfg.get("piper", {}) or {}
    max_opening = float(grasp_cfg.get("max_opening_m", 0.070))
    effort = float(piper_cfg.get("default_effort_mNm", 1000.0))

    try:
        piper.open_gripper(opening_m=max_opening, effort_mNm=effort)
    except Exception as exc:
        log.warning("safe_retreat: open_gripper failed (%s)", exc)

    retreat_dz = float(cfg.get("motion", {}).get("retreat_lift_m", 0.08))
    if retreat_dz <= 0.0:
        return
    try:
        cur = piper.get_end_pose_matrix()
        target = cur.copy()
        target[2, 3] += retreat_dz
        m_cfg = cfg.get("motion", {})
        piper.move_to_pose(
            target,
            linear=True,
            speed_pct=max(5, int(piper_cfg.get("cartesian_move_speed_pct", 20)) // 2),
            pos_tol_m=float(m_cfg.get("pos_tol_m", 0.003)),
            ang_tol_deg=float(m_cfg.get("ang_tol_deg", 1.0)),
            timeout_s=float(m_cfg.get("cartesian_timeout_s", 8.0)),
        )
    except Exception as exc:
        log.warning("safe_retreat: vertical lift failed (%s)", exc)


def _snapshot_T_B_O(
    piper: PiperController,
    cam: RealSenseCamera,
    detector: SegmentationDetector,
    cfg: dict,
    target_class: Optional[int],
    T_E_C: np.ndarray,
    log,
    classes: Optional[dict] = None,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    retries: int = 2,
    retry_delay_s: float = 0.3,
):
    # Ensure arm is fully settled so the end-pose feedback matches the frame.
    time.sleep(float(cfg["handeye"].get("settle_time_s", 1.0)) * 0.5)
    T_B_E = piper.get_end_pose_matrix()
    up_in_cam = _up_direction_in_cam(T_B_E, T_E_C)

    last_exc: Optional[Exception] = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            det, pose_cam, color, pcd_cam = _perceive_object_in_camera(
                cam, detector, cfg, target_class, log,
                K=K, dist=dist, up_in_cam=up_in_cam,
            )
            break
        except RuntimeError as exc:
            last_exc = exc
            if attempt < retries:
                log.warning(
                    "perception attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt + 1, retries + 1, exc, retry_delay_s,
                )
                time.sleep(retry_delay_s)
                T_B_E = piper.get_end_pose_matrix()
                up_in_cam = _up_direction_in_cam(T_B_E, T_E_C)
            else:
                # Out of retries; re-raise the last failure so the
                # pipeline's outer try/except can trigger safe retreat.
                raise

    # Optional ICP refinement when the class has a template point cloud.
    # No-op when perception.icp.enable is false (the default), so this
    # change does not alter existing behaviour.
    if classes is not None:
        pose_cam = _refine_pose_with_icp_if_enabled(
            pcd_cam, pose_cam, cfg, classes, det.class_id, log,
        )

    T_B_O = T_B_E @ T_E_C @ pose_cam.T_C_O
    return det, pose_cam, T_B_E, T_B_O, color, pcd_cam


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
    classes: Optional[dict] = None,
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
    # move_to_pose repeats EndPoseCtrl @ 50 Hz until arrival/timeout --
    # crucial for CAN reliability on the PiPER (a lone EndPoseCtrl
    # frame can silently get dropped during a mode switch).
    if not piper.move_to_pose(
        T_B_E_near,
        linear=False,
        speed_pct=speed,
        pos_tol_m=pos_tol,
        ang_tol_deg=ang_tol,
        timeout_s=cart_timeout,
    ):
        raise RuntimeError(
            "close-up pose was not reached within timeout; aborting active perception"
        )
    time.sleep(0.4)

    return _snapshot_T_B_O(
        piper, cam, detector, cfg, target_class, T_E_C, log,
        classes=classes, K=K, dist=dist,
    )


def _execute_grasp(
    piper: PiperController,
    cfg: dict,
    grasp: GraspCandidate,
    place_pose_base: Optional[np.ndarray],
    log,
    *,
    close_effort_mNm: Optional[float] = None,
) -> None:
    # Re-check CAN health at the start of the motion phase. The rough
    # + fine perception easily takes 10-30 s, during which the bus
    # could have silently degraded (e.g. USB power-management on the
    # CAN adapter, a cable wiggle on a real workbench). We do not want
    # to send MOVE_L commands into a half-dead link.
    min_fps = float(cfg.get("motion", {}).get("min_can_fps", 50.0))
    if not piper.is_ok() or piper.can_fps() < min_fps:
        raise RuntimeError(
            f"CAN link degraded before grasp execution: ok={piper.is_ok()}, "
            f"fps={piper.can_fps():.0f} (min {min_fps:.0f}). "
            "Check cabling / power before retrying."
        )

    g_cfg = cfg["grasp"]
    speed_cart = int(cfg["piper"]["cartesian_move_speed_pct"])
    effort_mNm = float(cfg["piper"]["default_effort_mNm"])
    # ``close_effort_mNm`` may be overridden per-class by the caller
    # (classes.yaml -> <class>.close_effort_mNm). Fall back through
    # the global grasp block and finally the generic piper effort.
    if close_effort_mNm is None:
        close_effort_mNm = float(g_cfg.get("close_effort_mNm", effort_mNm))
    else:
        close_effort_mNm = float(close_effort_mNm)

    # Read cartesian arrival tolerances from [motion] so they are tunable
    # from system.yaml rather than scattered hard-coded values.
    m_cfg = cfg.get("motion", {})
    pos_tol = float(m_cfg.get("pos_tol_m", 0.003))
    ang_tol = float(m_cfg.get("ang_tol_deg", 1.0))
    cart_timeout = float(m_cfg.get("cartesian_timeout_s", 8.0))

    def _move(tag: str, pose: np.ndarray, *, linear: bool, speed: int) -> None:
        """Command a cartesian target and fail loud on timeout.

        Wraps :py:meth:`PiperController.move_to_pose` so every waypoint
        in the grasp sequence checks its own arrival. Previously we
        called ``end_pose_ctrl`` + ``wait_cartesian_arrive`` and threw
        away the boolean -- a timeout on the approach would silently
        continue into ``close_until`` and the lift motion, potentially
        driving the arm through the object / table.
        """
        log.info(
            "moving to %s: xyz=%s",
            tag, np.array2string(pose[:3, 3], precision=3),
        )
        ok = piper.move_to_pose(
            pose,
            linear=linear,
            speed_pct=speed,
            pos_tol_m=pos_tol,
            ang_tol_deg=ang_tol,
            timeout_s=cart_timeout,
        )
        if not ok:
            cur = piper.get_end_pose_matrix()
            err_xyz = cur[:3, 3] - pose[:3, 3]
            raise RuntimeError(
                f"timeout reaching {tag}; residual xyz error "
                f"({err_xyz[0]*1000:.1f}, {err_xyz[1]*1000:.1f}, {err_xyz[2]*1000:.1f}) mm"
            )

    log.info("opening gripper to %.3f m", float(g_cfg["max_opening_m"]))
    piper.open_gripper(
        opening_m=float(g_cfg["max_opening_m"]),
        effort_mNm=effort_mNm,
    )
    time.sleep(0.3)

    _move("pre-grasp", grasp.pre_grasp, linear=False, speed=speed_cart)
    time.sleep(0.2)

    _move(
        "grasp",
        grasp.T_B_grasp,
        linear=True,
        speed=max(5, speed_cart // 2),
    )
    time.sleep(0.2)

    # Close with feedback: stop when the jaws stall or reach the
    # commanded narrow opening.  ``close_until`` returns both the
    # result and the final opening so we can verify the grasp.
    log.info(
        "closing gripper to %.1f mm (object width %.1f mm)",
        grasp.close_opening_m * 1000.0,
        grasp.object_width_m * 1000.0,
    )
    stall_eps = float(g_cfg.get("close_stall_eps_m", 0.0003))
    close_timeout = float(g_cfg.get("close_timeout_s", 2.0))
    ok_close, final_opening = piper.close_until(
        target_opening_m=grasp.close_opening_m,
        effort_mNm=close_effort_mNm,
        timeout_s=close_timeout,
        stall_eps_m=stall_eps,
    )
    log.info(
        "gripper settled at %.1f mm (reached=%s)",
        final_opening * 1000.0, ok_close,
    )

    grasp_tol_m = float(g_cfg.get("grasp_check_tol_m", 0.006))
    min_grip_m = float(g_cfg.get("grasp_check_min_m", 0.003))
    is_held, _ = piper.check_grasp(
        expected_width_m=grasp.object_width_m,
        min_width_m=min_grip_m,
        tol_m=grasp_tol_m,
    )
    if not is_held:
        piper.open_gripper(
            opening_m=float(g_cfg["max_opening_m"]),
            effort_mNm=effort_mNm,
        )
        raise RuntimeError(
            f"grasp check failed: final opening {final_opening*1000:.1f} mm vs "
            f"expected object width {grasp.object_width_m*1000:.1f} mm "
            f"(tol {grasp_tol_m*1000:.1f} mm); object likely slipped or missing"
        )

    _move("lift", grasp.lift, linear=True, speed=max(5, speed_cart // 2))
    time.sleep(0.2)

    if place_pose_base is not None:
        _move("place", place_pose_base, linear=False, speed=speed_cart)
        piper.open_gripper(
            opening_m=float(g_cfg["max_opening_m"]),
            effort_mNm=effort_mNm,
        )
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _save_perception_snapshot(
    out_dir: Path,
    tag: str,
    color: np.ndarray,
    det: Detection,
    pose_cam,
    pcd_cam,
    T_B_E: np.ndarray,
    T_E_C: np.ndarray,
    K: np.ndarray,
    log,
) -> None:
    """Dump an annotated image + point cloud (in base frame) + transforms.

    ``tag`` becomes the filename stem: ``detection_<tag>.jpg``,
    ``cloud_<tag>.ply``, ``poses_<tag>.npz``. Called once per perception
    phase (rough / fine).
    """
    overlay = make_detection_overlay(
        color,
        det,
        center_cam=pose_cam.center,
        R_cam=pose_cam.R,
        extent=pose_cam.extent,
        K=K,
        title=f"{tag}: {det.label or det.class_id} conf={det.confidence:.2f}",
    )
    img_path = save_image(out_dir / f"detection_{tag}.jpg", overlay)
    log.info("viz: wrote %s", img_path)

    if o3d is not None and pcd_cam is not None and len(pcd_cam.points) > 0:
        # Express the cloud in the base frame so all phases share a common
        # coordinate system (easier to diff rough vs. fine in a PLY viewer).
        T_B_C = T_B_E @ T_E_C
        pcd_base = o3d.geometry.PointCloud(pcd_cam)
        pcd_base.transform(T_B_C)
        cloud_path = save_pointcloud(out_dir / f"cloud_{tag}.ply", pcd_base)
        if cloud_path is not None:
            log.info("viz: wrote %s", cloud_path)

    T_B_O = T_B_E @ T_E_C @ pose_cam.T_C_O
    save_poses_npz(
        out_dir / f"poses_{tag}.npz",
        T_B_E=T_B_E,
        T_E_C=T_E_C,
        T_C_O=pose_cam.T_C_O,
        T_B_O=T_B_O,
        extent=pose_cam.extent,
        K=K,
    )


def _save_plan_artifacts(
    out_dir: Path,
    T_B_O_final: np.ndarray,
    extent_final,
    grasp: GraspCandidate,
    place_pose_base: Optional[np.ndarray],
    log,
) -> None:
    save_poses_npz(
        out_dir / "plan.npz",
        T_B_O_final=T_B_O_final,
        extent_final=np.asarray(extent_final, dtype=np.float64),
        T_B_grasp=grasp.T_B_grasp,
        T_B_pregrasp=grasp.pre_grasp,
        T_B_lift=grasp.lift,
        opening_m=np.array([grasp.opening_m, grasp.close_opening_m, grasp.object_width_m]),
        place_pose_base=place_pose_base,
    )
    log.info("viz: wrote %s", out_dir / "plan.npz")


def _show_plan_3d(
    pcd_cam,
    T_B_E: np.ndarray,
    T_E_C: np.ndarray,
    T_B_O_final: np.ndarray,
    extent_final,
    grasp: GraspCandidate,
    place_pose_base: Optional[np.ndarray],
    log,
) -> None:
    """Blocking Open3D preview of the full plan in the base frame."""
    if o3d is None:
        log.warning("open3d not installed; skipping interactive 3D preview")
        return
    pcd_base = None
    if pcd_cam is not None and len(pcd_cam.points) > 0:
        pcd_base = o3d.geometry.PointCloud(pcd_cam)
        pcd_base.transform(T_B_E @ T_E_C)
    geoms = build_scene_geometries(
        scene_pcd_base=pcd_base,
        T_B_O=T_B_O_final,
        extent=extent_final,
        T_B_grasp=grasp.T_B_grasp,
        T_B_pregrasp=grasp.pre_grasp,
        T_B_lift=grasp.lift,
        T_B_place=place_pose_base,
    )
    log.info("viz: opening 3D preview (close window to continue)")
    show_scene(geoms, window_name="grasp plan (base frame)")


# ---------------------------------------------------------------------------
# Place pose helpers
# ---------------------------------------------------------------------------
def _resolve_place_pose(
    cfg: dict,
    classes: dict[int, dict],
    class_id: int,
    log,
) -> np.ndarray:
    """Build the release pose from (per-class or global) config.

    ``classes.yaml`` may carry a ``place: {xyz_m, tilt_deg, yaw_deg}``
    block per class (e.g. to sort bottles into one bin and cans into
    another). Any missing field falls back to the top-level ``place:``
    section in ``system.yaml``.
    """
    global_place = cfg.get("place", {}) or {}
    class_place = (classes.get(class_id, {}) or {}).get("place", {}) or {}

    xyz = class_place.get("xyz_m", global_place.get("xyz_m", [0.30, -0.15, 0.18]))
    tilt = float(class_place.get("tilt_deg", global_place.get("tilt_deg", 0.0)))
    yaw = float(class_place.get("yaw_deg", global_place.get("yaw_deg", 0.0)))

    R = camera_look_down_rotation(tilt_deg=tilt, yaw_deg=yaw)
    place = np.eye(4)
    place[:3, :3] = R
    place[:3, 3] = [float(v) for v in xyz]
    log.info(
        "place pose for class %d: xyz=(%.3f, %.3f, %.3f), tilt=%.1f deg, yaw=%.1f deg",
        class_id, xyz[0], xyz[1], xyz[2], tilt, yaw,
    )
    return place


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Eye-in-hand grasp loop.")
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="only grasp instances of this class id (int) or name (from classes.yaml)",
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
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="write detection overlays, point clouds, and poses to --viz-dir",
    )
    ap.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("runs/grasp_viz"),
        help="output directory for visualization artifacts (implies --visualize)",
    )
    ap.add_argument(
        "--viz-3d",
        action="store_true",
        help="open an interactive Open3D preview of the plan before motion",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirm-before-motion prompt used with --viz-3d",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Configure logging level from config (can still be overridden by
    # callers that reach into logging directly).
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log = get_logger("main", level=log_level)

    # Visualization artifacts (images / PLYs / npz) are written only when
    # --visualize is set. --viz-3d alone just opens the interactive
    # preview without touching the filesystem. Both flags keep the color
    # frame + cleaned cloud around from each perception phase.
    viz_dir: Optional[Path] = None
    if args.visualize:
        viz_dir = args.viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        log.info("viz: artifacts -> %s", viz_dir.resolve())

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

    target_class = _resolve_target_class(args.target_class, classes)
    if args.target_class is not None:
        log.info("filtering detections to class %s -> id %s", args.target_class, target_class)

    class_names = {k: v.get("name", str(k)) for k, v in classes.items()}
    detector = SegmentationDetector(
        model_path=project_path(cfg["paths"]["model"]),
        conf=float(cfg["perception"]["conf_threshold"]),
        device=cfg.get("perception", {}).get("device"),
        class_names=class_names,
    )

    # -- hardware ------------------------------------------------------
    piper_cfg = cfg["piper"]
    can_port = piper_cfg["can_port"]
    install_pos = piper_cfg.get("installation_pos")
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
        spatial_filter=bool(cfg["camera"].get("spatial_filter", False)),
        temporal_filter=bool(cfg["camera"].get("temporal_filter", False)),
        hole_filling_filter=bool(cfg["camera"].get("hole_filling_filter", False)),
    ) as cam, PiperController(
        can_port=can_port,
        installation_pos=install_pos,
    ) as piper:
        # Keep the arm enabled after the process exits. Disabling all joints
        # at shutdown is surprising during bring-up and can make the arm sag.
        piper.disable_on_disconnect = False

        # One-shot soft-stop guard: any exception or Ctrl+C from this
        # point on triggers the safe retreat + soft stop path, so the
        # arm never ends a run still holding an object or in the middle
        # of a MOVE_L.
        try:
            _run_pipeline(
                piper=piper,
                cam=cam,
                detector=detector,
                cfg=cfg,
                classes=classes,
                target_class=target_class,
                T_E_C=T_E_C,
                intr=intr,
                K_cal=K_cal,
                dist_cal=dist_cal,
                args=args,
                viz_dir=viz_dir,
                log=log,
            )
        except KeyboardInterrupt:
            log.warning("user interrupt; soft-stopping arm")
            try:
                piper.stop()
            except Exception as exc:
                log.warning("piper.stop() raised: %s", exc)
            raise
        except Exception:
            log.exception("pipeline error; attempting safe retreat")
            try:
                _safe_retreat(piper, cfg, log)
            except Exception as exc:
                log.warning("safe_retreat raised: %s", exc)
            raise


def _run_pipeline(
    *,
    piper: PiperController,
    cam: RealSenseCamera,
    detector: SegmentationDetector,
    cfg: dict,
    classes: dict,
    target_class: Optional[int],
    T_E_C: np.ndarray,
    intr: dict,
    K_cal: np.ndarray,
    dist_cal: np.ndarray,
    args,
    viz_dir: Optional[Path],
    log,
) -> None:
    """The full observe -> detect -> plan -> execute -> home loop.

    Extracted from :func:`main` so the caller can wrap the whole thing
    in a single ``try/except`` for uniform safe-retreat handling.
    """
    piper_cfg = cfg["piper"]
    can_port = piper_cfg["can_port"]

    # CAN health check: GetArmJointMsgs will silently return stale
    # zeros if the reader thread is not actually receiving frames.
    # Fail loud here rather than moving to a bogus observe pose.
    if not piper.is_ok():
        raise RuntimeError(
            f"CAN reader is not receiving frames on {can_port}; check "
            f"`ip link show {can_port}`, CAN wiring, and that the arm is powered."
        )
    min_fps = float(cfg.get("motion", {}).get("min_can_fps", 50.0))
    fps = piper.can_fps()
    log.info("piper CAN fps ~ %.0f Hz", fps)
    if fps < min_fps:
        raise RuntimeError(
            f"CAN fps {fps:.0f} Hz below minimum {min_fps:.0f} Hz; "
            "check CAN wiring / bitrate before attempting motion"
        )

    # Optional first-time gripper parameter push. Safe to re-send on
    # every run: the controller just writes the same values back.
    gtp_cfg = piper_cfg.get("gripper_teach_pendant", {}) or {}
    if gtp_cfg.get("enable", False):
        stroke_mm = int(gtp_cfg.get("stroke_mm", 100))
        max_range_mm = int(gtp_cfg.get("max_range_mm", 70))
        log.info(
            "pushing gripper teach-pendant params: stroke=%d mm, max_range=%d mm",
            stroke_mm, max_range_mm,
        )
        piper.gripper_teaching_pendant_param_config(
            teach_pendant_stroke_mm=stroke_mm,
            max_range_mm=max_range_mm,
        )

    # Sanity-check calibration vs. runtime stream. Previous versions only
    # logged a warning and then used the calibrated K against a differently
    # sized image, producing back-projections that were off by the ratio.
    # Now we rescale the calibrated K (and leave ``dist`` alone, which is
    # dimensionless) so the pipeline keeps working when the user calibrated
    # at 1280x720 but runs the live stream at 640x480, or similar.
    rs_intr = cam.intrinsics
    if (rs_intr.width, rs_intr.height) != (intr["width"], intr["height"]):
        log.warning(
            "calibration resolution %dx%d differs from live stream %dx%d; "
            "rescaling calibrated K to match the live stream",
            intr["width"], intr["height"], rs_intr.width, rs_intr.height,
        )
        K_cal = rescale_intrinsics(
            K_cal,
            src_size=(intr["width"], intr["height"]),
            dst_size=(rs_intr.width, rs_intr.height),
        )
        intr["width"] = rs_intr.width
        intr["height"] = rs_intr.height

    # 1) observe pose
    _go_to_observe(piper, cfg, log)

    # 2) rough perception
    det, pose_cam_rough, T_B_E_rough, T_B_O_rough, color_rough, pcd_rough_cam = _snapshot_T_B_O(
        piper, cam, detector, cfg, target_class, T_E_C, log,
        classes=classes,
        K=K_cal, dist=dist_cal,
        retries=int(cfg.get("perception", {}).get("retries", 2)),
    )
    log.info(
        "T_B_O (rough) translation: %s",
        np.array2string(T_B_O_rough[:3, 3], precision=4),
    )
    if viz_dir is not None:
        _save_perception_snapshot(
            viz_dir, "rough", color_rough, det, pose_cam_rough,
            pcd_rough_cam, T_B_E_rough, T_E_C, K_cal, log,
        )

    # Track the freshest scene cloud + EEF pose so the final 3D preview
    # can be in the same frame as the grasp plan. Active perception
    # overwrites these if it succeeds.
    pcd_best_cam = pcd_rough_cam
    T_B_E_best = T_B_E_rough

    # 3) optional close-up active perception
    T_B_O_final = T_B_O_rough
    extent_final = pose_cam_rough.extent
    if cfg["active_perception"].get("enable", True) and not args.no_active_perception:
        try:
            det_f, pose_cam_fine, T_B_E_fine, T_B_O_fine, color_fine, pcd_fine_cam = (
                _active_perception_close_up(
                    piper,
                    cam,
                    detector,
                    cfg,
                    T_E_C,
                    T_B_O_rough,
                    pose_cam_rough.extent,
                    target_class,
                    log,
                    classes=classes,
                    K=K_cal,
                    dist=dist_cal,
                )
            )
            if viz_dir is not None:
                _save_perception_snapshot(
                    viz_dir, "fine", color_fine, det_f, pose_cam_fine,
                    pcd_fine_cam, T_B_E_fine, T_E_C, K_cal, log,
                )
            pcd_best_cam = pcd_fine_cam
            T_B_E_best = T_B_E_fine
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
    # Per-class over-close override wins over the global default.
    global_overclose = float(cfg["grasp"].get("close_overclose_m", 0.002))
    close_overclose_m = float(
        cinfo.get("close_overclose_mm", global_overclose * 1000.0)
    ) / 1000.0
    # Per-class close torque override. Falls through to
    # grasp.close_effort_mNm inside _execute_grasp when not set.
    class_close_effort = cinfo.get("close_effort_mNm")
    close_effort_mNm: Optional[float] = (
        float(class_close_effort) if class_close_effort is not None else None
    )

    grasp = plan_topdown_grasp(
        T_B_O=T_B_O_final,
        extent=extent_final,
        approach_world=tuple(cfg["grasp"]["approach_direction_world"]),
        opening_margin_m=float(cfg["grasp"]["opening_margin_m"]),
        close_overclose_m=close_overclose_m,
        approach_m=float(cfg["grasp"]["pre_grasp_offset_m"]),
        lift_m=lift_m,
        max_opening_m=max_opening_m,
        workspace=cfg.get("workspace"),
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

    # 5) Build place pose (possibly per-class) before the dry-run short-
    # circuit so visualization reflects what will actually be executed.
    place = _resolve_place_pose(cfg, classes, det.class_id, log)

    # Plan visualization: on-disk artifacts + optional interactive preview.
    # Done before motion so a bad plan can be caught at a glance.
    if viz_dir is not None:
        _save_plan_artifacts(viz_dir, T_B_O_final, extent_final, grasp, place, log)
    if args.viz_3d:
        _show_plan_3d(
            pcd_best_cam, T_B_E_best, T_E_C,
            T_B_O_final, extent_final, grasp, place, log,
        )
        if not args.dry_run and not args.yes:
            try:
                # Explicit opt-in after the 3D review, mirrors what a
                # human would do in a bring-up context. --yes bypasses
                # this for scripted runs (e.g. overnight tests).
                answer = input("Proceed with motion? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                log.info("user declined motion; aborting")
                return

    if args.dry_run:
        log.info("dry-run; skipping motion")
        return

    _execute_grasp(
        piper, cfg, grasp, place_pose_base=place, log=log,
        close_effort_mNm=close_effort_mNm,
    )

    # 6) Always open gripper + lift before going home, so a stray
    # object (or a partially-succeeded place) doesn't get dragged
    # through the workspace on the way back to observe.
    _safe_retreat(piper, cfg, log)
    _go_to_observe(piper, cfg, log)
    log.info("done")


if __name__ == "__main__":
    main()
