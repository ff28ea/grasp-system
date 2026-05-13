"""Move to the saved observe pose and show the camera preview.

This is a safer bring-up helper than running the full grasp loop. It only
checks that ``configs/observe_joints.npy`` can be reached and that the saved
pose gives a useful RealSense view.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..common import get_logger, load_config, load_npy, project_path
from ..control.piper_controller import PiperController
from ..perception.camera import RealSenseCamera


def main() -> None:
    cfg = load_config()
    default_pose = project_path(cfg["paths"]["observe_joints"])
    can_port = cfg.get("piper", {}).get("can_port", "can0")
    observe_cfg = cfg.get("observe", {})

    ap = argparse.ArgumentParser(description="Test the saved observe pose.")
    ap.add_argument("--pose", type=Path, default=default_pose)
    ap.add_argument("--can-port", type=str, default=can_port)
    ap.add_argument("--speed", type=int, default=int(observe_cfg.get("move_speed_pct", 20)))
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(observe_cfg.get("arrive_timeout_s", 8.0)),
        help="seconds to wait for the joint move to arrive",
    )
    ap.add_argument(
        "--no-move",
        action="store_true",
        help="do not command motion; only preview from the current arm pose",
    )
    args = ap.parse_args()

    log = get_logger("test-observe")
    target = load_npy(args.pose)
    if target.shape != (6,):
        raise RuntimeError(f"{args.pose} must contain 6 joint angles, got {target.shape}")

    log.info("target joints (deg): %s", np.array2string(np.rad2deg(target), precision=3))
    piper = PiperController(can_port=args.can_port, enable_on_connect=not args.no_move)
    piper.connect()
    piper.disable_on_disconnect = False
    try:
        current = piper.get_joints_rad()
        log.info("current joints (deg): %s", np.array2string(np.rad2deg(current), precision=3))
        log.info(
            "joint error to observe (deg): %s",
            np.array2string(np.rad2deg(target - current), precision=3),
        )

        if not args.no_move:
            input(
                "Release the physical teach/drag button, keep clear of the arm, "
                "then press ENTER to move to observe pose..."
            )
            log.info("moving to observe pose at speed %d%%", args.speed)
            if not piper.move_joints_rad(
                target,
                speed_pct=args.speed,
                tol_deg=1.0,
                timeout_s=args.timeout,
            ):
                current_after = piper.get_joints_rad()
                log.warning("observe-pose arrival timeout; check enable state or increase --timeout")
                log.warning(
                    "current joints after command (deg): %s",
                    np.array2string(np.rad2deg(current_after), precision=3),
                )
                log.warning(
                    "remaining joint error (deg): %s",
                    np.array2string(np.rad2deg(target - current_after), precision=3),
                )
            else:
                log.info("observe pose reached")
            time.sleep(0.5)

        cam_cfg = cfg.get("camera", {})
        window_name = "observe pose preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        log.info("showing preview; ESC=quit")
        with RealSenseCamera(
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
            color_fps=int(cam_cfg.get("color_fps", 30)),
            depth_fps=int(cam_cfg.get("depth_fps", 30)),
            align_to=str(cam_cfg.get("align_to", "color")),
            depth_scale=float(cam_cfg.get("depth_scale", 0.001)),
            warmup_frames=int(cam_cfg.get("warmup_frames", 30)),
        ) as cam:
            while True:
                color, _ = cam.grab_aligned()
                vis = color.copy()
                cv2.putText(
                    vis,
                    "observe preview  ESC=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cv2.destroyAllWindows()
        piper.disconnect(disable_arm=False)


if __name__ == "__main__":
    main()
