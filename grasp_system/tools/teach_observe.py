"""Teach-and-save helper for the fixed observe pose.

Use the arm's physical teach/drag controls to move the PiPER to a pose from
which the camera comfortably views the workspace. This script opens a live
camera preview; press SPACE or left-click the preview window to record the
current 6 joint angles into ``configs/observe_joints.npy`` so
``main_grasp.py`` can reliably reproduce the observe pose via MOVE_J.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..common import ensure_parent, get_logger, load_config, project_path
from ..control.piper_controller import PiperController
from ..perception.camera import RealSenseCamera


def _save_observe_pose(piper: PiperController, out_path: Path, log) -> None:
    joints_rad = piper.get_joints_rad()
    piper.validate_joints_rad(joints_rad)
    joints_deg = np.rad2deg(joints_rad)
    log.info("captured joints (deg): %s", np.array2string(joints_deg, precision=3))
    ensure_parent(out_path)
    np.save(out_path, joints_rad)
    log.info("saved %s", out_path)


def main() -> None:
    cfg = load_config()
    default_out = project_path(cfg["paths"]["observe_joints"])
    can_port = cfg.get("piper", {}).get("can_port", "can0")

    ap = argparse.ArgumentParser(description="Capture observe-pose joint angles.")
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--can-port", type=str, default=can_port)
    ap.add_argument(
        "--no-preview",
        action="store_true",
        help="do not open the RealSense preview; press ENTER in the terminal to save",
    )
    args = ap.parse_args()

    log = get_logger("teach")

    piper = PiperController(can_port=args.can_port, enable_on_connect=False)
    piper.connect()
    try:
        if args.no_preview:
            input("Move / teach the arm to the observe pose, then press ENTER: ")
            _save_observe_pose(piper, args.out, log)
            return

        window_name = "teach observe pose"
        mouse_save_requested = False

        def _request_save_on_left_click(
            event: int,
            _x: int,
            _y: int,
            _flags: int,
            _param: object,
        ) -> None:
            nonlocal mouse_save_requested
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_save_requested = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _request_save_on_left_click)
        log.info(
            "use the arm's physical teach button to position the camera; "
            "SPACE/left-click=save observe pose, ESC=quit"
        )

        cam_cfg = cfg.get("camera", {})
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
                joints_deg = piper.get_joints_deg()

                vis = color.copy()
                cv2.putText(
                    vis,
                    "SPACE/click=save observe pose  ESC=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    "joints deg: " + np.array2string(joints_deg, precision=1),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow(window_name, vis)

                key = cv2.waitKey(1) & 0xFF
                clicked = mouse_save_requested
                mouse_save_requested = False
                if key == 27:
                    log.info("quit without saving observe pose")
                    break
                if key == 32 or clicked:
                    _save_observe_pose(piper, args.out, log)
                    break
    finally:
        cv2.destroyAllWindows()
        piper.disconnect(disable_arm=False)


if __name__ == "__main__":
    main()
