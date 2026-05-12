"""Teach-and-save helper for the fixed observe pose.

Jog / teach the PiPER to a pose from which the camera comfortably views the
workspace, then run this script and press ENTER. It records the 6 joint
angles into ``configs/observe_joints.npy`` so ``main_grasp.py`` can
reliably reproduce the observe pose via MOVE_J.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common import ensure_parent, get_logger, load_config, project_path
from ..control.piper_controller import PiperController


def main() -> None:
    cfg = load_config()
    default_out = project_path(cfg["paths"]["observe_joints"])
    can_port = cfg.get("piper", {}).get("can_port", "can0")

    ap = argparse.ArgumentParser(description="Capture observe-pose joint angles.")
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--can-port", type=str, default=can_port)
    args = ap.parse_args()

    log = get_logger("teach")

    piper = PiperController(can_port=args.can_port, enable_on_connect=False)
    piper.connect()
    try:
        input("Move / teach the arm to the observe pose, then press ENTER: ")
        joints_rad = piper.get_joints_rad()
        joints_deg = np.rad2deg(joints_rad)
        log.info("captured joints (deg): %s", np.array2string(joints_deg, precision=3))
        ensure_parent(args.out)
        np.save(args.out, joints_rad)
        log.info("saved %s", args.out)
    finally:
        piper.disconnect()


if __name__ == "__main__":
    main()
