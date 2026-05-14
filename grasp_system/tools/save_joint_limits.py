"""Read current PiPER joints and save a joint-limit template."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..common import ensure_parent, get_logger, load_config
from ..control.piper_controller import PiperController


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    cfg = load_config()
    can_port = cfg.get("piper", {}).get("can_port", "can0")

    ap = argparse.ArgumentParser(description="Save current joint angles and a joint-limit YAML template.")
    ap.add_argument("--can-port", default=can_port)
    ap.add_argument(
        "--margin-deg",
        type=float,
        default=10.0,
        help="half-width around the current angle for the generated limit template",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("grasp_system/configs/joint_limits_snapshot.json"),
        help="JSON output path",
    )
    args = ap.parse_args()

    log = get_logger("joint-limits")
    piper = PiperController(can_port=args.can_port, enable_on_connect=False)
    piper.connect()
    piper.disable_on_disconnect = False
    try:
        joints_rad = piper.get_joints_rad()
        joints_deg = np.rad2deg(joints_rad)
        margin = abs(float(args.margin_deg))
        limits_deg = np.column_stack([joints_deg - margin, joints_deg + margin])

        payload = {
            "timestamp": _timestamp(),
            "joints_rad": joints_rad.tolist(),
            "joints_deg": joints_deg.tolist(),
            "margin_deg": margin,
            "joint_limits_deg": limits_deg.tolist(),
        }
        out = ensure_parent(args.out)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log.info("current joints deg: %s", np.array2string(joints_deg, precision=3))
        log.info("saved %s", out)
        print("\nPaste/edit this under piper: in grasp_system/configs/system.yaml:")
        print("  joint_limits_deg:")
        for lo, hi in limits_deg:
            print(f"    - [{lo:.2f}, {hi:.2f}]")
    finally:
        piper.disconnect(disable_arm=False)


if __name__ == "__main__":
    main()
