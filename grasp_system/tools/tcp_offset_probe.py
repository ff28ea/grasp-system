"""Probe whether the configured TCP offset sign is consistent with PiPER poses.

This tool does not command motion. Put the gripper TCP at a known contact point,
then run this with candidate offsets to see the computed base-frame TCP z.
"""
from __future__ import annotations

import argparse

import numpy as np

from ..common import get_logger, load_config
from ..control.piper_controller import PiperController


def main() -> None:
    cfg0 = load_config()
    can_port = cfg0.get("piper", {}).get("can_port", "can0")

    ap = argparse.ArgumentParser(description="Read current EEF pose and candidate TCP positions.")
    ap.add_argument("--config", default="configs/system.yaml")
    ap.add_argument("--can-port", default=can_port)
    ap.add_argument(
        "--length",
        type=float,
        default=0.145,
        help="candidate distance from EEF origin to TCP, meters",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    configured = np.asarray(
        cfg.get("grasp", {}).get("tool_offset_eef_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    candidates = [
        ("configured", configured),
        ("+eef_z", np.asarray([0.0, 0.0, float(args.length)])),
        ("-eef_z", np.asarray([0.0, 0.0, -float(args.length)])),
    ]

    log = get_logger("tcp-probe")
    piper = PiperController(can_port=args.can_port, enable_on_connect=False)
    piper.connect()
    piper.disable_on_disconnect = False
    try:
        T_B_E = piper.get_end_pose_matrix()
        log.info("EEF xyz: %s m", np.array2string(T_B_E[:3, 3], precision=5))
        log.info("EEF z axis in base: %s", np.array2string(T_B_E[:3, 2], precision=5))
        for name, offset in candidates:
            tcp = T_B_E[:3, 3] + T_B_E[:3, :3] @ offset
            log.info(
                "%s offset %s -> TCP xyz %s m",
                name,
                np.array2string(offset, precision=4),
                np.array2string(tcp, precision=5),
            )
    finally:
        piper.disconnect(disable_arm=False)


if __name__ == "__main__":
    main()
