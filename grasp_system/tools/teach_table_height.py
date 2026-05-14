"""Teach the physical table height from a hand-placed gripper contact.

Move the arm with the physical teach/drag controls until the configured TCP
touches the tabletop, then press ENTER. The script reads the current PiPER
end-effector pose, applies ``grasp.tool_offset_eef_m``, and writes the measured
TCP z value into ``workspace.table_z_m``.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from ..common import get_logger, load_config, project_path
from ..control.piper_controller import PiperController


def _tool_offset_eef_m(cfg: dict, override: list[float] | None) -> np.ndarray:
    if override is not None:
        return np.asarray(override, dtype=np.float64).reshape(3)
    return np.asarray(
        cfg.get("grasp", {}).get("tool_offset_eef_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)


def _tcp_position_base(T_B_E: np.ndarray, tool_offset_eef_m: np.ndarray) -> np.ndarray:
    return T_B_E[:3, 3] + T_B_E[:3, :3] @ tool_offset_eef_m


def _write_table_z(config_path: Path, table_z_m: float) -> None:
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*table_z_m:\s*)([-+0-9.eE]+)(\s*(?:#.*)?)$", re.MULTILINE)
    replacement = rf"\g<1>{table_z_m:.6f}\g<3>"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"could not find exactly one table_z_m entry in {config_path}")
    config_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    cfg0 = load_config()
    can_port = cfg0.get("piper", {}).get("can_port", "can0")

    ap = argparse.ArgumentParser(description="Teach workspace.table_z_m from gripper contact.")
    ap.add_argument("--config", type=Path, default=Path("configs/system.yaml"))
    ap.add_argument("--can-port", type=str, default=can_port)
    ap.add_argument(
        "--tool-offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="EEF-frame vector from end-effector origin to tabletop contact point, meters",
    )
    ap.add_argument("--samples", type=int, default=20, help="number of pose samples to average")
    ap.add_argument("--period", type=float, default=0.03, help="seconds between samples")
    ap.add_argument("--no-write", action="store_true", help="print the measured height without editing config")
    ap.add_argument("--yes", action="store_true", help="do not ask before writing the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    config_path = project_path(args.config)
    tool_offset = _tool_offset_eef_m(cfg, args.tool_offset)

    log = get_logger("teach-table")
    log.info("using TCP/contact offset in EEF frame: %s m", np.array2string(tool_offset, precision=4))
    log.warning(
        "This tool does not command teach mode. Use the physical teach/drag control, "
        "place the TCP lightly on the tabletop, then release motion and press ENTER."
    )

    piper = PiperController(can_port=args.can_port, enable_on_connect=False)
    piper.connect()
    piper.disable_on_disconnect = False
    try:
        input("Put the gripper TCP on the tabletop, then press ENTER to sample: ")
        samples: list[np.ndarray] = []
        n = max(1, int(args.samples))
        for _ in range(n):
            T_B_E = piper.get_end_pose_matrix()
            samples.append(_tcp_position_base(T_B_E, tool_offset))
            time.sleep(max(0.0, float(args.period)))

        pts = np.stack(samples, axis=0)
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        table_z_m = float(mean[2])
        log.info("measured TCP/contact xyz mean: %s m", np.array2string(mean, precision=5))
        log.info("sample std xyz: %s mm", np.array2string(std * 1000.0, precision=3))
        log.info("new workspace.table_z_m = %.6f m (%.1f mm)", table_z_m, table_z_m * 1000.0)

        if args.no_write:
            return
        if not args.yes:
            answer = input(f"Write table_z_m={table_z_m:.6f} to {config_path}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                log.info("not writing config")
                return
        _write_table_z(config_path, table_z_m)
        log.info("updated %s", config_path)
    finally:
        piper.disconnect(disable_arm=False)


if __name__ == "__main__":
    main()
