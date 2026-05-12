"""Record RGB-D snapshots from the D435i for dataset building.

Each SPACE press writes four files into ``out_dir``::

    frame_0000_color.png
    frame_0000_depth.png    # raw uint16 depth in millimeters
    frame_0000_depth.npy    # float32 depth in meters (convenience)
    frame_0000_pose.npy     # 4x4 T_B_E (only if --with-arm is set)

Useful for capturing the YOLOv8-seg training set from the actual
observe poses used at inference time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..common import ensure_parent, get_logger
from ..perception.camera import RealSenseCamera, depth_to_meters


def main() -> None:
    ap = argparse.ArgumentParser(description="Record RGB-D snapshots.")
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/raw"))
    ap.add_argument(
        "--with-arm",
        action="store_true",
        help="also log the PiPER end-pose with each frame",
    )
    ap.add_argument("--can-port", type=str, default="can0")
    args = ap.parse_args()

    log = get_logger("record")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    piper = None
    if args.with_arm:
        from ..control.piper_controller import PiperController

        piper = PiperController(can_port=args.can_port, enable_on_connect=False)
        piper.connect()

    idx = 0
    try:
        with RealSenseCamera() as cam:
            while True:
                color, depth_raw = cam.grab_aligned()
                depth_m = depth_to_meters(depth_raw, cam.depth_scale)

                vis = color.copy()
                cv2.putText(
                    vis,
                    f"frame {idx}  SPACE=save  ESC=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("record", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key != 32:
                    continue

                stem = out_dir / f"frame_{idx:04d}"
                ensure_parent(stem)
                cv2.imwrite(str(stem) + "_color.png", color)
                cv2.imwrite(str(stem) + "_depth.png", depth_raw)
                np.save(str(stem) + "_depth.npy", depth_m)
                if piper is not None:
                    T_BE = piper.get_end_pose_matrix()
                    np.save(str(stem) + "_pose.npy", T_BE)
                log.info("saved %s_*", stem)
                idx += 1
    finally:
        cv2.destroyAllWindows()
        if piper is not None:
            piper.disconnect()


if __name__ == "__main__":
    main()
