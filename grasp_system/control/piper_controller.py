"""PiPER SDK V2 wrapper.

This module centralises every unit-conversion and mode-selection rule the
PiPER arm expects so that the rest of the codebase can stay in SI units
(meters, radians, seconds, newton-meters) and clean 4x4 transforms.

Key PiPER quirks handled here:
  * EndPoseCtrl X/Y/Z in 0.001 mm (i.e. multiply meters by 1e6).
  * EndPoseCtrl RX/RY/RZ in 0.001 deg (multiply degrees by 1e3).
  * Euler convention: extrinsic xyz ("xyz" lowercase in scipy).
  * JointCtrl takes joint angles in 0.001 deg.
  * GripperCtrl position in 0.001 mm, effort in 0.001 N*m.
  * MotionCtrl_2 move_mode: 0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L,
    0x03=MOVE_C, 0x04=MOVE_M.

Safety note:
  * ``MotionCtrl_1(0x02, 0, 0)`` (a.k.a. ``ResetPiper`` in the SDK) cuts
    power on every joint -- the arm will physically drop. It is NOT a
    "clear errors" command. Use :meth:`PiperController.reset_and_drop`
    only when that behaviour is exactly what you want, and call
    :meth:`PiperController.recover_errors` (disable + re-enable) to clear
    a driver error latch while the arm stays powered.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..common import (
    get_logger,
    matrix_to_euler_xyz_deg,
    pose_xyzrpy_to_matrix,
    rotation_angle_deg,
)

try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore
except ImportError:  # pragma: no cover - hardware dependency
    C_PiperInterface_V2 = None  # type: ignore[assignment]


# ---- Scale factors ----------------------------------------------------
_M_TO_PIPER_POS = 1.0e6     # meters -> 0.001 mm
_DEG_TO_PIPER_ANG = 1.0e3   # degrees -> 0.001 deg
_RAD_TO_PIPER_ANG = _DEG_TO_PIPER_ANG * 180.0 / math.pi
_MM_TO_PIPER_GRIP = 1.0e3   # mm -> 0.001 mm
# Note: the gripper effort unit accepted by GripperCtrl is already
# 0.001 N*m (milli-N*m). Callers pass ``effort_mNm`` through directly,
# so there is no N*m->PiPER conversion factor here.

# ---- Move modes -------------------------------------------------------
MOVE_P = 0x00
MOVE_J = 0x01
MOVE_L = 0x02
MOVE_C = 0x03
MOVE_M = 0x04


@dataclass
class EndPose:
    """End-effector pose in SI units + extrinsic xyz Euler (degrees)."""

    x_m: float
    y_m: float
    z_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def as_matrix(self) -> np.ndarray:
        return pose_xyzrpy_to_matrix(
            self.x_m, self.y_m, self.z_m, self.rx_deg, self.ry_deg, self.rz_deg
        )


class PiperController:
    """High-level, unit-safe wrapper around ``C_PiperInterface_V2``.

    Parameters
    ----------
    can_port:
        CAN interface name, e.g. ``"can0"``.
    enable_on_connect:
        If True, enable the arm as part of ``connect``.
    """

    def __init__(self, can_port: str = "can0", enable_on_connect: bool = True) -> None:
        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. `pip install piper_sdk` and configure "
                "your CAN interface before using PiperController."
            )
        self.can_port = can_port
        self.enable_on_connect = enable_on_connect
        self._piper: Optional[C_PiperInterface_V2] = None
        self._log = get_logger("piper")

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "PiperController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def connect(self) -> None:
        # C_PiperInterface_V2 is a singleton per can_name: repeated __init__
        # returns the same instance. Re-calling ConnectPort on an already
        # connected instance is harmless (it early-returns under the lock).
        self._piper = C_PiperInterface_V2(self.can_port)
        self._piper.ConnectPort()
        # Give the reader thread a moment to populate feedback buffers.
        time.sleep(0.05)
        if self.enable_on_connect:
            self.enable_arm()

    def disconnect(self) -> None:
        if self._piper is None:
            return
        try:
            self.disable_arm()
        except Exception:
            pass
        # Properly tear down the CAN reader thread and close socketcan.
        # Without this the daemon thread stays alive and the port remains
        # bound, making a subsequent reconnect flaky.
        try:
            self._piper.DisconnectPort()
        except Exception:
            pass
        self._piper = None

    # -- raw handle -----------------------------------------------------
    @property
    def raw(self) -> "C_PiperInterface_V2":
        if self._piper is None:
            raise RuntimeError("PiperController is not connected.")
        return self._piper

    # -- enable / disable ----------------------------------------------
    def enable_arm(self, timeout_s: float = 5.0) -> None:
        """Enable all 6 joints, polling feedback until they report enabled.

        Raises
        ------
        RuntimeError
            If the arm fails to enable within ``timeout_s``. This is fatal:
            continuing would issue motion commands that the drivers ignore,
            leaving the caller unable to tell simulation from real motion.
        """
        piper = self.raw
        t0 = time.time()
        last_states: Sequence[bool] = []
        while time.time() - t0 < timeout_s:
            # EnablePiper() is the canonical SDK helper: it sends EnableArm(7)
            # and returns True iff every motor already reports enabled.
            if piper.EnablePiper():
                self._log.info("arm enabled")
                return
            try:
                last_states = piper.GetArmEnableStatus()
            except Exception:
                last_states = []
            time.sleep(0.05)
        raise RuntimeError(
            f"arm enable timeout after {timeout_s:.1f}s; last motor states={list(last_states)}. "
            "Check e-stop, CAN wiring, and that the teach pendant is not holding the arm."
        )

    def disable_arm(self) -> None:
        self.raw.DisableArm(7)

    def recover_errors(self) -> None:
        """Clear a latched driver-error state without dropping the arm.

        Implemented as ``DisableArm(7)`` followed by a re-enable. This is
        the safe alternative to ``MotionCtrl_1(0x02, 0, 0)`` (which cuts
        motor power).
        """
        self.raw.DisableArm(7)
        time.sleep(0.05)
        self.enable_arm()

    def reset_and_drop(self) -> None:
        """Hard reset via MotionCtrl_1(0x02, 0, 0).

        .. warning::
            This removes power from every motor; the arm will physically
            drop. Use only when the arm is already supported (e.g. resting
            on the table) or when you deliberately want to exit MIT /
            teach mode back to position control.
        """
        self.raw.MotionCtrl_1(0x02, 0, 0)

    # -- motion-mode helpers -------------------------------------------
    def _set_mode(self, move_mode: int, speed_pct: int) -> None:
        speed_pct = int(max(0, min(100, speed_pct)))
        self.raw.MotionCtrl_2(0x01, int(move_mode), speed_pct)

    # -- joint control --------------------------------------------------
    def joint_ctrl_rad(self, joints_rad: Sequence[float], speed_pct: int = 30) -> None:
        if len(joints_rad) != 6:
            raise ValueError("joints_rad must have exactly 6 elements")
        self._set_mode(MOVE_J, speed_pct)
        vals = [round(float(j) * _RAD_TO_PIPER_ANG) for j in joints_rad]
        self.raw.JointCtrl(*vals)

    def joint_ctrl_deg(self, joints_deg: Sequence[float], speed_pct: int = 30) -> None:
        if len(joints_deg) != 6:
            raise ValueError("joints_deg must have exactly 6 elements")
        self._set_mode(MOVE_J, speed_pct)
        vals = [round(float(j) * _DEG_TO_PIPER_ANG) for j in joints_deg]
        self.raw.JointCtrl(*vals)

    def get_joints_rad(self) -> np.ndarray:
        msg = self.raw.GetArmJointMsgs().joint_state
        deg = np.asarray(
            [
                msg.joint_1,
                msg.joint_2,
                msg.joint_3,
                msg.joint_4,
                msg.joint_5,
                msg.joint_6,
            ],
            dtype=np.float64,
        ) / _DEG_TO_PIPER_ANG
        return np.deg2rad(deg)

    def get_joints_deg(self) -> np.ndarray:
        msg = self.raw.GetArmJointMsgs().joint_state
        return np.asarray(
            [
                msg.joint_1,
                msg.joint_2,
                msg.joint_3,
                msg.joint_4,
                msg.joint_5,
                msg.joint_6,
            ],
            dtype=np.float64,
        ) / _DEG_TO_PIPER_ANG

    # -- cartesian end-pose control ------------------------------------
    def end_pose_ctrl(
        self,
        pose: EndPose | np.ndarray,
        linear: bool = False,
        speed_pct: int = 20,
    ) -> None:
        """Send a cartesian end-effector target.

        ``pose`` may be either an :class:`EndPose` (SI units + degrees) or a
        4x4 transform in meters. Set ``linear=True`` to use MOVE_L
        (cartesian straight line) instead of MOVE_P (point-to-point).
        """
        if isinstance(pose, np.ndarray):
            T = np.asarray(pose, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError("pose matrix must be 4x4")
            x, y, z = T[:3, 3]
            rx, ry, rz = matrix_to_euler_xyz_deg(T[:3, :3])
            pose = EndPose(x, y, z, rx, ry, rz)

        move_mode = MOVE_L if linear else MOVE_P
        self._set_mode(move_mode, speed_pct)
        self.raw.EndPoseCtrl(
            round(pose.x_m * _M_TO_PIPER_POS),
            round(pose.y_m * _M_TO_PIPER_POS),
            round(pose.z_m * _M_TO_PIPER_POS),
            round(pose.rx_deg * _DEG_TO_PIPER_ANG),
            round(pose.ry_deg * _DEG_TO_PIPER_ANG),
            round(pose.rz_deg * _DEG_TO_PIPER_ANG),
        )

    def get_end_pose(self) -> EndPose:
        msg = self.raw.GetArmEndPoseMsgs().end_pose
        return EndPose(
            x_m=msg.X_axis / _M_TO_PIPER_POS,
            y_m=msg.Y_axis / _M_TO_PIPER_POS,
            z_m=msg.Z_axis / _M_TO_PIPER_POS,
            rx_deg=msg.RX_axis / _DEG_TO_PIPER_ANG,
            ry_deg=msg.RY_axis / _DEG_TO_PIPER_ANG,
            rz_deg=msg.RZ_axis / _DEG_TO_PIPER_ANG,
        )

    def get_end_pose_matrix(self) -> np.ndarray:
        return self.get_end_pose().as_matrix()

    # -- arrival / wait -------------------------------------------------
    def wait_cartesian_arrive(
        self,
        target: EndPose | np.ndarray,
        pos_tol_m: float = 0.003,
        ang_tol_deg: float = 1.0,
        timeout_s: float = 10.0,
        poll_s: float = 0.05,
    ) -> bool:
        if isinstance(target, EndPose):
            target_mat = target.as_matrix()
        else:
            target_mat = np.asarray(target, dtype=np.float64)
        t_target = target_mat[:3, 3]
        R_target = target_mat[:3, :3]

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = self.get_end_pose_matrix()
            dt = float(np.linalg.norm(cur[:3, 3] - t_target))
            # Geodesic angular distance is the only metric that stays
            # meaningful near gimbal lock; Euler per-axis diffs used to
            # cause both false arrivals and false timeouts here.
            da = rotation_angle_deg(cur[:3, :3], R_target)
            if dt <= pos_tol_m and da <= ang_tol_deg:
                return True
            time.sleep(poll_s)
        return False

    def wait_joints_arrive(
        self,
        target_rad: Sequence[float],
        tol_deg: float = 0.5,
        timeout_s: float = 10.0,
        poll_s: float = 0.05,
    ) -> bool:
        target_deg = np.rad2deg(np.asarray(target_rad, dtype=np.float64))
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = self.get_joints_deg()
            if np.max(np.abs(_wrap_deg(cur - target_deg))) <= tol_deg:
                return True
            time.sleep(poll_s)
        return False

    # -- gripper --------------------------------------------------------
    def set_gripper_opening_mm(
        self,
        opening_mm: float,
        effort_mNm: float = 1000.0,
        enable: bool = True,
    ) -> None:
        """Set a target gripper opening in millimeters.

        ``effort_mNm`` is in milli-newton-metres (i.e. 0.001 N*m), matching the
        PiPER SDK unit exactly. So pass 1000 for 1 N*m.
        """
        code = 0x01 if enable else 0x00
        self.raw.GripperCtrl(
            round(float(opening_mm) * _MM_TO_PIPER_GRIP),
            round(float(effort_mNm)),
            code,
            0,
        )

    def open_gripper(self, opening_m: float = 0.070, effort_mNm: float = 1000.0) -> None:
        self.set_gripper_opening_mm(opening_m * 1000.0, effort_mNm=effort_mNm)

    def close_gripper(self, effort_mNm: float = 1000.0) -> None:
        self.set_gripper_opening_mm(0.0, effort_mNm=effort_mNm)

    def zero_gripper(self) -> None:
        """Set the current gripper position as the zero reference."""
        self.raw.GripperCtrl(0, 0, 0x01, 0xAE)


def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap an angular difference (degrees) into [-180, 180]."""
    return (np.asarray(delta) + 180.0) % 360.0 - 180.0
