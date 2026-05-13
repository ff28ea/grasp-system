"""PiPER SDK V2 wrapper.

This module centralises every unit-conversion and mode-selection rule the
PiPER arm expects so that the rest of the codebase can stay in SI units
(meters, radians, seconds, newton-meters) and clean 4x4 transforms.

The behaviour is cross-checked against the three upstream references:

  * https://github.com/agilexrobotics/piper_sdk           (SDK source)
  * https://github.com/agilexrobotics/piper_sdk_demo      (V2 demos)
  * https://github.com/agilexrobotics/Piper_sdk_ui        (GUI tool)

Key PiPER quirks handled here
-----------------------------
  * EndPoseCtrl X/Y/Z in 0.001 mm (i.e. multiply meters by 1e6).
  * EndPoseCtrl RX/RY/RZ in 0.001 deg (multiply degrees by 1e3).
  * Euler convention: extrinsic xyz ("xyz" lowercase in scipy).
  * JointCtrl takes joint angles in 0.001 deg.
  * GripperCtrl position in 0.001 mm, effort in 0.001 N*m (mN*m).
  * GripperCtrl code: 0x00=disable, 0x01=enable, 0x02=disable+clear-err,
    0x03=enable+clear-err. set_zero: 0xAE writes current pos as zero.
  * MotionCtrl_2 move_mode: 0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L,
    0x03=MOVE_C, 0x04=MOVE_M, 0x05=MOVE_CPV.
  * MotionCtrl_2 installation_pos (byte 6, firmware >= V1.5-2):
    0x01=horizontal, 0x02=side-left, 0x03=side-right.

Safety notes
------------
  * ``MotionCtrl_1(0x02, 0, 0)`` (a.k.a. ``ResetPiper`` in the SDK) cuts
    power on every joint -- the arm will physically drop. It is NOT a
    "clear errors" command.  :meth:`PiperController.reset_and_drop`
    additionally sends ``MotionCtrl_2(0, 0, 0, 0x00)`` to return the arm
    to position-velocity standby (matches ``piper_reset.py``).
  * ``MotionCtrl_1(0x01, 0, 0)`` (a.k.a. ``EmergencyStop(0x01)`` in the
    SDK) is the soft e-stop. After it you must ``reset_and_drop()`` and
    re-enable *twice* to resume motion (see :meth:`PiperController.stop`).
  * ``DisableArm(7)`` removes holding torque. Do not call
    :meth:`PiperController.clear_errors` unless the arm is supported and
    a deliberate driver-error recovery is required.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

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
# PiPER's internal representation is uniform at 0.001 mm for every
# position-like quantity. We derive the mm-scale from the m-scale
# rather than hard-coding both so a copy/paste typo cannot make
# EndPoseCtrl and GripperCtrl disagree on what "1 mm" means.
_M_TO_PIPER_POS = 1.0e6                        # meters -> 0.001 mm
_MM_TO_PIPER_GRIP = _M_TO_PIPER_POS / 1.0e3    # mm -> 0.001 mm (derived)
_DEG_TO_PIPER_ANG = 1.0e3                      # degrees -> 0.001 deg
_RAD_TO_PIPER_ANG = _DEG_TO_PIPER_ANG * 180.0 / math.pi
# Note: the gripper effort unit accepted by GripperCtrl is mN*m
# (SDK docstring: 0-5000 = 0-5 N/m). Callers pass ``effort_mNm`` through
# directly, so there is no N*m->PiPER conversion factor here.

# ---- Move modes (MotionCtrl_2 byte 1) ---------------------------------
MOVE_P = 0x00
MOVE_J = 0x01
MOVE_L = 0x02
MOVE_C = 0x03
MOVE_M = 0x04
MOVE_CPV = 0x05

# ---- Gripper op codes (GripperCtrl byte 6) ----------------------------
GRIPPER_DISABLE = 0x00
GRIPPER_ENABLE = 0x01
GRIPPER_DISABLE_CLEAR_ERR = 0x02
GRIPPER_ENABLE_CLEAR_ERR = 0x03
GRIPPER_SET_ZERO_CODE = 0xAE

# ---- Installation position (MotionCtrl_2 byte 6, firmware >= V1.5-2) --
InstallPos = Literal["horizontal", "left", "right"]
_INSTALL_POS_CODE = {"horizontal": 0x01, "left": 0x02, "right": 0x03}

# ---- Joint soft limits (rad) ------------------------------------------
# These match the Piper_sdk_ui joint_control_window sliders. The SDK's
# own hard limits (documented in piper_interface_v2.py) are only enforced
# when the interface is constructed with start_sdk_joint_limit=True; we
# reproduce the UI limits here so that out-of-range targets fail loudly
# in Python rather than being silently clipped by firmware.
_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.618, 2.618],     # J1
        [0.0, 3.14],         # J2
        [-2.697, 0.0],       # J3
        [-1.832, 1.832],     # J4
        [-1.22, 1.22],       # J5
        [-2.0944, 2.0944],   # J6
    ],
    dtype=np.float64,
)
_JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")


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
        If True, enable the arm as part of :meth:`connect`.
    installation_pos:
        If not None, write the arm installation position on connect.
        Required for firmware >= V1.5-2 so gravity comp is correct.
        One of ``"horizontal"``, ``"left"``, ``"right"``.
    """

    def __init__(
        self,
        can_port: str = "can0",
        enable_on_connect: bool = True,
        installation_pos: Optional[InstallPos] = None,
    ) -> None:
        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. `pip install piper_sdk` and configure "
                "your CAN interface before using PiperController."
            )
        self.can_port = can_port
        self.enable_on_connect = enable_on_connect
        self.installation_pos: Optional[InstallPos] = installation_pos
        self.disable_on_disconnect = True
        self._piper: Optional[C_PiperInterface_V2] = None
        self._log = get_logger("piper")

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "PiperController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect(disable_arm=self.disable_on_disconnect)

    def connect(self) -> None:
        """Construct the SDK interface, start reader threads, optionally enable.

        Mirrors the canonical demo sequence: construct the singleton, call
        ``ConnectPort()`` *once*, wait for feedback frames, then enable.
        We do **not** call ``ConnectPort(True)`` here -- ``can_init=True``
        re-initialises the CAN socket and is only appropriate after a
        prior ``DisconnectPort()`` (see :meth:`reconnect`).
        """
        # C_PiperInterface_V2 is a per-``can_name`` singleton: repeated
        # construction returns the cached instance and ``__init__`` is a
        # no-op after the first call. That also means flags like
        # ``judge_flag`` passed on later constructions are ignored.
        self._piper = C_PiperInterface_V2(self.can_port)
        self._piper.ConnectPort()
        # Low-speed feedback arrives at ~100 Hz; give the reader thread
        # enough time to fill the first frame before we look at enable
        # status, otherwise the first GetArmLowSpdInfoMsgs returns zeros.
        self._wait_for_feedback(timeout_s=1.0)
        if self.installation_pos is not None:
            self.set_installation_pos(self.installation_pos)
        if self.enable_on_connect:
            self.enable_arm()

    def reconnect(self) -> None:
        """Re-initialise the CAN bus after a ``DisconnectPort`` call.

        Equivalent to the Piper_sdk_ui sequence ``ConnectPort(True)``.
        Only call this if ``DisconnectPort`` or a CAN down/up cycle
        happened; a plain :meth:`connect` is sufficient otherwise.
        """
        if self._piper is None:
            self.connect()
            return
        self._piper.ConnectPort(True)
        self._wait_for_feedback(timeout_s=1.0)

    def disconnect(self, disable_arm: bool = True) -> None:
        """Shut down cleanly.

        When ``disable_arm`` is True this mirrors ``piper_disable.py`` by
        disabling the gripper with the clear-error code in addition to
        calling ``DisableArm(7)``. Then ``DisconnectPort`` stops the
        reader thread and closes the socketcan handle; without this the
        daemon thread stays alive and port-reconnect is flaky.
        """
        if self._piper is None:
            return
        if disable_arm:
            try:
                self.disable_arm()
            except Exception as e:
                self._log.warning("disable_arm during disconnect failed: %s", e)
        try:
            self._piper.DisconnectPort()
        except Exception as e:
            self._log.warning("DisconnectPort failed: %s", e)
        self._piper = None

    # -- raw handle -----------------------------------------------------
    @property
    def raw(self) -> "C_PiperInterface_V2":
        if self._piper is None:
            raise RuntimeError("PiperController is not connected.")
        return self._piper

    # -- health ---------------------------------------------------------
    def is_ok(self) -> bool:
        """True if the CAN reader thread is alive and receiving frames."""
        try:
            return bool(self.raw.isOk())
        except Exception:
            return False

    def can_fps(self) -> float:
        """Current CAN receive frame rate (Hz)."""
        return float(self.raw.GetCanFps())

    def get_arm_status(self):
        """Proxy to ``C_PiperInterface_V2.GetArmStatus()``."""
        return self.raw.GetArmStatus()

    def _wait_for_feedback(self, timeout_s: float = 1.0) -> bool:
        """Block until at least one low-speed feedback frame has arrived.

        Returns True if feedback is flowing, False on timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            low = self.raw.GetArmLowSpdInfoMsgs()
            if getattr(low, "Hz", 0.0) > 0.0:
                return True
            time.sleep(0.02)
        self._log.warning(
            "no low-speed feedback within %.1fs on %s; is the arm powered and "
            "CAN up? (`ip link show %s`)", timeout_s, self.can_port, self.can_port,
        )
        return False

    # -- enable / disable ----------------------------------------------
    def enable_arm(self, timeout_s: float = 5.0, poll_s: float = 0.25) -> None:
        """Enable all 6 joints and the gripper, polling until they report enabled.

        The cadence (``poll_s >= 0.25s``) matches the official demos.
        Sending EnableArm faster than feedback arrives spams the bus and
        can actually *prevent* the drivers from latching enabled.

        Raises
        ------
        RuntimeError
            If the arm fails to enable within ``timeout_s``. This is
            fatal: continuing would issue motion commands that the
            drivers ignore, leaving the caller unable to tell simulation
            from real motion.
        """
        piper = self.raw
        t0 = time.time()
        last_states: Sequence[bool] = []
        while True:
            # Read feedback first (demo order): if already enabled we can
            # return immediately without another redundant write.
            last_states = self.get_arm_enable_status()
            if all(last_states):
                self._log.info("arm enabled")
                return
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, GRIPPER_ENABLE, 0)
            if time.time() - t0 >= timeout_s:
                break
            time.sleep(poll_s)

        raise RuntimeError(
            f"arm enable timeout after {timeout_s:.1f}s; "
            f"last motor states={list(last_states)}. "
            "Check e-stop, CAN wiring, and that the teach pendant is not "
            "holding the arm."
        )

    def disable_arm(self, timeout_s: float = 5.0, poll_s: float = 0.25) -> None:
        """Disable all joints and the gripper.

        Mirrors ``piper_disable.py``: loops with ``DisableArm(7)`` +
        ``GripperCtrl(..., 0x02, 0)`` until every motor reports disabled
        or ``timeout_s`` expires. Unlike :meth:`enable_arm` this is best-
        effort: we log a warning instead of raising so ``disconnect``
        still completes even if the bus has just gone down.
        """
        piper = self.raw
        t0 = time.time()
        while True:
            states = self.get_arm_enable_status()
            if not any(states):
                self._log.info("arm disabled")
                return
            piper.DisableArm(7)
            piper.GripperCtrl(0, 1000, GRIPPER_DISABLE_CLEAR_ERR, 0)
            if time.time() - t0 >= timeout_s:
                self._log.warning(
                    "disable_arm: timeout after %.1fs, last states=%s",
                    timeout_s, states,
                )
                return
            time.sleep(poll_s)

    def clear_errors(self) -> None:
        """Clear a latched driver-error state by disable+clear / enable+clear.

        Uses the single-shot clear-err codes (``0x02`` on disable,
        ``0x03`` on re-enable) rather than the plain disable/enable
        cycle: this keeps the gripper latched error out of the way on
        the first recovery pass.

        .. warning::
            This removes holding torque between the disable and the
            re-enable. Only use it when the arm is supported.
        """
        self.raw.DisableArm(7)
        self.raw.GripperCtrl(0, 1000, GRIPPER_DISABLE_CLEAR_ERR, 0)
        time.sleep(0.1)
        # Use the enable-and-clear-err gripper code on re-enable so any
        # remaining gripper latched fault also clears in one shot.
        piper = self.raw
        t0 = time.time()
        while time.time() - t0 < 5.0:
            states = self.get_arm_enable_status()
            if all(states):
                self._log.info("arm re-enabled after clear_errors")
                return
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, GRIPPER_ENABLE_CLEAR_ERR, 0)
            time.sleep(0.25)
        raise RuntimeError("clear_errors: failed to re-enable within 5s")

    # -- stop / reset --------------------------------------------------
    def stop(self) -> None:
        """Soft emergency stop (``MotionCtrl_1(0x01, 0, 0)``).

        The arm holds position but refuses motion commands. To resume:
        call :meth:`reset_and_drop` (with the arm supported), then
        :meth:`enable_arm` **twice** (this is the SDK-documented
        recovery ritual for the stop code).
        """
        self.raw.MotionCtrl_1(0x01, 0, 0)

    def reset_and_drop(self) -> None:
        """Hard reset via ``MotionCtrl_1(0x02, 0, 0)`` + standby switch.

        Clears all internal flags and returns the arm to position-
        velocity standby, mirroring ``piper_sdk_demo/V2/piper_reset.py``.

        .. warning::
            This removes power from every motor; the arm will physically
            drop. Use only when the arm is already supported (e.g.
            resting on the table) or when you deliberately want to exit
            MIT / teach mode back to position control.
        """
        self.raw.MotionCtrl_1(0x02, 0, 0)
        # Back to position-velocity standby (ctrl_mode=0, move_mode=0).
        self.raw.MotionCtrl_2(0, 0, 0, 0x00)

    # -- installation / config -----------------------------------------
    def set_installation_pos(self, pos: InstallPos) -> None:
        """Tell the controller how the arm is mounted.

        Required on firmware >= V1.5-2 so gravity compensation and IK
        are consistent. Corresponds to the UI "Installation position"
        combobox.
        """
        if pos not in _INSTALL_POS_CODE:
            raise ValueError(
                f"installation_pos must be one of "
                f"{sorted(_INSTALL_POS_CODE.keys())!r}, got {pos!r}"
            )
        code = _INSTALL_POS_CODE[pos]
        # V2_installation_pos.py: MotionCtrl_2(0x01, 0x01, 0, 0, 0, code)
        self.raw.MotionCtrl_2(0x01, 0x01, 0, 0, 0, code)
        self._log.info("installation position set to %s (0x%02X)", pos, code)
        self.installation_pos = pos

    def config_init(self) -> None:
        """Restore joint limits / max speed / max accel to factory defaults.

        Equivalent to the UI "Config Init" button
        (``ArmParamEnquiryAndConfig(0x01, 0x02, 0, 0, 0x02)`` + a
        sanity read of the motor speed limits).
        """
        self.raw.ArmParamEnquiryAndConfig(0x01, 0x02, 0, 0, 0x02)
        time.sleep(0.05)
        self.raw.SearchAllMotorMaxAngleSpd()

    # -- motion-mode helpers -------------------------------------------
    def _set_mode(self, move_mode: int, speed_pct: int) -> None:
        speed_pct = int(max(0, min(100, speed_pct)))
        # ctrl_mode=0x01 -> CAN command control, is_mit_mode=0x00 -> pos/vel.
        self.raw.MotionCtrl_2(0x01, int(move_mode), speed_pct, 0x00)

    # -- joint control --------------------------------------------------
    def joint_ctrl_rad(self, joints_rad: Sequence[float], speed_pct: int = 30) -> None:
        if len(joints_rad) != 6:
            raise ValueError("joints_rad must have exactly 6 elements")
        self.validate_joints_rad(joints_rad)
        self._set_mode(MOVE_J, speed_pct)
        vals = [round(float(j) * _RAD_TO_PIPER_ANG) for j in joints_rad]
        self.raw.JointCtrl(*vals)

    def joint_ctrl_deg(self, joints_deg: Sequence[float], speed_pct: int = 30) -> None:
        if len(joints_deg) != 6:
            raise ValueError("joints_deg must have exactly 6 elements")
        self.validate_joints_rad(np.deg2rad(np.asarray(joints_deg, dtype=np.float64)))
        self._set_mode(MOVE_J, speed_pct)
        vals = [round(float(j) * _DEG_TO_PIPER_ANG) for j in joints_deg]
        self.raw.JointCtrl(*vals)

    def move_joints_rad(
        self,
        joints_rad: Sequence[float],
        speed_pct: int = 30,
        tol_deg: float = 0.5,
        timeout_s: float = 10.0,
        command_period_s: float = 0.005,
    ) -> bool:
        """Move to a joint target, repeating commands until arrival/timeout.

        The demos (``piper_joint_ctrl.py``) send ``MotionCtrl_2`` +
        ``JointCtrl`` at ~200 Hz; we keep the same pattern so the servo
        loop never starves a command.
        """
        self.validate_joints_rad(joints_rad)
        target_rad = np.asarray(joints_rad, dtype=np.float64)
        target_units = [round(float(j) * _RAD_TO_PIPER_ANG) for j in target_rad]
        target_deg = np.rad2deg(target_rad)
        speed_pct = int(max(0, min(100, speed_pct)))
        command_speed = 50 if speed_pct <= 0 else speed_pct

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self.raw.MotionCtrl_2(0x01, MOVE_J, command_speed, 0x00)
            self.raw.JointCtrl(*target_units)
            cur = self.get_joints_deg()
            if np.max(np.abs(_wrap_deg(cur - target_deg))) <= tol_deg:
                return True
            time.sleep(command_period_s)
        return False

    def validate_joints_rad(self, joints_rad: Sequence[float]) -> None:
        joints = np.asarray(joints_rad, dtype=np.float64)
        if joints.shape != (6,):
            raise ValueError(f"expected 6 joint angles, got shape {joints.shape}")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint target contains NaN or Inf")

        eps = 1e-6
        below = joints < (_JOINT_LIMITS_RAD[:, 0] - eps)
        above = joints > (_JOINT_LIMITS_RAD[:, 1] + eps)
        bad = np.where(below | above)[0]
        if bad.size == 0:
            return

        joints_deg = np.rad2deg(joints)
        limits_deg = np.rad2deg(_JOINT_LIMITS_RAD)
        details = ", ".join(
            f"{_JOINT_NAMES[i]}={joints_deg[i]:.3f} deg "
            f"outside [{limits_deg[i, 0]:.3f}, {limits_deg[i, 1]:.3f}] deg"
            for i in bad
        )
        raise ValueError(
            "joint target is outside the Piper UI joint limits: "
            f"{details}. Re-teach the observe pose inside these limits."
        )

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

    def get_arm_enable_status(self) -> list[bool]:
        """Per-joint driver_enable_status, read from low-speed feedback."""
        low_spd = self.raw.GetArmLowSpdInfoMsgs()
        return [
            bool(low_spd.motor_1.foc_status.driver_enable_status),
            bool(low_spd.motor_2.foc_status.driver_enable_status),
            bool(low_spd.motor_3.foc_status.driver_enable_status),
            bool(low_spd.motor_4.foc_status.driver_enable_status),
            bool(low_spd.motor_5.foc_status.driver_enable_status),
            bool(low_spd.motor_6.foc_status.driver_enable_status),
        ]

    # -- cartesian end-pose control ------------------------------------
    def end_pose_ctrl(
        self,
        pose: EndPose | np.ndarray,
        linear: bool = False,
        speed_pct: int = 20,
    ) -> None:
        """Send a cartesian end-effector target.

        ``pose`` may be either an :class:`EndPose` (SI units + degrees)
        or a 4x4 transform in meters. Set ``linear=True`` to use MOVE_L
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

    def move_to_pose(
        self,
        pose: EndPose | np.ndarray,
        linear: bool = False,
        speed_pct: int = 20,
        pos_tol_m: float = 0.003,
        ang_tol_deg: float = 1.0,
        timeout_s: float = 8.0,
        command_period_s: float = 0.02,
    ) -> bool:
        """Command a cartesian target and keep re-sending it until arrival.

        ``end_pose_ctrl`` is a single-shot write: one dropped CAN frame
        or an intervening mode switch can leave the arm "not moving".
        This helper mirrors :meth:`move_joints_rad` by repeating the
        MotionCtrl_2 + EndPoseCtrl pair at ~50 Hz until the arm actually
        reaches the pose or the timeout expires. Returns True on
        arrival, False on timeout.
        """
        # Normalise to matrix form once (avoid recomputing Euler every loop).
        if isinstance(pose, np.ndarray):
            T_target = np.asarray(pose, dtype=np.float64)
            if T_target.shape != (4, 4):
                raise ValueError("pose matrix must be 4x4")
        else:
            T_target = pose.as_matrix()
        x, y, z = T_target[:3, 3]
        rx, ry, rz = matrix_to_euler_xyz_deg(T_target[:3, :3])
        cmd = (
            round(float(x) * _M_TO_PIPER_POS),
            round(float(y) * _M_TO_PIPER_POS),
            round(float(z) * _M_TO_PIPER_POS),
            round(float(rx) * _DEG_TO_PIPER_ANG),
            round(float(ry) * _DEG_TO_PIPER_ANG),
            round(float(rz) * _DEG_TO_PIPER_ANG),
        )
        move_mode = MOVE_L if linear else MOVE_P
        speed_pct = int(max(0, min(100, speed_pct)))
        command_speed = 20 if speed_pct <= 0 else speed_pct

        R_target = T_target[:3, :3]
        t_target = T_target[:3, 3]
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self.raw.MotionCtrl_2(0x01, move_mode, command_speed, 0x00)
            self.raw.EndPoseCtrl(*cmd)
            cur = self.get_end_pose_matrix()
            if (
                float(np.linalg.norm(cur[:3, 3] - t_target)) <= pos_tol_m
                and rotation_angle_deg(cur[:3, :3], R_target) <= ang_tol_deg
            ):
                return True
            time.sleep(command_period_s)
        return False

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

        ``effort_mNm`` is in milli-newton-metres (i.e. 0.001 N*m),
        matching the PiPER SDK unit exactly. The SDK clamps to
        [0, 5000]; pass 1000 for 1 N*m.
        """
        code = GRIPPER_ENABLE if enable else GRIPPER_DISABLE
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

    # -- gripper feedback ---------------------------------------------
    def get_gripper_opening_m(self) -> float:
        """Return the current gripper opening in meters.

        Reads :py:meth:`GetArmGripperMsgs` from the SDK. The message
        carries the gripper position in 0.001 mm (the same unit we
        write with :py:meth:`GripperCtrl`), so we convert back to
        meters here. Negative values (can happen briefly during zero
        calibration) are clamped to 0 so callers don't have to.
        """
        msg = self.raw.GetArmGripperMsgs().gripper_state
        # Newer SDK uses ``grippers_angle``; older versions use
        # ``gripper_angle``. ``getattr(..., default)`` still returns the
        # attribute when it exists but is ``None``, which would then
        # crash ``float()``; fall through explicitly in that case.
        raw_pos = getattr(msg, "grippers_angle", None)
        if raw_pos is None:
            raw_pos = getattr(msg, "gripper_angle", 0.0)
        if raw_pos is None:
            raw_pos = 0.0
        opening_m = float(raw_pos) / _MM_TO_PIPER_GRIP / 1000.0
        return max(0.0, opening_m)

    def get_gripper_effort_mNm(self) -> float:
        """Return the current gripper torque reading in mN*m.

        Useful for detecting that the jaws have actually clamped on
        something (as opposed to closing into empty air).
        """
        msg = self.raw.GetArmGripperMsgs().gripper_state
        return float(getattr(msg, "grippers_effort", 0.0))

    def close_until(
        self,
        target_opening_m: float = 0.0,
        effort_mNm: float = 1000.0,
        tol_m: float = 0.0015,
        timeout_s: float = 2.0,
        poll_s: float = 0.05,
        stall_window_s: float = 0.25,
        stall_eps_m: float = 0.0003,
    ) -> tuple[bool, float]:
        """Close the gripper towards ``target_opening_m`` with feedback.

        Sends a GripperCtrl command once, then polls
        :py:meth:`get_gripper_opening_m` until one of:

        * the current opening is within ``tol_m`` of ``target_opening_m``
          (success, reached commanded position);
        * the opening stops changing by more than ``stall_eps_m`` over
          ``stall_window_s`` (success, jaws clamped on an object);
        * ``timeout_s`` expires (failure).

        Returns ``(reached_or_stalled, final_opening_m)``. Callers
        typically compare ``final_opening_m`` against the expected
        object width to decide whether the grip is good.
        """
        target = max(0.0, float(target_opening_m))
        self.set_gripper_opening_mm(
            target * 1000.0, effort_mNm=effort_mNm, enable=True
        )

        # Ring buffer of (timestamp, opening_m) samples covering the
        # trailing ``stall_window_s`` of readings. We only need
        # min/max over this window to decide "jaws stopped moving",
        # and the window at 20 Hz poll * 0.25 s holds <10 entries, so
        # scanning it on each poll stays O(1) amortised while keeping
        # memory bounded for long timeouts.
        history: deque[tuple[float, float]] = deque()
        t0 = time.time()
        final = target
        while time.time() - t0 < timeout_s:
            time.sleep(poll_s)
            cur = self.get_gripper_opening_m()
            final = cur
            now = time.time()
            history.append((now, cur))
            # Success 1: reached the commanded opening.
            if abs(cur - target) <= tol_m:
                return True, cur
            # Success 2: stalled -- jaws are no longer moving, probably
            # because they have clamped on an object wider than
            # ``target``. Drop samples older than the stall window and
            # check min/max over what remains; require at least one
            # full window of elapsed time since the close started so
            # we don't mis-fire on the very first couple of samples.
            cutoff = now - stall_window_s
            while history and history[0][0] < cutoff:
                history.popleft()
            if (
                now - t0 >= stall_window_s
                and len(history) >= 2
                and (max(v for _, v in history) - min(v for _, v in history))
                <= stall_eps_m
            ):
                return True, cur
        return False, final

    def check_grasp(
        self,
        expected_width_m: float,
        min_width_m: float = 0.003,
        tol_m: float = 0.006,
    ) -> tuple[bool, float]:
        """Heuristic post-close check: are we actually holding something?

        Reads the current gripper opening and compares against the
        object's expected short-axis width. Returns ``(ok, opening_m)``.

        * ``ok`` is False if the jaws closed below ``min_width_m`` --
          they met each other, no object in between.
        * ``ok`` is False if the jaws are more than ``tol_m`` away
          from ``expected_width_m`` in either direction -- the object
          either slipped or was a completely different size than
          perception estimated.
        """
        opening = self.get_gripper_opening_m()
        if opening < min_width_m:
            return False, opening
        if abs(opening - expected_width_m) > tol_m:
            return False, opening
        return True, opening

    def gripper_clear_errors(self) -> None:
        """Clear gripper-only latched faults (equivalent to the UI
        "Gripper disable and clear err" button)."""
        self.raw.GripperCtrl(0, 1000, GRIPPER_DISABLE_CLEAR_ERR, 0)

    def zero_gripper(self) -> None:
        """Calibrate the current gripper position as the zero reference.

        Follows ``piper_sdk_demo/V2/piper_gripper_zero_set.py`` exactly:
        disable the gripper, wait 1.5 s for it to come to rest, then
        write 0xAE while still disabled. Calling this while the gripper
        is actively holding torque (code 0x01) would latch a bad zero.
        """
        self.raw.GripperCtrl(0, 1000, GRIPPER_DISABLE, 0)
        time.sleep(1.5)
        self.raw.GripperCtrl(0, 1000, GRIPPER_DISABLE, GRIPPER_SET_ZERO_CODE)

    def gripper_teaching_pendant_param_config(
        self,
        teach_pendant_stroke_mm: int = 100,
        max_range_mm: int = 70,
    ) -> None:
        """First-time gripper parameter push (see ``V2_gripper_param_config.py``).

        Without this the gripper + teach pendant won't report back or
        respond to control. Typical values are
        ``teach_pendant_stroke_mm=100`` and ``max_range_mm=70``.
        """
        self.raw.GripperTeachingPendantParamConfig(
            teach_pendant_stroke_mm, max_range_mm
        )


def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap an angular difference (degrees) into [-180, 180]."""
    return (np.asarray(delta) + 180.0) % 360.0 - 180.0


# Backwards-compat alias: the previous public API exposed ``recover_errors``.
# Keep the name working so external scripts don't break; the new canonical
# name (``clear_errors``) aligns with SDK / UI terminology.
PiperController.recover_errors = PiperController.clear_errors  # type: ignore[attr-defined]
