"""RealSense D435i wrapper.

Provides a context manager that streams aligned color + depth frames and
exposes the color intrinsics. The ``grab_aligned`` method returns a color
image in BGR (uint8) and an aligned depth image in millimeters (uint16).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - hardware dependency
    rs = None  # type: ignore[assignment]

from ..common import get_logger

_log = get_logger("camera")


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics of the color stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"
    coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def dist(self) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=np.float64).reshape(1, -1)


class RealSenseCamera:
    """Context-managed RealSense D435i camera.

    Usage:
        with RealSenseCamera() as cam:
            color, depth_m = cam.grab_aligned()

    Depth post-processing
    ---------------------
    D435i depth is noisy at mask boundaries (classic "flying pixels"):
    the shadow edge of an object picks up background range, so mask +
    depth back-projection produces points that are 5-20 cm behind the
    real object. The official RealSense recommendation is to chain
    spatial + temporal filters on the depth frame *before* alignment
    and masking. We expose each filter as a constructor flag so the
    caller can opt in from yaml without pulling pyrealsense2 into
    every call site.

    The filters run only when pyrealsense2 reports them available (they
    are part of the post_processing module, which is present in every
    supported SDK build, so this is effectively always-on once enabled).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        color_fps: int = 30,
        depth_fps: int = 30,
        align_to: str = "color",
        depth_scale: float = 0.001,
        warmup_frames: int = 30,
        spatial_filter: bool = False,
        temporal_filter: bool = False,
        hole_filling_filter: bool = False,
    ) -> None:
        if rs is None:
            raise ImportError(
                "pyrealsense2 is not installed. Install the RealSense SDK and the "
                "pyrealsense2 Python bindings before using RealSenseCamera."
            )
        self.width = width
        self.height = height
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        # ``depth_scale`` here is the *expected* scale from the yaml config;
        # the actual scale is queried from the device in ``start()`` and
        # overwrites this attribute if the device reports a different value
        # (rare, but happens if the firmware has been flashed with custom
        # units). Keeping the configured value as a fallback lets unit
        # tests run without a physical camera.
        self.depth_scale = float(depth_scale)
        self._expected_depth_scale = float(depth_scale)
        self.warmup_frames = int(max(0, warmup_frames))

        self._pipeline: Optional["rs.pipeline"] = None
        self._profile = None
        self._align = None
        self._intr: Optional[CameraIntrinsics] = None
        self._align_to_color = align_to == "color"

        # Lazy-constructed filter chain. Order matches Intel's
        # documented recommendation: spatial (edge-preserving smoothing)
        # -> temporal (time averaging) -> hole filling (last). We keep
        # the raw objects so they can be inspected / tuned after start().
        self._enable_spatial = bool(spatial_filter)
        self._enable_temporal = bool(temporal_filter)
        self._enable_hole_filling = bool(hole_filling_filter)
        self._depth_filters: list = []

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.color_fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps
        )
        self._profile = pipeline.start(config)
        self._pipeline = pipeline

        target = rs.stream.color if self._align_to_color else rs.stream.depth
        self._align = rs.align(target)

        # Build the depth-filter chain. Do this *after* pipeline.start()
        # so if the SDK build we are on lacks any of these filters, the
        # AttributeError is caught and we fall back gracefully instead
        # of failing the whole camera start.
        self._depth_filters = []
        try:
            if self._enable_spatial:
                # Edge-preserving smoothing: removes quantisation stripes
                # without blurring depth discontinuities.
                self._depth_filters.append(rs.spatial_filter())
            if self._enable_temporal:
                # Time-average: reduces jitter; assumes a relatively
                # stable scene. Not ideal if the camera is still
                # swinging to the observe pose, but the warmup_frames
                # loop above already gives it ~1 s to settle.
                self._depth_filters.append(rs.temporal_filter())
            if self._enable_hole_filling:
                # Fills small invalid pixels. Use sparingly: on thin
                # objects this can hallucinate depth at mask edges.
                self._depth_filters.append(rs.hole_filling_filter())
        except AttributeError as exc:  # pragma: no cover - SDK-dependent
            _log.warning(
                "pyrealsense2 is missing a depth filter class (%s); continuing without it",
                exc,
            )
            self._depth_filters = []

        # Pull the *actual* depth scale from the firmware; the yaml constant
        # is only a fallback. D435i typically reports 0.001 m/unit, but a
        # wrong value silently turns a 50 cm object into 5 m and breaks
        # every downstream computation.
        try:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            reported = float(depth_sensor.get_depth_scale())
            if abs(reported - self._expected_depth_scale) > 1e-6:
                _log.warning(
                    "device depth_scale=%g m/unit differs from yaml %g; using device value",
                    reported, self._expected_depth_scale,
                )
            self.depth_scale = reported
        except Exception as exc:  # pragma: no cover - hardware dependent
            _log.warning(
                "could not read device depth_scale (%s); falling back to %g",
                exc, self._expected_depth_scale,
            )
            self.depth_scale = self._expected_depth_scale

        # Warm up auto-exposure / auto white-balance. Five frames (the old
        # value) is too short on D435i: exposure typically needs 20-30
        # frames at 30 fps to settle, otherwise the first captured image
        # is visibly dark/over-exposed and YOLO misses low-contrast
        # objects. Configurable via ``warmup_frames``.
        for _ in range(self.warmup_frames):
            pipeline.wait_for_frames()

        self._intr = self._read_color_intrinsics()

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:  # pragma: no cover - hardware dependent
                _log.warning("RealSense pipeline.stop() raised: %s", exc)
        self._pipeline = None
        self._profile = None
        self._align = None

    # -- I/O ------------------------------------------------------------
    def grab_aligned(
        self, timeout_ms: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (color_bgr uint8, depth_mm uint16).

        Both images share the color-stream intrinsics after alignment.
        """
        if self._pipeline is None or self._align is None:
            raise RuntimeError("Camera is not started; use `with RealSenseCamera()`.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        # Apply depth-only post-processing *before* alignment. The
        # RealSense docs recommend this order because alignment
        # resamples depth onto the color grid, which would otherwise
        # smear the effect of a spatial filter.
        if self._depth_filters:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                for f in self._depth_filters:
                    depth_frame = f.process(depth_frame)
                # Reinsert the filtered depth into the composite frame
                # so ``align.process`` operates on the denoised stream.
                # Older SDKs expose as_frameset() on the returned frame;
                # we use it when present, otherwise fall back to the
                # original frames (no filtering applied).
                try:
                    frames = depth_frame.as_frameset()
                except Exception:  # pragma: no cover - SDK-dependent
                    pass
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to obtain aligned color/depth frames.")
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        return color, depth

    def grab_color(self, timeout_ms: int = 2000) -> np.ndarray:
        color, _ = self.grab_aligned(timeout_ms=timeout_ms)
        return color

    # -- intrinsics -----------------------------------------------------
    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intr is None:
            raise RuntimeError("Camera must be started before reading intrinsics.")
        return self._intr

    def _read_color_intrinsics(self) -> CameraIntrinsics:
        assert self._profile is not None
        color_stream = self._profile.get_stream(rs.stream.color)
        ri = color_stream.as_video_stream_profile().get_intrinsics()
        return CameraIntrinsics(
            width=int(ri.width),
            height=int(ri.height),
            fx=float(ri.fx),
            fy=float(ri.fy),
            cx=float(ri.ppx),
            cy=float(ri.ppy),
            model=str(ri.model),
            coeffs=tuple(float(c) for c in ri.coeffs),
        )


def depth_to_meters(depth_raw: np.ndarray, depth_scale: float = 0.001) -> np.ndarray:
    """Convert uint16 depth (millimeters, typically) to float32 meters."""
    return depth_raw.astype(np.float32) * float(depth_scale)
