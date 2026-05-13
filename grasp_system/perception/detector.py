"""YOLOv8-seg instance segmentation wrapper.

Wraps ``ultralytics.YOLO`` and normalises the output into a list of
:class:`Detection` records containing an instance mask resized to the
depth image shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - optional at import time
    YOLO = None  # type: ignore[assignment]


@dataclass
class Detection:
    """A single YOLOv8-seg result, post-processed for our geometry pipeline."""

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: np.ndarray   # shape (4,), integer pixel coordinates on image grid
    mask: np.ndarray        # shape (H, W), bool, aligned to the input image

    @property
    def num_pixels(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox_center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


class SegmentationDetector:
    """Thin wrapper over :class:`ultralytics.YOLO`.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` checkpoint. Defaults to ``models/best.pt``.
    conf:
        Confidence threshold for predictions.
    device:
        Inference device, e.g. ``"cuda"`` or ``"cpu"``. ``None`` lets
        Ultralytics decide.
    class_names:
        Optional overriding mapping from class index to human label.
    """

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        conf: float = 0.5,
        device: Optional[str] = None,
        class_names: Optional[dict[int, str]] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. `pip install ultralytics` first."
            )
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.device = device
        self._class_names = class_names
        self._model = YOLO(self.model_path)
        # Move the model to the requested device up front; otherwise the
        # first ``predict(device=...)`` call has to shuffle weights
        # between host and GPU, which adds ~1 s to the first frame and
        # complicates reasoning about where inference actually ran.
        if device is not None:
            try:
                self._model.to(device)
            except Exception:  # pragma: no cover - optional pt backend
                # e.g. requested cuda but only cpu is available -- let
                # ultralytics fall back at predict() time and just log.
                pass

    # -- inference ------------------------------------------------------
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        min_pixels: int = 0,
        mask_erode_px: int = 0,
        target_hw: Optional[tuple[int, int]] = None,
    ) -> List[Detection]:
        """Run segmentation on ``image_bgr`` and return detections.

        Parameters
        ----------
        target_hw:
            Optional (height, width) to resize each mask to. Pass the depth
            image shape when you want masks aligned to the depth grid.
        min_pixels:
            Drop detections whose mask has fewer than this many pixels.
        mask_erode_px:
            If > 0, morphologically erode each mask to shrink away flying
            pixels at the object boundary.
        """
        conf = self.conf if conf is None else float(conf)
        res = self._model.predict(
            image_bgr,
            conf=conf,
            device=self.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if res.masks is None or res.boxes is None:
            return detections

        masks = res.masks.data.cpu().numpy()  # (N, h, w), float32 0/1
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy().astype(float)
        names_from_model = getattr(res, "names", None) or getattr(self._model, "names", {})

        H_img, W_img = image_bgr.shape[:2]
        target_hw = target_hw or (H_img, W_img)
        H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
        # Scale factors from the image grid (where YOLO returned bboxes)
        # to the requested target grid (where we will store the mask).
        sx = W_tgt / float(W_img)
        sy = H_tgt / float(H_img)

        for m, xyxy, cid, score in zip(masks, boxes, classes, scores):
            if m.shape != (H_tgt, W_tgt):
                mask_resized = cv2.resize(
                    m.astype(np.uint8),
                    (W_tgt, H_tgt),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask_resized = m.astype(bool)

            # Keep the bbox expressed in the same coordinate system as
            # the mask. Without this, any consumer that uses both
            # fields together (e.g. visualize.draw_detections) breaks
            # silently whenever target_hw != image_bgr.shape[:2].
            x1, y1, x2, y2 = xyxy
            xyxy_scaled = np.array(
                [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                dtype=np.float64,
            )

            if mask_erode_px > 0:
                k = 2 * int(mask_erode_px) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask_resized = cv2.erode(
                    mask_resized.astype(np.uint8), kernel
                ).astype(bool)

            if mask_resized.sum() < min_pixels:
                continue

            label = ""
            if self._class_names is not None and int(cid) in self._class_names:
                label = str(self._class_names[int(cid)])
            elif isinstance(names_from_model, dict) and int(cid) in names_from_model:
                label = str(names_from_model[int(cid)])

            detections.append(
                Detection(
                    class_id=int(cid),
                    label=label,
                    confidence=float(score),
                    bbox_xyxy=xyxy_scaled.astype(np.int32),
                    mask=mask_resized,
                )
            )

        return detections

    def best(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
