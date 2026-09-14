"""Ultralytics YOLO detector.

This wraps a real model. It reports exactly what the loaded weights can and
cannot detect, and it never substitutes fabricated results when inference
fails - a failed frame returns an empty list and is logged.

Note on the stock weights: ``yolo11n.pt`` and friends are trained on COCO.
COCO contains person, car, bus, truck, motorcycle, bicycle, cow, dog, horse
and a traffic light, but it contains **no pothole, auto-rickshaw, bullock cart,
debris, cone or barrier class**. Those require custom weights; see
``models/README.md``. :meth:`YOLODetector.coverage_report` states this at
startup so the limitation is visible rather than assumed away.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.classes import NOT_IN_COCO, canonicalize
from ..core.types import Detection, DetectionSource
from .detector import BaseDetector, DetectorInfo, ModelNotFoundError

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["YOLODetector"]

_AUTO_DOWNLOADABLE = frozenset({
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
})


class YOLODetector(BaseDetector):
    """Object detector backed by Ultralytics YOLO."""

    def __init__(self, config: "Config") -> None:
        super().__init__(config)
        self.model_path = Path(config.detector.model)
        self._model: Any = None
        self._device: str = "cpu"
        self._class_names: tuple[str, ...] = ()
        self._is_custom = False
        self._failed_frames = 0

    # ---- loading ----------------------------------------------------------

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelNotFoundError(
                "The 'ultralytics' package is not installed, so the YOLO "
                "detector cannot run.\n"
                "  Install it with:\n"
                "    pip install ultralytics\n"
                "  Or run without a model using the mock detector:\n"
                "    python main.py --source image --input <path> --detector mock"
            ) from exc

        weights = self._resolve_weights()
        self._device = self._select_device(self.config.detector.device)

        logger.info("Loading YOLO weights: %s (device=%s)", weights, self._device)
        try:
            os.environ.setdefault("YOLO_VERBOSE", "false")
            self._model = YOLO(str(weights))
            self._model.to(self._device)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Failed to load YOLO weights from '{weights}': {exc}\n"
                "  - Confirm the file is a valid Ultralytics .pt checkpoint.\n"
                "  - If it was trained with a much older Ultralytics release, "
                "re-export it with the current version."
            ) from exc

        names = self._model.names
        self._class_names = tuple(
            names[i] for i in sorted(names)
        ) if isinstance(names, dict) else tuple(names)

        self._is_custom = len(self._class_names) != 80

        logger.info(
            "YOLO ready: %d classes, %s weights.",
            len(self._class_names),
            "custom" if self._is_custom else "COCO (stock)",
        )
        for line in self.coverage_report().splitlines():
            logger.info("%s", line)

    def _resolve_weights(self) -> Path:
        """Locate the weights file, with an actionable error when missing."""
        if self.model_path.exists():
            return self.model_path

        if self.model_path.name in _AUTO_DOWNLOADABLE:
            logger.warning(
                "Weights %s not found locally - Ultralytics will attempt to "
                "download them on first use (requires network access).",
                self.model_path,
            )
            return self.model_path

        raise ModelNotFoundError(
            f"Model weights not found: {self.model_path}\n"
            "\n"
            "  What is missing:\n"
            f"    A YOLO checkpoint at '{self.model_path}'.\n"
            "\n"
            "  How to fix it, pick one:\n"
            "    1. Use stock COCO weights (downloads automatically):\n"
            "         detector.model: models/yolo11n.pt\n"
            "       Note: stock weights detect people, vehicles and animals, but\n"
            "       NOT potholes, auto-rickshaws, bullock carts or road debris.\n"
            "    2. Place your custom-trained weights at the configured path.\n"
            "       See models/README.md for the expected class names.\n"
            "    3. Run the mock detector to exercise the pipeline without a model:\n"
            "         python main.py --source image --input <path> --detector mock\n"
        )

    @staticmethod
    def _select_device(requested: str) -> str:
        """Resolve ``auto`` to the best available device."""
        requested = (requested or "auto").lower()
        try:
            import torch
        except ImportError:
            return "cpu"

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable - falling back to CPU.")
            return "cpu"
        return requested

    # ---- inference --------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Run inference on one frame.

        A failure returns an empty list. It never returns invented detections.
        """
        if self._model is None:
            raise RuntimeError("YOLODetector.detect() called before load().")
        if frame is None or frame.size == 0:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.config.detector.confidence,
                iou=self.config.detector.iou,
                imgsz=self.config.detector.imgsz,
                max_det=self.config.detector.max_detections,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            self._failed_frames += 1
            if self._failed_frames <= 3 or self._failed_frames % 50 == 0:
                logger.error(
                    "YOLO inference failed on frame %d (%d total failures): %s",
                    frame_index, self._failed_frames, exc,
                )
            return []

        return self._parse(results, frame.shape, frame_index)

    def _parse(
        self, results: Any, frame_shape: tuple[int, ...], frame_index: int
    ) -> list[Detection]:
        """Turn Ultralytics results into canonical detections."""
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception as exc:
            logger.error("Could not read YOLO boxes on frame %d: %s", frame_index, exc)
            return detections

        for box, conf, class_id in zip(xyxy, confs, class_ids, strict=False):
            raw_label = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names)
                else "unknown"
            )
            detection = self._make_detection(
                raw_label=raw_label,
                confidence=float(conf),
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                frame_shape=frame_shape,
                frame_index=frame_index,
                class_id=int(class_id),
                source=DetectionSource.MODEL,
            )
            if detection is not None:
                detections.append(detection)

        return detections

    # ---- reporting --------------------------------------------------------

    def coverage_report(self) -> str:
        """State plainly which required classes these weights cannot detect.

        Called at startup. The point is that nobody should ever believe the
        system is looking for potholes when the loaded weights have no such
        class.
        """
        canonical_available = {canonicalize(name) for name in self._class_names}
        missing = [c for c in NOT_IN_COCO if c not in canonical_available]

        lines = [
            f"Detector class coverage ({len(self._class_names)} model classes):",
        ]
        if missing:
            lines.append(
                "  NOT DETECTABLE with these weights: " + ", ".join(missing)
            )
            lines.append(
                "  These classes need custom-trained weights - see models/README.md."
            )
            if "pothole" in missing:
                lines.append(
                    "  Potholes will NOT be detected by the main model. Enable a "
                    "pothole model via pothole.type: yolo, or the clearly-labelled "
                    "heuristic via pothole.type: heuristic."
                )
        else:
            lines.append("  All target classes are present in the loaded weights.")
        return "\n".join(lines)

    @property
    def detectable_classes(self) -> frozenset[str]:
        """Canonical classes these specific weights can produce."""
        return frozenset(canonicalize(name) for name in self._class_names)

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="YOLO",
            model_path=str(self.model_path),
            device=self._device,
            class_names=self._class_names,
            is_custom=self._is_custom,
            notes="custom weights" if self._is_custom else "stock COCO weights",
        )

    def release(self) -> None:
        self._model = None
