"""Detector interface and detector factory.

The rest of the application depends only on :class:`BaseDetector`. Swapping
YOLO for a different model, or for the deterministic :class:`MockDetector`, is
a configuration change and nothing more.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core.classes import canonicalize, category_for
from ..core.types import BBox, Detection, DetectionSource

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["BaseDetector", "DetectorInfo", "ModelNotFoundError", "create_detector"]


class ModelNotFoundError(FileNotFoundError):
    """Raised when configured model weights are missing.

    The message is written to be directly actionable - it says exactly what is
    missing and how to obtain or configure it, rather than failing obscurely.
    """


@dataclass(slots=True)
class DetectorInfo:
    """Describes a loaded detector, for logging and the UI banner."""

    name: str
    model_path: str
    device: str
    class_names: tuple[str, ...] = ()
    is_custom: bool = False
    notes: str = ""


class BaseDetector(ABC):
    """Contract every detector must satisfy."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.confidence = config.detector.confidence
        self._ignore = {c.lower() for c in config.detector.ignore_classes}

    @abstractmethod
    def load(self) -> None:
        """Load weights / initialise. Raises :class:`ModelNotFoundError`."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Run detection on one BGR frame and return canonical detections."""

    @property
    @abstractmethod
    def info(self) -> DetectorInfo:
        """Metadata about the loaded model."""

    # ---- shared helpers ---------------------------------------------------

    def _make_detection(
        self,
        raw_label: str,
        confidence: float,
        xyxy: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
        frame_index: int,
        class_id: int = -1,
        source: DetectionSource = DetectionSource.MODEL,
    ) -> Optional[Detection]:
        """Convert a raw model output into a canonical :class:`Detection`.

        Returns ``None`` when the class is ignored or the box is degenerate.
        """
        if raw_label.lower() in self._ignore:
            return None

        height, width = frame_shape[0], frame_shape[1]
        bbox = BBox(*xyxy).clip(float(width), float(height))
        if bbox.width < 2.0 or bbox.height < 2.0:
            return None

        canonical = canonicalize(raw_label)
        return Detection(
            class_name=canonical,
            raw_class_name=raw_label,
            confidence=float(confidence),
            bbox=bbox,
            category=category_for(canonical),
            source=source,
            class_id=class_id,
            frame_index=frame_index,
        )

    def release(self) -> None:
        """Release model resources. Default is a no-op."""

    def __enter__(self) -> "BaseDetector":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def create_detector(config: "Config") -> BaseDetector:
    """Build the detector named in ``config.detector.type``."""
    detector_type = config.detector.type.lower()

    if detector_type == "yolo":
        from .yolo_detector import YOLODetector
        return YOLODetector(config)
    if detector_type == "mock":
        from .mock_detector import MockDetector
        return MockDetector(config)

    raise ValueError(
        f"Unknown detector type '{config.detector.type}'. Expected: yolo, mock."
    )
