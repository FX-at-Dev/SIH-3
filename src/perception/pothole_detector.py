"""Pothole detection.

Potholes are a first-class requirement of this system and a genuine problem for
off-the-shelf detectors: **no COCO-trained model contains a pothole class**, so
the stock YOLO weights cannot find them at any confidence threshold. Pretending
otherwise would be the single most misleading thing this codebase could do.

Three modes, and the mode in force is reported everywhere - in the logs, in the
dashboard, and on every detection record:

``none`` (default)
    No pothole detection. The system states clearly that potholes are not being
    detected rather than silently finding nothing.

``yolo``
    A custom-trained pothole model. This is the only mode that performs real
    pothole detection. See ``models/README.md`` for training and class names.
    Detections are tagged ``DetectionSource.MODEL``.

``heuristic``
    A classical computer-vision approximation: dark, roughly convex blobs in
    the road region that are darker than their surroundings. It is **not**
    machine learning and it is **not** reliable - shadows, wet patches, tar
    repairs and manhole covers all trigger it. Every detection is tagged
    ``DetectionSource.HEURISTIC``, rendered with a "HEUR" marker, and carries
    reduced confidence. It exists so the pipeline can be demonstrated without
    custom weights, not as a substitute for them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

from ..core.classes import canonicalize
from ..core.types import (
    BBox,
    Detection,
    DetectionSource,
    PotholeInfo,
    RiskLevel,
    SemanticCategory,
)
from ..geometry.ego_path import EgoPath

if TYPE_CHECKING:
    from ..core.config import Config
    from ..geometry.distance import BaseDistanceEstimator

logger = logging.getLogger(__name__)

__all__ = [
    "BasePotholeDetector",
    "HeuristicPotholeDetector",
    "NullPotholeDetector",
    "YOLOPotholeDetector",
    "assess_pothole",
    "create_pothole_detector",
]


# ---------------------------------------------------------------------------
# Severity assessment - shared by every detector mode
# ---------------------------------------------------------------------------


def assess_pothole(
    detection: Detection,
    config: "Config",
    ego_path: EgoPath,
    distance_estimator: Optional["BaseDistanceEstimator"],
    frame_width: int,
    frame_height: int,
) -> PotholeInfo:
    """Derive severity, size and path relation for a pothole detection.

    Severity rises with physical width and with how much of the ego corridor
    the pothole occupies: a wide one squarely in the wheel track matters far
    more than a wider one on the shoulder.
    """
    severity_cfg = config.pothole.severity
    overlap = ego_path.obstruction(detection.bbox)

    width_m = 0.0
    if distance_estimator is not None:
        result = distance_estimator.estimate(
            detection.bbox, "pothole", frame_width, frame_height
        )
        if result is not None:
            focal = frame_width / (
                2.0 * np.tan(np.radians(config.camera.horizontal_fov_deg) / 2.0)
            )
            if focal > 1e-6:
                width_m = float(detection.bbox.width * result.distance_m / focal)

    if width_m >= severity_cfg.high_width_m:
        size_label = "large"
    elif width_m >= severity_cfg.medium_width_m:
        size_label = "medium"
    else:
        size_label = "small"

    if width_m >= severity_cfg.high_width_m:
        severity = RiskLevel.HIGH
    elif width_m >= severity_cfg.medium_width_m:
        severity = RiskLevel.MEDIUM
    else:
        severity = RiskLevel.LOW

    if overlap >= severity_cfg.high_overlap:
        severity = RiskLevel.max(severity, RiskLevel.MEDIUM)
        if width_m >= severity_cfg.medium_width_m:
            severity = RiskLevel.HIGH

    return PotholeInfo(
        severity=severity,
        estimated_width_m=width_m,
        ego_path_overlap=overlap,
        size_label=size_label,
        source=detection.source,
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class BasePotholeDetector(ABC):
    """Contract for any pothole detection strategy."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Return pothole detections for one frame."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """``none``, ``yolo`` or ``heuristic``."""

    @property
    @abstractmethod
    def is_ml_detection(self) -> bool:
        """True only for real model inference."""

    @property
    def status_text(self) -> str:
        """One line describing what this detector is actually doing."""
        return f"pothole detection: {self.mode}"

    def load(self) -> None:
        """Optional initialisation hook."""

    def set_road_mask(self, mask: Optional[np.ndarray]) -> None:
        """Supply a drivable-area mask to constrain the search (optional)."""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class NullPotholeDetector(BasePotholeDetector):
    """Detects nothing, and says so plainly."""

    def __init__(self, config: "Config") -> None:
        self.config = config

    def load(self) -> None:
        logger.warning(
            "Pothole detection is DISABLED (pothole.type: none). No potholes "
            "will be detected. The main detector cannot find potholes either "
            "unless it was trained on a pothole class - see models/README.md."
        )

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        return []

    @property
    def mode(self) -> str:
        return "none"

    @property
    def is_ml_detection(self) -> bool:
        return False

    @property
    def status_text(self) -> str:
        return "POTHOLE DETECTION OFF (no model configured)"


class YOLOPotholeDetector(BasePotholeDetector):
    """Real pothole detection using a custom-trained YOLO model."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.model_path = Path(config.pothole.model)
        self._model: Any = None
        self._class_names: tuple[str, ...] = ()
        self._failures = 0

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pothole model not found: {self.model_path}\n"
                "\n"
                "  What is missing:\n"
                "    A YOLO checkpoint trained to detect potholes.\n"
                "\n"
                "  There is no stock model for this. COCO has no pothole class,\n"
                "  so pothole detection requires weights you train or obtain.\n"
                "\n"
                "  Options:\n"
                "    1. Train one - see models/README.md for the dataset and\n"
                "       command. A few hundred labelled Indian road images is\n"
                "       enough for a usable prototype model.\n"
                "    2. Point pothole.model at existing weights you already have.\n"
                "    3. Disable it:            pothole.type: none\n"
                "    4. Use the CV heuristic:  pothole.type: heuristic\n"
                "       (approximate, not machine learning, high false-positive\n"
                "       rate on shadows and tar patches)\n"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for pothole.type: yolo.\n"
                "  Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(str(self.model_path))
        names = self._model.names
        self._class_names = tuple(
            names[i] for i in sorted(names)
        ) if isinstance(names, dict) else tuple(names)

        canonical = {canonicalize(n) for n in self._class_names}
        if "pothole" not in canonical:
            logger.warning(
                "Pothole model %s has classes %s, none of which map to "
                "'pothole'. Detections from it will not be treated as potholes. "
                "Check models/README.md for the expected class names.",
                self.model_path, list(self._class_names),
            )
        else:
            logger.info(
                "Pothole model loaded: %s (classes: %s)",
                self.model_path, ", ".join(self._class_names),
            )

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        if self._model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.config.pothole.confidence,
                imgsz=self.config.detector.imgsz,
                verbose=False,
            )
        except Exception as exc:
            self._failures += 1
            if self._failures <= 3:
                logger.error("Pothole inference failed on frame %d: %s",
                             frame_index, exc)
            return []

        detections: list[Detection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        height, width = frame.shape[0], frame.shape[1]
        for box, conf, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            strict=False,
        ):
            raw = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names) else "pothole"
            )
            canonical = canonicalize(raw)
            if canonical not in ("pothole", "road_damage"):
                continue
            detections.append(
                Detection(
                    class_name="pothole",
                    raw_class_name=raw,
                    confidence=float(conf),
                    bbox=BBox(*[float(v) for v in box]).clip(
                        float(width), float(height)
                    ),
                    category=SemanticCategory.OBSTACLE,
                    source=DetectionSource.MODEL,
                    frame_index=frame_index,
                )
            )
        return detections

    @property
    def mode(self) -> str:
        return "yolo"

    @property
    def is_ml_detection(self) -> bool:
        return True

    @property
    def status_text(self) -> str:
        return f"pothole model: {self.model_path.name}"


class HeuristicPotholeDetector(BasePotholeDetector):
    """Classical-CV pothole approximation. NOT machine learning.

    Looks for dark, compact, roughly convex regions within the road area. The
    physical intuition is sound - a pothole is a shadowed depression, darker
    than the surrounding surface - but so are shadows, oil stains, tar repairs
    and manhole covers, and this method cannot tell them apart.

    Confidence is capped well below what a trained model would report, and
    every detection is tagged as heuristic so it can never be mistaken for one.
    """

    MAX_CONFIDENCE = 0.45

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.params = config.pothole.heuristic
        self._road_mask: Optional[np.ndarray] = None

    def load(self) -> None:
        logger.warning(
            "Pothole detection is running in HEURISTIC mode. This is a "
            "classical computer-vision approximation, NOT machine learning. "
            "Expect false positives on shadows, tar patches and manhole "
            "covers. Every such detection is labelled 'HEUR'. For reliable "
            "results train a pothole model - see models/README.md."
        )

    def set_road_mask(self, mask: Optional[np.ndarray]) -> None:
        self._road_mask = mask

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        frame_area = float(height * width)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        region = np.zeros((height, width), dtype=np.uint8)
        horizon = int(height * self.config.drivable_area.horizon_ratio)
        region[horizon:, :] = 255
        if self._road_mask is not None and self._road_mask.shape == region.shape:
            region = cv2.bitwise_and(region, self._road_mask)

        road_pixels = blurred[region > 0]
        if road_pixels.size < 500:
            return []

        threshold = float(np.percentile(road_pixels, self.params.darkness_percentile))
        dark = ((blurred < threshold) & (region > 0)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / frame_area
            if not (self.params.min_area_ratio <= area_ratio
                    <= self.params.max_area_ratio):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 6:
                continue

            aspect = w / float(h)
            if not (self.params.min_aspect <= aspect <= self.params.max_aspect):
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < self.params.min_solidity:
                continue

            contrast = self._local_contrast(blurred, x, y, w, h)
            if contrast < 12.0:
                continue

            confidence = min(
                self.MAX_CONFIDENCE,
                0.18 + 0.25 * solidity + min(0.12, contrast / 200.0),
            )

            detections.append(
                Detection(
                    class_name="pothole",
                    raw_class_name="pothole(heuristic)",
                    confidence=float(confidence),
                    bbox=BBox(float(x), float(y), float(x + w), float(y + h)),
                    category=SemanticCategory.OBSTACLE,
                    source=DetectionSource.HEURISTIC,
                    frame_index=frame_index,
                    attributes={"solidity": round(solidity, 3),
                                "contrast": round(contrast, 1)},
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:6]

    @staticmethod
    def _local_contrast(
        gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """Mean brightness difference between a region's surround and itself."""
        height, width = gray.shape
        pad_x, pad_y = max(6, w // 2), max(6, h // 2)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)

        surround = gray[y0:y1, x0:x1]
        inner = gray[y:y + h, x:x + w]
        if surround.size == 0 or inner.size == 0:
            return 0.0
        return float(np.mean(surround)) - float(np.mean(inner))

    @property
    def mode(self) -> str:
        return "heuristic"

    @property
    def is_ml_detection(self) -> bool:
        return False

    @property
    def status_text(self) -> str:
        return "pothole: HEURISTIC (not ML - approximate)"


def create_pothole_detector(config: "Config") -> BasePotholeDetector:
    """Build the configured pothole detector."""
    if not config.pothole.enabled:
        return NullPotholeDetector(config)

    mode = config.pothole.type.lower()
    if mode == "yolo":
        return YOLOPotholeDetector(config)
    if mode == "heuristic":
        return HeuristicPotholeDetector(config)
    if mode == "none":
        return NullPotholeDetector(config)

    raise ValueError(
        f"Unknown pothole detector type '{config.pothole.type}'. "
        "Expected: none, yolo, heuristic."
    )
