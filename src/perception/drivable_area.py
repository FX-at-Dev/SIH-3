"""Drivable-area estimation.

Deliberately does not depend on lane markings. On the roads this system targets,
markings are frequently faded, absent, or simply not followed, so a
lane-detection approach fails precisely where the system is most needed.

The MVP uses a colour/texture heuristic: sample the road surface immediately in
front of the vehicle - where the road almost certainly is - and grow a mask of
pixels resembling it. Crude, but it needs no training data and no markings.

:class:`DrivableAreaDetector` is the seam where a real segmentation model
(SegFormer, BiSeNet, YOLOP, or a YOLO-seg checkpoint) drops in later without
touching anything downstream.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "DrivableArea",
    "DrivableAreaDetector",
    "HeuristicDrivableAreaDetector",
    "NullDrivableAreaDetector",
    "create_drivable_area_detector",
]


@dataclass(slots=True)
class DrivableArea:
    """Estimated drivable region for one frame."""

    mask: Optional[np.ndarray] = None
    drivable_ratio: float = 1.0
    available: bool = False
    method: str = "none"

    def ratio_in_box(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Fraction of a box's area that falls on drivable surface."""
        if self.mask is None or not self.available:
            return 1.0
        height, width = self.mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        patch = self.mask[y1:y2, x1:x2]
        return float(np.count_nonzero(patch)) / float(patch.size)


class DrivableAreaDetector(ABC):
    """Interface for drivable-area estimation."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DrivableArea:
        """Estimate the drivable region for one frame."""

    @property
    @abstractmethod
    def method(self) -> str:
        ...


class NullDrivableAreaDetector(DrivableAreaDetector):
    """Disabled: assumes everything ahead is drivable."""

    def detect(self, frame: np.ndarray) -> DrivableArea:
        return DrivableArea(available=False, method="none")

    @property
    def method(self) -> str:
        return "none"


class HeuristicDrivableAreaDetector(DrivableAreaDetector):
    """Colour-similarity road estimate seeded from the near field.

    Assumes the strip directly in front of the vehicle is road surface. That
    assumption fails if the vehicle is already up against an obstacle, so the
    result is reported as an estimate and never used as a hard constraint - it
    informs the pothole search region and a dashboard indicator only.
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.horizon_ratio = config.drivable_area.horizon_ratio

    def detect(self, frame: np.ndarray) -> DrivableArea:
        if frame is None or frame.size == 0:
            return DrivableArea(available=False, method="heuristic")

        try:
            return self._detect(frame)
        except cv2.error as exc:
            logger.debug("Drivable-area estimation failed: %s", exc)
            return DrivableArea(available=False, method="heuristic")

    def _detect(self, frame: np.ndarray) -> DrivableArea:
        height, width = frame.shape[:2]
        horizon = int(height * self.horizon_ratio)

        small = cv2.resize(frame, (width // 2, height // 2))
        blurred = cv2.GaussianBlur(small, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sh, sw = hsv.shape[:2]

        seed = hsv[int(sh * 0.86):sh, int(sw * 0.35):int(sw * 0.65)]
        if seed.size == 0:
            return DrivableArea(available=False, method="heuristic")

        median = np.median(seed.reshape(-1, 3), axis=0)
        spread = np.std(seed.reshape(-1, 3), axis=0)

        tolerance = np.clip(spread * 2.6, [8, 26, 30], [26, 78, 92])
        lower = np.clip(median - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(median + tolerance, 0, 255).astype(np.uint8)

        mask_small = cv2.inRange(hsv, lower, upper)

        mask_small[: int(sh * self.horizon_ratio), :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)

        mask_small = self._largest_bottom_component(mask_small)

        mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)

        lower_area = mask[horizon:, :]
        ratio = (
            float(np.count_nonzero(lower_area)) / float(lower_area.size)
            if lower_area.size else 0.0
        )

        return DrivableArea(
            mask=mask,
            drivable_ratio=ratio,
            available=True,
            method="heuristic",
        )

    @staticmethod
    def _largest_bottom_component(mask: np.ndarray) -> np.ndarray:
        """Keep the largest blob that touches the bottom edge of the frame."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return mask

        height = mask.shape[0]
        best_label, best_area = -1, 0
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            bottom = stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]
            if bottom >= height - 2 and area > best_area:
                best_label, best_area = label, area

        if best_label < 0:
            return mask
        return ((labels == best_label).astype(np.uint8)) * 255

    @property
    def method(self) -> str:
        return "heuristic"


class SegmentationDrivableAreaDetector(DrivableAreaDetector):
    """Placeholder for a real segmentation model.

    Intentionally raises rather than silently degrading to the heuristic: a
    config asking for segmentation should not quietly get something weaker.
    """

    def __init__(self, config: "Config") -> None:
        raise NotImplementedError(
            "drivable_area.type: segmentation is an interface stub, not an "
            "implementation.\n"
            "  To add one, implement DrivableAreaDetector.detect() in "
            "src/perception/drivable_area.py using a segmentation checkpoint "
            "(for example a YOLO-seg model via ultralytics, or SegFormer).\n"
            "  For now use drivable_area.type: heuristic or none."
        )

    def detect(self, frame: np.ndarray) -> DrivableArea:  # pragma: no cover
        raise NotImplementedError

    @property
    def method(self) -> str:  # pragma: no cover
        return "segmentation"


def create_drivable_area_detector(config: "Config") -> DrivableAreaDetector:
    """Build the configured drivable-area detector."""
    kind = config.drivable_area.type.lower()
    if kind == "heuristic":
        return HeuristicDrivableAreaDetector(config)
    if kind == "segmentation":
        return SegmentationDrivableAreaDetector(config)
    return NullDrivableAreaDetector()
