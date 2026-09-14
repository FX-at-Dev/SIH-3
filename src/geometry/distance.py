"""Monocular distance estimation.

IMPORTANT - what these numbers are and are not.

A single camera cannot measure distance. Everything produced here is an
*inference* from image geometry plus assumptions about how large objects
typically are. Every value carries ``distance_is_estimated=True`` and is
rendered with a ``~`` suffix so an approximation is never mistaken for a
sensor reading.

Two independent methods are implemented:

1. **Known-size (pinhole).** ``Z = f * H_real / h_pixels``. Accurate when the
   object's real height matches the assumed typical height and the whole object
   is visible. Degrades for partially-occluded or truncated boxes.
2. **Ground-plane.** Projects the box's bottom edge onto an assumed flat road
   surface at a known camera height. Independent of object size, so it is
   robust for unusual objects, but it depends on the object touching the road
   and on the road actually being flat.

The ``hybrid`` method fuses the two, weighting each by how trustworthy it is
for the box in question. Expect errors on the order of 15-30% with uncalibrated
intrinsics; this is a decision-support prototype, not a ranging instrument.

Future sensors (stereo, depth camera, LiDAR) implement
:class:`BaseDistanceEstimator` and set ``distance_is_estimated=False``.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.types import BBox, TrackedObject

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "BaseDistanceEstimator",
    "DistanceResult",
    "MonocularDistanceEstimator",
    "create_distance_estimator",
]


@dataclass(slots=True)
class DistanceResult:
    """A distance estimate plus provenance."""

    distance_m: float
    method: str
    is_estimated: bool = True
    confidence: float = 0.5

    def __str__(self) -> str:
        suffix = " (estimated)" if self.is_estimated else ""
        return f"{self.distance_m:.1f} m{suffix} [{self.method}]"


class BaseDistanceEstimator(ABC):
    """Interface for any distance source: monocular, stereo, depth, LiDAR."""

    @abstractmethod
    def estimate(
        self, bbox: BBox, class_name: str, frame_width: int, frame_height: int
    ) -> Optional[DistanceResult]:
        """Estimate range to an object, or ``None`` when not determinable."""

    @property
    @abstractmethod
    def provides_true_measurement(self) -> bool:
        """False for inference-based estimators, True for real range sensors."""


class MonocularDistanceEstimator(BaseDistanceEstimator):
    """Distance from a single uncalibrated camera."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.camera = config.camera
        self.settings = config.distance
        self.method = config.distance.method
        self._focal_px: Optional[float] = None
        self._frame_width = 0
        self._frame_height = 0

        if not self.camera.calibrated:
            logger.info(
                "Camera is not calibrated - distances are approximations derived "
                "from a nominal %.0f deg field of view and typical object sizes.",
                self.camera.horizontal_fov_deg,
            )

    # ---- intrinsics -------------------------------------------------------

    def _focal_length(self, frame_width: int, frame_height: int) -> float:
        """Focal length in pixels, derived from the nominal horizontal FOV."""
        if self._focal_px is None or frame_width != self._frame_width:
            fov_rad = math.radians(self.camera.horizontal_fov_deg)
            self._focal_px = (frame_width / 2.0) / math.tan(fov_rad / 2.0)
            self._frame_width = frame_width
            self._frame_height = frame_height
            logger.debug(
                "Derived focal length %.1f px for %dpx-wide frames.",
                self._focal_px, frame_width,
            )
        return self._focal_px

    # ---- individual methods ----------------------------------------------

    def _known_size_distance(
        self, bbox: BBox, class_name: str, frame_width: int, frame_height: int
    ) -> Optional[tuple[float, float]]:
        """Pinhole estimate from an assumed real-world height.

        Returns ``(distance_m, confidence)``.
        """
        if bbox.height < 2.0:
            return None

        dims = self.settings.dimension_for(class_name)
        focal = self._focal_length(frame_width, frame_height)

        dims_width = dims.width
        use_width = bbox.aspect_ratio > 1.6 and dims_width > 0

        if use_width:
            distance = (dims_width * focal) / max(1.0, bbox.width)
        else:
            distance = (dims.height * focal) / max(1.0, bbox.height)

        touches_edge = (
            bbox.x1 <= 2.0 or bbox.y1 <= 2.0
            or bbox.x2 >= frame_width - 2.0 or bbox.y2 >= frame_height - 2.0
        )
        confidence = 0.75
        if touches_edge:
            confidence = 0.35
        if class_name == "unknown":
            confidence = 0.20

        return distance, confidence

    def _ground_plane_distance(
        self, bbox: BBox, frame_width: int, frame_height: int
    ) -> Optional[tuple[float, float]]:
        """Estimate from where the object's base meets an assumed flat road.

        Returns ``(distance_m, confidence)``.
        """
        focal = self._focal_length(frame_width, frame_height)
        horizon_y = frame_height / 2.0 + focal * math.tan(
            math.radians(self.camera.pitch_deg)
        )

        base_y = bbox.y2
        dy = base_y - horizon_y
        if dy <= 2.0:
            return None

        distance = (self.camera.mount_height_m * focal) / dy

        normalised = dy / max(1.0, frame_height / 2.0)
        confidence = float(min(0.85, 0.25 + 0.7 * normalised))
        return distance, confidence

    # ---- public API -------------------------------------------------------

    def estimate(
        self, bbox: BBox, class_name: str, frame_width: int, frame_height: int
    ) -> Optional[DistanceResult]:
        """Estimate distance using the configured method."""
        if frame_width <= 0 or frame_height <= 0 or bbox.area <= 0:
            return None

        known = None
        ground = None

        if self.method in ("known_size", "hybrid"):
            known = self._known_size_distance(
                bbox, class_name, frame_width, frame_height
            )
        if self.method in ("ground_plane", "hybrid"):
            ground = self._ground_plane_distance(bbox, frame_width, frame_height)

        if self.method == "known_size":
            if known is None:
                return None
            distance, confidence, method = known[0], known[1], "known_size"
        elif self.method == "ground_plane":
            if ground is None:
                return None
            distance, confidence, method = ground[0], ground[1], "ground_plane"
        else:
            distance, confidence, method = self._fuse(known, ground)
            if distance is None:
                return None

        distance = float(
            min(self.settings.max_distance_m,
                max(self.settings.min_distance_m, distance))
        )
        return DistanceResult(
            distance_m=distance,
            method=method,
            is_estimated=True,
            confidence=confidence,
        )

    @staticmethod
    def _fuse(
        known: Optional[tuple[float, float]],
        ground: Optional[tuple[float, float]],
    ) -> tuple[Optional[float], float, str]:
        """Confidence-weighted fusion of the two monocular methods."""
        if known is None and ground is None:
            return None, 0.0, "none"
        if known is None:
            return ground[0], ground[1], "ground_plane"
        if ground is None:
            return known[0], known[1], "known_size"

        kd, kc = known
        gd, gc = ground

        ratio = max(kd, gd) / max(1e-6, min(kd, gd))
        if ratio > 2.5:
            if kc >= gc:
                return kd, kc * 0.6, "known_size(divergent)"
            return gd, gc * 0.6, "ground_plane(divergent)"

        total = kc + gc
        if total <= 1e-6:
            return (kd + gd) / 2.0, 0.2, "hybrid"
        fused = (kd * kc + gd * gc) / total
        confidence = min(0.9, (total / 2.0) * 1.1)
        return fused, confidence, "hybrid"

    def apply_to_track(
        self, obj: TrackedObject, frame_width: int, frame_height: int
    ) -> None:
        """Estimate and attach distance to a tracked object, with smoothing.

        Smoothing across a track's history suppresses the frame-to-frame jitter
        that raw box-size estimates suffer from, which matters because closing
        speed is differentiated from this series.
        """
        result = self.estimate(obj.bbox, obj.class_name, frame_width, frame_height)
        if result is None:
            obj.estimated_distance = None
            obj.distance_method = "none"
            return

        distance = result.distance_m
        alpha = self.settings.smoothing
        if obj.distance_history and alpha > 0.0:
            previous = obj.distance_history[-1]
            distance = alpha * previous + (1.0 - alpha) * distance

        obj.estimated_distance = distance
        obj.distance_is_estimated = not self.provides_true_measurement
        obj.distance_method = result.method
        obj.attributes["distance_confidence"] = result.confidence

    @property
    def provides_true_measurement(self) -> bool:
        return False


def create_distance_estimator(config: "Config") -> BaseDistanceEstimator:
    """Build the configured distance estimator.

    Stereo, depth-camera and LiDAR estimators slot in here when added; they
    implement the same interface and report
    ``provides_true_measurement = True``.
    """
    return MonocularDistanceEstimator(config)
