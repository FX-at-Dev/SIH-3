"""Perception manager: orchestrates the perception half of the pipeline.

Runs, in order::

    frame -> detector -> pothole detector -> merge
          -> tracker -> distance -> road position -> prediction
          -> tracked objects

and hands the result to the risk and decision stages. It owns the per-stage
timing that feeds the performance metrics, and it isolates each stage so that
one failing component degrades the output rather than crashing the run.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core.types import Detection, FrameMetrics, TrackedObject
from ..geometry.distance import BaseDistanceEstimator, create_distance_estimator
from ..geometry.ego_path import EgoPath
from ..geometry.road_position import RoadPositionAnalyzer
from ..prediction.constant_velocity import ConstantVelocityPredictor
from .detector import BaseDetector, create_detector
from .drivable_area import DrivableArea, create_drivable_area_detector
from .pothole_detector import assess_pothole, create_pothole_detector
from .tracker import KalmanIoUTracker, create_tracker

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["PerceptionManager", "PerceptionResult"]


class PerceptionResult:
    """Everything perception produced for one frame."""

    __slots__ = ("detections", "drivable", "dt", "metrics", "objects")

    def __init__(
        self,
        objects: list[TrackedObject],
        detections: list[Detection],
        drivable: DrivableArea,
        metrics: FrameMetrics,
        dt: float,
    ) -> None:
        self.objects = objects
        self.detections = detections
        self.drivable = drivable
        self.metrics = metrics
        self.dt = dt


class PerceptionManager:
    """Owns and sequences every perception component."""

    def __init__(self, config: "Config", ego_path: EgoPath) -> None:
        self.config = config
        self.ego_path = ego_path

        self.detector: BaseDetector = create_detector(config)
        self.pothole_detector = create_pothole_detector(config)
        self.drivable_detector = create_drivable_area_detector(config)
        self.tracker = create_tracker(config)
        self.distance_estimator: BaseDistanceEstimator = create_distance_estimator(config)
        self.road_position = RoadPositionAnalyzer(config, ego_path)
        self.predictor = ConstantVelocityPredictor(config, ego_path)

        self._last_timestamp: Optional[float] = None
        self._frame_failures = 0

    # ---- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Load every model. Errors here are fatal and clearly explained."""
        self.detector.load()
        self.pothole_detector.load()

    def reset(self) -> None:
        """Clear all temporal state (tracks, timing)."""
        self.tracker.reset()
        self._last_timestamp = None
        logger.info("Perception state reset.")

    def release(self) -> None:
        self.detector.release()

    # ---- per-frame processing --------------------------------------------

    def process(
        self, frame: np.ndarray, frame_index: int, timestamp: float
    ) -> PerceptionResult:
        """Run the full perception stack on one frame."""
        metrics = FrameMetrics(frame_index=frame_index, timestamp=timestamp)
        height, width = frame.shape[:2]
        self.ego_path.configure(width, height)

        dt = 0.0
        if self._last_timestamp is not None:
            dt = max(0.0, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        # ---- 1. Drivable area (advisory; failure is non-fatal) -----------
        drivable = self.drivable_detector.detect(frame)
        self.pothole_detector.set_road_mask(drivable.mask)

        # ---- 2. Detection ------------------------------------------------
        t0 = time.perf_counter()
        detections = self._safe_detect(frame, frame_index)
        pothole_detections = self._safe_detect_potholes(frame, frame_index)
        metrics.inference_ms = (time.perf_counter() - t0) * 1000.0

        for detection in pothole_detections:
            detection.pothole = assess_pothole(
                detection, self.config, self.ego_path,
                self.distance_estimator, width, height,
            )

        all_detections = detections + pothole_detections

        # ---- 3. Tracking -------------------------------------------------
        t0 = time.perf_counter()
        try:
            objects = self.tracker.update(all_detections, timestamp, frame_index)
        except Exception as exc:
            logger.error(
                "Tracker failed on frame %d (%s); resetting tracks and continuing.",
                frame_index, exc,
            )
            self.tracker.reset()
            objects = []
        metrics.tracking_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 4. Geometry: distance, then position -------------------------
        t0 = time.perf_counter()
        for obj in objects:
            self.distance_estimator.apply_to_track(obj, width, height)
            if obj.estimated_distance is not None and isinstance(
                self.tracker, KalmanIoUTracker
            ):
                self.tracker.store_distance(obj.track_id, obj.estimated_distance)
                obj.distance_history.append(obj.estimated_distance)

        self.road_position.analyze(objects, width, height, dt)
        metrics.geometry_ms = (time.perf_counter() - t0) * 1000.0

        # ---- 5. Trajectory prediction -------------------------------------
        t0 = time.perf_counter()
        if self.config.prediction.enabled:
            try:
                self.predictor.predict_all(objects, width, height)
            except Exception as exc:
                logger.error("Prediction failed on frame %d: %s", frame_index, exc)
        metrics.prediction_ms = (time.perf_counter() - t0) * 1000.0

        metrics.object_count = len(objects)
        return PerceptionResult(objects, all_detections, drivable, metrics, dt)

    # ---- fault isolation --------------------------------------------------

    def _safe_detect(
        self, frame: np.ndarray, frame_index: int
    ) -> list[Detection]:
        """Detection that degrades to an empty list rather than crashing."""
        try:
            return self.detector.detect(frame, frame_index)
        except Exception as exc:
            self._frame_failures += 1
            if self._frame_failures <= 3 or self._frame_failures % 100 == 0:
                logger.error(
                    "Detection failed on frame %d (%d failures total): %s",
                    frame_index, self._frame_failures, exc,
                )
            return []

    def _safe_detect_potholes(
        self, frame: np.ndarray, frame_index: int
    ) -> list[Detection]:
        try:
            return self.pothole_detector.detect(frame, frame_index)
        except Exception as exc:
            logger.error(
                "Pothole detection failed on frame %d: %s", frame_index, exc
            )
            return []

    # ---- reporting --------------------------------------------------------

    @property
    def status_lines(self) -> list[str]:
        """Short status summary shown at startup and in the dashboard."""
        info = self.detector.info
        return [
            f"detector: {info.name} ({info.notes}) on {info.device}",
            f"model: {info.model_path}",
            f"tracker: {self.tracker.name}",
            f"distance: {self.config.distance.method} (monocular, ESTIMATED)",
            self.pothole_detector.status_text,
            f"drivable area: {self.drivable_detector.method}",
        ]
