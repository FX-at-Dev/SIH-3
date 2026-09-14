"""Optional ByteTrack backend, kept behind the :class:`BaseTracker` interface.

ByteTrack generally holds identities better than plain IoU-SORT through
occlusion, which matters in dense Indian traffic where a pedestrian can be
briefly hidden by an auto-rickshaw. It is optional because it pulls in the
``supervision`` package.

Enable with::

    pip install supervision

and in the config::

    tracking:
      type: bytetrack
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..core.classes import category_for
from ..core.types import (
    BBox,
    Detection,
    DetectionSource,
    MovementStatus,
    Point,
    TrackedObject,
)
from .tracker import BaseTracker

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["ByteTrackTracker"]


class ByteTrackTracker(BaseTracker):
    """Adapter around ``supervision.ByteTrack``."""

    def __init__(self, config: "Config") -> None:
        import supervision as sv

        self._sv = sv
        self.config = config
        self.max_history = config.tracking.max_history
        self._tracker = sv.ByteTrack(
            track_activation_threshold=max(0.1, config.detector.confidence),
            lost_track_buffer=config.tracking.max_age,
            minimum_matching_threshold=max(0.1, 1.0 - config.tracking.iou_threshold),
            minimum_consecutive_frames=config.tracking.min_hits,
        )
        self._history: dict[int, list[Point]] = {}
        self._last_centers: dict[int, tuple[Point, float]] = {}
        logger.info("ByteTrack tracker initialised (supervision backend).")

    @property
    def name(self) -> str:
        return "bytetrack"

    def reset(self) -> None:
        self._tracker.reset()
        self._history.clear()
        self._last_centers.clear()

    def update(
        self, detections: list[Detection], timestamp: float, frame_index: int
    ) -> list[TrackedObject]:
        sv = self._sv

        if detections:
            xyxy = np.array([d.bbox.as_tuple() for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_ids = np.arange(len(detections), dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_ids)
        tracked = self._tracker.update_with_detections(sv_dets)

        objects: list[TrackedObject] = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            src_idx = int(tracked.class_id[i])
            source_det = (
                detections[src_idx] if 0 <= src_idx < len(detections) else None
            )
            bbox = BBox(*[float(v) for v in tracked.xyxy[i]])
            center = bbox.center

            history = self._history.setdefault(track_id, [])
            history.append(center)
            if len(history) > self.max_history:
                del history[0]

            vx = vy = 0.0
            previous = self._last_centers.get(track_id)
            if previous is not None:
                prev_center, prev_time = previous
                dt = timestamp - prev_time
                if dt > 1e-6:
                    vx = (center.x - prev_center.x) / dt
                    vy = (center.y - prev_center.y) / dt
            self._last_centers[track_id] = (center, timestamp)

            class_name = source_det.class_name if source_det else "unknown"
            speed_px = float(np.hypot(vx, vy))
            objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=float(tracked.confidence[i]),
                    bbox=bbox,
                    category=category_for(class_name),
                    source=(
                        source_det.source if source_det else DetectionSource.MODEL
                    ),
                    age=len(history),
                    hits=len(history),
                    time_since_update=0,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    velocity_px=(vx, vy),
                    speed_px=speed_px,
                    movement=(
                        MovementStatus.STATIONARY
                        if speed_px < self.config.prediction.static_speed_px
                        else MovementStatus.MOVING
                    ),
                    history=list(history),
                    pothole=source_det.pothole if source_det else None,
                )
            )
        return objects
