"""Multi-object tracking.

Implements a SORT-style tracker: a constant-velocity Kalman filter per track,
IoU-based association solved optimally with the Hungarian algorithm, and an
age/hit lifecycle that suppresses one-frame flickers.

The point of tracking here is not just pretty IDs. Persistent identity is what
makes velocity, trajectory prediction and time-to-collision possible at all -
none of which can be computed from a single frame.

``BaseTracker`` keeps ByteTrack (via the optional ``supervision`` package)
behind the same interface, so the choice is a config switch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core.types import (
    BBox,
    Detection,
    MovementStatus,
    Point,
    TrackedObject,
)

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["BaseTracker", "KalmanIoUTracker", "create_tracker"]

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only without SciPy
    _linear_sum_assignment = None
    logger.info(
        "SciPy is not installed; the tracker will use greedy IoU association. "
        "Install it with: pip install scipy"
    )


# ---------------------------------------------------------------------------
# Kalman filter for a single track
# ---------------------------------------------------------------------------


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter over box state ``[cx, cy, s, r]``.

    ``s`` is box area and ``r`` is aspect ratio; velocities are tracked for
    ``cx``, ``cy`` and ``s``. Aspect ratio is treated as constant, which holds
    well enough for road users that do not rotate in the image plane.
    """

    __slots__ = ("F", "H", "P", "Q", "R", "_dim", "x")

    def __init__(self, bbox: BBox) -> None:
        self._dim = 7

        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01
        self.Q[6, 6] *= 0.0001

        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4, 0] = self._to_z(bbox)

    @staticmethod
    def _to_z(bbox: BBox) -> np.ndarray:
        w = max(1e-3, bbox.width)
        h = max(1e-3, bbox.height)
        center = bbox.center
        return np.array([center.x, center.y, w * h, w / h], dtype=np.float64)

    @staticmethod
    def _to_bbox(state: np.ndarray) -> BBox:
        cx, cy, s, r = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        s = max(1.0, s)
        r = max(1e-3, r)
        w = float(np.sqrt(s * r))
        h = s / w if w > 1e-6 else 1.0
        return BBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self) -> BBox:
        """Advance the state one step and return the predicted box."""
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox(self.x[:4, 0])

    def update(self, bbox: BBox) -> None:
        """Correct the state with a measured box."""
        z = self._to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.debug("Singular Kalman innovation covariance; skipping update.")
            return
        self.x = self.x + K @ y
        identity = np.eye(self._dim, dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P

    @property
    def bbox(self) -> BBox:
        return self._to_bbox(self.x[:4, 0])

    @property
    def velocity_px(self) -> tuple[float, float]:
        """Per-frame centre velocity in pixels."""
        return (float(self.x[4, 0]), float(self.x[5, 0]))


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------


class _Track:
    """Internal bookkeeping for one tracked object."""

    __slots__ = (
        "age",
        "attributes",
        "category",
        "class_name",
        "class_votes",
        "confidence",
        "distance_history",
        "first_seen",
        "history",
        "hits",
        "kf",
        "last_timestamp",
        "pothole",
        "source",
        "time_since_update",
        "track_id",
    )

    def __init__(
        self, track_id: int, detection: Detection, max_history: int
    ) -> None:
        self.track_id = track_id
        self.kf = _KalmanBoxTracker(detection.bbox)
        self.class_name = detection.class_name
        self.category = detection.category
        self.confidence = detection.confidence
        self.source = detection.source
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history: list[Point] = [detection.bbox.center]
        self.distance_history: list[float] = []
        self.first_seen = detection.timestamp
        self.last_timestamp = detection.timestamp
        self.class_votes: dict[str, float] = {detection.class_name: detection.confidence}
        self.pothole = detection.pothole
        self.attributes = dict(detection.attributes)

    def update(self, detection: Detection, max_history: int) -> None:
        self.kf.update(detection.bbox)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection.confidence
        self.source = detection.source
        self.last_timestamp = detection.timestamp
        self.pothole = detection.pothole or self.pothole
        if detection.attributes:
            self.attributes.update(detection.attributes)

        self.class_votes[detection.class_name] = (
            self.class_votes.get(detection.class_name, 0.0) + detection.confidence
        )
        best = max(self.class_votes.items(), key=lambda kv: kv[1])[0]
        if best != self.class_name:
            from ..core.classes import category_for
            self.class_name = best
            self.category = category_for(best)

        self.history.append(detection.bbox.center)
        if len(self.history) > max_history:
            del self.history[0]

    def mark_missed(self, max_history: int) -> None:
        """Record a frame with no matching detection, coasting on prediction."""
        self.time_since_update += 1
        self.history.append(self.kf.bbox.center)
        if len(self.history) > max_history:
            del self.history[0]


# ---------------------------------------------------------------------------
# Tracker interface
# ---------------------------------------------------------------------------


class BaseTracker(ABC):
    """Contract every tracker implementation must satisfy."""

    @abstractmethod
    def update(
        self, detections: list[Detection], timestamp: float, frame_index: int
    ) -> list[TrackedObject]:
        """Associate detections with existing tracks and return live tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracks and restart ID numbering."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class KalmanIoUTracker(BaseTracker):
    """SORT-style tracker: Kalman prediction + Hungarian IoU association."""

    def __init__(self, config: "Config") -> None:
        cfg = config.tracking
        self.max_age = cfg.max_age
        self.min_hits = cfg.min_hits
        self.iou_threshold = cfg.iou_threshold
        self.max_history = cfg.max_history
        self.static_speed_px = config.prediction.static_speed_px

        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_timestamp: Optional[float] = None

    @property
    def name(self) -> str:
        return "kalman_iou"

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
        logger.info("Tracker reset; track IDs restart at 1.")

    # ---- main step --------------------------------------------------------

    def update(
        self, detections: list[Detection], timestamp: float, frame_index: int
    ) -> list[TrackedObject]:
        dt = 0.0
        if self._last_timestamp is not None:
            dt = max(0.0, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        predicted: list[BBox] = []
        for track in self._tracks:
            track.age += 1
            predicted.append(track.kf.predict())

        matches, unmatched_dets, unmatched_tracks = self._associate(
            detections, predicted
        )

        for det_idx, track_idx in matches:
            self._tracks[track_idx].update(detections[det_idx], self.max_history)

        for track_idx in unmatched_tracks:
            self._tracks[track_idx].mark_missed(self.max_history)

        for det_idx in unmatched_dets:
            self._tracks.append(
                _Track(self._next_id, detections[det_idx], self.max_history)
            )
            self._next_id += 1

        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        if before != len(self._tracks):
            logger.debug("Retired %d stale track(s).", before - len(self._tracks))

        return self._emit(timestamp, frame_index, dt)

    def _associate(
        self, detections: list[Detection], predicted: list[BBox]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match detections to tracks by IoU, gated by class compatibility."""
        if not detections:
            return [], [], list(range(len(predicted)))
        if not predicted:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(predicted)), dtype=np.float64)
        for d, detection in enumerate(detections):
            for t, track_box in enumerate(predicted):
                if self._tracks[t].category is not detection.category:
                    iou_matrix[d, t] = 0.0
                    continue
                iou_matrix[d, t] = detection.bbox.iou(track_box)

        matches: list[tuple[int, int]] = []
        if _linear_sum_assignment is not None:
            row_idx, col_idx = _linear_sum_assignment(-iou_matrix)
            pairs = zip(row_idx, col_idx, strict=False)
        else:
            pairs = self._greedy_pairs(iou_matrix)

        for d, t in pairs:
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append((int(d), int(t)))

        matched_d = {d for d, _ in matches}
        matched_t = {t for _, t in matches}
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_d]
        unmatched_tracks = [t for t in range(len(predicted)) if t not in matched_t]
        return matches, unmatched_dets, unmatched_tracks

    @staticmethod
    def _greedy_pairs(iou_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Greedy highest-IoU-first matching, used when SciPy is absent."""
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        order = np.dstack(np.unravel_index(
            np.argsort(-iou_matrix, axis=None), iou_matrix.shape
        ))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        return pairs

    # ---- output -----------------------------------------------------------

    def _emit(
        self, timestamp: float, frame_index: int, dt: float
    ) -> list[TrackedObject]:
        """Convert internal tracks into public :class:`TrackedObject` records."""
        objects: list[TrackedObject] = []
        fps = (1.0 / dt) if dt > 1e-6 else 0.0

        for track in self._tracks:
            if track.hits < self.min_hits and track.age > self.min_hits:
                continue
            if track.time_since_update > 0 and track.hits < self.min_hits:
                continue

            vx_frame, vy_frame = track.kf.velocity_px
            vx, vy = vx_frame * fps, vy_frame * fps
            speed_px = float(np.hypot(vx, vy))

            obj = TrackedObject(
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=track.confidence,
                bbox=track.kf.bbox,
                category=track.category,
                source=track.source,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
                first_seen=track.first_seen,
                timestamp=timestamp,
                frame_index=frame_index,
                velocity_px=(vx, vy),
                speed_px=speed_px,
                movement=self._movement_status(speed_px, vy),
                heading_deg=(
                    float(np.degrees(np.arctan2(vy, vx)))
                    if speed_px > self.static_speed_px else None
                ),
                history=list(track.history),
                distance_history=list(track.distance_history),
                pothole=track.pothole,
                attributes=dict(track.attributes),
            )
            objects.append(obj)

        return objects

    def _movement_status(self, speed_px: float, vy: float) -> MovementStatus:
        """Coarse motion label from image-space velocity.

        Refined later by :mod:`src.geometry.road_position`, which has access to
        distance history and can distinguish approaching from receding properly.
        """
        if speed_px < self.static_speed_px:
            return MovementStatus.STATIONARY
        return MovementStatus.APPROACHING if vy > 0 else MovementStatus.MOVING

    def store_distance(self, track_id: int, distance: float) -> None:
        """Record a distance sample against a track, for closing-speed maths."""
        for track in self._tracks:
            if track.track_id == track_id:
                track.distance_history.append(distance)
                if len(track.distance_history) > self.max_history:
                    del track.distance_history[0]
                return

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)


class _PassThroughTracker(BaseTracker):
    """Assigns a fresh ID per detection. Used when tracking is disabled.

    Velocity, TTC and trajectory prediction are all unavailable in this mode,
    which is exactly why it is not the default.
    """

    def __init__(self, config: "Config") -> None:
        self._next_id = 1

    @property
    def name(self) -> str:
        return "passthrough"

    def reset(self) -> None:
        self._next_id = 1

    def update(
        self, detections: list[Detection], timestamp: float, frame_index: int
    ) -> list[TrackedObject]:
        objects = []
        for detection in detections:
            objects.append(
                TrackedObject(
                    track_id=self._next_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    category=detection.category,
                    source=detection.source,
                    age=1,
                    hits=1,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    history=[detection.bbox.center],
                    movement=MovementStatus.UNKNOWN,
                    pothole=detection.pothole,
                )
            )
            self._next_id += 1
        return objects


def create_tracker(config: "Config") -> BaseTracker:
    """Build the tracker named in ``config.tracking``."""
    if not config.tracking.enabled:
        logger.warning(
            "Tracking is disabled: velocity, TTC and trajectory prediction "
            "will be unavailable."
        )
        return _PassThroughTracker(config)

    tracker_type = config.tracking.type.lower()
    if tracker_type == "kalman_iou":
        return KalmanIoUTracker(config)

    if tracker_type == "bytetrack":
        try:
            from .bytetrack_tracker import ByteTrackTracker
            return ByteTrackTracker(config)
        except ImportError as exc:
            logger.warning(
                "ByteTrack requested but unavailable (%s). "
                "Install it with: pip install supervision. "
                "Falling back to the built-in kalman_iou tracker.",
                exc,
            )
            return KalmanIoUTracker(config)

    raise ValueError(
        f"Unknown tracker type '{config.tracking.type}'. "
        "Expected: kalman_iou, bytetrack."
    )
