"""Core data structures shared by every stage of the perception pipeline.

These types are the contract between modules. A detector produces
:class:`Detection`, a tracker turns those into :class:`TrackedObject`, the risk
engine annotates them with :class:`ObjectRisk`, and the decision engine consumes
the assembled :class:`WorldState` to emit a :class:`DrivingRecommendation`.

Keeping the contract here means ``src/app.py`` never needs to know that the
detector happens to be YOLO, or that the tracker happens to use a Kalman filter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

__all__ = [
    "ActionType",
    "BBox",
    "Detection",
    "DetectionSource",
    "Direction",
    "DrivingRecommendation",
    "FrameMetrics",
    "LateralPosition",
    "MovementStatus",
    "ObjectRisk",
    "PathRelation",
    "Point",
    "PotholeInfo",
    "RiskLevel",
    "SceneRisk",
    "SemanticCategory",
    "TrackedObject",
    "WorldState",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Ordered risk bands used consistently across the whole system."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _RISK_ORDER[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank >= other.rank

    @classmethod
    def max(cls, *levels: "RiskLevel") -> "RiskLevel":
        """Return the most severe of the supplied levels (LOW when empty)."""
        best = cls.LOW
        for level in levels:
            if level.rank > best.rank:
                best = level
        return best

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Map a continuous 0-1 risk score onto a band."""
        if score >= 0.80:
            return cls.CRITICAL
        if score >= 0.55:
            return cls.HIGH
        if score >= 0.30:
            return cls.MEDIUM
        return cls.LOW


_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class ActionType(str, Enum):
    """The five driving recommendations this system may output."""

    CRUISE = "CRUISE"
    NUDGE = "NUDGE"
    YIELD = "YIELD"
    CREEP = "CREEP"
    STOP = "STOP"

    @property
    def caution_rank(self) -> int:
        """Higher means more conservative. Used for hysteresis decisions."""
        return _ACTION_ORDER[self]


_ACTION_ORDER: dict[ActionType, int] = {
    ActionType.CRUISE: 0,
    ActionType.NUDGE: 1,
    ActionType.YIELD: 2,
    ActionType.CREEP: 3,
    ActionType.STOP: 4,
}


class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class SemanticCategory(str, Enum):
    """Coarse behavioural grouping that drives risk policy.

    The risk engine reasons about categories rather than raw class names so
    that adding a new class only requires extending the class map.
    """

    VULNERABLE = "VULNERABLE"
    ANIMAL = "ANIMAL"
    VEHICLE = "VEHICLE"
    OBSTACLE = "OBSTACLE"
    STATIC = "STATIC"
    UNKNOWN = "UNKNOWN"


class MovementStatus(str, Enum):
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    CROSSING = "CROSSING"
    UNKNOWN = "UNKNOWN"


class PathRelation(str, Enum):
    """Where an object sits relative to the ego vehicle's likely corridor."""

    DIRECTLY_AHEAD = "DIRECTLY_AHEAD"
    IN_PATH = "IN_PATH"
    CROSSING_PATH = "CROSSING_PATH"
    BESIDE_VEHICLE = "BESIDE_VEHICLE"
    OUTSIDE_PATH = "OUTSIDE_PATH"
    BEHIND = "BEHIND"


class LateralPosition(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class DetectionSource(str, Enum):
    """How a detection was produced.

    This exists so the UI and the logs can never blur the line between a real
    model inference and a classical-CV heuristic guess.
    """

    MODEL = "model"
    HEURISTIC = "heuristic"
    MOCK = "mock"


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            object.__setattr__(self, "x1", self.x2)
            object.__setattr__(self, "x2", self.x1)
        if self.y2 < self.y1:
            object.__setattr__(self, "y1", self.y2)
            object.__setattr__(self, "y2", self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Ground-contact point, the anchor used for ground-plane geometry."""
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 1e-6 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    def iou(self, other: "BBox") -> float:
        """Intersection over union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 1e-6 else 0.0

    def clip(self, width: float, height: float) -> "BBox":
        return BBox(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def scaled(self, sx: float, sy: float) -> "BBox":
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)


# ---------------------------------------------------------------------------
# Detection and tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PotholeInfo:
    """Pothole-specific attributes attached to a pothole detection."""

    severity: RiskLevel = RiskLevel.LOW
    estimated_width_m: float = 0.0
    ego_path_overlap: float = 0.0
    size_label: str = "unknown"
    source: DetectionSource = DetectionSource.MODEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "estimated_width_m": round(self.estimated_width_m, 2),
            "ego_path_overlap": round(self.ego_path_overlap, 3),
            "size_label": self.size_label,
            "source": self.source.value,
        }


@dataclass(slots=True)
class Detection:
    """A single raw detection produced by a detector for one frame."""

    class_name: str
    confidence: float
    bbox: BBox
    category: SemanticCategory = SemanticCategory.UNKNOWN
    source: DetectionSource = DetectionSource.MODEL
    raw_class_name: str = ""
    class_id: int = -1
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0
    pothole: Optional[PotholeInfo] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_class_name:
            self.raw_class_name = self.class_name

    @property
    def center(self) -> Point:
        return self.bbox.center

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def is_pothole(self) -> bool:
        return self.class_name == "pothole"

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "raw_class_name": self.raw_class_name,
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox.as_tuple()],
            "source": self.source.value,
            "frame_index": self.frame_index,
        }


@dataclass(slots=True)
class TrackedObject:
    """A detection promoted to a persistent, temporally-consistent track.

    Distances are *estimated* from monocular geometry unless a depth sensor
    populates them; ``distance_is_estimated`` records which is the case so the
    UI never presents an approximation as a measurement.
    """

    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    category: SemanticCategory = SemanticCategory.UNKNOWN
    source: DetectionSource = DetectionSource.MODEL

    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    first_seen: float = field(default_factory=time.time)
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0

    estimated_distance: Optional[float] = None
    distance_is_estimated: bool = True
    distance_method: str = "none"
    lateral_offset_m: Optional[float] = None
    relative_position: LateralPosition = LateralPosition.CENTER
    path_relation: PathRelation = PathRelation.OUTSIDE_PATH
    ego_path_overlap: float = 0.0
    corridor_obstruction: float = 0.0

    velocity_px: tuple[float, float] = (0.0, 0.0)
    speed_px: float = 0.0
    velocity_mps: Optional[tuple[float, float]] = None
    closing_speed_mps: Optional[float] = None
    movement: MovementStatus = MovementStatus.UNKNOWN
    heading_deg: Optional[float] = None

    history: list[Point] = field(default_factory=list)
    distance_history: list[float] = field(default_factory=list)
    predicted_path: list[Point] = field(default_factory=list)
    predicted_overlap: float = 0.0
    time_to_path_entry: Optional[float] = None

    ttc: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    risk_reasons: list[str] = field(default_factory=list)

    pothole: Optional[PotholeInfo] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Point:
        return self.bbox.center

    @property
    def is_confirmed(self) -> bool:
        return self.time_since_update == 0

    @property
    def is_vulnerable(self) -> bool:
        return self.category in (SemanticCategory.VULNERABLE, SemanticCategory.ANIMAL)

    @property
    def is_pothole(self) -> bool:
        return self.class_name == "pothole"

    @property
    def velocity(self) -> float:
        """Best available scalar speed estimate in m/s, else image speed scaled."""
        if self.velocity_mps is not None:
            vx, vy = self.velocity_mps
            return math.hypot(vx, vy)
        return 0.0

    def distance_text(self) -> str:
        """Human-readable distance, always marked when it is an estimate."""
        if self.estimated_distance is None:
            return "dist n/a"
        suffix = "m~" if self.distance_is_estimated else "m"
        return f"{self.estimated_distance:.1f}{suffix}"

    def ttc_text(self) -> str:
        """TTC for display. Never emits inf or NaN."""
        if self.ttc is None or not math.isfinite(self.ttc):
            return "--"
        return f"{self.ttc:.1f}s"

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-serialisable snapshot used by the loggers."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox.as_tuple()],
            "center": [round(self.center.x, 1), round(self.center.y, 1)],
            "width": round(self.bbox.width, 1),
            "height": round(self.bbox.height, 1),
            "timestamp": round(self.timestamp, 3),
            "frame_index": self.frame_index,
            "estimated_distance": (
                round(self.estimated_distance, 2)
                if self.estimated_distance is not None else None
            ),
            "distance_is_estimated": self.distance_is_estimated,
            "distance_method": self.distance_method,
            "relative_position": self.relative_position.value,
            "path_relation": self.path_relation.value,
            "ego_path_overlap": round(self.ego_path_overlap, 3),
            "corridor_obstruction": round(self.corridor_obstruction, 3),
            "velocity": round(self.velocity, 2),
            "closing_speed_mps": (
                round(self.closing_speed_mps, 2)
                if self.closing_speed_mps is not None else None
            ),
            "movement": self.movement.value,
            "ttc": round(self.ttc, 2) if self.ttc is not None else None,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "risk_reasons": list(self.risk_reasons),
            "source": self.source.value,
            "pothole": self.pothole.to_dict() if self.pothole else None,
        }


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObjectRisk:
    """Risk assessment for one tracked object."""

    track_id: int
    level: RiskLevel
    score: float
    ttc: Optional[float]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "level": self.level.value,
            "score": round(self.score, 3),
            "ttc": round(self.ttc, 2) if self.ttc is not None else None,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class SceneRisk:
    """Aggregated risk for the whole scene, plus the drivers behind it."""

    level: RiskLevel = RiskLevel.LOW
    score: float = 0.0
    object_risks: list[ObjectRisk] = field(default_factory=list)
    critical_track_ids: list[int] = field(default_factory=list)
    min_ttc: Optional[float] = None
    nearest_in_path_distance: Optional[float] = None
    blocking_track_ids: list[int] = field(default_factory=list)
    traffic_density: str = "sparse"
    perception_confidence: float = 1.0
    left_clear: bool = True
    right_clear: bool = True
    left_score: float = 1.0
    right_score: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 3),
            "min_ttc": round(self.min_ttc, 2) if self.min_ttc is not None else None,
            "nearest_in_path_distance": (
                round(self.nearest_in_path_distance, 2)
                if self.nearest_in_path_distance is not None else None
            ),
            "critical_track_ids": list(self.critical_track_ids),
            "blocking_track_ids": list(self.blocking_track_ids),
            "traffic_density": self.traffic_density,
            "perception_confidence": round(self.perception_confidence, 3),
            "left_clear": self.left_clear,
            "right_clear": self.right_clear,
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DrivingRecommendation:
    """The single primary recommendation emitted for a frame.

    This is advisory output for a human or a downstream planner. It is never
    wired to actuation - see the safety notice in the project README.
    """

    action: ActionType = ActionType.CRUISE
    direction: Optional[Direction] = None
    target_speed_kmh: float = 0.0
    risk: RiskLevel = RiskLevel.LOW
    reason: str = ""
    confidence: float = 1.0
    factors: list[str] = field(default_factory=list)
    ttc: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0

    @property
    def label(self) -> str:
        """Display label, including direction for NUDGE."""
        if self.action is ActionType.NUDGE and self.direction is not None:
            return f"NUDGE {self.direction.value}"
        return self.action.value

    def same_as(self, other: Optional["DrivingRecommendation"]) -> bool:
        if other is None:
            return False
        return self.action is other.action and self.direction is other.direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "direction": self.direction.value if self.direction else None,
            "target_speed": round(self.target_speed_kmh, 1),
            "risk": self.risk.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "ttc": round(self.ttc, 2) if self.ttc is not None else None,
            "factors": list(self.factors),
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
        }


# ---------------------------------------------------------------------------
# World state and metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameMetrics:
    """Per-frame timing and counting metrics, in milliseconds."""

    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    geometry_ms: float = 0.0
    prediction_ms: float = 0.0
    risk_ms: float = 0.0
    decision_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    object_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "capture_ms": round(self.capture_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "tracking_ms": round(self.tracking_ms, 2),
            "geometry_ms": round(self.geometry_ms, 2),
            "prediction_ms": round(self.prediction_ms, 2),
            "risk_ms": round(self.risk_ms, 2),
            "decision_ms": round(self.decision_ms, 2),
            "render_ms": round(self.render_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "fps": round(self.fps, 2),
            "object_count": self.object_count,
        }


@dataclass(slots=True)
class WorldState:
    """Everything the decision engine is allowed to reason about for a frame."""

    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    frame_width: int = 0
    frame_height: int = 0
    objects: list[TrackedObject] = field(default_factory=list)
    scene_risk: SceneRisk = field(default_factory=SceneRisk)
    ego_path_polygon: list[Point] = field(default_factory=list)
    drivable_ratio: float = 1.0
    drivable_available: bool = False
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    dt: float = 0.0

    # ---- convenience queries used by the decision engine and the dashboard ---

    def by_category(self, category: SemanticCategory) -> list[TrackedObject]:
        return [o for o in self.objects if o.category is category]

    @property
    def pedestrians(self) -> list[TrackedObject]:
        return [o for o in self.objects if o.class_name == "person"]

    @property
    def vehicles(self) -> list[TrackedObject]:
        return self.by_category(SemanticCategory.VEHICLE)

    @property
    def animals(self) -> list[TrackedObject]:
        return self.by_category(SemanticCategory.ANIMAL)

    @property
    def potholes(self) -> list[TrackedObject]:
        return [o for o in self.objects if o.is_pothole]

    @property
    def in_path_objects(self) -> list[TrackedObject]:
        return [
            o for o in self.objects
            if o.path_relation in (
                PathRelation.DIRECTLY_AHEAD,
                PathRelation.IN_PATH,
                PathRelation.CROSSING_PATH,
            )
        ]

    def counts(self) -> dict[str, int]:
        return {
            "objects": len(self.objects),
            "pedestrians": len(self.pedestrians),
            "vehicles": len(self.vehicles),
            "animals": len(self.animals),
            "potholes": len(self.potholes),
        }
