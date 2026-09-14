"""Event logging, metrics recording and session output.

Produces a self-contained session directory::

    results/session_2026_09_13_2142/
        events.json            structured event stream
        detections.csv         per-object, per-frame records
        recommendations.csv    per-frame decisions
        metrics.csv            per-frame timings
        summary.json           session totals
        screenshots/
        annotated_video.mp4

Events are emitted on *changes* rather than every frame, so the log stays
readable: an object entering the path is an event, an object remaining in the
path is not.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..core.types import (
    ActionType,
    DrivingRecommendation,
    FrameMetrics,
    MovementStatus,
    PathRelation,
    RiskLevel,
    SemanticCategory,
    WorldState,
)

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["Event", "EventLogger", "EventType", "SessionRecorder"]


class EventType(str, Enum):
    """Events the system emits."""

    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    OBJECT_LOST = "OBJECT_LOST"
    OBJECT_ENTERED_PATH = "OBJECT_ENTERED_PATH"
    OBJECT_LEFT_PATH = "OBJECT_LEFT_PATH"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL_RISK = "CRITICAL_RISK"
    POTHOLE_DETECTED = "POTHOLE_DETECTED"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
    ANIMAL_CROSSING = "ANIMAL_CROSSING"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    RECOMMENDATION_CHANGED = "RECOMMENDATION_CHANGED"
    PERCEPTION_DEGRADED = "PERCEPTION_DEGRADED"


@dataclass(slots=True)
class Event:
    """One logged event."""

    type: EventType
    timestamp: float
    frame_index: int
    message: str
    track_id: Optional[int] = None
    risk: Optional[RiskLevel] = None
    ttc: Optional[float] = None
    action: Optional[ActionType] = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def time_text(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": round(self.timestamp, 3),
            "time": self.time_text,
            "frame": self.frame_index,
            "message": self.message,
            "track_id": self.track_id,
            "risk": self.risk.value if self.risk else None,
            "ttc": round(self.ttc, 2) if self.ttc is not None else None,
            "action": self.action.value if self.action else None,
            **({"data": self.data} if self.data else {}),
        }

    def __str__(self) -> str:
        parts = [f"[{self.time_text}]", self.type.value]
        if self.ttc is not None:
            parts.append(f"TTC={self.ttc:.1f}")
        if self.risk is not None:
            parts.append(f"Risk={self.risk.value}")
        if self.action is not None:
            parts.append(f"Action={self.action.value}")
        return " ".join(parts) + f" | {self.message}"


class EventLogger:
    """Detects state changes between frames and emits events for them."""

    def __init__(self, config: "Config", echo: bool = True) -> None:
        self.config = config
        self.echo = echo
        self.events: list[Event] = []

        self._seen_tracks: set[int] = set()
        self._in_path: set[int] = set()
        self._high_risk: set[int] = set()
        self._critical_risk: set[int] = set()
        self._reported_potholes: set[int] = set()
        self._crossing: set[int] = set()
        self._last_action: Optional[ActionType] = None
        self._last_direction: Optional[str] = None
        self._degraded = False

        self.counters: dict[str, int] = {
            "high_risk_events": 0,
            "critical_events": 0,
            "recommendation_changes": 0,
            "potholes_detected": 0,
            "pedestrian_crossings": 0,
            "animal_crossings": 0,
            "replans": 0,
        }

    # ---- emission ---------------------------------------------------------

    def emit(self, event: Event) -> None:
        self.events.append(event)
        if self.echo:
            logger.info("%s", event)

    def log_session_start(self, source_name: str, details: dict[str, Any]) -> None:
        self.emit(Event(
            type=EventType.SESSION_START,
            timestamp=time.time(),
            frame_index=0,
            message=f"Session started on {source_name}",
            data=details,
        ))

    def log_session_end(self, frames: int) -> None:
        self.emit(Event(
            type=EventType.SESSION_END,
            timestamp=time.time(),
            frame_index=frames,
            message=f"Session ended after {frames} frames",
            data=dict(self.counters),
        ))

    # ---- per-frame analysis ----------------------------------------------

    def update(
        self, world: WorldState, recommendation: DrivingRecommendation
    ) -> list[Event]:
        """Compare this frame against the last and emit any state changes."""
        new_events: list[Event] = []
        timestamp = world.timestamp
        frame_index = world.frame_index

        def add(event: Event) -> None:
            new_events.append(event)
            self.emit(event)

        live_ids = {o.track_id for o in world.objects}

        for obj in world.objects:
            name = obj.class_name.replace("_", " ")

            # --- new object ---------------------------------------------
            if obj.track_id not in self._seen_tracks:
                self._seen_tracks.add(obj.track_id)
                add(Event(
                    EventType.OBJECT_DETECTED, timestamp, frame_index,
                    f"{name} detected (ID {obj.track_id}, "
                    f"confidence {obj.confidence:.2f})",
                    track_id=obj.track_id, risk=obj.risk_level,
                ))

                if obj.is_pothole and obj.track_id not in self._reported_potholes:
                    self._reported_potholes.add(obj.track_id)
                    self.counters["potholes_detected"] += 1
                    severity = obj.pothole.severity.value if obj.pothole else "UNKNOWN"
                    method = obj.source.value
                    add(Event(
                        EventType.POTHOLE_DETECTED, timestamp, frame_index,
                        f"Pothole (ID {obj.track_id}) severity {severity}, "
                        f"{obj.distance_text()}, detection method: {method}",
                        track_id=obj.track_id, risk=obj.risk_level,
                        data=obj.pothole.to_dict() if obj.pothole else {},
                    ))

            # --- path entry / exit ---------------------------------------
            in_path = obj.path_relation in (
                PathRelation.DIRECTLY_AHEAD,
                PathRelation.IN_PATH,
                PathRelation.CROSSING_PATH,
            )
            if in_path and obj.track_id not in self._in_path:
                self._in_path.add(obj.track_id)
                add(Event(
                    EventType.OBJECT_ENTERED_PATH, timestamp, frame_index,
                    f"{name} (ID {obj.track_id}) entered the ego path at "
                    f"{obj.distance_text()}",
                    track_id=obj.track_id, risk=obj.risk_level, ttc=obj.ttc,
                ))
            elif not in_path and obj.track_id in self._in_path:
                self._in_path.discard(obj.track_id)
                add(Event(
                    EventType.OBJECT_LEFT_PATH, timestamp, frame_index,
                    f"{name} (ID {obj.track_id}) left the ego path",
                    track_id=obj.track_id,
                ))

            # --- crossing ------------------------------------------------
            if (obj.movement is MovementStatus.CROSSING
                    and obj.track_id not in self._crossing):
                self._crossing.add(obj.track_id)
                if obj.category is SemanticCategory.VULNERABLE:
                    self.counters["pedestrian_crossings"] += 1
                    add(Event(
                        EventType.PEDESTRIAN_CROSSING, timestamp, frame_index,
                        f"{name} (ID {obj.track_id}) crossing the vehicle path, "
                        f"{obj.distance_text()}",
                        track_id=obj.track_id, risk=obj.risk_level, ttc=obj.ttc,
                    ))
                elif obj.category is SemanticCategory.ANIMAL:
                    self.counters["animal_crossings"] += 1
                    add(Event(
                        EventType.ANIMAL_CROSSING, timestamp, frame_index,
                        f"{name} (ID {obj.track_id}) crossing the vehicle path, "
                        f"{obj.distance_text()}",
                        track_id=obj.track_id, risk=obj.risk_level, ttc=obj.ttc,
                    ))
            elif obj.movement is not MovementStatus.CROSSING:
                self._crossing.discard(obj.track_id)

            # --- risk escalation ------------------------------------------
            if obj.risk_level is RiskLevel.CRITICAL:
                if obj.track_id not in self._critical_risk:
                    self._critical_risk.add(obj.track_id)
                    self.counters["critical_events"] += 1
                    reason = obj.risk_reasons[0] if obj.risk_reasons else "critical risk"
                    add(Event(
                        EventType.CRITICAL_RISK, timestamp, frame_index,
                        f"CRITICAL risk from {name} (ID {obj.track_id}): {reason}",
                        track_id=obj.track_id, risk=RiskLevel.CRITICAL, ttc=obj.ttc,
                    ))
            else:
                self._critical_risk.discard(obj.track_id)

            if obj.risk_level is RiskLevel.HIGH:
                if obj.track_id not in self._high_risk:
                    self._high_risk.add(obj.track_id)
                    self.counters["high_risk_events"] += 1
                    reason = obj.risk_reasons[0] if obj.risk_reasons else "elevated risk"
                    add(Event(
                        EventType.HIGH_RISK, timestamp, frame_index,
                        f"HIGH risk from {name} (ID {obj.track_id}): {reason}",
                        track_id=obj.track_id, risk=RiskLevel.HIGH, ttc=obj.ttc,
                    ))
            else:
                self._high_risk.discard(obj.track_id)

        # --- lost tracks --------------------------------------------------
        for track_id in list(self._seen_tracks - live_ids):
            if track_id in self._in_path:
                self._in_path.discard(track_id)
            self._seen_tracks.discard(track_id)
            self._high_risk.discard(track_id)
            self._critical_risk.discard(track_id)

        # --- recommendation change ---------------------------------------
        direction = (
            recommendation.direction.value if recommendation.direction else None
        )
        if (self._last_action is not None
                and (recommendation.action is not self._last_action
                     or direction != self._last_direction)):
            self.counters["recommendation_changes"] += 1
            add(Event(
                EventType.RECOMMENDATION_CHANGED, timestamp, frame_index,
                f"{self._last_action.value} -> {recommendation.label}: "
                f"{recommendation.reason}",
                risk=recommendation.risk, ttc=recommendation.ttc,
                action=recommendation.action,
            ))
            if recommendation.action.caution_rank > self._last_action.caution_rank:
                self.counters["replans"] += 1
                add(Event(
                    EventType.REPLAN_REQUIRED, timestamp, frame_index,
                    f"Escalation to {recommendation.label} requires a new plan",
                    risk=recommendation.risk, action=recommendation.action,
                ))

        self._last_action = recommendation.action
        self._last_direction = direction

        # --- perception degradation ---------------------------------------
        degraded = (
            world.scene_risk.perception_confidence < self.config.risk.low_confidence
        )
        if degraded and not self._degraded:
            self._degraded = True
            add(Event(
                EventType.PERCEPTION_DEGRADED, timestamp, frame_index,
                "Perception confidence fell below the configured threshold "
                f"({world.scene_risk.perception_confidence:.2f}); the system is "
                "behaving more conservatively",
                risk=world.scene_risk.level,
            ))
        elif not degraded:
            self._degraded = False

        return new_events


class SessionRecorder:
    """Writes annotated media, CSV metrics and the JSON event log to disk."""

    def __init__(self, config: "Config", enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled
        self.session_dir: Optional[Path] = None
        self._video_writer: Optional[Any] = None
        self._detection_rows: list[dict[str, Any]] = []
        self._recommendation_rows: list[dict[str, Any]] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._screenshot_count = 0

    # ---- lifecycle --------------------------------------------------------

    def start(self, source_name: str) -> Optional[Path]:
        """Create the session directory."""
        if not self.enabled:
            return None
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        root = Path(self.config.output.results_dir)
        self.session_dir = root / f"session_{stamp}"
        (self.session_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        logger.info("Recording session output to %s", self.session_dir)
        return self.session_dir

    def open_video(self, width: int, height: int, fps: float) -> None:
        """Open the annotated-video writer."""
        if not self.enabled or self.session_dir is None:
            return
        if not self.config.output.save_video:
            return

        import cv2

        path = self.session_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (width, height))
        if not writer.isOpened():
            logger.warning(
                "Could not open a video writer at %s; annotated video will not "
                "be saved. The rest of the session output is unaffected.", path,
            )
            return
        self._video_writer = writer
        logger.info("Writing annotated video to %s (%dx%d @ %.1f fps)",
                    path, width, height, fps)

    # ---- per-frame capture ------------------------------------------------

    def record_frame(
        self,
        world: WorldState,
        recommendation: DrivingRecommendation,
        metrics: FrameMetrics,
        rendered: Optional[np.ndarray] = None,
    ) -> None:
        if not self.enabled:
            return

        if self._video_writer is not None and rendered is not None:
            try:
                self._video_writer.write(rendered)
            except Exception as exc:
                logger.debug("Video write failed on frame %d: %s",
                             metrics.frame_index, exc)

        if self.config.output.save_detections:
            for obj in world.objects:
                row = obj.to_dict()
                row["risk_reasons"] = "; ".join(row.get("risk_reasons", []))
                row.pop("pothole", None)
                if obj.pothole is not None:
                    row["pothole_severity"] = obj.pothole.severity.value
                    row["pothole_width_m"] = round(obj.pothole.estimated_width_m, 2)
                row["bbox"] = ",".join(str(v) for v in row["bbox"])
                row["center"] = ",".join(str(v) for v in row["center"])
                self._detection_rows.append(row)

        if self.config.output.save_recommendations:
            row = recommendation.to_dict()
            row["factors"] = "; ".join(row.get("factors", []))
            row["scene_risk"] = world.scene_risk.level.value
            row["object_count"] = len(world.objects)
            row["traffic_density"] = world.scene_risk.traffic_density
            row["perception_confidence"] = round(
                world.scene_risk.perception_confidence, 3
            )
            self._recommendation_rows.append(row)

        self._metric_rows.append(metrics.to_dict())

    def save_screenshot(
        self, image: np.ndarray, label: str = "manual"
    ) -> Optional[Path]:
        """Save a still. Works even when session recording is disabled."""
        import cv2

        root = (
            self.session_dir / "screenshots"
            if self.session_dir is not None
            else Path(self.config.output.results_dir) / "screenshots"
        )
        root.mkdir(parents=True, exist_ok=True)

        self._screenshot_count += 1
        stamp = datetime.now().strftime("%H%M%S")
        path = root / f"{label}_{stamp}_{self._screenshot_count:03d}.jpg"
        try:
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise OSError("JPEG encoding failed")
            buffer.tofile(str(path))
            logger.info("Screenshot saved: %s", path)
            return path
        except (OSError, Exception) as exc:
            logger.error("Could not save screenshot to %s: %s", path, exc)
            return None

    # ---- finalisation -----------------------------------------------------

    def finish(
        self,
        event_logger: EventLogger,
        summary: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Flush everything to disk and return the session directory."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if not self.enabled or self.session_dir is None:
            return None

        if self.config.output.save_events:
            self._write_json(
                self.session_dir / "events.json",
                {
                    "counters": event_logger.counters,
                    "events": [e.to_dict() for e in event_logger.events],
                },
            )

        self._write_csv(self.session_dir / "detections.csv", self._detection_rows)
        self._write_csv(
            self.session_dir / "recommendations.csv", self._recommendation_rows
        )
        self._write_csv(self.session_dir / "metrics.csv", self._metric_rows)

        if summary is not None:
            self._write_json(self.session_dir / "summary.json", summary)

        logger.info("Session output written to %s", self.session_dir)
        return self.session_dir

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except OSError as exc:
            logger.error("Could not write %s: %s", path, exc)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            logger.error("Could not write %s: %s", path, exc)
