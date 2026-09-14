"""Constant-velocity trajectory prediction.

Propagates each track's current image-space velocity forward over the horizon
and tests, at each step, whether the object's footprint would fall inside the
ego corridor. The first step that does gives the time-to-path-entry - the
quantity that separates "a pedestrian is standing at the kerb" (YIELD) from
"a pedestrian will be in front of me shortly" (STOP).

Assumptions, stated plainly: constant speed, constant heading, no interaction
between objects, no lane following. Beyond roughly two seconds these are weak,
which is why prediction confidence decays with horizon length and why the
decision engine never relies on prediction alone.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from ..core.types import BBox, TrackedObject
from ..geometry.ego_path import EgoPath
from .predictor import BasePredictor, PredictedTrajectory

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["ConstantVelocityPredictor"]


class ConstantVelocityPredictor(BasePredictor):
    """Linear extrapolation of image-space motion."""

    def __init__(self, config: "Config", ego_path: EgoPath) -> None:
        self.config = config
        self.ego_path = ego_path
        self.horizon = config.prediction.horizon_seconds
        self.timestep = config.prediction.timestep
        self.steps = config.prediction.steps
        self.min_track_age = config.prediction.min_track_age
        self.static_speed_px = config.prediction.static_speed_px

    def predict(
        self, obj: TrackedObject, frame_width: int, frame_height: int
    ) -> PredictedTrajectory:
        trajectory = PredictedTrajectory(track_id=obj.track_id)

        if obj.hits < self.min_track_age:
            trajectory.confidence = 0.0
            trajectory.max_path_overlap = obj.ego_path_overlap
            return trajectory

        vx, vy = obj.velocity_px
        speed = math.hypot(vx, vy)

        if speed < self.static_speed_px:
            trajectory.confidence = 0.85
            trajectory.max_path_overlap = obj.ego_path_overlap
            if obj.ego_path_overlap >= self.config.risk.in_path_overlap:
                trajectory.time_to_path_entry = 0.0
            trajectory.points = [obj.bbox.center]
            trajectory.timestamps = [0.0]
            return trajectory

        max_overlap = obj.ego_path_overlap
        entry_time = 0.0 if obj.ego_path_overlap >= self.config.risk.in_path_overlap else None

        base_bbox = obj.bbox
        for step in range(1, self.steps + 1):
            t = step * self.timestep
            dx, dy = vx * t, vy * t

            future_bbox = BBox(
                base_bbox.x1 + dx, base_bbox.y1 + dy,
                base_bbox.x2 + dx, base_bbox.y2 + dy,
            )
            center = future_bbox.center

            if (center.x < -frame_width * 0.2 or center.x > frame_width * 1.2
                    or center.y < 0 or center.y > frame_height * 1.2):
                break

            trajectory.points.append(center)
            trajectory.timestamps.append(t)

            overlap = self.ego_path.overlap(future_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
            if entry_time is None and overlap >= self.config.risk.in_path_overlap:
                entry_time = t

        trajectory.max_path_overlap = max_overlap
        trajectory.time_to_path_entry = entry_time
        trajectory.confidence = self._confidence(obj)
        return trajectory

    def _confidence(self, obj: TrackedObject) -> float:
        """Trust in this prediction.

        Grows with observation length (more evidence for the velocity) and
        falls for erratic motion, since constant velocity is a poor model for
        anything that keeps changing direction.
        """
        age_factor = min(1.0, obj.hits / 12.0)

        consistency = 1.0
        if len(obj.history) >= 5:
            recent = obj.history[-5:]
            deltas = [
                (recent[i + 1].x - recent[i].x, recent[i + 1].y - recent[i].y)
                for i in range(len(recent) - 1)
            ]
            magnitudes = [math.hypot(dx, dy) for dx, dy in deltas]
            mean_mag = sum(magnitudes) / len(magnitudes)
            if mean_mag > 1e-3:
                variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
                consistency = float(
                    max(0.25, 1.0 - min(1.0, math.sqrt(variance) / mean_mag))
                )

        if obj.category.value in ("ANIMAL", "VULNERABLE"):
            consistency *= 0.8

        return float(min(0.95, age_factor * consistency))
