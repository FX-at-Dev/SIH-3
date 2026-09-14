"""Road position: where each object sits relative to the ego vehicle.

Answers two questions per object, both from image-space geometry:

* **Lateral position** - left, center or right of the ego corridor.
* **Path relation** - directly ahead, in path, crossing the path, beside the
  vehicle, behind it, or outside the likely path entirely.

Also derives world-frame motion (closing speed, lateral speed) by
differentiating the smoothed distance series, which is what makes
time-to-collision computable.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from ..core.types import (
    LateralPosition,
    MovementStatus,
    PathRelation,
    TrackedObject,
)
from .ego_path import EgoPath

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["RoadPositionAnalyzer"]


class RoadPositionAnalyzer:
    """Assigns lateral position, path relation and world motion to tracks."""

    def __init__(self, config: "Config", ego_path: EgoPath) -> None:
        self.config = config
        self.ego_path = ego_path
        self.in_path_overlap = config.risk.in_path_overlap
        self.static_speed_px = config.prediction.static_speed_px

    def analyze(
        self,
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
        dt: float,
    ) -> None:
        """Annotate every object in place."""
        for obj in objects:
            obj.ego_path_overlap = self.ego_path.overlap(obj.bbox)
            obj.corridor_obstruction = self.ego_path.obstruction(obj.bbox)
            obj.relative_position = self._lateral_position(obj, frame_width)
            self._world_motion(obj, dt, frame_height)
            obj.path_relation = self._path_relation(obj, frame_height)
            obj.lateral_offset_m = self._lateral_offset_m(obj, frame_height)
            obj.movement = self._movement_status(obj)

    # ---- lateral position -------------------------------------------------

    def _lateral_position(
        self, obj: TrackedObject, frame_width: int
    ) -> LateralPosition:
        """Left / center / right, judged against the corridor rather than the
        image centre - an object can be image-centre yet outside the corridor
        on a curve or with an offset camera mount."""
        base_y = obj.bbox.y2
        center_x = self.ego_path.center_x_at(base_y)
        half = self.ego_path.half_width_at(base_y)
        obj_x = obj.bbox.center.x

        if obj_x < center_x - half:
            return LateralPosition.LEFT
        if obj_x > center_x + half:
            return LateralPosition.RIGHT
        return LateralPosition.CENTER

    def _lateral_offset_m(
        self, obj: TrackedObject, frame_height: int
    ) -> Optional[float]:
        """Approximate sideways offset from the corridor centreline, in metres.

        Signed: negative is left, positive is right. Approximate by
        construction; used only for clearance reasoning.
        """
        base_y = obj.bbox.y2
        mpp = self.ego_path.pixels_to_metres_at(base_y)
        if mpp <= 0.0:
            return None
        dx_px = obj.bbox.center.x - self.ego_path.center_x_at(base_y)
        return float(dx_px * mpp)

    # ---- motion -----------------------------------------------------------

    def _world_motion(
        self, obj: TrackedObject, dt: float, frame_height: int
    ) -> None:
        """Derive closing and lateral speed in m/s from the distance series.

        Closing speed is the rate at which range is shrinking. It is the only
        quantity from which a meaningful TTC can be computed, and it needs at
        least two distance samples, which is why tracking is a prerequisite.
        """
        if obj.estimated_distance is None:
            obj.closing_speed_mps = None
            obj.velocity_mps = None
            return

        history = obj.distance_history
        if len(history) < 2 or dt <= 1e-6:
            obj.closing_speed_mps = None
            obj.velocity_mps = None
            return

        window = history[-min(6, len(history)):]
        n = len(window)
        mean_t = (n - 1) / 2.0
        mean_d = sum(window) / n
        numerator = sum((i - mean_t) * (d - mean_d) for i, d in enumerate(window))
        denominator = sum((i - mean_t) ** 2 for i in range(n))
        if denominator <= 1e-9:
            obj.closing_speed_mps = None
            obj.velocity_mps = None
            return

        slope_per_frame = numerator / denominator
        range_rate = slope_per_frame / max(1e-6, dt)
        closing_speed = -range_rate

        mpp = self.ego_path.pixels_to_metres_at(obj.bbox.y2)
        lateral_mps = obj.velocity_px[0] * mpp if mpp > 0 else 0.0

        obj.closing_speed_mps = float(closing_speed)
        obj.velocity_mps = (float(lateral_mps), float(range_rate))

    def _movement_status(self, obj: TrackedObject) -> MovementStatus:
        """Refine the tracker's coarse label using world motion."""
        if obj.speed_px < self.static_speed_px:
            return MovementStatus.STATIONARY

        vx, vy = obj.velocity_px
        lateral_dominant = abs(vx) > abs(vy) * 1.2

        if lateral_dominant:
            center_x = self.ego_path.center_x_at(obj.bbox.y2)
            moving_inward = (
                (obj.bbox.center.x < center_x and vx > 0)
                or (obj.bbox.center.x > center_x and vx < 0)
            )
            if moving_inward:
                return MovementStatus.CROSSING

        if obj.closing_speed_mps is not None:
            if obj.closing_speed_mps > 0.4:
                return MovementStatus.APPROACHING
            if obj.closing_speed_mps < -0.4:
                return MovementStatus.RECEDING

        return MovementStatus.MOVING

    # ---- path relation ----------------------------------------------------

    def _path_relation(
        self, obj: TrackedObject, frame_height: int
    ) -> PathRelation:
        """Classify how the object relates to the ego corridor."""
        overlap = obj.ego_path_overlap
        base_y = obj.bbox.y2

        if base_y >= frame_height - 2 and obj.bbox.height > frame_height * 0.55:
            return PathRelation.BESIDE_VEHICLE

        if overlap >= self.in_path_overlap:
            if overlap >= 0.55 and obj.relative_position is LateralPosition.CENTER:
                return PathRelation.DIRECTLY_AHEAD
            return PathRelation.IN_PATH

        if obj.movement is MovementStatus.CROSSING or self._is_converging(obj):
            return PathRelation.CROSSING_PATH

        if base_y > frame_height * 0.82:
            return PathRelation.BESIDE_VEHICLE

        return PathRelation.OUTSIDE_PATH

    def _is_converging(self, obj: TrackedObject) -> bool:
        """Whether the object's image-space motion points into the corridor."""
        if obj.speed_px < self.static_speed_px:
            return False
        vx, _ = obj.velocity_px
        if abs(vx) < self.static_speed_px * 0.5:
            return False

        center_x = self.ego_path.center_x_at(obj.bbox.y2)
        dx = center_x - obj.bbox.center.x
        return math.copysign(1.0, dx) == math.copysign(1.0, vx)
