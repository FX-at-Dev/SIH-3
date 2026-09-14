"""Ego-path corridor: the region of road the vehicle is likely to occupy.

The corridor is a trapezoid in image space, widening toward the bottom of the
frame the way a road does under perspective. This is deliberately a geometric
prior rather than a lane-marking detector: Indian roads frequently have faded,
absent or simply ignored lane markings, so a system that depends on painted
lines fails exactly where it is needed most.

Everything here is image-space geometry. No claim is made about world
coordinates or GPS position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core.types import BBox, Point

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["EgoPath", "SideClearance"]


@dataclass(slots=True)
class SideClearance:
    """How much free lateral room exists on each side of the corridor."""

    left_free: float
    right_free: float
    left_blockers: list[int]
    right_blockers: list[int]

    @property
    def left_clear(self) -> bool:
        return self.left_free > 0.55

    @property
    def right_clear(self) -> bool:
        return self.right_free > 0.55


class EgoPath:
    """The ego corridor for one frame size, with overlap queries."""

    def __init__(self, config: "Config") -> None:
        cfg = config.ego_path
        self.near_width = cfg.near_width
        self.far_width = cfg.far_width
        self.far_y = cfg.far_y
        self.lateral_offset = cfg.lateral_offset
        self.vehicle_width_m = cfg.vehicle_width_m
        self.lane_width_m = cfg.lane_width_m
        self.nudge_clearance_m = cfg.nudge_clearance_m

        self._width = 0
        self._height = 0
        self._polygon: list[Point] = []
        self._mask: Optional[np.ndarray] = None

    # ---- construction -----------------------------------------------------

    def configure(self, width: int, height: int) -> None:
        """Build the corridor for a given frame size (cached between frames)."""
        if width == self._width and height == self._height and self._polygon:
            return

        self._width, self._height = width, height
        cx = 0.5 + self.lateral_offset

        near_half = self.near_width / 2.0
        far_half = self.far_width / 2.0
        y_far = self.far_y

        self._polygon = [
            Point((cx - near_half) * width, float(height)),
            Point((cx - far_half) * width, y_far * height),
            Point((cx + far_half) * width, y_far * height),
            Point((cx + near_half) * width, float(height)),
        ]
        self._mask = None
        logger.debug("Ego corridor configured for %dx%d.", width, height)

    @property
    def polygon(self) -> list[Point]:
        return list(self._polygon)

    @property
    def polygon_array(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self._polygon], dtype=np.int32)

    # ---- geometry queries -------------------------------------------------

    def half_width_at(self, y: float) -> float:
        """Half-width of the corridor in pixels at image row ``y``."""
        if self._height <= 0:
            return 0.0
        y_far_px = self.far_y * self._height
        if y <= y_far_px:
            return self.far_width * self._width / 2.0
        span = max(1e-6, self._height - y_far_px)
        t = (y - y_far_px) / span
        half = (self.far_width + (self.near_width - self.far_width) * t) / 2.0
        return half * self._width

    def center_x_at(self, y: float) -> float:
        """Corridor centreline x at image row ``y``."""
        return (0.5 + self.lateral_offset) * self._width

    def contains(self, point: Point) -> bool:
        """Whether a point falls inside the corridor."""
        if self._height <= 0 or point.y < self.far_y * self._height:
            return False
        half = self.half_width_at(point.y)
        return abs(point.x - self.center_x_at(point.y)) <= half

    def overlap(self, bbox: BBox) -> float:
        """Fraction of a box that lies inside the corridor.

        The box's lower portion is weighted most heavily: an object's footprint
        is what actually occupies road space, while its upper body may overhang
        harmlessly. Sampling several rows also handles the trapezoid's taper
        far better than a single centre-point test.
        """
        if self._height <= 0 or bbox.area <= 0:
            return 0.0

        y_start = bbox.y1 + 0.4 * bbox.height
        rows = np.linspace(y_start, bbox.y2, num=7)

        total_weight = 0.0
        covered_weight = 0.0
        for i, y in enumerate(rows):
            weight = 1.0 + i * 0.35
            total_weight += weight
            if y < self.far_y * self._height:
                continue
            half = self.half_width_at(float(y))
            cx = self.center_x_at(float(y))
            lo, hi = cx - half, cx + half
            inter = max(0.0, min(bbox.x2, hi) - max(bbox.x1, lo))
            if bbox.width > 1e-6:
                covered_weight += weight * (inter / bbox.width)

        return float(min(1.0, covered_weight / total_weight)) if total_weight else 0.0

    def obstruction(self, bbox: BBox) -> float:
        """Fraction of the corridor's *width* that this box blocks.

        This is the complement of :meth:`overlap` and the two must not be
        confused. ``overlap`` asks "how much of the object is in my lane" - the
        right question for risk. ``obstruction`` asks "how much of my lane does
        it take up" - the right question for whether the vehicle can get past.

        A narrow pothole sitting squarely in a wide lane scores ``overlap``
        near 1.0 but ``obstruction`` near 0.25: fully in the path, yet easily
        driven around. A bus in the same lane scores high on both.
        """
        if self._height <= 0 or bbox.area <= 0:
            return 0.0

        y = max(bbox.y2, self.far_y * self._height + 1.0)
        half = self.half_width_at(y)
        if half <= 1e-6:
            return 0.0

        cx = self.center_x_at(y)
        lo, hi = cx - half, cx + half
        blocked = max(0.0, min(bbox.x2, hi) - max(bbox.x1, lo))
        return float(min(1.0, blocked / (2.0 * half)))

    def outside_room_m(self, y: float) -> tuple[float, float]:
        """Approximate drivable room outside the corridor at row ``y``.

        Returns ``(left_m, right_m)``. This is the space a lateral adjustment
        could use - the shoulder, or the unoccupied part of a wide carriageway.
        It is bounded by the frame edge, so it under-reports on roads that
        extend beyond the camera's field of view.
        """
        if self._width <= 0:
            return (0.0, 0.0)
        half = self.half_width_at(y)
        cx = self.center_x_at(y)
        mpp = self.pixels_to_metres_at(y)
        if mpp <= 0.0:
            return (0.0, 0.0)
        left_px = max(0.0, (cx - half) - 0.0)
        right_px = max(0.0, float(self._width) - (cx + half))
        return (left_px * mpp, right_px * mpp)

    def lateral_gap_px(self, bbox: BBox) -> tuple[float, float]:
        """Free pixels between the box and each corridor edge at its base row.

        Returns ``(left_gap, right_gap)``. A negative gap means the box crosses
        that edge.
        """
        y = bbox.y2
        half = self.half_width_at(y)
        cx = self.center_x_at(y)
        left_edge, right_edge = cx - half, cx + half
        return (bbox.x1 - left_edge, right_edge - bbox.x2)

    def pixels_to_metres_at(self, y: float) -> float:
        """Approximate metres-per-pixel laterally at image row ``y``.

        Derived by assuming the corridor spans one lane width. Rough, and only
        used for clearance reasoning, never presented as a measurement.
        """
        half_px = self.half_width_at(y)
        if half_px <= 1e-6:
            return 0.0
        return (self.lane_width_m / 2.0) / half_px

    # ---- clearance reasoning ---------------------------------------------

    def side_clearance(self, obstacles: list[tuple[int, BBox]]) -> SideClearance:
        """Estimate free space either side of the corridor.

        Args:
            obstacles: ``(track_id, bbox)`` pairs for objects to consider.

        Returns:
            A :class:`SideClearance` describing each side. This is what makes a
            NUDGE recommendation defensible: a direction is only offered when
            the space on that side is actually estimated to be free.
        """
        if self._width <= 0:
            return SideClearance(0.0, 0.0, [], [])

        band_top = self._height * (self.far_y + 1.0) / 2.0
        cx = self.center_x_at(self._height)
        half = self.half_width_at(self._height * 0.9)

        left_span = max(1.0, (cx - half) - 0.0)
        right_span = max(1.0, float(self._width) - (cx + half))

        left_blocked = 0.0
        right_blocked = 0.0
        left_ids: list[int] = []
        right_ids: list[int] = []

        for track_id, bbox in obstacles:
            if bbox.y2 < band_top:
                continue
            left_overlap = max(0.0, min(bbox.x2, cx - half) - max(bbox.x1, 0.0))
            if left_overlap > 1.0:
                left_blocked += left_overlap
                left_ids.append(track_id)
            right_overlap = max(
                0.0, min(bbox.x2, float(self._width)) - max(bbox.x1, cx + half)
            )
            if right_overlap > 1.0:
                right_blocked += right_overlap
                right_ids.append(track_id)

        left_free = float(np.clip(1.0 - left_blocked / left_span, 0.0, 1.0))
        right_free = float(np.clip(1.0 - right_blocked / right_span, 0.0, 1.0))

        mpp = self.pixels_to_metres_at(self._height * 0.9)
        if mpp > 0.0:
            if left_span * mpp < self.nudge_clearance_m:
                left_free = min(left_free, 0.2)
            if right_span * mpp < self.nudge_clearance_m:
                right_free = min(right_free, 0.2)

        return SideClearance(left_free, right_free, left_ids, right_ids)
