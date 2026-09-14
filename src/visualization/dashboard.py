"""Side dashboard: recommendation, risk, counts and performance.

Rendered as a fixed-width panel beside the video so annotations never obscure
the footage they describe. Everything shown is derived from the live world
state; nothing here is static text pretending to be output.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..core.types import (
    ActionType,
    DrivingRecommendation,
    FrameMetrics,
    WorldState,
)
from .overlays import (
    COLORS,
    RISK_MARKERS,
    action_color,
    draw_text,
    risk_color,
    text_size,
)

if TYPE_CHECKING:
    from ..core.config import Config

__all__ = ["Dashboard"]

SAFETY_NOTICE = (
    "PROTOTYPE - research / driver-assistance decision support only. "
    "NOT for control of a real vehicle."
)


class Dashboard:
    """Renders the information panel."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.width = config.visualization.panel_width
        self.title = config.app.name

    def render(
        self,
        height: int,
        world: WorldState,
        recommendation: DrivingRecommendation,
        metrics: FrameMetrics,
        status_lines: Sequence[str] = (),
        extra: Optional[dict[str, str]] = None,
    ) -> np.ndarray:
        """Build the panel image for one frame."""
        panel = np.full((height, self.width, 3), COLORS["panel"], dtype=np.uint8)
        y = 0

        y = self._header(panel, y)
        y = self._recommendation(panel, y, recommendation)
        y = self._risk(panel, y, world, recommendation)
        y = self._counts(panel, y, world)
        y = self._performance(panel, y, metrics, height)
        if extra and y + 14 * len(extra) < height - 64:
            y = self._extra(panel, y, extra)
        self._footer(panel, height, status_lines, min_y=y)
        return panel

    # ---- sections ---------------------------------------------------------

    def _section_rule(self, panel: np.ndarray, y: int) -> int:
        cv2.line(panel, (10, y), (self.width - 10, y), (60, 60, 68), 1)
        return y + 14

    def _header(self, panel: np.ndarray, y: int) -> int:
        cv2.rectangle(panel, (0, 0), (self.width, 34), (40, 40, 48), -1)
        draw_text(panel, "INDIAN ROAD PERCEPTION", (12, 15), 0.44,
                  COLORS["white"], 1)
        draw_text(panel, "Perception + Decision Support", (12, 28), 0.36,
                  COLORS["grey"], 1)
        return 48

    def _recommendation(
        self, panel: np.ndarray, y: int, rec: DrivingRecommendation
    ) -> int:
        draw_text(panel, "RECOMMENDATION", (12, y), 0.4, COLORS["grey"], 1)
        y += 10

        color = action_color(rec.action)

        cv2.rectangle(panel, (10, y), (self.width - 10, y + 44), color, -1)
        cv2.rectangle(panel, (10, y), (self.width - 10, y + 44), COLORS["black"], 1)

        label = rec.label
        scale = 0.95 if len(label) <= 11 else 0.72
        tw, th = text_size(label, scale, 2)
        cv2.putText(
            panel, label,
            ((self.width - tw) // 2, y + 22 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS["black"], 2, cv2.LINE_AA,
        )
        y += 54

        speed = f"Target speed: {rec.target_speed_kmh:.0f} km/h"
        if rec.action is ActionType.STOP:
            speed = "Target speed: 0 km/h (full stop)"
        draw_text(panel, speed, (12, y), 0.42, COLORS["white"], 1)
        y += 18

        ttc_text = f"{rec.ttc:.1f} s" if rec.ttc is not None else "--"
        draw_text(panel, f"TTC: {ttc_text}", (12, y), 0.42, COLORS["white"], 1)
        draw_text(panel, f"Conf: {rec.confidence:.2f}", (self.width // 2 + 20, y),
                  0.42, COLORS["white"], 1)
        y += 20

        draw_text(panel, "REASON", (12, y), 0.38, COLORS["grey"], 1)
        y += 14
        for line in textwrap.wrap(rec.reason, width=46)[:6]:
            draw_text(panel, line, (12, y), 0.375, COLORS["white"], 1)
            y += 13

        return self._section_rule(panel, y + 6)

    def _risk(
        self,
        panel: np.ndarray,
        y: int,
        world: WorldState,
        rec: DrivingRecommendation,
    ) -> int:
        scene = world.scene_risk
        color = risk_color(scene.level)

        draw_text(panel, "SCENE RISK", (12, y), 0.4, COLORS["grey"], 1)
        y += 18

        marker = RISK_MARKERS[scene.level]
        draw_text(panel, f"{scene.level.value} {marker}", (12, y), 0.58, color, 2)

        bar_x = 150
        bar_w = self.width - bar_x - 14
        cv2.rectangle(panel, (bar_x, y - 11), (bar_x + bar_w, y + 2),
                      (55, 55, 62), -1)
        filled = int(bar_w * min(1.0, max(0.02, scene.score)))
        cv2.rectangle(panel, (bar_x, y - 11), (bar_x + filled, y + 2), color, -1)
        y += 18

        if scene.perception_confidence < self.config.risk.low_confidence:
            draw_text(panel, "! LOW PERCEPTION CONFIDENCE", (12, y), 0.36,
                      COLORS["high"], 1)
            y += 14

        for reason in scene.reasons[:3]:
            for line in textwrap.wrap(f"- {reason}", width=48)[:2]:
                draw_text(panel, line, (12, y), 0.355, COLORS["white"], 1)
                y += 12

        left = "clear" if scene.left_clear else "blocked"
        right = "clear" if scene.right_clear else "blocked"
        y += 4
        draw_text(panel, f"Left: {left}    Right: {right}", (12, y), 0.38,
                  COLORS["white"], 1)
        y += 14

        return self._section_rule(panel, y + 2)

    def _counts(self, panel: np.ndarray, y: int, world: WorldState) -> int:
        counts = world.counts()
        draw_text(panel, "DETECTED OBJECTS", (12, y), 0.4, COLORS["grey"], 1)
        y += 18

        rows = [
            ("Objects", counts["objects"]),
            ("Pedestrians", counts["pedestrians"]),
            ("Vehicles", counts["vehicles"]),
            ("Animals", counts["animals"]),
            ("Potholes", counts["potholes"]),
        ]
        for label, value in rows:
            draw_text(panel, label, (16, y), 0.4, COLORS["white"], 1)
            draw_text(panel, str(value), (self.width - 40, y), 0.4,
                      COLORS["white"], 1)
            y += 16

        density = world.scene_risk.traffic_density
        draw_text(panel, f"Traffic density: {density}", (16, y), 0.38,
                  COLORS["grey"], 1)
        y += 16
        return self._section_rule(panel, y)

    def _performance(
        self, panel: np.ndarray, y: int, metrics: FrameMetrics, height: int
    ) -> int:
        draw_text(panel, "PERFORMANCE", (12, y), 0.4, COLORS["grey"], 1)
        y += 18

        rows = [
            ("FPS", f"{metrics.fps:.1f}"),
            ("Frame", str(metrics.frame_index)),
            ("Inference", f"{metrics.inference_ms:.0f} ms"),
            ("Track", f"{metrics.tracking_ms:.1f} ms"),
            ("Risk+Decide", f"{metrics.risk_ms + metrics.decision_ms:.1f} ms"),
            ("Total", f"{metrics.total_ms:.0f} ms"),
        ]

        available = (height - 64) - y
        max_rows = max(2, available // 14)
        if len(rows) > max_rows:
            keep = ["FPS", "Inference", "Total", "Frame"][:max_rows]
            rows = [r for r in rows if r[0] in keep]
        for label, value in rows:
            draw_text(panel, label, (16, y), 0.375, COLORS["grey"], 1)
            draw_text(panel, value, (self.width - 100, y), 0.375,
                      COLORS["white"], 1)
            y += 14
        return self._section_rule(panel, y + 2)

    def _extra(self, panel: np.ndarray, y: int, extra: dict[str, str]) -> int:
        for key, value in extra.items():
            draw_text(panel, f"{key}: {value}", (12, y), 0.37, COLORS["white"], 1)
            y += 14
        return self._section_rule(panel, y + 2)

    def _footer(
        self,
        panel: np.ndarray,
        height: int,
        status_lines: Sequence[str],
        min_y: int = 0,
    ) -> None:
        """Status lines plus the permanent safety notice.

        The safety notice is mandatory, so it gets a reserved band at the base
        of the panel that content above can never occupy. Status lines fill the
        remaining gap and are dropped when there is no room for them.
        """
        notice_lines = textwrap.wrap(SAFETY_NOTICE, width=44)
        notice_h = 12 * len(notice_lines) + 12

        band_top = height - notice_h
        cv2.rectangle(panel, (0, band_top), (self.width, height),
                      COLORS["panel"], -1)
        cv2.line(panel, (10, band_top + 2), (self.width - 10, band_top + 2),
                 (60, 60, 68), 1)

        y = band_top + 16
        for line in notice_lines:
            draw_text(panel, line, (10, y), 0.33, COLORS["medium"], 1)
            y += 12

        status = [line[:52] for line in status_lines]
        while status and (band_top - 8 - 12 * len(status)) < min_y:
            status.pop()
        if not status:
            return

        y = band_top - 8 - 12 * len(status)
        for line in status:
            draw_text(panel, line, (10, y), 0.33, COLORS["grey"], 1)
            y += 12
