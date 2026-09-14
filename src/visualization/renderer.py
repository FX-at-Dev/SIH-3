"""Renderer: composes the annotated video view and the dashboard panel."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from ..core.types import (
    ActionType,
    DetectionSource,
    DrivingRecommendation,
    FrameMetrics,
    PathRelation,
    RiskLevel,
    WorldState,
)
from .dashboard import Dashboard
from .overlays import (
    COLORS,
    action_color,
    draw_box,
    draw_drivable_overlay,
    draw_ego_corridor,
    draw_nudge_arrow,
    draw_text,
    draw_trajectory,
    text_size,
)

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["Renderer"]


class Renderer:
    """Turns a world state into the frame the user sees."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.cfg = config.visualization
        self.dashboard = Dashboard(config)

        self.show_detections = self.cfg.show_detections
        self.show_details = self.cfg.show_details
        self.show_trajectories = self.cfg.show_trajectories
        self.show_ego_path = self.cfg.show_ego_path
        self.show_dashboard = self.cfg.show_dashboard
        self.show_drivable = False

    # ---- toggles ----------------------------------------------------------

    def toggle_details(self) -> bool:
        self.show_details = not self.show_details
        return self.show_details

    def toggle_dashboard(self) -> bool:
        self.show_dashboard = not self.show_dashboard
        return self.show_dashboard

    def toggle_trajectories(self) -> bool:
        self.show_trajectories = not self.show_trajectories
        return self.show_trajectories

    def toggle_drivable(self) -> bool:
        self.show_drivable = not self.show_drivable
        return self.show_drivable

    # ---- rendering --------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        world: WorldState,
        recommendation: DrivingRecommendation,
        metrics: FrameMetrics,
        drivable_mask: Optional[np.ndarray] = None,
        status_lines: Sequence[str] = (),
        extra: Optional[dict[str, str]] = None,
        banner: Optional[str] = None,
    ) -> np.ndarray:
        """Render the full view: annotated video plus dashboard."""
        view = frame.copy()

        if self.show_drivable and drivable_mask is not None:
            draw_drivable_overlay(view, drivable_mask)

        if self.show_ego_path and world.ego_path_polygon:
            conflict = any(
                o.path_relation in (
                    PathRelation.DIRECTLY_AHEAD,
                    PathRelation.IN_PATH,
                    PathRelation.CROSSING_PATH,
                )
                and o.risk_level >= RiskLevel.HIGH
                for o in world.objects
            )
            draw_ego_corridor(view, world.ego_path_polygon, conflict)

        if self.show_detections:
            for obj in sorted(world.objects, key=lambda o: o.risk_level.rank):
                if self.show_trajectories:
                    draw_trajectory(view, obj)
                draw_box(
                    view, obj,
                    thickness_map=self.cfg.box_thickness,
                    show_details=self.show_details,
                    font_scale=self.cfg.font_scale * 0.84,
                )

        if recommendation.action is ActionType.NUDGE and recommendation.direction:
            draw_nudge_arrow(
                view, recommendation.direction,
                action_color(recommendation.action),
            )

        self._overlay_banner(view, recommendation, world, banner)

        if not self.show_dashboard:
            return view

        panel = self.dashboard.render(
            height=view.shape[0],
            world=world,
            recommendation=recommendation,
            metrics=metrics,
            status_lines=status_lines,
            extra=extra,
        )
        return np.hstack([view, panel])

    # ---- in-frame overlays ------------------------------------------------

    def _overlay_banner(
        self,
        view: np.ndarray,
        recommendation: DrivingRecommendation,
        world: WorldState,
        banner: Optional[str],
    ) -> None:
        """Top strip: title, action, and any warning banner."""
        height, width = view.shape[:2]

        strip = view[0:30, 0:width]
        backing = np.full_like(strip, COLORS["black"], dtype=np.uint8)
        cv2.addWeighted(backing, 0.55, strip, 0.45, 0, strip)

        draw_text(view, "INDIAN ROAD PERCEPTION SYSTEM", (10, 20), 0.46,
                  COLORS["white"], 1)

        label = recommendation.label
        color = action_color(recommendation.action)
        tw, _ = text_size(label, 0.62, 2)
        draw_text(view, label, (width - tw - 14, 21), 0.62, color, 2)

        warnings: list[str] = []
        if banner:
            warnings.append(banner)
        if any(o.source is DetectionSource.MOCK for o in world.objects):
            warnings.append("MOCK DETECTOR - SYNTHETIC DATA, NOT REAL DETECTION")
        if any(o.source is DetectionSource.HEURISTIC for o in world.objects):
            warnings.append("Potholes shown are CV heuristic, not ML detection")

        y = 46
        for text in warnings[:3]:
            tw, th = text_size(text, 0.42, 1)
            cv2.rectangle(view, (8, y - th - 6), (14 + tw, y + 6),
                          COLORS["black"], -1)
            cv2.rectangle(view, (8, y - th - 6), (14 + tw, y + 6),
                          COLORS["medium"], 1)
            draw_text(view, text, (11, y), 0.42, COLORS["medium"], 1)
            y += th + 16

        notice = "PROTOTYPE - not for vehicle control"
        tw, _ = text_size(notice, 0.36, 1)
        draw_text(view, notice, (width - tw - 10, height - 10), 0.36,
                  COLORS["grey"], 1)

    # ---- helper for image mode -------------------------------------------

    @staticmethod
    def fit_to_screen(
        image: np.ndarray, max_width: int = 1700, max_height: int = 950
    ) -> np.ndarray:
        """Scale an image down to fit a typical screen, preserving aspect."""
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 1.0:
            return image
        return cv2.resize(
            image, (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
