"""Deterministic synthetic detector, for testing the pipeline without a model.

Every detection this class emits is tagged ``DetectionSource.MOCK`` and the UI
renders a permanent "MOCK DETECTOR - SYNTHETIC DATA" banner. It exists so the
tracker, risk engine, decision engine and renderer can be exercised and
unit-tested deterministically. It is never a stand-in for real inference, and
the real :class:`YOLODetector` never falls back to it.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ..core.classes import category_for
from ..core.types import BBox, Detection, DetectionSource
from .detector import BaseDetector, DetectorInfo

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["SCENARIOS", "MockDetector", "ScriptedObject"]


class ScriptedObject:
    """One synthetic object moving along a scripted linear path.

    Positions are given in normalised frame coordinates so a scenario renders
    identically at any resolution.
    """

    __slots__ = ("class_name", "confidence", "end", "period", "size", "start")

    def __init__(
        self,
        class_name: str,
        start: tuple[float, float],
        end: tuple[float, float],
        size: tuple[float, float],
        confidence: float = 0.85,
        period: float = 120.0,
    ) -> None:
        self.class_name = class_name
        self.start = start
        self.end = end
        self.size = size
        self.confidence = confidence
        self.period = period

    def bbox_at(self, frame_index: int, width: int, height: int) -> BBox:
        """Box for this object at a given frame, interpolated start -> end."""
        t = min(1.0, (frame_index % self.period) / self.period)
        cx = self.start[0] + (self.end[0] - self.start[0]) * t
        cy = self.start[1] + (self.end[1] - self.start[1]) * t
        scale = 0.6 + 0.8 * cy
        bw = self.size[0] * scale * width
        bh = self.size[1] * scale * height
        return BBox(
            cx * width - bw / 2.0,
            cy * height - bh / 2.0,
            cx * width + bw / 2.0,
            cy * height + bh / 2.0,
        )


SCENARIOS: dict[str, list[ScriptedObject]] = {
    "clear_road": [],
    "pothole": [
        ScriptedObject("pothole", (0.62, 0.72), (0.66, 0.92), (0.10, 0.045), 0.81),
    ],
    "pedestrian_crossing": [
        ScriptedObject("person", (0.88, 0.62), (0.46, 0.70), (0.05, 0.20), 0.93),
    ],
    "cow_blocking": [
        ScriptedObject("cow", (0.50, 0.60), (0.50, 0.86), (0.17, 0.20), 0.89),
    ],
    "bullock_cart": [
        ScriptedObject("bullock_cart", (0.46, 0.58), (0.46, 0.70), (0.20, 0.22), 0.77),
    ],
    "slow_vehicle": [
        ScriptedObject("truck", (0.50, 0.52), (0.50, 0.62), (0.24, 0.26), 0.91),
    ],
    "motorcycle_overtaking": [
        ScriptedObject("motorcycle", (0.82, 0.60), (0.58, 0.80), (0.07, 0.12), 0.86),
        ScriptedObject("car", (0.44, 0.56), (0.44, 0.60), (0.18, 0.16), 0.90),
    ],
    "dense_market": [
        ScriptedObject("person", (0.20, 0.66), (0.26, 0.74), (0.05, 0.18), 0.72),
        ScriptedObject("person", (0.34, 0.62), (0.30, 0.72), (0.045, 0.17), 0.68),
        ScriptedObject("person", (0.74, 0.64), (0.66, 0.72), (0.05, 0.18), 0.70),
        ScriptedObject("motorcycle", (0.58, 0.60), (0.54, 0.68), (0.07, 0.11), 0.66),
        ScriptedObject("auto_rickshaw", (0.44, 0.58), (0.46, 0.64), (0.12, 0.15), 0.63),
        ScriptedObject("pushcart", (0.84, 0.66), (0.80, 0.70), (0.10, 0.12), 0.58),
        ScriptedObject("cow", (0.12, 0.62), (0.16, 0.66), (0.13, 0.15), 0.61),
    ],
    "multiple_obstacles": [
        ScriptedObject("pothole", (0.36, 0.78), (0.36, 0.86), (0.09, 0.04), 0.74),
        ScriptedObject("traffic_cone", (0.62, 0.70), (0.62, 0.78), (0.04, 0.08), 0.80),
        ScriptedObject("debris", (0.50, 0.74), (0.50, 0.82), (0.08, 0.05), 0.55),
    ],
    "unknown_obstacle": [
        ScriptedObject("unknown", (0.50, 0.62), (0.50, 0.80), (0.16, 0.18), 0.41),
    ],
}


class MockDetector(BaseDetector):
    """Emits scripted detections. Synthetic data only - never real inference."""

    def __init__(self, config: "Config", scenario: str = "pedestrian_crossing") -> None:
        super().__init__(config)
        self.scenario = scenario
        self._objects: list[ScriptedObject] = []

    def load(self) -> None:
        if self.scenario not in SCENARIOS:
            available = ", ".join(sorted(SCENARIOS))
            raise ValueError(
                f"Unknown mock scenario '{self.scenario}'. Available: {available}"
            )
        self._objects = SCENARIOS[self.scenario]
        logger.warning(
            "MOCK DETECTOR ACTIVE - scenario '%s'. All detections are SYNTHETIC "
            "and do not come from any model.",
            self.scenario,
        )

    def set_scenario(self, scenario: str) -> None:
        """Switch scenario at runtime (used by the demo script)."""
        self.scenario = scenario
        self.load()

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[0], frame.shape[1]

        detections: list[Detection] = []
        for scripted in self._objects:
            bbox = scripted.bbox_at(frame_index, width, height).clip(
                float(width), float(height)
            )
            if bbox.width < 2.0 or bbox.height < 2.0:
                continue
            jitter = 0.03 * math.sin(frame_index * 0.2)
            detections.append(
                Detection(
                    class_name=scripted.class_name,
                    raw_class_name=scripted.class_name,
                    confidence=float(
                        min(0.99, max(0.05, scripted.confidence + jitter))
                    ),
                    bbox=bbox,
                    category=category_for(scripted.class_name),
                    source=DetectionSource.MOCK,
                    frame_index=frame_index,
                )
            )
        return detections

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="Mock",
            model_path=f"<scenario:{self.scenario}>",
            device="none",
            class_names=tuple(o.class_name for o in self._objects),
            is_custom=False,
            notes="SYNTHETIC DATA - not a real model",
        )
