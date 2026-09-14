"""Shared test fixtures and builders.

The helpers here construct :class:`TrackedObject` instances that have been run
through the real geometry analyser, so tests exercise the same path the live
pipeline does rather than a hand-faked approximation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.classes import category_for
from src.core.config import Config, load_config
from src.core.types import (
    BBox,
    PotholeInfo,
    TrackedObject,
    WorldState,
)
from src.decision.decision_engine import DecisionEngine
from src.geometry.ego_path import EgoPath
from src.geometry.road_position import RoadPositionAnalyzer
from src.risk.risk_engine import RiskEngine

FRAME_WIDTH = 960
FRAME_HEIGHT = 540


@pytest.fixture
def config() -> Config:
    """A fresh config for each test, loaded from the project default."""
    path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    return load_config(path) if path.exists() else Config()


@pytest.fixture
def ego_path(config: Config) -> EgoPath:
    path = EgoPath(config)
    path.configure(FRAME_WIDTH, FRAME_HEIGHT)
    return path


@pytest.fixture
def analyzer(config: Config, ego_path: EgoPath) -> RoadPositionAnalyzer:
    return RoadPositionAnalyzer(config, ego_path)


@pytest.fixture
def risk_engine(config: Config, ego_path: EgoPath) -> RiskEngine:
    return RiskEngine(config, ego_path)


@pytest.fixture
def decision_engine(config: Config, ego_path: EgoPath) -> DecisionEngine:
    return DecisionEngine(config, ego_path)


def make_object(
    track_id: int = 1,
    class_name: str = "person",
    bbox: tuple[float, float, float, float] = (430, 240, 500, 430),
    distance: Optional[float] = 10.0,
    confidence: float = 0.9,
    hits: int = 10,
    velocity_px: tuple[float, float] = (0.0, 0.0),
    closing_speed: Optional[float] = None,
    pothole: Optional[PotholeInfo] = None,
) -> TrackedObject:
    """Build a tracked object with sensible defaults."""
    obj = TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BBox(*bbox),
        category=category_for(class_name),
        age=hits,
        hits=hits,
        estimated_distance=distance,
        velocity_px=velocity_px,
        speed_px=(velocity_px[0] ** 2 + velocity_px[1] ** 2) ** 0.5,
        pothole=pothole,
    )
    if distance is not None:
        obj.distance_history = [distance] * 6
    obj.closing_speed_mps = closing_speed
    return obj


def build_world(
    objects: list[TrackedObject],
    analyzer: RoadPositionAnalyzer,
    risk_engine: RiskEngine,
    ego_path: EgoPath,
    dt: float = 0.05,
) -> WorldState:
    """Run geometry + risk over objects and assemble a world state.

    Mirrors the live pipeline order so tests cannot pass against a shortcut
    the real application never takes.
    """
    preserved = {o.track_id: o.closing_speed_mps for o in objects}
    analyzer.analyze(objects, FRAME_WIDTH, FRAME_HEIGHT, dt)

    for obj in objects:
        if preserved.get(obj.track_id) is not None:
            obj.closing_speed_mps = preserved[obj.track_id]

    scene = risk_engine.evaluate(objects, FRAME_WIDTH, FRAME_HEIGHT)
    return WorldState(
        frame_index=1,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        objects=objects,
        scene_risk=scene,
        ego_path_polygon=ego_path.polygon,
    )
