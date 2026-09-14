"""Typed, validated configuration loaded from YAML.

Every tunable threshold in the system lives in ``configs/*.yaml`` and is parsed
into the Pydantic models below. Validation happens once at startup so a typo in
a config file fails loudly and immediately rather than producing a subtly wrong
risk assessment ten minutes into a video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class _Base(BaseModel):
    """Base model: reject unknown keys so config typos cannot pass silently."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------


class AppConfig(_Base):
    name: str = "Indian Road Autonomy Perception System"
    window_title: str = "Indian Road Perception - Prototype"
    process_width: int = Field(960, ge=320, le=3840)
    log_level: str = "INFO"


class DetectorConfig(_Base):
    type: Literal["yolo", "mock"] = "yolo"
    model: str = "models/yolo11n.pt"
    confidence: float = Field(0.35, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    imgsz: int = Field(640, ge=160, le=1920)
    device: str = "auto"
    max_detections: int = Field(60, ge=1, le=1000)
    ignore_classes: list[str] = Field(default_factory=list)


class PotholeHeuristicConfig(_Base):
    min_area_ratio: float = Field(0.0008, ge=0.0, le=1.0)
    max_area_ratio: float = Field(0.08, ge=0.0, le=1.0)
    darkness_percentile: float = Field(12.0, ge=0.5, le=50.0)
    min_solidity: float = Field(0.55, ge=0.0, le=1.0)
    min_aspect: float = Field(0.35, ge=0.0)
    max_aspect: float = Field(3.0, ge=0.0)


class PotholeSeverityConfig(_Base):
    medium_width_m: float = Field(0.35, gt=0.0)
    high_width_m: float = Field(0.70, gt=0.0)
    high_overlap: float = Field(0.35, ge=0.0, le=1.0)


class PotholeConfig(_Base):
    enabled: bool = True
    type: Literal["none", "yolo", "heuristic"] = "none"
    model: str = "models/pothole.pt"
    confidence: float = Field(0.30, ge=0.0, le=1.0)
    heuristic: PotholeHeuristicConfig = Field(default_factory=PotholeHeuristicConfig)
    severity: PotholeSeverityConfig = Field(default_factory=PotholeSeverityConfig)


class DrivableAreaConfig(_Base):
    type: Literal["none", "heuristic", "segmentation"] = "heuristic"
    horizon_ratio: float = Field(0.45, ge=0.0, le=0.95)


class TrackingConfig(_Base):
    enabled: bool = True
    type: Literal["kalman_iou", "bytetrack"] = "kalman_iou"
    max_age: int = Field(15, ge=1, le=300)
    min_hits: int = Field(2, ge=1, le=50)
    iou_threshold: float = Field(0.25, ge=0.0, le=1.0)
    max_history: int = Field(30, ge=2, le=500)


class CameraConfig(_Base):
    horizontal_fov_deg: float = Field(65.0, gt=1.0, lt=180.0)
    mount_height_m: float = Field(1.25, gt=0.0, lt=10.0)
    pitch_deg: float = Field(0.0, ge=-45.0, le=45.0)
    calibrated: bool = False


class ObjectDimension(_Base):
    height: float = Field(..., gt=0.0)
    width: float = Field(..., gt=0.0)


class DistanceConfig(_Base):
    method: Literal["known_size", "ground_plane", "hybrid"] = "hybrid"
    min_distance_m: float = Field(0.8, gt=0.0)
    max_distance_m: float = Field(120.0, gt=0.0)
    smoothing: float = Field(0.55, ge=0.0, le=1.0)
    object_dimensions: dict[str, ObjectDimension] = Field(default_factory=dict)

    def dimension_for(self, class_name: str) -> ObjectDimension:
        """Dimensions for a class, falling back to the ``unknown`` entry."""
        dim = self.object_dimensions.get(class_name)
        if dim is not None:
            return dim
        fallback = self.object_dimensions.get("unknown")
        if fallback is not None:
            return fallback
        return ObjectDimension(height=1.5, width=1.0)


class EgoPathConfig(_Base):
    near_width: float = Field(0.52, gt=0.0, le=1.0)
    far_width: float = Field(0.13, gt=0.0, le=1.0)
    far_y: float = Field(0.52, ge=0.0, le=1.0)
    lateral_offset: float = Field(0.0, ge=-0.5, le=0.5)
    vehicle_width_m: float = Field(1.8, gt=0.0)
    lane_width_m: float = Field(3.2, gt=0.0)
    nudge_clearance_m: float = Field(0.6, ge=0.0)


class PredictionConfig(_Base):
    enabled: bool = True
    horizon_seconds: float = Field(3.0, gt=0.0, le=10.0)
    timestep: float = Field(0.25, gt=0.0, le=2.0)
    min_track_age: int = Field(3, ge=1)
    static_speed_px: float = Field(6.0, ge=0.0)

    @property
    def steps(self) -> int:
        return max(1, int(round(self.horizon_seconds / self.timestep)))


class TTCConfig(_Base):
    medium: float = Field(5.0, gt=0.0)
    high: float = Field(3.0, gt=0.0)
    critical: float = Field(1.5, gt=0.0)
    min_closing_speed: float = Field(0.25, gt=0.0)
    max_valid_ttc: float = Field(60.0, gt=0.0)

    @field_validator("critical")
    @classmethod
    def _ordered(cls, v: float, info: Any) -> float:
        high = info.data.get("high")
        if high is not None and v >= high:
            raise ValueError("risk.ttc.critical must be smaller than risk.ttc.high")
        return v


class DistanceRiskConfig(_Base):
    medium: float = Field(25.0, gt=0.0)
    high: float = Field(12.0, gt=0.0)
    critical: float = Field(5.0, gt=0.0)


class DensityConfig(_Base):
    moderate: int = Field(6, ge=1)
    dense: int = Field(11, ge=1)


class RiskConfig(_Base):
    ttc: TTCConfig = Field(default_factory=TTCConfig)
    distance: DistanceRiskConfig = Field(default_factory=DistanceRiskConfig)
    in_path_overlap: float = Field(0.12, ge=0.0, le=1.0)
    blocking_overlap: float = Field(0.45, ge=0.0, le=1.0)
    low_confidence: float = Field(0.45, ge=0.0, le=1.0)
    density: DensityConfig = Field(default_factory=DensityConfig)
    priority: dict[str, float] = Field(default_factory=dict)
    class_priority: dict[str, float] = Field(default_factory=dict)

    def priority_for(self, class_name: str, category_name: str) -> float:
        """Per-class weight when configured, else the category weight."""
        if class_name in self.class_priority:
            return float(self.class_priority[class_name])
        return float(self.priority.get(category_name, 0.6))


class DecisionConfig(_Base):
    cruise_speed_kmh: float = Field(30.0, ge=0.0)
    nudge_speed_kmh: float = Field(18.0, ge=0.0)
    yield_speed_kmh: float = Field(12.0, ge=0.0)
    creep_speed_kmh: float = Field(5.0, ge=0.0)
    hysteresis_frames: int = Field(3, ge=0, le=100)
    immediate_escalation: bool = True
    nudge_margin: float = Field(0.15, ge=0.0, le=1.0)
    min_cruise_confidence: float = Field(0.50, ge=0.0, le=1.0)


class VisualizationConfig(_Base):
    enabled: bool = True
    show_detections: bool = True
    show_details: bool = True
    show_trajectories: bool = True
    show_ego_path: bool = True
    show_dashboard: bool = True
    panel_width: int = Field(360, ge=200, le=900)
    font_scale: float = Field(0.5, gt=0.1, le=2.0)
    box_thickness: dict[str, int] = Field(
        default_factory=lambda: {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    )


class OutputConfig(_Base):
    results_dir: str = "results"
    save_video: bool = False
    save_events: bool = True
    save_detections: bool = True
    save_recommendations: bool = True
    save_screenshots: bool = True
    video_fps: float = Field(20.0, gt=0.0)


class Config(_Base):
    """Root configuration object."""

    app: AppConfig = Field(default_factory=AppConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    pothole: PotholeConfig = Field(default_factory=PotholeConfig)
    drivable_area: DrivableAreaConfig = Field(default_factory=DrivableAreaConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    distance: DistanceConfig = Field(default_factory=DistanceConfig)
    ego_path: EgoPathConfig = Field(default_factory=EgoPathConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    source_path: Optional[str] = None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config.

    Falls back to built-in defaults when no file is supplied or found, so the
    application still runs on a fresh checkout.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    cfg_path = Path(path)

    if not cfg_path.exists():
        if str(path) == str(DEFAULT_CONFIG_PATH):
            logger.warning(
                "Config file %s not found - using built-in defaults.", cfg_path
            )
            return Config()
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {cfg_path}: {exc}") from exc

    config = Config(**raw)
    config.source_path = str(cfg_path)
    logger.info("Loaded configuration from %s", cfg_path)
    return config


def apply_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted-key CLI overrides, e.g. ``{"detector.confidence": 0.5}``."""
    for dotted, value in overrides.items():
        if value is None:
            continue
        target: Any = config
        parts = dotted.split(".")
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
            logger.debug("Config override: %s = %r", dotted, value)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Cannot apply override {dotted}={value!r}: {exc}") from exc
    return config
