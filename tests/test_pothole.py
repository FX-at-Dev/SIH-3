"""Pothole detection and severity tests.

Two things are being protected here:

1. **Severity scales correctly** with physical size and path overlap.
2. **Provenance is never lost.** A classical-CV heuristic guess must never be
   presentable as a model detection, and a configuration with no pothole model
   must report that it is detecting nothing rather than silently finding none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import BBox, Detection, DetectionSource, RiskLevel, SemanticCategory
from src.geometry.distance import MonocularDistanceEstimator
from src.perception.pothole_detector import (
    HeuristicPotholeDetector,
    NullPotholeDetector,
    assess_pothole,
    create_pothole_detector,
)
from tests.conftest import FRAME_HEIGHT, FRAME_WIDTH


def pothole_detection(
    bbox: tuple[float, float, float, float],
    confidence: float = 0.8,
    source: DetectionSource = DetectionSource.MODEL,
) -> Detection:
    return Detection(
        class_name="pothole",
        confidence=confidence,
        bbox=BBox(*bbox),
        category=SemanticCategory.OBSTACLE,
        source=source,
    )


class TestSeverity:
    def test_larger_pothole_is_more_severe(self, config, ego_path) -> None:
        estimator = MonocularDistanceEstimator(config)
        small = assess_pothole(
            pothole_detection((470, 440, 500, 455)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        large = assess_pothole(
            pothole_detection((380, 420, 620, 490)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        assert large.estimated_width_m > small.estimated_width_m
        assert large.severity >= small.severity

    def test_severity_bands_follow_config(self, config, ego_path) -> None:
        estimator = MonocularDistanceEstimator(config)
        widths = []
        severities = []
        for width in (20, 60, 140, 260):
            info = assess_pothole(
                pothole_detection((480 - width // 2, 430, 480 + width // 2, 470)),
                config, ego_path, estimator, FRAME_WIDTH, FRAME_HEIGHT,
            )
            widths.append(info.estimated_width_m)
            severities.append(info.severity.rank)
        assert widths == sorted(widths)
        assert severities == sorted(severities)

    def test_size_label_is_populated(self, config, ego_path) -> None:
        estimator = MonocularDistanceEstimator(config)
        info = assess_pothole(
            pothole_detection((440, 430, 520, 465)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        assert info.size_label in ("small", "medium", "large")

    def test_path_overlap_is_recorded(self, config, ego_path) -> None:
        estimator = MonocularDistanceEstimator(config)
        in_path = assess_pothole(
            pothole_detection((400, 430, 560, 480)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        off_path = assess_pothole(
            pothole_detection((20, 430, 120, 480)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        assert in_path.ego_path_overlap > off_path.ego_path_overlap
        assert off_path.ego_path_overlap == pytest.approx(0.0)

    def test_overlap_escalates_severity(self, config, ego_path) -> None:
        """The same pothole matters more when it sits in the wheel track."""
        estimator = MonocularDistanceEstimator(config)
        config.pothole.severity.high_overlap = 0.1

        centred = assess_pothole(
            pothole_detection((330, 430, 630, 485)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        assert centred.ego_path_overlap > 0.1
        assert centred.severity >= RiskLevel.MEDIUM

    def test_serialises(self, config, ego_path) -> None:
        estimator = MonocularDistanceEstimator(config)
        info = assess_pothole(
            pothole_detection((440, 430, 520, 465)), config, ego_path,
            estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        payload = info.to_dict()
        assert payload["severity"] in {level.value for level in RiskLevel}
        assert "source" in payload


class TestProvenance:
    """The heuristic must never be mistakable for machine learning."""

    def test_heuristic_is_not_ml(self, config) -> None:
        detector = HeuristicPotholeDetector(config)
        assert detector.is_ml_detection is False
        assert detector.mode == "heuristic"
        assert "not ML" in detector.status_text

    def test_heuristic_detections_are_tagged(self, config) -> None:
        detector = HeuristicPotholeDetector(config)
        detector.load()

        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 110, dtype=np.uint8)
        rng = np.random.default_rng(0)
        frame[250:] = np.clip(
            frame[250:].astype(np.int16)
            + rng.integers(-6, 6, frame[250:].shape),
            0, 255,
        ).astype(np.uint8)
        import cv2
        cv2.ellipse(frame, (480, 420), (46, 22), 0, 0, 360, (28, 28, 30), -1)

        detections = detector.detect(frame, 0)
        assert detections, "the heuristic should find an obvious dark blob"
        for detection in detections:
            assert detection.source is DetectionSource.HEURISTIC
            assert detection.confidence <= HeuristicPotholeDetector.MAX_CONFIDENCE

    def test_heuristic_confidence_is_capped(self, config) -> None:
        """A heuristic must never report model-grade confidence."""
        assert HeuristicPotholeDetector.MAX_CONFIDENCE < 0.5

    def test_severity_carries_the_detection_source(
        self, config, ego_path
    ) -> None:
        estimator = MonocularDistanceEstimator(config)
        info = assess_pothole(
            pothole_detection((440, 430, 520, 465),
                              source=DetectionSource.HEURISTIC),
            config, ego_path, estimator, FRAME_WIDTH, FRAME_HEIGHT,
        )
        assert info.source is DetectionSource.HEURISTIC


class TestNullMode:
    def test_null_detector_finds_nothing_and_says_so(self, config) -> None:
        detector = NullPotholeDetector(config)
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        assert detector.detect(frame, 0) == []
        assert detector.is_ml_detection is False
        assert "OFF" in detector.status_text

    def test_default_config_does_not_claim_pothole_detection(
        self, config
    ) -> None:
        """The shipped default must not imply potholes are being found."""
        detector = create_pothole_detector(config)
        if config.pothole.type == "none":
            assert detector.is_ml_detection is False
            assert "OFF" in detector.status_text


class TestFactory:
    @pytest.mark.parametrize(
        ("mode", "expected"), [("none", "none"), ("heuristic", "heuristic")]
    )
    def test_factory_modes(self, config, mode: str, expected: str) -> None:
        config.pothole.type = mode
        config.pothole.enabled = True
        assert create_pothole_detector(config).mode == expected

    def test_disabled_returns_null(self, config) -> None:
        config.pothole.enabled = False
        assert create_pothole_detector(config).mode == "none"

    def test_missing_custom_model_raises_actionable_error(self, config) -> None:
        """The error must say what to do, not just that a file is missing."""
        from src.perception.pothole_detector import YOLOPotholeDetector

        config.pothole.model = "models/definitely_not_here.pt"
        detector = YOLOPotholeDetector(config)
        with pytest.raises(FileNotFoundError) as excinfo:
            detector.load()
        message = str(excinfo.value)
        assert "models/README.md" in message
        assert "pothole.type: none" in message
        assert "COCO has no pothole class" in message


class TestCocoLimitation:
    def test_pothole_is_documented_as_absent_from_coco(self) -> None:
        """Guard against anyone claiming stock weights detect potholes."""
        from src.core.classes import NOT_IN_COCO, is_supported_by_coco

        assert "pothole" in NOT_IN_COCO
        assert not is_supported_by_coco("pothole")

    def test_indian_specific_classes_are_flagged(self) -> None:
        from src.core.classes import NOT_IN_COCO

        for name in ("auto_rickshaw", "bullock_cart", "pushcart"):
            assert name in NOT_IN_COCO
