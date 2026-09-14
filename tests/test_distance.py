"""Distance-estimation tests.

The critical property is not accuracy - a single uncalibrated camera cannot be
accurate - but **honesty and monotonicity**: estimates must always be labelled
as estimates, and a nearer object must always read nearer than a farther one.
"""

from __future__ import annotations

import math

import pytest

from src.core.types import BBox
from src.geometry.distance import (
    MonocularDistanceEstimator,
    create_distance_estimator,
)
from tests.conftest import FRAME_HEIGHT, FRAME_WIDTH, make_object


@pytest.fixture
def estimator(config) -> MonocularDistanceEstimator:
    return MonocularDistanceEstimator(config)


class TestHonesty:
    """The system must never present an inference as a measurement."""

    def test_monocular_never_claims_true_measurement(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        assert estimator.provides_true_measurement is False

    def test_results_are_flagged_as_estimated(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        result = estimator.estimate(
            BBox(430, 300, 500, 430), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert result is not None
        assert result.is_estimated is True
        assert "estimated" in str(result)

    def test_track_distance_text_marks_the_estimate(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        obj = make_object(class_name="person", bbox=(430, 300, 500, 430))
        estimator.apply_to_track(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert obj.distance_is_estimated
        assert obj.distance_text().endswith("m~")

    def test_missing_distance_renders_safely(self) -> None:
        obj = make_object(distance=None)
        assert obj.distance_text() == "dist n/a"


class TestMonotonicity:
    def test_larger_box_reads_nearer(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        """The same object appearing bigger must be reported as closer."""
        near = estimator.estimate(
            BBox(400, 200, 560, 500), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        far = estimator.estimate(
            BBox(470, 330, 500, 400), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert near is not None and far is not None
        assert near.distance_m < far.distance_m

    def test_monotonic_across_a_size_sweep(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        distances = []
        for height in range(40, 320, 40):
            bbox = BBox(450, 460 - height, 510, 460)
            result = estimator.estimate(
                bbox, "person", FRAME_WIDTH, FRAME_HEIGHT
            )
            assert result is not None
            distances.append(result.distance_m)
        assert distances == sorted(distances, reverse=True), distances

    def test_smaller_object_class_reads_nearer_at_equal_size(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        """A bus filling the same box as a person must be much farther away."""
        bbox = BBox(400, 250, 560, 450)
        person = estimator.estimate(bbox, "person", FRAME_WIDTH, FRAME_HEIGHT)
        bus = estimator.estimate(bbox, "bus", FRAME_WIDTH, FRAME_HEIGHT)
        assert person is not None and bus is not None
        assert bus.distance_m > person.distance_m


class TestBounds:
    def test_results_are_clamped_to_the_configured_range(
        self, estimator: MonocularDistanceEstimator, config
    ) -> None:
        tiny = estimator.estimate(
            BBox(478, 268, 482, 272), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        huge = estimator.estimate(
            BBox(0, 0, FRAME_WIDTH, FRAME_HEIGHT), "person",
            FRAME_WIDTH, FRAME_HEIGHT,
        )
        for result in (tiny, huge):
            assert result is not None
            assert config.distance.min_distance_m <= result.distance_m
            assert result.distance_m <= config.distance.max_distance_m

    def test_always_finite(self, estimator: MonocularDistanceEstimator) -> None:
        for bbox in (
            BBox(0, 0, 1, 1), BBox(400, 0, 500, 3),
            BBox(400, 530, 500, 540), BBox(0, 0, 960, 540),
        ):
            result = estimator.estimate(bbox, "car", FRAME_WIDTH, FRAME_HEIGHT)
            if result is not None:
                assert math.isfinite(result.distance_m)

    def test_degenerate_inputs_return_none(
        self, estimator: MonocularDistanceEstimator
    ) -> None:
        assert estimator.estimate(BBox(100, 100, 100, 100), "car", 960, 540) is None
        assert estimator.estimate(BBox(1, 1, 20, 20), "car", 0, 0) is None


class TestConfidence:
    def test_truncated_box_is_lower_confidence(
        self, estimator: MonocularDistanceEstimator, config
    ) -> None:
        """A box cut off by the frame edge under-reports the object's size."""
        config.distance.method = "known_size"
        estimator = MonocularDistanceEstimator(config)

        whole = estimator.estimate(
            BBox(400, 200, 480, 400), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        truncated = estimator.estimate(
            BBox(0, 200, 80, 400), "person", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert whole is not None and truncated is not None
        assert truncated.confidence < whole.confidence

    def test_unknown_class_has_low_confidence(
        self, config
    ) -> None:
        """No size prior exists for an unclassified object."""
        config.distance.method = "known_size"
        estimator = MonocularDistanceEstimator(config)
        result = estimator.estimate(
            BBox(400, 250, 500, 450), "unknown", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert result is not None
        assert result.confidence <= 0.25


class TestMethods:
    @pytest.mark.parametrize("method", ["known_size", "ground_plane", "hybrid"])
    def test_all_methods_produce_a_value(self, config, method: str) -> None:
        config.distance.method = method
        estimator = MonocularDistanceEstimator(config)
        result = estimator.estimate(
            BBox(420, 280, 520, 450), "car", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert result is not None
        assert result.distance_m > 0
        assert method.split("_")[0] in result.method or result.method == "hybrid"

    def test_ground_plane_rejects_objects_above_the_horizon(
        self, config
    ) -> None:
        """A box floating above the horizon has no ground-plane solution."""
        config.distance.method = "ground_plane"
        estimator = MonocularDistanceEstimator(config)
        assert estimator.estimate(
            BBox(400, 10, 500, 60), "car", FRAME_WIDTH, FRAME_HEIGHT
        ) is None

    def test_hybrid_survives_when_one_method_fails(self, config) -> None:
        """Above the horizon, hybrid still returns the known-size estimate."""
        config.distance.method = "hybrid"
        estimator = MonocularDistanceEstimator(config)
        result = estimator.estimate(
            BBox(400, 10, 500, 60), "car", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert result is not None
        assert "known_size" in result.method


class TestSmoothing:
    def test_smoothing_damps_a_jump(self, config) -> None:
        config.distance.smoothing = 0.8
        estimator = MonocularDistanceEstimator(config)

        obj = make_object(class_name="car", bbox=(420, 280, 520, 450))
        obj.distance_history = [30.0]
        estimator.apply_to_track(obj, FRAME_WIDTH, FRAME_HEIGHT)

        raw = estimator.estimate(
            BBox(420, 280, 520, 450), "car", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert raw is not None
        low, high = sorted((30.0, raw.distance_m))
        assert low <= obj.estimated_distance <= high

    def test_zero_smoothing_passes_the_raw_value(self, config) -> None:
        config.distance.smoothing = 0.0
        estimator = MonocularDistanceEstimator(config)
        obj = make_object(class_name="car", bbox=(420, 280, 520, 450))
        obj.distance_history = [99.0]
        estimator.apply_to_track(obj, FRAME_WIDTH, FRAME_HEIGHT)
        raw = estimator.estimate(
            BBox(420, 280, 520, 450), "car", FRAME_WIDTH, FRAME_HEIGHT
        )
        assert obj.estimated_distance == pytest.approx(raw.distance_m)


class TestFactory:
    def test_factory_returns_an_estimator(self, config) -> None:
        estimator = create_distance_estimator(config)
        assert estimator.provides_true_measurement is False
