"""Time-to-collision tests.

The arithmetic is trivial; the value of these tests is in the edge cases, where
a naive implementation emits ``inf`` or ``NaN`` into the UI.
"""

from __future__ import annotations

import math

import pytest

from src.core.types import RiskLevel
from src.risk.ttc import TTCCalculator


@pytest.fixture
def calculator(config) -> TTCCalculator:
    return TTCCalculator(config.risk.ttc)


class TestBasicComputation:
    def test_simple_case(self, calculator: TTCCalculator) -> None:
        result = calculator.compute(distance_m=20.0, closing_speed_mps=10.0)
        assert result.valid
        assert result.seconds == pytest.approx(2.0)

    def test_contact_range_is_critical(self, calculator: TTCCalculator) -> None:
        result = calculator.compute(distance_m=0.0, closing_speed_mps=5.0)
        assert result.valid
        assert result.seconds == 0.0
        assert result.risk_level is RiskLevel.CRITICAL


class TestInvalidInputs:
    """Every one of these must yield no TTC rather than a bogus number."""

    def test_zero_closing_speed(self, calculator: TTCCalculator) -> None:
        result = calculator.compute(20.0, 0.0)
        assert not result.valid
        assert result.seconds is None
        assert result.display == "--"

    def test_negative_closing_speed_is_not_approaching(
        self, calculator: TTCCalculator
    ) -> None:
        result = calculator.compute(20.0, -5.0)
        assert not result.valid
        assert "not approaching" in result.reason

    def test_missing_distance(self, calculator: TTCCalculator) -> None:
        result = calculator.compute(None, 5.0)
        assert not result.valid
        assert result.seconds is None

    def test_missing_velocity(self, calculator: TTCCalculator) -> None:
        result = calculator.compute(20.0, None)
        assert not result.valid
        assert result.seconds is None

    def test_speed_below_noise_floor_rejected(
        self, calculator: TTCCalculator, config
    ) -> None:
        """A crawl slower than the estimator's noise is not a real closing rate."""
        speed = config.risk.ttc.min_closing_speed * 0.5
        result = calculator.compute(50.0, speed)
        assert not result.valid
        assert "noise floor" in result.reason

    def test_nan_input_rejected(self, calculator: TTCCalculator) -> None:
        assert not calculator.compute(float("nan"), 5.0).valid
        assert not calculator.compute(20.0, float("nan")).valid

    def test_infinite_input_rejected(self, calculator: TTCCalculator) -> None:
        assert not calculator.compute(float("inf"), 5.0).valid

    def test_beyond_horizon_rejected(
        self, calculator: TTCCalculator, config
    ) -> None:
        result = calculator.compute(10_000.0, config.risk.ttc.min_closing_speed)
        assert not result.valid
        assert "horizon" in result.reason

    @pytest.mark.parametrize(
        ("distance", "speed"),
        [(20.0, 0.0), (20.0, -1.0), (None, 5.0), (20.0, None),
         (float("nan"), 1.0), (0.0, 0.0)],
    )
    def test_never_emits_non_finite(
        self, calculator: TTCCalculator, distance, speed
    ) -> None:
        """The UI must never be handed inf or NaN, whatever the inputs."""
        result = calculator.compute(distance, speed)
        if result.seconds is not None:
            assert math.isfinite(result.seconds)
        assert result.display == "--" or result.display.endswith("s")


class TestClassification:
    @pytest.mark.parametrize(
        ("ttc", "expected"),
        [
            (10.0, RiskLevel.LOW),
            (5.5, RiskLevel.LOW),
            (4.0, RiskLevel.MEDIUM),
            (2.5, RiskLevel.HIGH),
            (1.0, RiskLevel.CRITICAL),
            (0.1, RiskLevel.CRITICAL),
        ],
    )
    def test_bands(
        self, calculator: TTCCalculator, ttc: float, expected: RiskLevel
    ) -> None:
        assert calculator.classify(ttc) is expected

    def test_boundaries_are_exclusive_below(
        self, calculator: TTCCalculator, config
    ) -> None:
        """A TTC exactly at a threshold belongs to the calmer band."""
        assert calculator.classify(config.risk.ttc.critical) is RiskLevel.HIGH
        assert calculator.classify(config.risk.ttc.high) is RiskLevel.MEDIUM
        assert calculator.classify(config.risk.ttc.medium) is RiskLevel.LOW

    def test_none_classifies_low(self, calculator: TTCCalculator) -> None:
        assert calculator.classify(None) is RiskLevel.LOW

    def test_thresholds_come_from_config(self, config) -> None:
        """Thresholds are configurable, not hard-coded constants."""
        config.risk.ttc.critical = 0.5
        config.risk.ttc.high = 1.0
        calculator = TTCCalculator(config.risk.ttc)
        assert calculator.classify(0.75) is RiskLevel.HIGH
        assert calculator.classify(0.25) is RiskLevel.CRITICAL


class TestTrackIntegration:
    def test_attaches_to_track(self, calculator: TTCCalculator) -> None:
        from tests.conftest import make_object

        obj = make_object(distance=15.0, closing_speed=5.0)
        result = calculator.compute_for_track(obj)
        assert obj.ttc == pytest.approx(3.0)
        assert result.valid

    def test_clears_ttc_when_invalid(self, calculator: TTCCalculator) -> None:
        from tests.conftest import make_object

        obj = make_object(distance=15.0, closing_speed=0.0)
        calculator.compute_for_track(obj)
        assert obj.ttc is None
        assert obj.ttc_text() == "--"
