"""Risk-engine tests: classification, path overlap, priorities, confidence."""

from __future__ import annotations

import pytest

from src.core.types import (
    LateralPosition,
    PathRelation,
    RiskLevel,
)
from tests.conftest import FRAME_HEIGHT, build_world, make_object


class TestRiskLevelOrdering:
    def test_levels_are_ordered(self) -> None:
        assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_max_picks_the_most_severe(self) -> None:
        assert RiskLevel.max(RiskLevel.LOW, RiskLevel.HIGH) is RiskLevel.HIGH
        assert RiskLevel.max() is RiskLevel.LOW

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0.0, RiskLevel.LOW), (0.35, RiskLevel.MEDIUM),
         (0.60, RiskLevel.HIGH), (0.95, RiskLevel.CRITICAL)],
    )
    def test_from_score(self, score: float, expected: RiskLevel) -> None:
        assert RiskLevel.from_score(score) is expected


class TestEgoPathGeometry:
    """`overlap` and `obstruction` answer different questions and must differ."""

    def test_object_outside_corridor_has_no_overlap(self, ego_path) -> None:
        from src.core.types import BBox

        assert ego_path.overlap(BBox(10, 300, 90, 400)) == pytest.approx(0.0)

    def test_object_in_corridor_has_overlap(self, ego_path) -> None:
        from src.core.types import BBox

        assert ego_path.overlap(BBox(430, 300, 530, 460)) > 0.5

    def test_narrow_object_is_in_path_but_barely_obstructs(
        self, ego_path
    ) -> None:
        """A pothole fully in the lane still leaves room to drive around."""
        from src.core.types import BBox

        pothole = BBox(460, 430, 540, 470)
        assert ego_path.overlap(pothole) > 0.9
        assert ego_path.obstruction(pothole) < 0.4

    def test_wide_object_obstructs_heavily(self, ego_path) -> None:
        from src.core.types import BBox

        bus = BBox(290, 300, 670, 470)
        assert ego_path.obstruction(bus) > 0.8

    def test_corridor_widens_toward_the_vehicle(self, ego_path) -> None:
        """Perspective: the lane is wider near the camera than far away."""
        near = ego_path.half_width_at(FRAME_HEIGHT - 1)
        far = ego_path.half_width_at(FRAME_HEIGHT * 0.55)
        assert near > far


class TestObjectRisk:
    def test_vulnerable_user_in_path_is_at_least_high(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        person = make_object(
            class_name="person", bbox=(430, 250, 510, 450), distance=10.0
        )
        build_world([person], analyzer, risk_engine, ego_path)
        assert person.risk_level >= RiskLevel.HIGH

    def test_distant_object_off_path_is_low(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        car = make_object(
            track_id=2, class_name="car", bbox=(780, 250, 880, 330), distance=60.0
        )
        build_world([car], analyzer, risk_engine, ego_path)
        assert car.risk_level is RiskLevel.LOW

    def test_critical_ttc_forces_critical_risk(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        car = make_object(
            track_id=3, class_name="car", bbox=(400, 300, 560, 440),
            distance=6.0, closing_speed=10.0,
        )
        build_world([car], analyzer, risk_engine, ego_path)
        assert car.risk_level is RiskLevel.CRITICAL

    def test_risk_carries_reasons(self, risk_engine, analyzer, ego_path) -> None:
        person = make_object(
            class_name="person", bbox=(430, 250, 510, 450),
            distance=8.0, closing_speed=5.0,
        )
        build_world([person], analyzer, risk_engine, ego_path)
        assert person.risk_reasons
        assert all(isinstance(r, str) and r for r in person.risk_reasons)


class TestCategoryPriority:
    def test_pedestrian_outranks_car_in_the_same_position(
        self, risk_engine, analyzer, ego_path, config
    ) -> None:
        """Identical geometry, different consequence."""
        bbox = (430, 250, 510, 450)
        person = make_object(track_id=1, class_name="person",
                             bbox=bbox, distance=14.0)
        build_world([person], analyzer, risk_engine, ego_path)
        person_score = person.risk_score

        car = make_object(track_id=2, class_name="car", bbox=bbox, distance=14.0)
        build_world([car], analyzer, risk_engine, ego_path)

        assert person_score > car.risk_score

    def test_priorities_are_configurable(self, config) -> None:
        assert config.risk.priority_for("person", "VULNERABLE") == pytest.approx(1.0)
        config.risk.class_priority["car"] = 0.99
        assert config.risk.priority_for("car", "VEHICLE") == pytest.approx(0.99)

    def test_category_fallback_when_class_not_listed(self, config) -> None:
        value = config.risk.priority_for("totally_unlisted", "VULNERABLE")
        assert value == pytest.approx(config.risk.priority["VULNERABLE"])


class TestSceneRisk:
    def test_empty_scene_is_low_risk_and_fully_confident(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        world = build_world([], analyzer, risk_engine, ego_path)
        assert world.scene_risk.level is RiskLevel.LOW
        assert world.scene_risk.perception_confidence == pytest.approx(1.0)

    def test_scene_takes_the_worst_object_not_the_average(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        """One critical pedestrian must not be diluted by harmless traffic."""
        harmless = [
            make_object(track_id=10 + i, class_name="car",
                        bbox=(780, 250 + i * 8, 860, 300 + i * 8), distance=70.0)
            for i in range(8)
        ]
        dangerous = make_object(
            track_id=99, class_name="person", bbox=(430, 250, 510, 450),
            distance=5.0, closing_speed=9.0,
        )
        world = build_world(
            [*harmless, dangerous], analyzer, risk_engine, ego_path
        )
        assert world.scene_risk.level is RiskLevel.CRITICAL

    def test_traffic_density_bands(
        self, risk_engine, analyzer, ego_path, config
    ) -> None:
        def density(count: int) -> str:
            objects = [
                make_object(track_id=i, class_name="car",
                            bbox=(40 + i * 60, 300, 90 + i * 60, 380), distance=30.0)
                for i in range(count)
            ]
            return build_world(
                objects, analyzer, risk_engine, ego_path
            ).scene_risk.traffic_density

        assert density(2) == "sparse"
        assert density(config.risk.density.moderate) == "moderate"
        assert density(config.risk.density.dense + 2) == "dense"

    def test_signage_does_not_count_toward_density(
        self, risk_engine, analyzer, ego_path, config
    ) -> None:
        signs = [
            make_object(track_id=i, class_name="traffic_sign",
                        bbox=(40 + i * 60, 200, 80 + i * 60, 250), distance=30.0)
            for i in range(config.risk.density.dense + 3)
        ]
        world = build_world(signs, analyzer, risk_engine, ego_path)
        assert world.scene_risk.traffic_density == "sparse"

    def test_min_ttc_is_reported(self, risk_engine, analyzer, ego_path) -> None:
        near = make_object(track_id=1, class_name="car",
                           bbox=(430, 300, 530, 420), distance=10.0,
                           closing_speed=5.0)
        far = make_object(track_id=2, class_name="car",
                          bbox=(440, 260, 520, 300), distance=40.0,
                          closing_speed=5.0)
        world = build_world([near, far], analyzer, risk_engine, ego_path)
        assert world.scene_risk.min_ttc == pytest.approx(2.0, abs=0.5)


class TestConfidenceHandling:
    def test_low_confidence_detections_reduce_scene_confidence(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        confident = make_object(
            track_id=1, class_name="car", bbox=(430, 300, 530, 420),
            distance=20.0, confidence=0.95,
        )
        world_high = build_world([confident], analyzer, risk_engine, ego_path)

        unsure = make_object(
            track_id=2, class_name="car", bbox=(430, 300, 530, 420),
            distance=20.0, confidence=0.30,
        )
        world_low = build_world([unsure], analyzer, risk_engine, ego_path)

        assert (world_low.scene_risk.perception_confidence
                < world_high.scene_risk.perception_confidence)

    def test_unknown_object_in_path_caps_confidence(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        unknown = make_object(
            class_name="unknown", bbox=(430, 300, 530, 440), distance=15.0,
            confidence=0.8,
        )
        world = build_world([unknown], analyzer, risk_engine, ego_path)
        assert world.scene_risk.perception_confidence <= 0.5

    def test_unknown_class_raises_risk_rather_than_ignoring_it(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        """Not knowing what something is, is itself a hazard."""
        unknown = make_object(
            track_id=1, class_name="unknown", bbox=(430, 280, 540, 450),
            distance=12.0, confidence=0.4,
        )
        build_world([unknown], analyzer, risk_engine, ego_path)
        assert unknown.risk_level >= RiskLevel.MEDIUM


class TestLateralClearance:
    def test_open_road_reports_both_sides_clear(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        world = build_world([], analyzer, risk_engine, ego_path)
        assert world.scene_risk.left_clear and world.scene_risk.right_clear

    def test_vehicle_on_the_right_blocks_that_side(
        self, risk_engine, analyzer, ego_path
    ) -> None:
        blocker = make_object(
            track_id=1, class_name="car", bbox=(700, 330, 950, 510), distance=8.0
        )
        world = build_world([blocker], analyzer, risk_engine, ego_path)
        assert not world.scene_risk.right_clear
        assert world.scene_risk.left_clear


class TestPositionClassification:
    def test_lateral_positions(self, analyzer, risk_engine, ego_path) -> None:
        left = make_object(track_id=1, class_name="car",
                           bbox=(40, 330, 180, 450), distance=15.0)
        center = make_object(track_id=2, class_name="car",
                             bbox=(430, 330, 530, 450), distance=15.0)
        right = make_object(track_id=3, class_name="car",
                            bbox=(800, 330, 930, 450), distance=15.0)
        build_world([left, center, right], analyzer, risk_engine, ego_path)
        assert left.relative_position is LateralPosition.LEFT
        assert center.relative_position is LateralPosition.CENTER
        assert right.relative_position is LateralPosition.RIGHT

    def test_object_in_corridor_is_marked_in_path(
        self, analyzer, risk_engine, ego_path
    ) -> None:
        obj = make_object(class_name="car", bbox=(410, 300, 550, 450),
                          distance=15.0)
        build_world([obj], analyzer, risk_engine, ego_path)
        assert obj.path_relation in (
            PathRelation.IN_PATH, PathRelation.DIRECTLY_AHEAD
        )
