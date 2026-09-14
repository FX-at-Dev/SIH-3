"""Decision-engine tests: the scenario-to-recommendation contract.

These encode the behaviour the system promises. If one of these fails, the
system is recommending something different from what it claims.
"""

from __future__ import annotations

import pytest

from src.core.types import (
    ActionType,
    Direction,
    MovementStatus,
    PotholeInfo,
    RiskLevel,
)
from tests.conftest import build_world, make_object


def decide(objects, analyzer, risk_engine, decision_engine, ego_path):
    world = build_world(objects, analyzer, risk_engine, ego_path)
    return decision_engine.decide(world), world


class TestCruise:
    def test_clear_road_cruises(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        rec, _ = decide([], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.CRUISE
        assert rec.target_speed_kmh > 0
        assert rec.reason

    def test_distant_object_off_path_cruises(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        car = make_object(
            track_id=1, class_name="car", bbox=(770, 250, 870, 330), distance=45.0
        )
        rec, _ = decide([car], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.CRUISE


class TestStop:
    def test_pedestrian_in_path_with_critical_ttc_stops(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """The headline safety case: someone in front, closing fast."""
        person = make_object(
            class_name="person", bbox=(430, 240, 500, 430),
            distance=7.0, closing_speed=6.0,
        )
        rec, _ = decide([person], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP
        assert rec.target_speed_kmh == 0
        assert rec.risk is RiskLevel.CRITICAL

    def test_animal_blocking_path_stops(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        cow = make_object(
            track_id=3, class_name="cow", bbox=(400, 280, 560, 460), distance=9.0
        )
        rec, _ = decide([cow], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP
        assert "cow" in rec.reason.lower()

    def test_no_safe_side_stops(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """Blocked ahead with both sides occupied leaves no trajectory."""
        objects = [
            make_object(track_id=1, class_name="truck",
                        bbox=(330, 230, 590, 450), distance=9.0),
            make_object(track_id=2, class_name="car",
                        bbox=(660, 320, 940, 500), distance=8.0),
            make_object(track_id=3, class_name="car",
                        bbox=(20, 320, 300, 500), distance=8.0),
        ]
        rec, _ = decide(objects, analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP

    def test_unknown_obstacle_very_close_stops(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """An unclassified object close ahead cannot be safely driven around."""
        unknown = make_object(
            track_id=9, class_name="unknown", bbox=(400, 270, 560, 460),
            distance=4.0, confidence=0.41,
        )
        rec, _ = decide([unknown], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP

    def test_stop_never_downgraded_by_hysteresis(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """Escalation to STOP takes effect on the very first frame."""
        decide([], analyzer, risk_engine, decision_engine, ego_path)
        person = make_object(
            class_name="person", bbox=(430, 240, 500, 430),
            distance=5.0, closing_speed=8.0,
        )
        rec, _ = decide([person], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP


class TestYield:
    def test_pedestrian_predicted_to_cross_yields(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        person = make_object(
            track_id=2, class_name="person", bbox=(700, 250, 760, 420),
            distance=16.0, velocity_px=(-60.0, 10.0), closing_speed=0.1,
        )
        person.time_to_path_entry = 1.8
        person.movement = MovementStatus.CROSSING
        rec, _ = decide([person], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.YIELD
        assert rec.target_speed_kmh > 0
        assert "1.8" in rec.reason or "path" in rec.reason.lower()

    def test_unpassable_obstacle_ahead_yields(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """A cart filling the lane with a blocked side slows rather than stops."""
        cart = make_object(
            track_id=6, class_name="bullock_cart",
            bbox=(370, 290, 590, 460), distance=16.0,
        )
        blocker = make_object(
            track_id=7, class_name="car", bbox=(660, 330, 940, 500), distance=15.0
        )
        rec, _ = decide(
            [cart, blocker], analyzer, risk_engine, decision_engine, ego_path
        )
        assert rec.action in (ActionType.YIELD, ActionType.CREEP)
        assert rec.action is not ActionType.CRUISE


class TestNudge:
    def test_partial_obstruction_with_clear_side_nudges(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        pothole = make_object(
            track_id=4, class_name="pothole", bbox=(505, 425, 615, 468),
            distance=11.0,
            pothole=PotholeInfo(
                severity=RiskLevel.MEDIUM, estimated_width_m=0.5,
                ego_path_overlap=0.0, size_label="medium",
            ),
        )
        rec, _ = decide([pothole], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.NUDGE
        assert rec.direction is not None

    def test_nudge_direction_avoids_the_blocked_side(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """A direction is only offered when that side is actually clearer."""
        obstacle = make_object(
            track_id=1, class_name="debris", bbox=(500, 420, 610, 470),
            distance=12.0,
        )
        right_blocker = make_object(
            track_id=2, class_name="car", bbox=(700, 330, 950, 500), distance=10.0
        )
        rec, _ = decide(
            [obstacle, right_blocker], analyzer, risk_engine, decision_engine, ego_path
        )
        assert not (
            rec.action is ActionType.NUDGE and rec.direction is Direction.RIGHT
        )

    def test_no_nudge_past_close_pedestrian(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """Never squeeze past a vulnerable road user: they may step sideways."""
        person = make_object(
            class_name="person", bbox=(520, 300, 580, 450), distance=8.0
        )
        rec, _ = decide([person], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is not ActionType.NUDGE


class TestCreep:
    def test_dense_traffic_creeps(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        objects = [
            make_object(
                track_id=10 + i,
                class_name="person" if i % 3 else "motorcycle",
                bbox=(60 + i * 70, 300, 105 + i * 70, 420),
                distance=22.0 + i, confidence=0.62,
            )
            for i in range(12)
        ]
        rec, _ = decide(objects, analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.CREEP
        assert rec.target_speed_kmh == pytest.approx(5.0, abs=0.01)
        assert "dense" in rec.reason.lower()

    def test_creep_is_not_used_when_risk_is_critical(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """CREEP must never substitute for STOP under a critical risk."""
        crowd = [
            make_object(
                track_id=20 + i, class_name="person",
                bbox=(60 + i * 70, 300, 105 + i * 70, 420),
                distance=20.0 + i, confidence=0.6,
            )
            for i in range(12)
        ]
        imminent = make_object(
            track_id=99, class_name="person", bbox=(430, 240, 500, 440),
            distance=4.0, closing_speed=9.0,
        )
        rec, _ = decide(
            [*crowd, imminent], analyzer, risk_engine, decision_engine, ego_path
        )
        assert rec.action is ActionType.STOP


class TestConfidenceAwareBehaviour:
    def test_low_confidence_prevents_cruise(
        self, analyzer, risk_engine, decision_engine, ego_path, config
    ) -> None:
        """Weak perception makes the system more conservative, not less."""
        vague = make_object(
            track_id=1, class_name="unknown", bbox=(470, 330, 560, 420),
            distance=30.0, confidence=0.20,
        )
        rec, _ = decide([vague], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is not ActionType.CRUISE


class TestExplanations:
    def test_explanation_reflects_the_actual_objects(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        """Explanations are generated from state, not canned per action."""
        cow = make_object(
            track_id=3, class_name="cow", bbox=(400, 280, 560, 460), distance=9.0
        )
        rec_cow, _ = decide([cow], analyzer, risk_engine, decision_engine, ego_path)

        decision_engine.reset()
        person = make_object(
            track_id=4, class_name="person", bbox=(420, 250, 500, 450),
            distance=7.0, closing_speed=6.0,
        )
        rec_person, _ = decide(
            [person], analyzer, risk_engine, decision_engine, ego_path
        )

        assert rec_cow.reason != rec_person.reason
        assert "cow" in rec_cow.reason.lower()
        assert "pedestrian" in rec_person.reason.lower()

    def test_every_recommendation_has_an_explanation(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        scenarios = [
            [],
            [make_object(class_name="cow", bbox=(400, 280, 560, 460), distance=9.0)],
            [make_object(class_name="person", bbox=(430, 240, 500, 430),
                         distance=7.0, closing_speed=6.0)],
        ]
        for objects in scenarios:
            decision_engine.reset()
            rec, _ = decide(objects, analyzer, risk_engine, decision_engine, ego_path)
            assert rec.reason.strip(), f"empty explanation for {rec.action}"
            assert len(rec.reason) > 20


class TestHysteresis:
    def test_escalation_is_immediate(self, config, ego_path) -> None:
        from src.decision.decision_engine import DecisionEngine

        engine = DecisionEngine(config, ego_path)
        assert engine.cfg.immediate_escalation

    def test_deescalation_requires_persistence(
        self, analyzer, risk_engine, decision_engine, ego_path, config
    ) -> None:
        """Dropping to a calmer action takes several consistent frames."""
        person = make_object(
            class_name="person", bbox=(430, 240, 500, 430),
            distance=6.0, closing_speed=7.0,
        )
        rec, _ = decide([person], analyzer, risk_engine, decision_engine, ego_path)
        assert rec.action is ActionType.STOP

        first, _ = decide([], analyzer, risk_engine, decision_engine, ego_path)
        assert first.action is ActionType.STOP

        for _ in range(config.decision.hysteresis_frames + 1):
            final, _ = decide([], analyzer, risk_engine, decision_engine, ego_path)
        assert final.action is ActionType.CRUISE


class TestOutputContract:
    def test_exactly_one_primary_action(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        rec, _ = decide([], analyzer, risk_engine, decision_engine, ego_path)
        assert isinstance(rec.action, ActionType)
        assert rec.action in set(ActionType)

    def test_direction_only_present_for_nudge(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        for objects in (
            [],
            [make_object(class_name="cow", bbox=(400, 280, 560, 460), distance=9.0)],
        ):
            decision_engine.reset()
            rec, _ = decide(objects, analyzer, risk_engine, decision_engine, ego_path)
            if rec.action is not ActionType.NUDGE:
                assert rec.direction is None

    def test_serialises_cleanly(
        self, analyzer, risk_engine, decision_engine, ego_path
    ) -> None:
        import json

        rec, _ = decide([], analyzer, risk_engine, decision_engine, ego_path)
        payload = json.loads(json.dumps(rec.to_dict()))
        assert payload["action"] in {a.value for a in ActionType}
        assert "reason" in payload
