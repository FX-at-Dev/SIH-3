"""Trajectory-prediction tests.

Prediction is what turns "a pedestrian is at the kerb" into "a pedestrian will
be in the lane in 1.8 seconds". These tests check that the entry time is
computed, that confidence reflects how much evidence exists, and that the model
does not assume road users follow lanes.
"""

from __future__ import annotations

import pytest

from src.core.types import Point
from src.prediction.constant_velocity import ConstantVelocityPredictor
from tests.conftest import FRAME_HEIGHT, FRAME_WIDTH, make_object


@pytest.fixture
def predictor(config, ego_path) -> ConstantVelocityPredictor:
    return ConstantVelocityPredictor(config, ego_path)


class TestBasicPrediction:
    def test_stationary_object_predicts_no_movement(
        self, predictor: ConstantVelocityPredictor, ego_path
    ) -> None:
        obj = make_object(class_name="car", bbox=(430, 300, 530, 430))
        obj.ego_path_overlap = ego_path.overlap(obj.bbox)
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert trajectory.confidence > 0.5
        assert len(trajectory.points) <= 1

    def test_moving_object_produces_a_path(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        obj = make_object(
            class_name="person", bbox=(700, 300, 760, 430),
            velocity_px=(-80.0, 20.0),
        )
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert len(trajectory.points) > 1
        xs = [p.x for p in trajectory.points]
        assert xs == sorted(xs, reverse=True)

    def test_horizon_length_respects_config(
        self, predictor: ConstantVelocityPredictor, config
    ) -> None:
        obj = make_object(
            class_name="person", bbox=(500, 300, 560, 430),
            velocity_px=(-20.0, 5.0),
        )
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        if trajectory.timestamps:
            assert max(trajectory.timestamps) <= config.prediction.horizon_seconds + 1e-6


class TestPathEntry:
    def test_object_crossing_into_the_corridor_reports_entry_time(
        self, predictor: ConstantVelocityPredictor, ego_path
    ) -> None:
        """The signal that drives YIELD."""
        obj = make_object(
            class_name="person", bbox=(800, 320, 860, 450),
            velocity_px=(-150.0, 20.0),
        )
        obj.ego_path_overlap = ego_path.overlap(obj.bbox)
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert trajectory.enters_path
        assert trajectory.time_to_path_entry is not None
        assert 0.0 < trajectory.time_to_path_entry <= 3.0

    def test_object_moving_away_never_enters(
        self, predictor: ConstantVelocityPredictor, ego_path
    ) -> None:
        obj = make_object(
            class_name="person", bbox=(700, 320, 760, 450),
            velocity_px=(220.0, 10.0),
        )
        obj.ego_path_overlap = ego_path.overlap(obj.bbox)
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert trajectory.time_to_path_entry is None

    def test_object_already_in_path_reports_zero(
        self, predictor: ConstantVelocityPredictor, ego_path
    ) -> None:
        obj = make_object(class_name="cow", bbox=(430, 300, 560, 460))
        obj.ego_path_overlap = ego_path.overlap(obj.bbox)
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert trajectory.time_to_path_entry == 0.0

    def test_predict_all_annotates_tracks(
        self, predictor: ConstantVelocityPredictor, ego_path
    ) -> None:
        obj = make_object(
            class_name="person", bbox=(800, 320, 860, 450),
            velocity_px=(-150.0, 20.0),
        )
        obj.ego_path_overlap = ego_path.overlap(obj.bbox)
        predictor.predict_all([obj], FRAME_WIDTH, FRAME_HEIGHT)
        assert obj.predicted_path
        assert obj.time_to_path_entry is not None


class TestConfidence:
    def test_new_track_is_not_trusted(
        self, predictor: ConstantVelocityPredictor, config
    ) -> None:
        """Velocity from a one-frame-old track is Kalman noise, not motion."""
        obj = make_object(
            class_name="person", bbox=(700, 300, 760, 430),
            velocity_px=(-80.0, 20.0),
            hits=1,
        )
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert trajectory.confidence == 0.0
        assert not trajectory.points

    def test_confidence_grows_with_observation_length(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        def confidence_for(hits: int) -> float:
            obj = make_object(
                class_name="car", bbox=(500, 300, 600, 420),
                velocity_px=(-40.0, 10.0), hits=hits,
            )
            obj.history = [Point(500 - i * 8, 300 + i * 2) for i in range(6)]
            return predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT).confidence

        assert confidence_for(4) < confidence_for(15)

    def test_unpredictable_categories_are_trusted_less(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        """A straight-line model suits a truck better than a stray dog."""
        history = [Point(500 - i * 8, 300 + i * 2) for i in range(6)]

        vehicle = make_object(
            track_id=1, class_name="truck", bbox=(500, 300, 600, 420),
            velocity_px=(-40.0, 10.0), hits=15,
        )
        vehicle.history = list(history)

        animal = make_object(
            track_id=2, class_name="cow", bbox=(500, 300, 600, 420),
            velocity_px=(-40.0, 10.0), hits=15,
        )
        animal.history = list(history)

        assert (predictor.predict(animal, FRAME_WIDTH, FRAME_HEIGHT).confidence
                < predictor.predict(vehicle, FRAME_WIDTH, FRAME_HEIGHT).confidence)

    def test_confidence_is_bounded(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        obj = make_object(
            class_name="car", bbox=(500, 300, 600, 420),
            velocity_px=(-40.0, 10.0), hits=500,
        )
        obj.history = [Point(500 - i * 8, 300 + i * 2) for i in range(6)]
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert 0.0 <= trajectory.confidence <= 1.0


class TestNoLaneAssumption:
    def test_diagonal_motion_is_extrapolated_as_given(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        """Road users here do not follow lanes, and prediction must not force it."""
        obj = make_object(
            class_name="motorcycle", bbox=(700, 300, 760, 380),
            velocity_px=(-120.0, 60.0),
        )
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        assert len(trajectory.points) >= 2
        first, last = trajectory.points[0], trajectory.points[-1]
        assert last.x < first.x
        assert last.y > first.y

    def test_prediction_stops_at_the_frame_edge(
        self, predictor: ConstantVelocityPredictor
    ) -> None:
        obj = make_object(
            class_name="person", bbox=(60, 300, 120, 430),
            velocity_px=(-400.0, 0.0),
        )
        trajectory = predictor.predict(obj, FRAME_WIDTH, FRAME_HEIGHT)
        for point in trajectory.points:
            assert point.x > -FRAME_WIDTH * 0.25
