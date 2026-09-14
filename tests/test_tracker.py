"""Tracker tests.

The property that matters is identity persistence: the same physical object
must keep the same ID across frames. Without that, velocity, TTC and trajectory
prediction are all meaningless.
"""

from __future__ import annotations

import pytest

from src.core.classes import category_for
from src.core.types import BBox, Detection, DetectionSource
from src.perception.tracker import KalmanIoUTracker, create_tracker


def detection(
    class_name: str,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
    frame_index: int = 0,
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=BBox(*bbox),
        category=category_for(class_name),
        source=DetectionSource.MOCK,
        frame_index=frame_index,
    )


@pytest.fixture
def tracker(config) -> KalmanIoUTracker:
    return KalmanIoUTracker(config)


class TestIdentityPersistence:
    def test_stationary_object_keeps_one_id(
        self, tracker: KalmanIoUTracker
    ) -> None:
        ids = set()
        for i in range(20):
            objects = tracker.update(
                [detection("car", (400, 300, 500, 380), frame_index=i)],
                timestamp=i / 20.0, frame_index=i,
            )
            ids.update(o.track_id for o in objects)
        assert len(ids) == 1, f"ID switched: {ids}"

    def test_moving_object_keeps_one_id(self, tracker: KalmanIoUTracker) -> None:
        ids = set()
        for i in range(25):
            x = 200 + i * 9
            objects = tracker.update(
                [detection("person", (x, 300, x + 60, 430), frame_index=i)],
                timestamp=i / 20.0, frame_index=i,
            )
            ids.update(o.track_id for o in objects)
        assert len(ids) == 1, f"ID switched while moving: {ids}"

    def test_two_objects_get_distinct_stable_ids(
        self, tracker: KalmanIoUTracker
    ) -> None:
        per_class: dict[str, set[int]] = {}
        for i in range(20):
            detections = [
                detection("car", (150 + i * 4, 300, 250 + i * 4, 380), frame_index=i),
                detection("person", (600 - i * 5, 300, 650 - i * 5, 430),
                          frame_index=i),
            ]
            for obj in tracker.update(detections, i / 20.0, i):
                per_class.setdefault(obj.class_name, set()).add(obj.track_id)

        assert set(per_class) == {"car", "person"}
        for name, ids in per_class.items():
            assert len(ids) == 1, f"{name} switched IDs: {ids}"
        assert len(set().union(*per_class.values())) == 2

    def test_ids_are_not_reassigned_each_frame(
        self, tracker: KalmanIoUTracker
    ) -> None:
        """Regression guard against a tracker that just enumerates detections."""
        seen: list[int] = []
        for i in range(10):
            objects = tracker.update(
                [detection("car", (400, 300, 500, 380), frame_index=i)], i / 20.0, i
            )
            seen.extend(o.track_id for o in objects)
        assert len(set(seen)) == 1


class TestOcclusion:
    def test_track_survives_brief_occlusion(
        self, tracker: KalmanIoUTracker
    ) -> None:
        """A pedestrian hidden behind an auto-rickshaw keeps their identity."""
        for i in range(10):
            tracker.update(
                [detection("person", (300 + i * 6, 300, 360 + i * 6, 430),
                           frame_index=i)],
                i / 20.0, i,
            )
        first_id = tracker.update(
            [detection("person", (360, 300, 420, 430))], 0.5, 10
        )[0].track_id

        for i in range(11, 16):
            tracker.update([], i / 20.0, i)

        objects = tracker.update(
            [detection("person", (400, 300, 460, 430), frame_index=16)], 0.8, 16
        )
        assert objects
        assert objects[0].track_id == first_id

    def test_track_retires_after_max_age(
        self, tracker: KalmanIoUTracker, config
    ) -> None:
        for i in range(6):
            tracker.update([detection("car", (400, 300, 500, 380))], i / 20.0, i)
        for i in range(6, 6 + config.tracking.max_age + 3):
            tracker.update([], i / 20.0, i)
        assert tracker.active_track_count == 0


class TestMotionEstimation:
    def test_velocity_direction_is_correct(
        self, tracker: KalmanIoUTracker
    ) -> None:
        objects = []
        for i in range(15):
            x = 200 + i * 10
            objects = tracker.update(
                [detection("car", (x, 300, x + 100, 380), frame_index=i)],
                i / 20.0, i,
            )
        assert objects
        vx, _ = objects[0].velocity_px
        assert vx > 0, "velocity should point right"

    def test_stationary_object_reports_near_zero_speed(
        self, tracker: KalmanIoUTracker
    ) -> None:
        objects = []
        for i in range(15):
            objects = tracker.update(
                [detection("car", (400, 300, 500, 380), frame_index=i)],
                i / 20.0, i,
            )
        assert objects[0].speed_px < 5.0

    def test_history_is_accumulated_and_bounded(
        self, tracker: KalmanIoUTracker, config
    ) -> None:
        objects = []
        for i in range(config.tracking.max_history + 20):
            x = 100 + i * 3
            objects = tracker.update(
                [detection("car", (x, 300, x + 80, 380), frame_index=i)],
                i / 20.0, i,
            )
        assert len(objects[0].history) > 1
        assert len(objects[0].history) <= config.tracking.max_history


class TestLifecycle:
    def test_reset_clears_tracks_and_ids(
        self, tracker: KalmanIoUTracker
    ) -> None:
        for i in range(5):
            tracker.update([detection("car", (400, 300, 500, 380))], i / 20.0, i)
        tracker.reset()
        assert tracker.active_track_count == 0
        objects = tracker.update([detection("car", (400, 300, 500, 380))], 0.0, 0)
        assert objects[0].track_id == 1

    def test_empty_detections_are_handled(
        self, tracker: KalmanIoUTracker
    ) -> None:
        assert tracker.update([], 0.0, 0) == []

    def test_class_switch_does_not_steal_identity(
        self, tracker: KalmanIoUTracker
    ) -> None:
        """A pedestrian must not inherit a car's track just by overlapping."""
        for i in range(8):
            tracker.update(
                [detection("car", (400, 300, 500, 380), frame_index=i)], i / 20.0, i
            )
        objects = tracker.update(
            [detection("person", (405, 300, 495, 380), frame_index=8)], 0.4, 8
        )
        classes = {o.class_name for o in objects}
        assert "person" in classes


class TestFactory:
    def test_factory_builds_configured_tracker(self, config) -> None:
        assert create_tracker(config).name == "kalman_iou"

    def test_disabled_tracking_falls_back_to_passthrough(self, config) -> None:
        config.tracking.enabled = False
        assert create_tracker(config).name == "passthrough"

    def test_unknown_type_rejected(self, config) -> None:
        with pytest.raises(ValueError):
            config.tracking.type = "nonexistent"
