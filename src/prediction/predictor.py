"""Trajectory-prediction interface.

Prediction turns "there is a pedestrian at the kerb" into "that pedestrian will
be in the lane in 1.8 seconds", which is the difference between reacting and
anticipating.

Only constant-velocity prediction is implemented for the prototype. It is a
deliberately weak model - it assumes each object keeps its current heading and
speed, and it explicitly does **not** assume road users follow lanes, because
on Indian roads they frequently do not. Social-force or learned predictors
(Social-LSTM, Trajectron++) implement this same interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import Point, TrackedObject

__all__ = ["BasePredictor", "PredictedTrajectory"]


@dataclass(slots=True)
class PredictedTrajectory:
    """A predicted future path for one tracked object."""

    track_id: int
    points: list[Point] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    max_path_overlap: float = 0.0
    time_to_path_entry: Optional[float] = None
    confidence: float = 0.0

    @property
    def enters_path(self) -> bool:
        return self.time_to_path_entry is not None


class BasePredictor(ABC):
    """Contract for trajectory predictors."""

    @abstractmethod
    def predict(
        self,
        obj: TrackedObject,
        frame_width: int,
        frame_height: int,
    ) -> PredictedTrajectory:
        """Predict one object's future path over the configured horizon."""

    def predict_all(
        self,
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> dict[int, PredictedTrajectory]:
        """Predict for every object and annotate the tracks in place."""
        results: dict[int, PredictedTrajectory] = {}
        for obj in objects:
            trajectory = self.predict(obj, frame_width, frame_height)
            results[obj.track_id] = trajectory
            obj.predicted_path = trajectory.points
            obj.predicted_overlap = trajectory.max_path_overlap
            obj.time_to_path_entry = trajectory.time_to_path_entry
        return results
