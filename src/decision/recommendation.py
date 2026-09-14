"""Recommendation construction and explanation generation.

Every recommendation must carry an explanation built from the actual world
state that produced it. A canned string per action would be worse than useless:
it would look like reasoning while conveying nothing, and it would hide the
case where the engine fired for the wrong reason.

The builders below take the specific objects, distances and times involved and
turn them into a sentence a person can check against what they see on screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from ..core.types import (
    ActionType,
    Direction,
    DrivingRecommendation,
    RiskLevel,
    TrackedObject,
)

if TYPE_CHECKING:
    from ..core.config import DecisionConfig

__all__ = ["RecommendationBuilder", "describe_object"]


def describe_object(obj: TrackedObject, with_id: bool = True) -> str:
    """Short phrase naming an object, e.g. ``"pedestrian (ID 17)"``."""
    name = obj.class_name.replace("_", " ")
    if name == "person":
        name = "pedestrian"
    if with_id:
        return f"{name} (ID {obj.track_id})"
    return name


def _distance_phrase(obj: TrackedObject) -> str:
    """Distance phrase that always marks the value as an estimate."""
    if obj.estimated_distance is None:
        return ""
    return f" at an estimated {obj.estimated_distance:.1f} m"


class RecommendationBuilder:
    """Builds recommendations with state-derived explanations."""

    def __init__(self, config: "DecisionConfig") -> None:
        self.config = config

    # ---- one builder per action -------------------------------------------

    def cruise(
        self,
        risk: RiskLevel,
        confidence: float,
        object_count: int,
        frame_index: int = 0,
    ) -> DrivingRecommendation:
        if object_count == 0:
            reason = "No objects detected in the ego path. Road ahead appears clear."
        else:
            reason = (
                f"{object_count} object(s) tracked, none conflicting with the "
                "predicted vehicle path. No significant collision risk."
            )
        return DrivingRecommendation(
            action=ActionType.CRUISE,
            direction=None,
            target_speed_kmh=self.config.cruise_speed_kmh,
            risk=risk,
            reason=reason,
            confidence=confidence,
            factors=["path clear", "low collision risk"],
            frame_index=frame_index,
        )

    def nudge(
        self,
        direction: Direction,
        obstacle: TrackedObject,
        risk: RiskLevel,
        confidence: float,
        clearance_score: float,
        frame_index: int = 0,
    ) -> DrivingRecommendation:
        side = "left" if obstacle.relative_position.value == "left" else "right"
        if obstacle.relative_position.value == "center":
            side = "centre"

        reason = (
            f"{describe_object(obstacle).capitalize()}"
            f"{_distance_phrase(obstacle)} occupies the {side} side of the "
            f"predicted path ({obstacle.corridor_obstruction * 100:.0f}% blocked). "
            f"Space is estimated to be available on the "
            f"{direction.value.lower()}; a full stop is not required."
        )
        return DrivingRecommendation(
            action=ActionType.NUDGE,
            direction=direction,
            target_speed_kmh=self.config.nudge_speed_kmh,
            risk=risk,
            reason=reason,
            confidence=confidence,
            factors=[
                f"partial path obstruction by {describe_object(obstacle, False)}",
                f"{direction.value.lower()} clearance score "
                f"{clearance_score:.2f}",
            ],
            ttc=obstacle.ttc,
            frame_index=frame_index,
        )

    def yield_to(
        self,
        subject: TrackedObject,
        risk: RiskLevel,
        confidence: float,
        frame_index: int = 0,
    ) -> DrivingRecommendation:
        entry = subject.time_to_path_entry
        already_in_path = subject.ego_path_overlap >= 0.12 and entry in (None, 0.0)

        if already_in_path and subject.movement.value != "CROSSING":
            reason = (
                f"{describe_object(subject).capitalize()}"
                f"{_distance_phrase(subject)} is occupying the vehicle path "
                f"({subject.corridor_obstruction * 100:.0f}% of the corridor "
                "blocked). Reducing speed and holding position behind it."
            )
        else:
            if entry is not None and entry > 0.0:
                timing = f" and is predicted to enter it in {entry:.1f} s"
            elif subject.movement.value == "CROSSING":
                timing = " and is crossing the vehicle path"
            else:
                timing = " and may enter the vehicle path"

            reason = (
                f"{describe_object(subject).capitalize()}"
                f"{_distance_phrase(subject)} is approaching the ego corridor"
                f"{timing}. Reducing speed to give priority."
            )
        factors = [f"{describe_object(subject, False)} has right of way"]
        if subject.ttc is not None:
            factors.append(f"TTC {subject.ttc:.1f}s")
        return DrivingRecommendation(
            action=ActionType.YIELD,
            direction=None,
            target_speed_kmh=self.config.yield_speed_kmh,
            risk=risk,
            reason=reason,
            confidence=confidence,
            factors=factors,
            ttc=subject.ttc,
            frame_index=frame_index,
        )

    def creep(
        self,
        risk: RiskLevel,
        confidence: float,
        drivers: Sequence[str],
        frame_index: int = 0,
    ) -> DrivingRecommendation:
        driver_text = "; ".join(drivers) if drivers else "uncertain conditions"
        reason = (
            f"Proceeding at very low speed: {driver_text}. The path is "
            "technically open but conditions are not confidently readable."
        )
        return DrivingRecommendation(
            action=ActionType.CREEP,
            direction=None,
            target_speed_kmh=self.config.creep_speed_kmh,
            risk=risk,
            reason=reason,
            confidence=confidence,
            factors=list(drivers),
            frame_index=frame_index,
        )

    def stop(
        self,
        risk: RiskLevel,
        confidence: float,
        cause: str,
        factors: Sequence[str],
        ttc: Optional[float] = None,
        frame_index: int = 0,
    ) -> DrivingRecommendation:
        return DrivingRecommendation(
            action=ActionType.STOP,
            direction=None,
            target_speed_kmh=0.0,
            risk=risk,
            reason=cause,
            confidence=confidence,
            factors=list(factors),
            ttc=ttc,
            frame_index=frame_index,
        )
