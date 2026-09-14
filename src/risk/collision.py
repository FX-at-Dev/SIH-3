"""Collision / path-intersection analysis.

Decides whether an object's *predicted* path conflicts with the ego corridor,
and how urgent that conflict is. This is distinct from TTC: TTC asks "how long
until I reach it", path intersection asks "will it be where I am going".

An object can be dangerous with no valid TTC at all - a cow standing still in
the lane has zero closing speed, so TTC is undefined, yet it fully blocks the
path. Conversely a fast-approaching vehicle in the next lane has a short TTC
but no path conflict. Both signals are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..core.types import (
    MovementStatus,
    RiskLevel,
    SemanticCategory,
    TrackedObject,
)

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["CollisionAssessment", "CollisionRiskAnalyzer"]


@dataclass(slots=True)
class CollisionAssessment:
    """Result of analysing one object for path conflict."""

    track_id: int
    level: RiskLevel = RiskLevel.LOW
    score: float = 0.0
    conflict: bool = False
    time_to_conflict: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


class CollisionRiskAnalyzer:
    """Scores how likely an object is to conflict with the ego path."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.risk_cfg = config.risk
        self.in_path = config.risk.in_path_overlap
        self.blocking = config.risk.blocking_overlap

    def analyze(self, obj: TrackedObject) -> CollisionAssessment:
        """Assess one tracked object."""
        assessment = CollisionAssessment(track_id=obj.track_id)
        reasons: list[str] = []
        score = 0.0

        # ---- 1. Current occupancy of the corridor ------------------------
        overlap = obj.ego_path_overlap
        obstruction = obj.corridor_obstruction
        if obstruction >= self.blocking:
            score += 0.45
            assessment.conflict = True
            reasons.append(f"blocks {obstruction * 100:.0f}% of the ego path")
        elif overlap >= self.in_path:
            score += 0.25
            assessment.conflict = True
            reasons.append(f"partially in path ({overlap * 100:.0f}%)")

        # ---- 2. Predicted entry into the corridor ------------------------
        if obj.time_to_path_entry is not None:
            assessment.conflict = True
            entry = obj.time_to_path_entry
            assessment.time_to_conflict = entry
            if entry <= 0.0:
                pass
            elif entry < 1.0:
                score += 0.40
                reasons.append(f"predicted to enter path in {entry:.1f}s")
            elif entry < 2.0:
                score += 0.28
                reasons.append(f"predicted to enter path in {entry:.1f}s")
            else:
                score += 0.15
                reasons.append(f"may enter path in {entry:.1f}s")
        elif obj.predicted_overlap >= self.in_path > overlap:
            score += 0.18
            assessment.conflict = True
            reasons.append("predicted path approaches the ego corridor")

        # ---- 3. Proximity -------------------------------------------------
        distance = obj.estimated_distance
        if distance is not None:
            d_cfg = self.risk_cfg.distance
            if distance < d_cfg.critical:
                score += 0.35
                reasons.append(f"very close (~{distance:.1f} m)")
            elif distance < d_cfg.high:
                score += 0.20
                reasons.append(f"close (~{distance:.1f} m)")
            elif distance < d_cfg.medium:
                score += 0.08

        # ---- 4. Closing motion --------------------------------------------
        if obj.closing_speed_mps is not None and obj.closing_speed_mps > 0.5:
            score += min(0.20, obj.closing_speed_mps / 40.0)
            if obj.movement is MovementStatus.APPROACHING and assessment.conflict:
                reasons.append(
                    f"approaching at ~{obj.closing_speed_mps:.1f} m/s"
                )

        # ---- 5. Crossing motion -------------------------------------------
        if obj.movement is MovementStatus.CROSSING:
            assessment.conflict = True
            if obj.time_to_path_entry is None:
                score += 0.22
                reasons.append("crossing the vehicle path")
            else:
                reasons.append("crossing the vehicle path")

        # ---- 6. Unpredictability ------------------------------------------
        if (obj.category in (SemanticCategory.ANIMAL, SemanticCategory.VULNERABLE)
                and distance is not None
                and distance < self.risk_cfg.distance.medium):
            score += 0.12
            if not reasons:
                reasons.append("vulnerable road user nearby")

        assessment.score = float(min(1.0, score))
        assessment.level = self._level(assessment)
        assessment.reasons = reasons
        return assessment

    def _level(self, assessment: CollisionAssessment) -> RiskLevel:
        """Band the collision score, with an urgency floor from timing."""
        level = RiskLevel.from_score(assessment.score)

        if (assessment.time_to_conflict is not None
                and assessment.time_to_conflict < 1.0
                and assessment.conflict):
            level = RiskLevel.max(level, RiskLevel.HIGH)

        return level
