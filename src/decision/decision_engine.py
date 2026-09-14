"""Decision engine: WorldState in, DrivingRecommendation out.

The five states form a caution hierarchy (CRUISE < NUDGE < YIELD < CREEP <
STOP), but the engine does not simply walk up that ladder. It evaluates the
situation against ten considerations - collision risk, TTC, object
vulnerability, path obstruction, predicted trajectory, available lateral space,
perception confidence, traffic density, pothole severity, and progress - and
each candidate action must be *justified* by the state rather than merely
permitted by a threshold.

Two rules sit above everything:

1. **Safety dominates progress.** Where the evidence is ambiguous the engine
   takes the more conservative action.
2. **CREEP is not a substitute for STOP.** Slow movement is a response to
   *uncertainty*, never to a critical collision risk. Where a collision is
   imminent the answer is STOP, however tempting a slow crawl might look.

Output is advisory. Nothing here actuates a vehicle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.types import (
    ActionType,
    Direction,
    DrivingRecommendation,
    MovementStatus,
    PathRelation,
    RiskLevel,
    SemanticCategory,
    TrackedObject,
    WorldState,
)
from ..geometry.ego_path import EgoPath
from .recommendation import RecommendationBuilder, describe_object

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["DecisionEngine"]


@dataclass(slots=True)
class _Candidate:
    """An action the engine is considering, with its justification."""

    action: ActionType
    recommendation: DrivingRecommendation
    priority: int


class DecisionEngine:
    """Selects one recommendation per frame."""

    def __init__(self, config: "Config", ego_path: Optional["EgoPath"] = None) -> None:
        self.config = config
        self.cfg = config.decision
        self.risk_cfg = config.risk
        self.builder = RecommendationBuilder(config.decision)
        self.ego_path = ego_path

        self._current: Optional[DrivingRecommendation] = None
        self._pending: Optional[DrivingRecommendation] = None
        self._pending_count = 0
        self._change_count = 0
        self._nudge_target_id: Optional[int] = None

    # ---- public API -------------------------------------------------------

    def decide(self, world: WorldState) -> DrivingRecommendation:
        """Produce the recommendation for this frame."""
        raw = self._evaluate(world)
        raw.frame_index = world.frame_index
        raw.timestamp = world.timestamp
        return self._stabilise(raw)

    def reset(self) -> None:
        self._current = None
        self._pending = None
        self._pending_count = 0
        self._change_count = 0

    @property
    def change_count(self) -> int:
        """How many times the recommendation has changed this session."""
        return self._change_count

    # ---- core rule evaluation ---------------------------------------------

    def _evaluate(self, world: WorldState) -> DrivingRecommendation:
        scene = world.scene_risk
        confidence = scene.perception_confidence

        # ================= 1. STOP conditions ==============================
        stop = self._check_stop(world)
        if stop is not None:
            return stop

        # ================= 2. NUDGE conditions =============================
        nudge = self._check_nudge(world)

        # ================= 3. YIELD conditions =============================
        yield_rec = self._check_yield(
            world, skip_track_id=self._nudge_target_id if nudge else None
        )

        # ================= 4. CREEP conditions =============================
        creep = self._check_creep(world)

        candidates = [c for c in (creep, yield_rec, nudge) if c is not None]
        if candidates:
            return max(candidates, key=lambda r: r.action.caution_rank)

        # ================= 5. CRUISE (default) =============================
        if confidence < self.cfg.min_cruise_confidence and world.objects:
            return self.builder.creep(
                risk=RiskLevel.max(scene.level, RiskLevel.MEDIUM),
                confidence=confidence,
                drivers=[
                    f"perception confidence {confidence:.2f} below the "
                    f"{self.cfg.min_cruise_confidence:.2f} threshold for normal speed"
                ],
                frame_index=world.frame_index,
            )

        return self.builder.cruise(
            risk=scene.level,
            confidence=confidence,
            object_count=len(world.objects),
            frame_index=world.frame_index,
        )

    # ---- STOP -------------------------------------------------------------

    def _check_stop(self, world: WorldState) -> Optional[DrivingRecommendation]:
        """Stop when continuing would be unsafe."""
        scene = world.scene_risk
        confidence = scene.perception_confidence

        critical = [o for o in world.objects if o.risk_level is RiskLevel.CRITICAL]
        if critical:
            worst = min(
                critical,
                key=lambda o: (
                    o.ttc if o.ttc is not None else 999.0,
                    o.estimated_distance if o.estimated_distance is not None else 999.0,
                ),
            )
            in_path = worst.ego_path_overlap >= self.risk_cfg.in_path_overlap
            imminent = worst.ttc is not None and worst.ttc < self.risk_cfg.ttc.critical

            if in_path or imminent:
                detail = worst.risk_reasons[0] if worst.risk_reasons else "critical risk"
                cause = (
                    f"Critical collision risk: {describe_object(worst)} - {detail}. "
                )
                if not scene.left_clear and not scene.right_clear:
                    cause += "No safe avoidance path is available on either side."
                else:
                    cause += "Insufficient margin for a safe avoidance manoeuvre."
                return self.builder.stop(
                    risk=RiskLevel.CRITICAL,
                    confidence=confidence,
                    cause=cause,
                    factors=worst.risk_reasons or ["critical collision risk"],
                    ttc=worst.ttc,
                    frame_index=world.frame_index,
                )

        for obj in world.objects:
            if not obj.is_vulnerable:
                continue
            if obj.path_relation not in (
                PathRelation.DIRECTLY_AHEAD, PathRelation.IN_PATH
            ):
                continue
            distance = obj.estimated_distance
            if distance is None or distance >= self.risk_cfg.distance.high:
                continue
            if obj.corridor_obstruction < self.risk_cfg.blocking_overlap:
                continue

            kind = "Animal" if obj.category is SemanticCategory.ANIMAL else "Pedestrian"
            return self.builder.stop(
                risk=RiskLevel.CRITICAL,
                confidence=confidence,
                cause=(
                    f"{kind} directly in the vehicle path: "
                    f"{describe_object(obj)} at an estimated {distance:.1f} m, "
                    f"occupying {obj.corridor_obstruction * 100:.0f}% of the corridor. "
                    "Stopping is the only safe response at this range."
                ),
                factors=[f"{describe_object(obj, False)} blocking path", "close range"],
                ttc=obj.ttc,
                frame_index=world.frame_index,
            )

        blockers = [
            o for o in world.objects
            if o.corridor_obstruction >= self.risk_cfg.blocking_overlap
        ]
        if blockers and not scene.left_clear and not scene.right_clear:
            nearest = min(
                blockers,
                key=lambda o: o.estimated_distance if o.estimated_distance else 999.0,
            )
            distance = nearest.estimated_distance
            if distance is None or distance < self.risk_cfg.distance.high:
                return self.builder.stop(
                    risk=RiskLevel.max(scene.level, RiskLevel.HIGH),
                    confidence=confidence,
                    cause=(
                        f"Road blocked by {describe_object(nearest)}"
                        + (f" at an estimated {distance:.1f} m" if distance else "")
                        + ". No clear space detected on either side; no safe "
                        "trajectory available."
                    ),
                    factors=["path blocked", "no lateral clearance"],
                    ttc=nearest.ttc,
                    frame_index=world.frame_index,
                )

        for pothole in world.potholes:
            if pothole.pothole is None:
                continue
            if pothole.pothole.severity is not RiskLevel.HIGH:
                continue
            if pothole.corridor_obstruction < 0.6:
                continue
            if scene.left_clear or scene.right_clear:
                continue
            return self.builder.stop(
                risk=RiskLevel.HIGH,
                confidence=confidence,
                cause=(
                    "Large pothole spans the available path "
                    f"({pothole.corridor_obstruction * 100:.0f}% corridor blocked, "
                    f"estimated width {pothole.pothole.estimated_width_m:.1f} m) "
                    "with no clear space to either side."
                ),
                factors=["severe pothole", "no bypass available"],
                frame_index=world.frame_index,
            )

        for obj in world.objects:
            if obj.category is not SemanticCategory.UNKNOWN:
                continue
            if obj.ego_path_overlap < self.risk_cfg.in_path_overlap:
                continue
            distance = obj.estimated_distance
            if distance is not None and distance < self.risk_cfg.distance.critical:
                return self.builder.stop(
                    risk=RiskLevel.max(scene.level, RiskLevel.HIGH),
                    confidence=confidence,
                    cause=(
                        f"Unidentified obstacle at an estimated {distance:.1f} m "
                        f"in the vehicle path (detection confidence "
                        f"{obj.confidence:.2f}). Object class could not be "
                        "determined, so a safe avoidance manoeuvre cannot be "
                        "planned."
                    ),
                    factors=["unknown obstacle", "very close", "low confidence"],
                    frame_index=world.frame_index,
                )

        return None

    # ---- YIELD ------------------------------------------------------------

    def _check_yield(
        self, world: WorldState, skip_track_id: Optional[int] = None
    ) -> Optional[DrivingRecommendation]:
        """Give priority to a road user about to enter or cross the path."""
        scene = world.scene_risk
        best: Optional[TrackedObject] = None
        best_urgency = float("inf")

        for obj in world.objects:
            if obj.category is SemanticCategory.STATIC:
                continue
            if skip_track_id is not None and obj.track_id == skip_track_id:
                continue

            entering = (
                obj.time_to_path_entry is not None
                and obj.time_to_path_entry > 0.0
                and obj.time_to_path_entry <= self.config.prediction.horizon_seconds
            )
            crossing = obj.movement is MovementStatus.CROSSING
            converging = obj.path_relation is PathRelation.CROSSING_PATH

            approaching_in_path = (
                obj.is_vulnerable
                and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap
                and obj.risk_level >= RiskLevel.MEDIUM
            )

            occupying_path = (
                obj.ego_path_overlap >= self.risk_cfg.in_path_overlap
                and obj.path_relation in (
                    PathRelation.DIRECTLY_AHEAD, PathRelation.IN_PATH
                )
                and obj.estimated_distance is not None
                and obj.estimated_distance < self.risk_cfg.distance.medium
            )

            if not (entering or crossing or converging
                    or approaching_in_path or occupying_path):
                continue

            if not (obj.is_vulnerable or obj.risk_level >= RiskLevel.MEDIUM):
                continue

            urgency = (
                obj.time_to_path_entry
                if obj.time_to_path_entry is not None
                else (obj.ttc if obj.ttc is not None else 10.0)
            )
            if urgency < best_urgency:
                best_urgency = urgency
                best = obj

        if best is None:
            return None

        return self.builder.yield_to(
            subject=best,
            risk=RiskLevel.max(scene.level, RiskLevel.MEDIUM),
            confidence=scene.perception_confidence,
            frame_index=world.frame_index,
        )

    # ---- CREEP ------------------------------------------------------------

    def _check_creep(self, world: WorldState) -> Optional[DrivingRecommendation]:
        """Crawl through constrained or poorly-understood situations.

        CREEP is the answer to *uncertainty*, not to *danger*. Anything with a
        critical collision risk has already returned STOP above.
        """
        scene = world.scene_risk
        drivers: list[str] = []

        if scene.traffic_density == "dense":
            counts = world.counts()
            drivers.append(
                f"dense mixed traffic ({counts['objects']} objects: "
                f"{counts['pedestrians']} pedestrian(s), "
                f"{counts['vehicles']} vehicle(s), {counts['animals']} animal(s))"
            )

        if scene.perception_confidence < self.risk_cfg.low_confidence:
            drivers.append(
                f"low perception confidence ({scene.perception_confidence:.2f})"
            )

        unknowns = [
            o for o in world.objects
            if o.category is SemanticCategory.UNKNOWN
            and o.ego_path_overlap >= self.risk_cfg.in_path_overlap
        ]
        if unknowns:
            drivers.append(
                f"{len(unknowns)} unidentified object(s) in the path"
            )

        erratic = [
            o for o in world.objects
            if o.category in (SemanticCategory.ANIMAL, SemanticCategory.VULNERABLE)
            and o.movement in (MovementStatus.MOVING, MovementStatus.CROSSING)
            and o.estimated_distance is not None
            and o.estimated_distance < self.risk_cfg.distance.medium
        ]
        if len(erratic) >= 2:
            drivers.append(
                f"{len(erratic)} vulnerable road users moving unpredictably nearby"
            )

        if not scene.left_clear and not scene.right_clear and not scene.blocking_track_ids:
            drivers.append("narrow passage with obstructions on both sides")

        for pothole in world.potholes:
            if pothole.ego_path_overlap < self.risk_cfg.in_path_overlap:
                continue
            if pothole.pothole is None:
                continue
            if pothole.pothole.severity < RiskLevel.MEDIUM:
                continue
            if scene.left_clear or scene.right_clear:
                continue
            distance = pothole.estimated_distance
            drivers.append(
                f"{pothole.pothole.severity.value.lower()}-severity pothole in "
                f"the path"
                + (f" at an estimated {distance:.1f} m" if distance else "")
                + " with no clear space to either side"
            )
            break

        if not drivers:
            return None

        return self.builder.creep(
            risk=RiskLevel.max(scene.level, RiskLevel.MEDIUM),
            confidence=scene.perception_confidence,
            drivers=drivers,
            frame_index=world.frame_index,
        )

    # ---- NUDGE ------------------------------------------------------------

    def _check_nudge(self, world: WorldState) -> Optional[DrivingRecommendation]:
        """Offer a small lateral adjustment around a partial obstruction.

        A direction is only ever recommended when that side is estimated to be
        genuinely clearer than the other. If neither side is safe the engine
        does not nudge - a more conservative candidate will win instead.
        """
        scene = world.scene_risk
        self._nudge_target_id = None

        candidates = [
            obj for obj in world.objects
            if obj.category is not SemanticCategory.STATIC
            and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap
            and obj.corridor_obstruction < 0.80
        ]
        if not candidates:
            return None

        obstacle = max(candidates, key=lambda o: o.corridor_obstruction)

        if obstacle.is_vulnerable:
            distance = obstacle.estimated_distance
            if distance is None or distance < self.risk_cfg.distance.medium:
                return None

        if self.ego_path is not None:
            left_gap_px, right_gap_px = self.ego_path.lateral_gap_px(obstacle.bbox)
        else:
            left_gap_px = right_gap_px = 0.0
        left_gap_px = max(0.0, left_gap_px)
        right_gap_px = max(0.0, right_gap_px)
        total_gap = left_gap_px + right_gap_px
        if total_gap > 1e-6:
            left_inner = left_gap_px / total_gap
            right_inner = right_gap_px / total_gap
        else:
            left_inner = right_inner = 0.5

        left_score = 0.6 * left_inner + 0.4 * scene.left_score
        right_score = 0.6 * right_inner + 0.4 * scene.right_score

        left_free_m, right_free_m = (0.0, 0.0)
        if self.ego_path is not None:
            left_free_m, right_free_m = self.ego_path.outside_room_m(obstacle.bbox.y2)

        if left_score >= right_score:
            direction, score, clear = Direction.LEFT, left_score, scene.left_clear
            other, side_free_m = right_score, left_free_m * scene.left_score
        else:
            direction, score, clear = Direction.RIGHT, right_score, scene.right_clear
            other, side_free_m = left_score, right_free_m * scene.right_score

        if not clear or (score - other) < self.cfg.nudge_margin:
            return None

        chosen_gap = left_gap_px if direction is Direction.LEFT else right_gap_px
        if self.ego_path is not None:
            metres_per_px = self.ego_path.pixels_to_metres_at(obstacle.bbox.y2)
            if metres_per_px > 0.0:
                usable_m = chosen_gap * metres_per_px
                if clear:
                    usable_m += min(
                        self.config.ego_path.lane_width_m / 2.0,
                        side_free_m,
                    )
                required_m = (
                    self.config.ego_path.vehicle_width_m
                    + self.config.ego_path.nudge_clearance_m
                )
                if usable_m < required_m:
                    logger.debug(
                        "NUDGE %s declined: %.2f m usable, %.2f m required.",
                        direction.value, usable_m, required_m,
                    )
                    return None

        self._nudge_target_id = obstacle.track_id
        return self.builder.nudge(
            direction=direction,
            obstacle=obstacle,
            risk=RiskLevel.max(scene.level, RiskLevel.MEDIUM),
            confidence=scene.perception_confidence,
            clearance_score=score,
            frame_index=world.frame_index,
        )

    # ---- temporal stabilisation -------------------------------------------

    def _stabilise(
        self, candidate: DrivingRecommendation
    ) -> DrivingRecommendation:
        """Suppress flicker without ever delaying an escalation.

        A noisy detector can make the raw decision oscillate frame to frame. We
        require a new action to persist before adopting it - except when it is
        *more* conservative than the current one, which takes effect
        immediately. Safety is never held back by hysteresis.
        """
        if self._current is None:
            self._current = candidate
            self._pending = None
            self._pending_count = 0
            return candidate

        if candidate.same_as(self._current):
            self._current = candidate
            self._pending = None
            self._pending_count = 0
            return candidate

        escalating = candidate.action.caution_rank > self._current.action.caution_rank
        if escalating and self.cfg.immediate_escalation:
            logger.debug(
                "Immediate escalation %s -> %s",
                self._current.label, candidate.label,
            )
            self._current = candidate
            self._pending = None
            self._pending_count = 0
            self._change_count += 1
            return candidate

        if self._pending is not None and candidate.same_as(self._pending):
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1

        if self._pending_count >= self.cfg.hysteresis_frames:
            self._current = candidate
            self._pending = None
            self._pending_count = 0
            self._change_count += 1
            return candidate

        return self._current
