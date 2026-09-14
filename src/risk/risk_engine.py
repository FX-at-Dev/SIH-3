"""Risk engine: fuses every perception signal into object and scene risk.

Per object it combines TTC band, path conflict, proximity, category priority,
perception confidence and pothole severity. At scene level it aggregates those
into an overall risk band plus the context the decision engine needs: traffic
density, which side is clear, what is blocking the path, and how much the
perception stack should be trusted at all.

Design principle: safety dominates. Where two signals disagree the engine takes
the more conservative reading, and low perception confidence *raises* risk
rather than lowering it - not knowing what something is, is itself a hazard.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..core.types import (
    ObjectRisk,
    PathRelation,
    RiskLevel,
    SceneRisk,
    SemanticCategory,
    TrackedObject,
)
from ..geometry.ego_path import EgoPath
from .collision import CollisionRiskAnalyzer
from .ttc import TTCCalculator

if TYPE_CHECKING:
    from ..core.config import Config

logger = logging.getLogger(__name__)

__all__ = ["RiskEngine"]


class RiskEngine:
    """Evaluates object-level and scene-level risk for one frame."""

    def __init__(self, config: "Config", ego_path: EgoPath) -> None:
        self.config = config
        self.risk_cfg = config.risk
        self.ego_path = ego_path
        self.ttc_calculator = TTCCalculator(config.risk.ttc)
        self.collision_analyzer = CollisionRiskAnalyzer(config)

    # ---- object level -----------------------------------------------------

    def evaluate_object(self, obj: TrackedObject) -> ObjectRisk:
        """Score one object and write the result back onto the track."""
        reasons: list[str] = []

        ttc_result = self.ttc_calculator.compute_for_track(obj)
        ttc_level = ttc_result.risk_level if ttc_result.valid else RiskLevel.LOW
        if (ttc_result.valid and ttc_result.seconds is not None
                and ttc_level >= RiskLevel.MEDIUM):
            reasons.append(f"TTC {ttc_result.seconds:.1f}s")

        collision = self.collision_analyzer.analyze(obj)
        reasons.extend(collision.reasons)

        priority = self.risk_cfg.priority_for(obj.class_name, obj.category.value)

        base = collision.score
        score = base * (0.55 + 0.45 * priority)

        ttc_contribution = {
            RiskLevel.CRITICAL: 0.45,
            RiskLevel.HIGH: 0.28,
            RiskLevel.MEDIUM: 0.12,
            RiskLevel.LOW: 0.0,
        }[ttc_level]
        score = min(1.0, score + ttc_contribution)

        if obj.is_pothole and obj.pothole is not None:
            severity = obj.pothole.severity
            score = max(score, {
                RiskLevel.CRITICAL: 0.85,
                RiskLevel.HIGH: 0.62,
                RiskLevel.MEDIUM: 0.38,
                RiskLevel.LOW: 0.15,
            }[severity])
            reasons.append(f"pothole severity {severity.value}")
            if obj.pothole.source.value == "heuristic":
                reasons.append("heuristic detection (not model-confirmed)")

        if (obj.confidence < self.risk_cfg.low_confidence
                and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap):
            score = min(1.0, score + 0.12)
            reasons.append(f"low detection confidence ({obj.confidence:.2f})")

        if (obj.category is SemanticCategory.UNKNOWN
                and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap):
            score = min(1.0, score + 0.15)
            reasons.append("unidentified obstacle in path")

        level = RiskLevel.from_score(score)

        level = self._apply_floors(obj, level, ttc_level, reasons)

        obj.risk_level = level
        obj.risk_score = float(score)
        obj.risk_reasons = reasons
        return ObjectRisk(
            track_id=obj.track_id,
            level=level,
            score=float(score),
            ttc=obj.ttc,
            reasons=reasons,
        )

    def _apply_floors(
        self,
        obj: TrackedObject,
        level: RiskLevel,
        ttc_level: RiskLevel,
        reasons: list[str],
    ) -> RiskLevel:
        """Non-negotiable minimum risk levels.

        These encode that some situations are dangerous by their nature, and
        must not be averaged away by a weighted score.
        """
        if ttc_level is RiskLevel.CRITICAL:
            return RiskLevel.CRITICAL

        if (obj.is_vulnerable
                and obj.path_relation in (
                    PathRelation.DIRECTLY_AHEAD, PathRelation.IN_PATH)
                and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap):
            floor = RiskLevel.HIGH
            distance = obj.estimated_distance
            if distance is not None and distance < self.risk_cfg.distance.high:
                floor = RiskLevel.CRITICAL
                reasons.append("vulnerable road user close and in path")
            level = RiskLevel.max(level, floor)

        if obj.corridor_obstruction >= self.risk_cfg.blocking_overlap:
            distance = obj.estimated_distance
            if distance is not None and distance < self.risk_cfg.distance.high:
                level = RiskLevel.max(level, RiskLevel.HIGH)
            if distance is not None and distance < self.risk_cfg.distance.critical:
                level = RiskLevel.max(level, RiskLevel.CRITICAL)

        return level

    # ---- scene level ------------------------------------------------------

    def evaluate(
        self,
        objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> SceneRisk:
        """Evaluate every object and aggregate into scene risk."""
        scene = SceneRisk()
        object_risks: list[ObjectRisk] = []

        for obj in objects:
            object_risks.append(self.evaluate_object(obj))

        scene.object_risks = object_risks

        if not objects:
            scene.level = RiskLevel.LOW
            scene.score = 0.0
            scene.perception_confidence = 1.0
            scene.traffic_density = "sparse"
            scene.left_clear = scene.right_clear = True
            scene.reasons = ["no objects detected"]
            return scene

        # --- worst-case aggregation -------------------------------------
        worst = max(object_risks, key=lambda r: (r.level.rank, r.score))
        scene.level = worst.level
        scene.score = worst.score

        scene.critical_track_ids = [
            r.track_id for r in object_risks if r.level is RiskLevel.CRITICAL
        ]

        # --- minimum valid TTC across the scene --------------------------
        valid_ttcs = [
            obj.ttc for obj in objects
            if obj.ttc is not None and obj.ttc >= 0.0
        ]
        scene.min_ttc = min(valid_ttcs) if valid_ttcs else None

        # --- nearest in-path object and blockers --------------------------
        in_path = [
            o for o in objects
            if o.ego_path_overlap >= self.risk_cfg.in_path_overlap
        ]
        in_path_distances = [
            o.estimated_distance for o in in_path if o.estimated_distance is not None
        ]
        scene.nearest_in_path_distance = (
            min(in_path_distances) if in_path_distances else None
        )
        scene.blocking_track_ids = [
            o.track_id for o in objects
            if o.corridor_obstruction >= self.risk_cfg.blocking_overlap
        ]

        # --- traffic density ----------------------------------------------
        relevant = [
            o for o in objects if o.category is not SemanticCategory.STATIC
        ]
        count = len(relevant)
        if count >= self.risk_cfg.density.dense:
            scene.traffic_density = "dense"
        elif count >= self.risk_cfg.density.moderate:
            scene.traffic_density = "moderate"
        else:
            scene.traffic_density = "sparse"

        # --- perception confidence ----------------------------------------
        scene.perception_confidence = self._perception_confidence(objects)

        # --- lateral clearance --------------------------------------------
        obstacles = [
            (o.track_id, o.bbox) for o in objects
            if o.category is not SemanticCategory.STATIC
        ]
        clearance = self.ego_path.side_clearance(obstacles)
        scene.left_clear = clearance.left_clear
        scene.right_clear = clearance.right_clear
        scene.left_score = clearance.left_free
        scene.right_score = clearance.right_free

        scene.reasons = self._scene_reasons(scene, objects, worst)
        return scene

    def _perception_confidence(self, objects: list[TrackedObject]) -> float:
        """How much the perception output should be trusted this frame.

        Driven down by low-confidence detections, unidentified objects, and
        very short track histories. The decision engine uses this to become
        more conservative rather than less - uncertainty is a hazard.
        """
        if not objects:
            return 1.0

        total_weight = 0.0
        weighted = 0.0
        for obj in objects:
            weight = 1.0 + 2.0 * obj.ego_path_overlap
            total_weight += weight
            confidence = obj.confidence
            if obj.category is SemanticCategory.UNKNOWN:
                confidence *= 0.5
            if obj.hits < 3:
                confidence *= 0.85
            weighted += weight * confidence

        base = weighted / total_weight if total_weight > 0 else 1.0

        for obj in objects:
            if (obj.category is SemanticCategory.UNKNOWN
                    and obj.ego_path_overlap >= self.risk_cfg.in_path_overlap):
                base = min(base, 0.45)
                break

        return float(max(0.0, min(1.0, base)))

    def _scene_reasons(
        self,
        scene: SceneRisk,
        objects: list[TrackedObject],
        worst: ObjectRisk,
    ) -> list[str]:
        """Human-readable drivers of the scene risk, most important first."""
        reasons: list[str] = []
        worst_obj = next((o for o in objects if o.track_id == worst.track_id), None)

        if worst_obj is not None and worst.level >= RiskLevel.MEDIUM:
            label = worst_obj.class_name.replace("_", " ")
            detail = worst.reasons[0] if worst.reasons else "elevated risk"
            reasons.append(f"{label} (ID {worst_obj.track_id}): {detail}")

        if scene.min_ttc is not None and scene.min_ttc < self.risk_cfg.ttc.medium:
            reasons.append(f"shortest time-to-collision {scene.min_ttc:.1f}s")

        if scene.traffic_density == "dense":
            reasons.append("dense mixed traffic")

        if scene.perception_confidence < self.risk_cfg.low_confidence:
            reasons.append(
                f"low perception confidence ({scene.perception_confidence:.2f})"
            )

        if not scene.left_clear and not scene.right_clear:
            reasons.append("no clear space on either side")

        return reasons
