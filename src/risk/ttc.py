"""Time to collision.

TTC = distance / closing_speed, which is trivial arithmetic and entirely
treacherous in practice. The hard part is knowing when the answer is
meaningless:

* closing speed at or below zero - the object is not approaching, so there is
  no collision to be timed;
* closing speed below the noise floor - a value derived from the jitter of a
  monocular distance estimate, not from real motion;
* missing distance or missing velocity - nothing to divide;
* an object that is not on a converging path at all.

In every one of those cases this module returns ``None``. It never returns
``inf`` or ``NaN``, so those values can never reach the UI or the logs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.types import RiskLevel, TrackedObject

if TYPE_CHECKING:
    from ..core.config import TTCConfig

logger = logging.getLogger(__name__)

__all__ = ["TTCCalculator", "TTCResult"]


@dataclass(slots=True)
class TTCResult:
    """A TTC value, or an explanation of why one is not available."""

    seconds: Optional[float]
    valid: bool
    reason: str = ""
    risk_level: RiskLevel = RiskLevel.LOW

    @property
    def display(self) -> str:
        """UI-safe text. Never emits inf or NaN."""
        if not self.valid or self.seconds is None:
            return "--"
        return f"{self.seconds:.1f}s"


class TTCCalculator:
    """Computes and bands time-to-collision."""

    def __init__(self, config: "TTCConfig") -> None:
        self.config = config

    def compute(
        self,
        distance_m: Optional[float],
        closing_speed_mps: Optional[float],
    ) -> TTCResult:
        """Compute TTC from a range and a closing speed.

        Args:
            distance_m: Estimated range in metres.
            closing_speed_mps: Rate the gap is shrinking, m/s. Positive means
                closing; zero or negative means stable or opening.
        """
        if distance_m is None:
            return TTCResult(None, False, "no distance estimate")
        if closing_speed_mps is None:
            return TTCResult(None, False, "no velocity estimate")

        if not math.isfinite(distance_m) or not math.isfinite(closing_speed_mps):
            logger.debug(
                "Non-finite TTC input (d=%r, v=%r); treating as unavailable.",
                distance_m, closing_speed_mps,
            )
            return TTCResult(None, False, "invalid measurement")

        if closing_speed_mps <= 0.0:
            return TTCResult(None, False, "not approaching")

        if closing_speed_mps < self.config.min_closing_speed:
            return TTCResult(None, False, "closing speed below noise floor")

        if distance_m <= 0.0:
            return TTCResult(0.0, True, "contact range", RiskLevel.CRITICAL)

        ttc = distance_m / closing_speed_mps

        if not math.isfinite(ttc):
            return TTCResult(None, False, "invalid measurement")

        if ttc > self.config.max_valid_ttc:
            return TTCResult(None, False, "beyond planning horizon")

        return TTCResult(ttc, True, "", self.classify(ttc))

    def classify(self, ttc: Optional[float]) -> RiskLevel:
        """Band a TTC value.

        Thresholds come from config and are prototype defaults, not universal
        safety standards.
        """
        if ttc is None or not math.isfinite(ttc):
            return RiskLevel.LOW
        if ttc < self.config.critical:
            return RiskLevel.CRITICAL
        if ttc < self.config.high:
            return RiskLevel.HIGH
        if ttc < self.config.medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def compute_for_track(self, obj: TrackedObject) -> TTCResult:
        """Compute TTC for a tracked object and attach it to the track."""
        result = self.compute(obj.estimated_distance, obj.closing_speed_mps)
        obj.ttc = result.seconds if result.valid else None
        return result
