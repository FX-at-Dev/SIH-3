"""Drawing primitives for the annotated video view.

Accessibility rule applied throughout: **risk is never communicated by colour
alone**. Every risk-bearing element also carries a text label and a distinct
line weight, so the display remains readable in greyscale and for colour-vision
deficiencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import cv2
import numpy as np

from ..core.types import (
    ActionType,
    DetectionSource,
    Direction,
    Point,
    RiskLevel,
    TrackedObject,
)

__all__ = [
    "COLORS",
    "action_color",
    "draw_box",
    "draw_drivable_overlay",
    "draw_ego_corridor",
    "draw_label_block",
    "draw_nudge_arrow",
    "draw_text",
    "draw_trajectory",
    "risk_color",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS: dict[str, tuple[int, int, int]] = {
    "low": (120, 200, 120),
    "medium": (60, 200, 250),
    "high": (60, 140, 255),
    "critical": (60, 60, 255),
    "cruise": (120, 220, 120),
    "nudge": (220, 200, 90),
    "yield": (60, 190, 250),
    "creep": (200, 140, 240),
    "stop": (60, 60, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (150, 150, 150),
    "dark": (32, 32, 36),
    "panel": (24, 24, 28),
    "ego": (90, 220, 190),
    "trajectory": (250, 220, 120),
    "heuristic": (170, 120, 255),
}

_RISK_COLORS = {
    RiskLevel.LOW: COLORS["low"],
    RiskLevel.MEDIUM: COLORS["medium"],
    RiskLevel.HIGH: COLORS["high"],
    RiskLevel.CRITICAL: COLORS["critical"],
}

_ACTION_COLORS = {
    ActionType.CRUISE: COLORS["cruise"],
    ActionType.NUDGE: COLORS["nudge"],
    ActionType.YIELD: COLORS["yield"],
    ActionType.CREEP: COLORS["creep"],
    ActionType.STOP: COLORS["stop"],
}

RISK_MARKERS = {
    RiskLevel.LOW: "-",
    RiskLevel.MEDIUM: "!",
    RiskLevel.HIGH: "!!",
    RiskLevel.CRITICAL: "!!!",
}


def risk_color(level: RiskLevel) -> tuple[int, int, int]:
    return _RISK_COLORS.get(level, COLORS["grey"])


def action_color(action: ActionType) -> tuple[int, int, int]:
    return _ACTION_COLORS.get(action, COLORS["grey"])


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = COLORS["white"],
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Draw text with a dark outline so it stays legible over any footage."""
    if shadow:
        cv2.putText(image, text, origin, FONT, scale, COLORS["black"],
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def draw_label_block(
    image: np.ndarray,
    lines: Sequence[str],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.42,
    above: bool = True,
) -> None:
    """Draw a compact multi-line label anchored to a box corner."""
    if not lines:
        return

    pad = 3
    line_h = int(16 * (scale / 0.42))
    widths = [text_size(line, scale)[0] for line in lines]
    box_w = max(widths) + pad * 2
    box_h = line_h * len(lines) + pad

    x, y = anchor
    height, width = image.shape[:2]

    top = y - box_h if above else y
    if top < 0:
        top = y
    top = max(0, min(top, height - box_h - 1))
    x = max(0, min(x, width - box_w - 1))

    overlay = image[top:top + box_h, x:x + box_w]
    if overlay.size > 0:
        backing = np.full_like(overlay, COLORS["black"], dtype=np.uint8)
        cv2.addWeighted(backing, 0.55, overlay, 0.45, 0, overlay)

    for i, line in enumerate(lines):
        draw_text(
            image, line,
            (x + pad, top + line_h * (i + 1) - 4),
            scale=scale,
            color=color if i == 0 else COLORS["white"],
            shadow=False,
        )


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


def draw_box(
    image: np.ndarray,
    obj: TrackedObject,
    thickness_map: dict[str, int],
    show_details: bool = True,
    font_scale: float = 0.42,
) -> None:
    """Draw one tracked object: box, risk marker, and its detail label."""
    x1, y1, x2, y2 = obj.bbox.as_int()
    color = risk_color(obj.risk_level)
    thickness = thickness_map.get(obj.risk_level.value, 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if obj.risk_level >= RiskLevel.HIGH:
        tick = max(8, min(22, (x2 - x1) // 5))
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            cv2.line(image, (cx, cy), (cx + dx * tick, cy), color, thickness + 1)
            cv2.line(image, (cx, cy), (cx, cy + dy * tick), color, thickness + 1)

    if obj.is_pothole:
        cv2.ellipse(
            image,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            (max(4, (x2 - x1) // 2), max(3, (y2 - y1) // 2)),
            0, 0, 360, color, max(1, thickness - 1), cv2.LINE_AA,
        )

    name = obj.class_name.replace("_", " ")
    lines = [f"{name} {obj.confidence * 100:.0f}%"]

    if show_details:
        lines.append(f"ID {obj.track_id}  {obj.distance_text()}")
        marker = RISK_MARKERS[obj.risk_level]
        risk_line = f"{obj.risk_level.value} {marker}"
        if obj.ttc is not None:
            risk_line += f"  TTC {obj.ttc_text()}"
        lines.append(risk_line)

        if obj.is_pothole and obj.pothole is not None:
            detail = f"sev {obj.pothole.severity.value}"
            if obj.pothole.estimated_width_m > 0:
                detail += f"  ~{obj.pothole.estimated_width_m:.1f}m"
            lines.append(detail)

        if obj.source is DetectionSource.HEURISTIC:
            lines.append("HEUR (not ML)")
        elif obj.source is DetectionSource.MOCK:
            lines.append("MOCK")

    draw_label_block(image, lines, (x1, y1 - 2), color, scale=font_scale)


def draw_trajectory(
    image: np.ndarray,
    obj: TrackedObject,
    show_history: bool = True,
) -> None:
    """Draw an object's recent track and its predicted future path."""
    if show_history and len(obj.history) >= 2:
        points = np.array(
            [p.as_int() for p in obj.history[-14:]], dtype=np.int32
        )
        cv2.polylines(image, [points], False, COLORS["grey"], 1, cv2.LINE_AA)

    if len(obj.predicted_path) >= 2:
        color = risk_color(obj.risk_level)
        previous = obj.predicted_path[0].as_int()
        for i, point in enumerate(obj.predicted_path[1:], start=1):
            current = point.as_int()
            if i % 2 == 1:
                cv2.line(image, previous, current, color, 2, cv2.LINE_AA)
            previous = current

        end = obj.predicted_path[-1].as_int()
        cv2.circle(image, end, 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, end, 5, COLORS["black"], 1, cv2.LINE_AA)

        if obj.time_to_path_entry is not None and obj.time_to_path_entry > 0:
            draw_text(
                image, f"+{obj.time_to_path_entry:.1f}s",
                (end[0] + 7, end[1] + 4), 0.38, color,
            )


# ---------------------------------------------------------------------------
# Ego path
# ---------------------------------------------------------------------------


def draw_ego_corridor(
    image: np.ndarray,
    polygon: Sequence[Point],
    conflict: bool = False,
) -> None:
    """Draw the ego corridor, highlighted when something conflicts with it."""
    if len(polygon) < 3:
        return

    points = np.array([p.as_int() for p in polygon], dtype=np.int32)
    color = COLORS["critical"] if conflict else COLORS["ego"]

    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.16, image, 0.84, 0, image)

    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)

    if conflict:
        draw_text(
            image, "PATH CONFLICT",
            (points[:, 0].min() + 6, points[:, 1].min() + 20),
            0.5, color, 2,
        )


def draw_nudge_arrow(
    image: np.ndarray,
    direction: Direction,
    color: tuple[int, int, int],
) -> None:
    """Draw a large directional arrow for a NUDGE recommendation."""
    height, width = image.shape[:2]
    cy = int(height * 0.82)
    cx = int(width * 0.5)
    length = int(width * 0.10)
    sign = -1 if direction is Direction.LEFT else 1

    start = (cx, cy)
    end = (cx + sign * length, cy - int(length * 0.45))
    cv2.arrowedLine(image, start, end, COLORS["black"], 9, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(image, start, end, color, 5, cv2.LINE_AA, tipLength=0.35)

    label = f"NUDGE {direction.value}"
    tw, _ = text_size(label, 0.62, 2)
    draw_text(
        image, label,
        (cx + sign * length - (tw // 2), cy - int(length * 0.45) - 14),
        0.62, color, 2,
    )


def draw_drivable_overlay(image: np.ndarray, mask: Optional[np.ndarray]) -> None:
    """Tint the estimated drivable area."""
    if mask is None or mask.shape[:2] != image.shape[:2]:
        return
    tint = np.zeros_like(image)
    tint[:, :] = (70, 120, 70)
    region = mask > 0
    image[region] = cv2.addWeighted(image, 0.88, tint, 0.12, 0)[region]
