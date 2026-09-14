"""Class taxonomy: model labels to canonical Indian-road classes and categories.

A detector emits whatever labels its weights were trained on. COCO calls a
pedestrian ``person``; a custom Indian-road model may call an auto-rickshaw
``auto``, ``autorickshaw`` or ``three-wheeler``. Everything downstream reasons
about *canonical* class names and :class:`SemanticCategory`, so this module is
the single place that has to change when new weights are plugged in.

Adding a class requires one alias entry and one category entry - no changes to
the tracker, risk engine, or decision engine.
"""

from __future__ import annotations

import re
from typing import Final

from .types import SemanticCategory

__all__ = [
    "CANONICAL_CLASSES",
    "COCO_SUPPORTED",
    "INDIAN_ROAD_CLASSES",
    "NOT_IN_COCO",
    "canonicalize",
    "category_for",
    "describe_class",
    "is_supported_by_coco",
]


# ---------------------------------------------------------------------------
# Alias table: any label a model might emit -> canonical class name.
# Keys are matched after lower-casing and collapsing separators.
# ---------------------------------------------------------------------------

_ALIASES: Final[dict[str, str]] = {
    # --- road users -------------------------------------------------------
    "person": "person",
    "pedestrian": "person",
    "people": "person",
    "human": "person",
    "rider": "person",
    "car": "car",
    "auto": "auto_rickshaw",
    "autorickshaw": "auto_rickshaw",
    "auto_rickshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
    "tuktuk": "auto_rickshaw",
    "threewheeler": "auto_rickshaw",
    "three_wheeler": "auto_rickshaw",
    "truck": "truck",
    "lorry": "truck",
    "tempo": "truck",
    "tractor": "truck",
    "bus": "bus",
    "minibus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "scooter": "motorcycle",
    "twowheeler": "motorcycle",
    "two_wheeler": "motorcycle",
    "bicycle": "bicycle",
    "cycle": "bicycle",
    "train": "train",
    "boat": "unknown",
    "airplane": "unknown",
    # --- animals ----------------------------------------------------------
    "cow": "cow",
    "cattle": "cow",
    "buffalo": "cow",
    "bull": "bull",
    "ox": "bull",
    "dog": "dog",
    "stray_dog": "dog",
    "cat": "cat",
    "horse": "horse",
    "sheep": "sheep",
    "goat": "sheep",
    "elephant": "elephant",
    "bear": "other_animal",
    "zebra": "other_animal",
    "giraffe": "other_animal",
    "animal": "other_animal",
    # --- Indian-specific slow movers -------------------------------------
    "bullock_cart": "bullock_cart",
    "bullockcart": "bullock_cart",
    "bullock": "bullock_cart",
    "cart": "bullock_cart",
    "oxcart": "bullock_cart",
    "pushcart": "pushcart",
    "handcart": "pushcart",
    "thela": "pushcart",
    "vendor_cart": "pushcart",
    # --- hazards ----------------------------------------------------------
    "pothole": "pothole",
    "potholes": "pothole",
    "pot_hole": "pothole",
    "crack": "road_damage",
    "road_damage": "road_damage",
    "manhole": "road_damage",
    "speedbump": "speed_bump",
    "speed_bump": "speed_bump",
    "hump": "speed_bump",
    "debris": "debris",
    "road_debris": "debris",
    "fallen_object": "debris",
    "garbage": "debris",
    "rock": "debris",
    "barrier": "construction_barrier",
    "construction_barrier": "construction_barrier",
    "barricade": "construction_barrier",
    "roadblock": "road_blockage",
    "road_blockage": "road_blockage",
    "cone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "drum": "traffic_cone",
    # --- signage ----------------------------------------------------------
    "traffic_light": "traffic_light",
    "trafficlight": "traffic_light",
    "signal": "traffic_light",
    "stop_sign": "traffic_sign",
    "stopsign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "sign": "traffic_sign",
    "signboard": "traffic_sign",
    "parking_meter": "traffic_sign",
    "fire_hydrant": "static_object",
    "bench": "static_object",
    "potted_plant": "static_object",
    "unknown": "unknown",
    "obstacle": "unknown",
    "unknown_obstacle": "unknown",
}


# ---------------------------------------------------------------------------
# Canonical class -> semantic category.
# ---------------------------------------------------------------------------

_CATEGORIES: Final[dict[str, SemanticCategory]] = {
    "person": SemanticCategory.VULNERABLE,
    "bicycle": SemanticCategory.VULNERABLE,
    "cow": SemanticCategory.ANIMAL,
    "bull": SemanticCategory.ANIMAL,
    "dog": SemanticCategory.ANIMAL,
    "cat": SemanticCategory.ANIMAL,
    "horse": SemanticCategory.ANIMAL,
    "sheep": SemanticCategory.ANIMAL,
    "elephant": SemanticCategory.ANIMAL,
    "other_animal": SemanticCategory.ANIMAL,
    "car": SemanticCategory.VEHICLE,
    "truck": SemanticCategory.VEHICLE,
    "bus": SemanticCategory.VEHICLE,
    "motorcycle": SemanticCategory.VEHICLE,
    "auto_rickshaw": SemanticCategory.VEHICLE,
    "train": SemanticCategory.VEHICLE,
    "bullock_cart": SemanticCategory.OBSTACLE,
    "pushcart": SemanticCategory.OBSTACLE,
    "pothole": SemanticCategory.OBSTACLE,
    "road_damage": SemanticCategory.OBSTACLE,
    "speed_bump": SemanticCategory.OBSTACLE,
    "debris": SemanticCategory.OBSTACLE,
    "construction_barrier": SemanticCategory.OBSTACLE,
    "road_blockage": SemanticCategory.OBSTACLE,
    "traffic_cone": SemanticCategory.OBSTACLE,
    "traffic_light": SemanticCategory.STATIC,
    "traffic_sign": SemanticCategory.STATIC,
    "static_object": SemanticCategory.STATIC,
    "unknown": SemanticCategory.UNKNOWN,
}

CANONICAL_CLASSES: Final[tuple[str, ...]] = tuple(sorted(_CATEGORIES))

INDIAN_ROAD_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "road_users": ("car", "truck", "bus", "motorcycle", "bicycle",
                   "auto_rickshaw", "person"),
    "animals": ("cow", "bull", "dog", "horse", "sheep", "other_animal"),
    "indian_specific": ("auto_rickshaw", "bullock_cart", "pushcart"),
    "hazards": ("pothole", "road_damage", "speed_bump", "debris",
                "construction_barrier", "traffic_cone", "road_blockage"),
    "signage": ("traffic_light", "traffic_sign"),
    "other": ("unknown",),
}

COCO_SUPPORTED: Final[frozenset[str]] = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic_light", "traffic_sign", "cow", "dog", "cat", "horse", "sheep",
    "elephant", "other_animal", "static_object",
})

NOT_IN_COCO: Final[tuple[str, ...]] = (
    "pothole", "road_damage", "speed_bump", "auto_rickshaw", "bullock_cart",
    "pushcart", "debris", "construction_barrier", "traffic_cone",
    "road_blockage", "bull",
)

_NORMALISE_RE = re.compile(r"[\s\-]+")


def _normalise(label: str) -> str:
    """Lower-case a raw label and collapse spaces/hyphens to underscores."""
    return _NORMALISE_RE.sub("_", label.strip().lower())


def canonicalize(label: str) -> str:
    """Map a raw model label onto a canonical class name.

    Unrecognised labels become ``"unknown"`` rather than being invented into a
    plausible-sounding class. The system must never hallucinate a class it was
    not trained to produce.
    """
    key = _normalise(label)
    if key in _ALIASES:
        return _ALIASES[key]
    compact = key.replace("_", "")
    if compact in _ALIASES:
        return _ALIASES[compact]
    if key in _CATEGORIES:
        return key
    return "unknown"


def category_for(canonical_class: str) -> SemanticCategory:
    """Semantic category for a canonical class name."""
    return _CATEGORIES.get(canonical_class, SemanticCategory.UNKNOWN)


def describe_class(canonical_class: str) -> str:
    """Short human-readable label used in UI overlays and explanations."""
    return canonical_class.replace("_", " ")


def is_supported_by_coco(canonical_class: str) -> bool:
    """Whether stock COCO weights can detect this canonical class at all."""
    return canonical_class in COCO_SUPPORTED
