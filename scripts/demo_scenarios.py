#!/usr/bin/env python3
"""Run the ten built-in demo scenarios and report what each recommends.

    python scripts/demo_scenarios.py               # all scenarios, console table
    python scripts/demo_scenarios.py --save        # also write annotated frames
    python scripts/demo_scenarios.py --scenario cow_blocking --show

Every scenario is driven by the :class:`MockDetector`, so the detections are
**synthetic** and clearly labelled as such. The purpose is to exercise and
demonstrate the risk and decision layers deterministically, without needing
footage of a cow standing in a road.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import Application
from src.core.config import load_config
from src.input.base import Frame
from src.perception.mock_detector import SCENARIOS

logger = logging.getLogger("demo")

EXPECTATIONS: dict[str, str] = {
    "clear_road": "CRUISE - nothing conflicting with the path",
    "pothole": "NUDGE - partial obstruction with space beside it",
    "pedestrian_crossing": "YIELD then STOP as the pedestrian enters the path",
    "cow_blocking": "STOP - animal occupying the corridor at close range",
    "bullock_cart": "YIELD or NUDGE - slow wide obstacle ahead",
    "slow_vehicle": "NUDGE or YIELD - truck partially blocking the lane",
    "motorcycle_overtaking": "YIELD or NUDGE - motorcycle cutting across",
    "dense_market": "CREEP - dense mixed traffic, uncertain movement",
    "multiple_obstacles": "NUDGE or CREEP - several hazards to negotiate",
    "unknown_obstacle": "CREEP or STOP - unidentified object, low confidence",
}

WIDTH, HEIGHT, FRAMES = 960, 540, 45


def synthetic_road(width: int, height: int, seed: int = 3) -> np.ndarray:
    """A plain road backdrop, so the overlays are the visible content."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[: int(height * 0.45)] = (188, 172, 158)
    frame[int(height * 0.45):] = (78, 78, 84)
    road = frame[int(height * 0.45):]
    noise = rng.integers(-7, 7, road.shape).astype(np.int16)
    frame[int(height * 0.45):] = np.clip(
        road.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    for y in range(int(height * 0.52), height, 34):
        cv2.line(frame, (width // 2, y), (width // 2, y + 16), (122, 122, 126), 2)
    return frame


def run_scenario(
    name: str, config, save_dir: Optional[Path], show: bool
) -> dict[str, Any]:
    """Run one scenario to completion and return its outcome."""
    config.detector.type = "mock"
    app = Application(config, save_results=False, headless=not show,
                      mock_scenario=name)
    app.load()

    backdrop = synthetic_road(WIDTH, HEIGHT)
    actions: list[str] = []
    last: Any = None
    last_world: Any = None
    rendered: Optional[np.ndarray] = None

    for i in range(FRAMES):
        frame = Frame(
            image=backdrop.copy(), index=i, timestamp=i / 20.0, source_time=i / 20.0
        )
        rendered, world, recommendation = app.process_frame(frame)
        actions.append(recommendation.label)
        last, last_world = recommendation, world

        if show and rendered is not None:
            cv2.imshow("Demo scenario", app.renderer.fit_to_screen(rendered))
            if (cv2.waitKey(40) & 0xFF) in (ord("q"), 27):
                break

    if show:
        cv2.destroyAllWindows()

    if save_dir is not None and rendered is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.jpg"
        ok, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            buffer.tofile(str(path))

    from collections import Counter
    counts = Counter(actions)

    return {
        "scenario": name,
        "final_action": last.label if last else "NONE",
        "final_risk": last.risk.value if last else "NONE",
        "final_speed": last.target_speed_kmh if last else 0.0,
        "reason": last.reason if last else "",
        "objects": len(last_world.objects) if last_world else 0,
        "distribution": dict(counts),
        "expectation": EXPECTATIONS.get(name, ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the built-in demo scenarios (synthetic detections).",
    )
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), help="Run only this scenario."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--save", action="store_true",
        help="Save an annotated frame per scenario to results/demo/.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display each scenario in a window."
    )
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.ERROR),
        format="%(levelname)-7s %(message)s",
    )

    names = [args.scenario] if args.scenario else list(EXPECTATIONS)
    save_dir = Path("results/demo") if args.save else None

    print("\n" + "=" * 78)
    print("  DEMO SCENARIOS - synthetic detections from the mock detector")
    print("  These exercise the risk and decision layers, not the real detector.")
    print("=" * 78)

    results = []
    for name in names:
        config = load_config(args.config)
        try:
            result = run_scenario(name, config, save_dir, args.show)
        except Exception as exc:
            logger.error("Scenario %s failed: %s", name, exc)
            continue
        results.append(result)

        print(f"\n  {name.upper().replace('_', ' ')}")
        print(f"    objects        : {result['objects']}")
        print(f"    recommendation : {result['final_action']}  "
              f"(risk {result['final_risk']}, "
              f"{result['final_speed']:.0f} km/h)")
        print(f"    reason         : {result['reason'][:150]}")
        if result["expectation"]:
            print(f"    intended       : {result['expectation']}")
        distribution = ", ".join(
            f"{k}={v}" for k, v in sorted(
                result["distribution"].items(), key=lambda kv: -kv[1]
            )
        )
        print(f"    over {FRAMES} frames: {distribution}")

    print("\n" + "=" * 78)
    print(f"  {len(results)} scenario(s) run.")
    actions = {r["final_action"].split()[0] for r in results}
    print(f"  Distinct final actions: {', '.join(sorted(actions))}")
    if save_dir:
        print(f"  Annotated frames: {save_dir}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
