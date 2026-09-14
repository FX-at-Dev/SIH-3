#!/usr/bin/env python3
"""Batch-evaluate a dataset directory and produce a summary report.

    python scripts/evaluate_dataset.py --input data/
    python scripts/evaluate_dataset.py --input data/potholes --pothole-mode heuristic
    python scripts/evaluate_dataset.py --input data/ --output results/eval

Walks the directory, processes every image and video it finds, and reports the
recommendation distribution, risk distribution, detection counts and timing
per scenario folder. Intended for the scenario folders described in the README
(``village/``, ``urban/``, ``market/``, ``highway/``, ``cattle/``, ``potholes/``).

This measures *what the system recommends*, not accuracy against ground truth -
there is no labelled ground truth here. Treat it as a behavioural regression
report: it tells you whether a config or model change altered the system's
behaviour on your footage, and how.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import Application
from src.core.config import apply_overrides, load_config
from src.input.image_source import SUPPORTED_IMAGE_EXTENSIONS
from src.input.source_factory import create_source
from src.input.video_source import SUPPORTED_VIDEO_EXTENSIONS

logger = logging.getLogger("evaluate")


def find_media(root: Path, max_per_folder: Optional[int] = None) -> dict[str, list[Path]]:
    """Group media files by their immediate parent folder name."""
    groups: dict[str, list[Path]] = defaultdict(list)
    extensions = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        scenario = path.parent.name if path.parent != root else root.name
        if max_per_folder and len(groups[scenario]) >= max_per_folder:
            continue
        groups[scenario].append(path)

    return dict(groups)


def evaluate_file(
    app: Application, path: Path, max_frames: Optional[int]
) -> dict[str, Any]:
    """Run one file through the pipeline and collect per-frame outcomes."""
    is_video = path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    source = create_source("video" if is_video else "image", path)

    actions: Counter[str] = Counter()
    risks: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    frames = 0
    total_ms = 0.0
    max_objects = 0
    started = time.time()

    app.perception.reset()
    app.decision_engine.reset()

    try:
        source.open()
        while True:
            frame = source.read()
            if frame is None:
                break
            _, world, recommendation = app.process_frame(frame, source)

            frames += 1
            total_ms += world.metrics.total_ms
            max_objects = max(max_objects, len(world.objects))
            actions[recommendation.label] += 1
            risks[recommendation.risk.value] += 1
            for obj in world.objects:
                classes[obj.class_name] += 1

            if max_frames and frames >= max_frames:
                break
    except Exception as exc:
        logger.error("Failed on %s: %s", path.name, exc)
    finally:
        source.release()

    dominant = actions.most_common(1)[0][0] if actions else "NONE"
    return {
        "file": path.name,
        "type": "video" if is_video else "image",
        "frames": frames,
        "duration_s": round(time.time() - started, 2),
        "avg_ms_per_frame": round(total_ms / frames, 2) if frames else 0.0,
        "max_objects": max_objects,
        "dominant_recommendation": dominant,
        "recommendations": dict(actions),
        "risks": dict(risks),
        "classes": dict(classes),
    }


def summarise(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate per-file results into a scenario-level summary."""
    summary: dict[str, Any] = {"scenarios": {}, "totals": {}}

    all_actions: Counter[str] = Counter()
    all_risks: Counter[str] = Counter()
    all_classes: Counter[str] = Counter()
    total_frames = 0
    total_ms = 0.0

    for scenario, files in results.items():
        actions: Counter[str] = Counter()
        risks: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        frames = 0
        ms = 0.0

        for entry in files:
            actions.update(entry["recommendations"])
            risks.update(entry["risks"])
            classes.update(entry["classes"])
            frames += entry["frames"]
            ms += entry["avg_ms_per_frame"] * entry["frames"]

        summary["scenarios"][scenario] = {
            "files": len(files),
            "frames": frames,
            "avg_ms_per_frame": round(ms / frames, 2) if frames else 0.0,
            "avg_fps": round(1000.0 / (ms / frames), 1) if frames and ms else 0.0,
            "recommendations": dict(actions),
            "risks": dict(risks),
            "top_classes": dict(classes.most_common(8)),
            "per_file": files,
        }

        all_actions.update(actions)
        all_risks.update(risks)
        all_classes.update(classes)
        total_frames += frames
        total_ms += ms

    summary["totals"] = {
        "scenarios": len(results),
        "files": sum(len(f) for f in results.values()),
        "frames": total_frames,
        "avg_ms_per_frame": round(total_ms / total_frames, 2) if total_frames else 0.0,
        "avg_fps": round(1000.0 / (total_ms / total_frames), 1)
        if total_frames and total_ms else 0.0,
        "recommendations": dict(all_actions),
        "risks": dict(all_risks),
        "classes": dict(all_classes.most_common(20)),
    }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    """Print a readable console report."""
    line = "=" * 74
    print(f"\n{line}\n  DATASET EVALUATION REPORT\n{line}")

    for scenario, data in sorted(summary["scenarios"].items()):
        print(f"\n  SCENARIO: {scenario}")
        print(f"    files {data['files']}, frames {data['frames']}, "
              f"{data['avg_ms_per_frame']:.1f} ms/frame "
              f"({data['avg_fps']:.1f} fps)")

        total = max(1, sum(data["recommendations"].values()))
        print("    recommendations:")
        for action, count in sorted(
            data["recommendations"].items(), key=lambda kv: -kv[1]
        ):
            bar = "#" * int(30 * count / total)
            print(f"      {action:<13} {count:>6}  {100*count/total:>5.1f}%  {bar}")

        if data["top_classes"]:
            classes = ", ".join(
                f"{name} x{count}" for name, count in data["top_classes"].items()
            )
            print(f"    detected: {classes}")

    totals = summary["totals"]
    print(f"\n{line}\n  TOTALS")
    print(f"    {totals['scenarios']} scenario(s), {totals['files']} file(s), "
          f"{totals['frames']} frame(s)")
    print(f"    {totals['avg_ms_per_frame']:.1f} ms/frame "
          f"({totals['avg_fps']:.1f} fps average)")

    total = max(1, sum(totals["recommendations"].values()))
    print("    recommendation distribution:")
    for action, count in sorted(totals["recommendations"].items(), key=lambda kv: -kv[1]):
        print(f"      {action:<13} {count:>6}  {100*count/total:>5.1f}%")
    print(f"{line}")
    print(
        "  NOTE: this reports what the system RECOMMENDED, not accuracy against\n"
        "  ground truth. Use it to compare behaviour across config or model\n"
        "  changes on the same footage.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a dataset of road images and videos.",
    )
    parser.add_argument("--input", required=True, help="Dataset root directory.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--detector", choices=["yolo", "mock"])
    parser.add_argument("--model", help="Override the model weights path.")
    parser.add_argument("--pothole-mode", choices=["none", "yolo", "heuristic"])
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Frames to process per video (default: 150; 0 = all).",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Limit files per folder (0 = no limit).",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)-7s %(message)s",
    )

    root = Path(args.input)
    if not root.exists():
        print(f"Input directory not found: {root}", file=sys.stderr)
        return 2

    groups = find_media(root, args.max_files or None)
    if not groups:
        print(
            f"No images or videos found under {root}.\n"
            "Expected a layout such as:\n"
            "  data/village/  data/urban/  data/market/\n"
            "  data/highway/  data/cattle/  data/potholes/",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    overrides = {
        "detector.type": args.detector,
        "detector.model": args.model,
        "pothole.type": args.pothole_mode,
    }
    apply_overrides(config, {k: v for k, v in overrides.items() if v})
    if args.pothole_mode:
        config.pothole.enabled = args.pothole_mode != "none"
    config.visualization.enabled = False

    app = Application(config, save_results=False, headless=True)
    try:
        app.load()
    except Exception as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 3

    total_files = sum(len(v) for v in groups.values())
    print(f"Evaluating {total_files} file(s) across {len(groups)} scenario(s)...")

    results: dict[str, list[dict[str, Any]]] = {}
    processed = 0
    for scenario, files in sorted(groups.items()):
        results[scenario] = []
        for path in files:
            processed += 1
            print(f"  [{processed}/{total_files}] {scenario}/{path.name}", flush=True)
            results[scenario].append(
                evaluate_file(app, path, args.max_frames or None)
            )

    summary = summarise(results)
    summary["config"] = {
        "path": config.source_path,
        "detector": config.detector.type,
        "model": config.detector.model,
        "pothole_mode": config.pothole.type,
    }
    summary["notice"] = (
        "Prototype research system. Recommendations are advisory and are not "
        "validated against ground truth."
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_report(summary)
    print(f"  Full report: {report_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
