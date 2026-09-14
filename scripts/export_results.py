#!/usr/bin/env python3
"""Export and summarise a recorded session.

    python scripts/export_results.py --session results/session_2026_09_13_214233
    python scripts/export_results.py --latest
    python scripts/export_results.py --latest --plots

Reads the CSV and JSON a session produced and emits:

* a console summary of recommendations, risk and timing;
* ``session_report.md``, a readable Markdown report;
* optional matplotlib charts (``--plots``) of the recommendation timeline,
  risk distribution and per-frame latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def find_latest_session(root: Path) -> Optional[Path]:
    """Most recently modified ``session_*`` directory."""
    candidates = [p for p in root.glob("session_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyse(session: Path) -> dict[str, Any]:
    """Collect everything interesting from a session directory."""
    summary = read_json(session / "summary.json")
    events = read_json(session / "events.json")
    recommendations = read_csv(session / "recommendations.csv")
    detections = read_csv(session / "detections.csv")
    metrics = read_csv(session / "metrics.csv")

    actions = Counter(r.get("action", "") for r in recommendations)
    risks = Counter(r.get("risk", "") for r in recommendations)
    classes = Counter(d.get("class_name", "") for d in detections)
    event_types = Counter(e.get("type", "") for e in events.get("events", []))

    transitions: Counter[str] = Counter()
    previous: Optional[str] = None
    for row in recommendations:
        action = row.get("action", "")
        if previous is not None and action != previous:
            transitions[f"{previous} -> {action}"] += 1
        previous = action

    latencies = [
        float(m["total_ms"]) for m in metrics
        if m.get("total_ms") not in (None, "")
    ]
    latencies.sort()

    def percentile(p: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[index]

    return {
        "session": str(session),
        "summary": summary,
        "actions": dict(actions),
        "risks": dict(risks),
        "classes": dict(classes.most_common(15)),
        "event_types": dict(event_types),
        "transitions": dict(transitions.most_common(12)),
        "frames": len(recommendations),
        "unique_tracks": len({d.get("track_id") for d in detections}),
        "latency": {
            "p50_ms": round(percentile(0.50), 2),
            "p90_ms": round(percentile(0.90), 2),
            "p99_ms": round(percentile(0.99), 2),
            "max_ms": round(latencies[-1], 2) if latencies else 0.0,
        },
        "_recommendations": recommendations,
        "_metrics": metrics,
    }


def print_report(data: dict[str, Any]) -> None:
    line = "=" * 72
    print(f"\n{line}\n  SESSION REPORT\n  {data['session']}\n{line}")

    summary = data["summary"]
    if summary:
        print(f"  Frames      : {summary.get('frames_processed', data['frames'])}")
        print(f"  Duration    : {summary.get('duration_seconds', 0)} s")
        print(f"  Average FPS : {summary.get('average_fps', 0)}")
        print(f"  Detector    : {summary.get('detector')} "
              f"({summary.get('model')})")
        print(f"  Pothole mode: {summary.get('pothole_mode')}")

    print(f"  Tracks      : {data['unique_tracks']}")

    latency = data["latency"]
    print(f"\n  LATENCY  p50 {latency['p50_ms']} ms | p90 {latency['p90_ms']} ms "
          f"| p99 {latency['p99_ms']} ms | max {latency['max_ms']} ms")

    total = max(1, sum(data["actions"].values()))
    print("\n  RECOMMENDATIONS")
    for action, count in sorted(data["actions"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(34 * count / total)
        print(f"    {action:<10} {count:>6}  {100*count/total:>5.1f}%  {bar}")

    print("\n  RISK LEVELS")
    total_risk = max(1, sum(data["risks"].values()))
    for risk, count in sorted(data["risks"].items(), key=lambda kv: -kv[1]):
        print(f"    {risk:<10} {count:>6}  {100*count/total_risk:>5.1f}%")

    if data["transitions"]:
        print("\n  RECOMMENDATION TRANSITIONS")
        for transition, count in data["transitions"].items():
            print(f"    {transition:<26} x{count}")

    if data["classes"]:
        print("\n  DETECTED CLASSES (object-frames)")
        for name, count in data["classes"].items():
            print(f"    {name:<18} {count}")

    if data["event_types"]:
        print("\n  EVENTS")
        for name, count in sorted(data["event_types"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<26} {count}")

    print(f"{line}\n")


def write_markdown(data: dict[str, Any], path: Path) -> None:
    summary = data["summary"]
    lines = [
        "# Session report",
        "",
        f"**Session:** `{data['session']}`",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Frames processed | {summary.get('frames_processed', data['frames'])} |",
        f"| Duration (s) | {summary.get('duration_seconds', 0)} |",
        f"| Average FPS | {summary.get('average_fps', 0)} |",
        f"| Detector | {summary.get('detector')} |",
        f"| Model | `{summary.get('model')}` |",
        f"| Pothole mode | {summary.get('pothole_mode')} |",
        f"| Unique tracks | {data['unique_tracks']} |",
        f"| Latency p50 / p90 / p99 (ms) | "
        f"{data['latency']['p50_ms']} / {data['latency']['p90_ms']} / "
        f"{data['latency']['p99_ms']} |",
        "",
        "## Recommendation distribution",
        "",
        "| Action | Frames | Share |",
        "| --- | ---: | ---: |",
    ]
    total = max(1, sum(data["actions"].values()))
    for action, count in sorted(data["actions"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {action} | {count} | {100*count/total:.1f}% |")

    lines += ["", "## Risk distribution", "",
              "| Risk | Frames |", "| --- | ---: |"]
    for risk, count in sorted(data["risks"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {risk} | {count} |")

    if data["transitions"]:
        lines += ["", "## Recommendation transitions", "",
                  "| Transition | Count |", "| --- | ---: |"]
        for transition, count in data["transitions"].items():
            lines.append(f"| {transition} | {count} |")

    if data["classes"]:
        lines += ["", "## Detected classes", "",
                  "| Class | Object-frames |", "| --- | ---: |"]
        for name, count in data["classes"].items():
            lines.append(f"| {name} | {count} |")

    if data["event_types"]:
        lines += ["", "## Events", "", "| Event | Count |", "| --- | ---: |"]
        for name, count in sorted(data["event_types"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "---",
        "",
        "_Prototype driver-assistance / autonomous-driving research system. "
        "Recommendations are advisory output only and are not intended for "
        "direct control of a real vehicle. Distances are monocular estimates, "
        "not sensor measurements._",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(data: dict[str, Any], out_dir: Path) -> list[Path]:
    """Render summary charts. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.\n"
              "  Install it with: pip install matplotlib")
        return []

    written: list[Path] = []
    action_order = ["CRUISE", "NUDGE", "YIELD", "CREEP", "STOP"]

    # --- recommendation timeline -------------------------------------
    recommendations = data["_recommendations"]
    if recommendations:
        frames = [int(r["frame_index"]) for r in recommendations]
        levels = [
            action_order.index(r["action"]) if r["action"] in action_order else 0
            for r in recommendations
        ]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.step(frames, levels, where="post", linewidth=1.6)
        ax.set_yticks(range(len(action_order)))
        ax.set_yticklabels(action_order)
        ax.set_xlabel("frame")
        ax.set_title("Driving recommendation over time")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / "recommendation_timeline.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # --- risk distribution -------------------------------------------
    if data["risks"]:
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        values = [data["risks"].get(k, 0) for k in order]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.bar(order, values)
        ax.set_ylabel("frames")
        ax.set_title("Scene risk distribution")
        fig.tight_layout()
        path = out_dir / "risk_distribution.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # --- latency ------------------------------------------------------
    metrics = data["_metrics"]
    if metrics:
        frames = [int(m["frame_index"]) for m in metrics]
        total = [float(m["total_ms"]) for m in metrics]
        inference = [float(m.get("inference_ms", 0) or 0) for m in metrics]
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(frames, total, linewidth=1.0, label="total")
        ax.plot(frames, inference, linewidth=1.0, label="inference")
        ax.set_xlabel("frame")
        ax.set_ylabel("ms")
        ax.set_title("Per-frame latency")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / "latency.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export and summarise a recorded session."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Path to a session directory.")
    group.add_argument(
        "--latest", action="store_true", help="Use the most recent session."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--plots", action="store_true", help="Also render summary charts."
    )
    args = parser.parse_args(argv)

    if args.latest:
        session = find_latest_session(Path(args.results_dir))
        if session is None:
            print(
                f"No sessions found in {args.results_dir}/.\n"
                "  Record one first:\n"
                "    python main.py --source video --input <file> --save-results",
                file=sys.stderr,
            )
            return 1
    else:
        session = Path(args.session)

    if not session.exists():
        print(f"Session directory not found: {session}", file=sys.stderr)
        return 2

    data = analyse(session)
    if not data["frames"] and not data["summary"]:
        print(
            f"{session} contains no session data.\n"
            "  Expected recommendations.csv / summary.json produced by\n"
            "  a run with --save-results.",
            file=sys.stderr,
        )
        return 3

    print_report(data)

    report_path = session / "session_report.md"
    write_markdown(data, report_path)
    print(f"  Markdown report: {report_path}")

    if args.plots:
        for path in write_plots(data, session):
            print(f"  Chart: {path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
