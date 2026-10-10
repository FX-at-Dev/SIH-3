#!/usr/bin/env python3
"""Indian Road Perception + Driving Recommendation System - CLI entry point.

    python main.py --source camera
    python main.py --source camera --camera-id 2
    python main.py --source video  --input data/videos/test.mp4
    python main.py --source image  --input data/images/test.jpg --save
    python main.py --source video  --input data/videos/test.mp4 --save-results
    python main.py --source camera --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.config import Config, apply_overrides, load_config
from src.input.base import SourceError
from src.input.source_factory import create_source, infer_source_type

if TYPE_CHECKING:
    from src.app import Application

BANNER = r"""
+--------------------------------------------------------------+
|  INDIAN ROAD PERCEPTION + DRIVING RECOMMENDATION SYSTEM       |
|  Perception and decision-support research prototype           |
|                                                               |
|  Outputs: CRUISE / NUDGE / YIELD / CREEP / STOP               |
|  NOT for direct control of a real vehicle.                    |
+--------------------------------------------------------------+
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Analyse Indian road scenes from a camera, video or image and "
            "produce a driving recommendation. Research prototype - advisory "
            "output only, not for vehicle control."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --source camera\n"
            "  python main.py --source video --input data/videos/road.mp4\n"
            "  python main.py --source image --input data/images/street.jpg --save\n"
            "  python main.py --source video --input road.mp4 --save-results\n"
            "  python main.py --source image --input street.jpg --detector mock\n"
        ),
    )

    source = parser.add_argument_group("input")
    source.add_argument(
        "--source", choices=["camera", "video", "image"],
        help="Input mode. Inferred from --input when omitted.",
    )
    source.add_argument("--input", type=str, help="Path to a video or image file.")
    source.add_argument(
        "--camera-id", type=int, default=2, help="Camera device index (default: 0)."
    )
    source.add_argument(
        "--loop", action="store_true", help="Loop a video file continuously."
    )
    source.add_argument(
        "--recursive", action="store_true",
        help="Recurse into sub-directories in image-directory mode.",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="YAML config path (default: configs/default.yaml).",
    )
    model.add_argument(
        "--detector", choices=["yolo", "mock"],
        help="Override the configured detector type.",
    )
    model.add_argument("--model", type=str, help="Override the model weights path.")
    model.add_argument(
        "--confidence", type=float, help="Override the detection confidence threshold."
    )
    model.add_argument(
        "--device", type=str, help="Inference device: auto, cpu, cuda, mps."
    )
    model.add_argument(
        "--pothole-mode", choices=["none", "yolo", "heuristic"],
        help=(
            "Pothole detection mode. 'heuristic' is a classical-CV "
            "approximation, not machine learning."
        ),
    )
    model.add_argument(
        "--scenario", type=str,
        help="Mock detector scenario (only with --detector mock).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--save", action="store_true",
        help="Save the annotated result (image mode).",
    )
    output.add_argument(
        "--save-results", action="store_true",
        help="Record a full session: video, events, CSV metrics.",
    )
    output.add_argument(
        "--output", type=str, help="Output path for --save in image mode."
    )
    output.add_argument(
        "--headless", action="store_true",
        help="Process without opening a window (for servers and batch runs).",
    )
    output.add_argument(
        "--no-display", dest="headless", action="store_true",
        help=argparse.SUPPRESS,
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--process-width", type=int,
        help="Resize frames to this width before processing.",
    )
    misc.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: from config).",
    )
    misc.add_argument(
        "--quiet", action="store_true", help="Only show warnings and errors."
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def resolve_config(args: argparse.Namespace) -> Config:
    """Load the YAML config and apply CLI overrides on top."""
    config = load_config(args.config)

    overrides = {
        "detector.type": args.detector,
        "detector.model": args.model,
        "detector.confidence": args.confidence,
        "detector.device": args.device,
        "pothole.type": args.pothole_mode,
        "app.process_width": args.process_width,
    }
    apply_overrides(config, {k: v for k, v in overrides.items() if v is not None})

    if args.pothole_mode is not None:
        config.pothole.enabled = args.pothole_mode != "none"
    if args.save_results:
        config.output.save_video = True
    if args.headless:
        config.visualization.enabled = True

    return config


def run_image_mode(app: "Application", args: argparse.Namespace, source) -> int:
    """Image mode: analyse, report to the console, optionally save."""
    import cv2

    source.open()
    saved_any = False
    processed = 0
    exit_code = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            processed += 1

            rendered, world, recommendation = app.process_frame(frame, source)
            _report_image_result(world, recommendation, source.current_file)

            if args.save and rendered is not None:
                path = _save_annotated(
                    rendered, source.current_file, args.output
                )
                if path:
                    saved_any = True
                    print(f"\nAnnotated image saved to: {path}")

            if not app.headless and rendered is not None:
                display = app.renderer.fit_to_screen(rendered)
                cv2.imshow("Indian Road Perception", display)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("s"), ord("S")) and rendered is not None:
                    app.recorder.save_screenshot(rendered, "image")
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        source.release()
        if not app.headless:
            cv2.destroyAllWindows()
        app.stats.end_time = app.stats.start_time or 0.0

    if processed == 0:
        print(
            "\nNo images could be analysed. Every candidate file failed to "
            "decode.\n"
            "  Check that the file is a valid, non-empty image in a supported "
            "format:\n"
            "    .jpg .jpeg .png .bmp .webp .tif .tiff .jfif\n",
            file=sys.stderr,
        )
        return 1
    if args.save and not saved_any:
        exit_code = 1
    return exit_code


def _save_annotated(image, source_path, output_arg: str | None) -> Path | None:
    """Write the annotated image next to the source, or to --output."""
    import cv2

    if output_arg:
        path = Path(output_arg)
    else:
        stem = source_path.stem if source_path else "annotated"
        path = Path("results") / "images" / f"{stem}_annotated.jpg"

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError("JPEG encoding failed")
        buffer.tofile(str(path))
        return path
    except Exception as exc:
        logging.getLogger(__name__).error("Could not save %s: %s", path, exc)
        return None


def _report_image_result(world, recommendation, source_path) -> None:
    """Print a full text report for a single analysed image."""
    name = source_path.name if source_path else "image"
    line = "=" * 70

    print(f"\n{line}\n  ANALYSIS: {name}\n{line}")

    if not world.objects:
        print("  No objects detected.")
    else:
        print(f"  {len(world.objects)} object(s) detected:\n")
        header = (
            f"  {'ID':>3}  {'CLASS':<14} {'CONF':>5}  {'DISTANCE':>10}  "
            f"{'POSITION':<8} {'PATH':<15} {'RISK':<9}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for obj in sorted(world.objects, key=lambda o: -o.risk_level.rank):
            distance = obj.distance_text()
            print(
                f"  {obj.track_id:>3}  {obj.class_name:<14} "
                f"{obj.confidence * 100:>4.0f}%  {distance:>10}  "
                f"{obj.relative_position.value:<8} "
                f"{obj.path_relation.value:<15} {obj.risk_level.value:<9}"
            )
            if obj.is_pothole and obj.pothole:
                print(
                    f"       pothole: severity {obj.pothole.severity.value}, "
                    f"~{obj.pothole.estimated_width_m:.2f} m wide, "
                    f"{obj.pothole.ego_path_overlap * 100:.0f}% of path, "
                    f"source: {obj.pothole.source.value}"
                )
            for reason in obj.risk_reasons[:2]:
                print(f"       - {reason}")

    print(f"\n{line}")
    print(f"  RECOMMENDATION : {recommendation.label}")
    print(f"  RISK LEVEL     : {recommendation.risk.value}")
    print(f"  TARGET SPEED   : {recommendation.target_speed_kmh:.0f} km/h")
    ttc = f"{recommendation.ttc:.1f} s" if recommendation.ttc is not None else "--"
    print(f"  TTC            : {ttc}")
    print(f"  CONFIDENCE     : {recommendation.confidence:.2f}")
    print("\n  EXPLANATION:")
    import textwrap
    for wrapped in textwrap.wrap(recommendation.reason, width=64):
        print(f"    {wrapped}")

    scene = world.scene_risk
    print(f"\n  SCENE: density={scene.traffic_density}, "
          f"left={'clear' if scene.left_clear else 'blocked'}, "
          f"right={'clear' if scene.right_clear else 'blocked'}, "
          f"perception confidence={scene.perception_confidence:.2f}")

    print(
        "\n  NOTE: distances marked '~' are monocular ESTIMATES, not sensor\n"
        "  measurements. Prototype decision support - not for vehicle control."
    )
    print(f"{line}\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- resolve source type ------------------------------------------
    if args.source is None:
        if args.input:
            try:
                args.source = infer_source_type(args.input)
            except Exception as exc:
                parser.error(str(exc))
        else:
            parser.error(
                "Specify --source camera|video|image, or pass --input <path>."
            )

    if args.source in ("video", "image") and not args.input:
        parser.error(f"--source {args.source} requires --input <path>.")

    # ---- config and logging --------------------------------------------
    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    level = args.log_level or ("WARNING" if args.quiet else config.app.log_level)
    configure_logging(level)

    if not args.quiet:
        print(BANNER)

    logger = logging.getLogger("main")

    # ---- build the application ------------------------------------------
    from src.app import Application

    app = Application(
        config,
        save_results=args.save_results,
        headless=args.headless,
        mock_scenario=args.scenario,
    )

    try:
        app.load()
    except (FileNotFoundError, ImportError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 3
    except Exception as exc:
        logger.error("Failed to initialise the pipeline: %s", exc)
        return 3

    # ---- create the input source ----------------------------------------
    try:
        source = create_source(
            args.source,
            input_path=args.input,
            camera_id=args.camera_id,
            loop=args.loop,
            recursive=args.recursive,
        )
    except Exception as exc:
        print(f"\nInput error: {exc}\n", file=sys.stderr)
        return 4

    # ---- run --------------------------------------------------------------
    try:
        if args.source == "image":
            import time
            app.stats.start_time = time.time()
            return run_image_mode(app, args, source)
        app.run(source)
    except SourceError as exc:
        print(f"\nInput error: {exc}\n", file=sys.stderr)
        return 4
    except Exception as exc:
        logger.error("Run failed: %s", exc, exc_info=level == "DEBUG")
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
