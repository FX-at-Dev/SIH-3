"""Application orchestrator.

Wires the pipeline together and runs the main loop::

    frame -> PerceptionManager -> RiskEngine -> DecisionEngine -> Renderer

This module knows nothing about YOLO, Kalman filters or OpenCV window details;
it composes interfaces. Adding a depth camera or swapping the tracker changes
the factories, not this file.

Robustness contract: one bad frame never stops a run. Detection, tracking,
risk, decision and rendering are each isolated, and a failure in any of them
degrades that frame's output rather than terminating the session.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .core.config import Config
from .core.types import (
    ActionType,
    DrivingRecommendation,
    FrameMetrics,
    RiskLevel,
    WorldState,
)
from .decision.decision_engine import DecisionEngine
from .geometry.ego_path import EgoPath
from .input.base import Frame, InputSource, SourceError
from .input.image_source import ImageSource
from .input.video_source import VideoSource
from .logs.event_logger import EventLogger, SessionRecorder
from .perception.mock_detector import MockDetector
from .perception.perception_manager import PerceptionManager
from .risk.risk_engine import RiskEngine
from .visualization.renderer import Renderer

logger = logging.getLogger(__name__)

__all__ = ["Application", "SessionStats"]

WINDOW_NAME = "Indian Road Perception"

HELP_TEXT = """
Keyboard controls
  Q / ESC  quit                    S  save screenshot
  R        reset tracking          D  toggle detection details
  SPACE    pause / resume          M  toggle dashboard panel
  T        toggle trajectories     A  toggle drivable-area overlay
  H        print this help
"""


@dataclass(slots=True)
class SessionStats:
    """Aggregate statistics for a whole run."""

    frames: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    total_ms: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    decision_ms: float = 0.0
    risk_ms: float = 0.0
    render_ms: float = 0.0
    max_objects: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    risk_counts: dict[str, int] = field(default_factory=dict)
    dropped_frames: int = 0

    def record(
        self, metrics: FrameMetrics, recommendation: DrivingRecommendation
    ) -> None:
        self.frames += 1
        self.total_ms += metrics.total_ms
        self.inference_ms += metrics.inference_ms
        self.tracking_ms += metrics.tracking_ms
        self.decision_ms += metrics.decision_ms
        self.risk_ms += metrics.risk_ms
        self.render_ms += metrics.render_ms
        self.max_objects = max(self.max_objects, metrics.object_count)
        key = recommendation.label
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        risk = recommendation.risk.value
        self.risk_counts[risk] = self.risk_counts.get(risk, 0) + 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def average_fps(self) -> float:
        return self.frames / self.duration if self.duration > 1e-6 else 0.0

    def to_dict(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        n = max(1, self.frames)
        payload: dict[str, Any] = {
            "frames_processed": self.frames,
            "duration_seconds": round(self.duration, 2),
            "average_fps": round(self.average_fps, 2),
            "average_total_ms": round(self.total_ms / n, 2),
            "average_inference_ms": round(self.inference_ms / n, 2),
            "average_tracking_ms": round(self.tracking_ms / n, 2),
            "average_risk_ms": round(self.risk_ms / n, 2),
            "average_decision_ms": round(self.decision_ms / n, 2),
            "average_render_ms": round(self.render_ms / n, 2),
            "max_simultaneous_objects": self.max_objects,
            "recommendation_distribution": dict(self.action_counts),
            "risk_distribution": dict(self.risk_counts),
            "dropped_frames": self.dropped_frames,
        }
        if extra:
            payload.update(extra)
        return payload


class Application:
    """Owns the pipeline and the main loop."""

    def __init__(
        self,
        config: Config,
        save_results: bool = False,
        headless: bool = False,
        mock_scenario: Optional[str] = None,
    ) -> None:
        self.config = config
        self.save_results = save_results
        self.headless = headless

        self.ego_path = EgoPath(config)
        self.perception = PerceptionManager(config, self.ego_path)
        self.risk_engine = RiskEngine(config, self.ego_path)
        self.decision_engine = DecisionEngine(config, self.ego_path)
        self.renderer = Renderer(config)
        self.event_logger = EventLogger(config)
        self.recorder = SessionRecorder(config, enabled=save_results)

        if mock_scenario and isinstance(self.perception.detector, MockDetector):
            self.perception.detector.scenario = mock_scenario

        self.stats = SessionStats()
        self._paused = False
        self._quit = False
        self._last_render: Optional[np.ndarray] = None
        self._fps_window: list[float] = []
        self._last_total_ms = 0.0
        self._last_fps = 0.0

    # ---- setup ------------------------------------------------------------

    def load(self) -> None:
        """Load models and print what is actually active."""
        self.perception.load()
        logger.info("Pipeline ready:")
        for line in self.perception.status_lines:
            logger.info("  %s", line)
        logger.info(
            "  SAFETY: prototype decision-support output only; not connected "
            "to vehicle control."
        )

    # ---- main loop --------------------------------------------------------

    def run(self, source: InputSource) -> SessionStats:
        """Process a source to completion."""
        source.open()

        single_shot = isinstance(source, ImageSource) and source.is_single_shot
        self.recorder.start(source.name)
        self.event_logger.log_session_start(source.name, {
            "mode": source.mode,
            "resolution": source.resolution,
            "native_fps": source.native_fps,
            "total_frames": source.total_frames,
            "detector": self.perception.detector.info.name,
            "model": self.perception.detector.info.model_path,
        })

        if not self.headless:
            logger.info("%s", HELP_TEXT)

        self.stats.start_time = time.time()
        video_opened = False

        try:
            while not self._quit:
                if self._paused:
                    if self._handle_key(cv2.waitKey(30) & 0xFF, None):
                        continue
                    continue

                frame = self._read_frame(source)
                if frame is None:
                    break

                rendered, world, recommendation = self.process_frame(frame, source)

                if self.save_results and not video_opened and rendered is not None:
                    self.recorder.open_video(
                        rendered.shape[1], rendered.shape[0],
                        source.native_fps or self.config.output.video_fps,
                    )
                    video_opened = True

                self.recorder.record_frame(
                    world, recommendation, world.metrics, rendered
                )

                if not self.headless and rendered is not None:
                    display = self.renderer.fit_to_screen(rendered)
                    cv2.imshow(WINDOW_NAME, display)
                    wait = 0 if single_shot else 1
                    if self._handle_key(cv2.waitKey(wait) & 0xFF, rendered):
                        break
                    if single_shot:
                        break
                elif single_shot:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stats.end_time = time.time()
            source.release()
            self._shutdown()

        return self.stats

    def _read_frame(self, source: InputSource) -> Optional[Frame]:
        """Read one frame, tolerating transient failures."""
        try:
            return source.read()
        except SourceError as exc:
            logger.error("Input source failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error reading from source: %s", exc)
            return None

    # ---- per-frame pipeline ----------------------------------------------

    def process_frame(
        self, frame: Frame, source: Optional[InputSource] = None
    ) -> tuple[Optional[np.ndarray], WorldState, DrivingRecommendation]:
        """Run the full pipeline on one frame and render the result."""
        loop_start = time.perf_counter()

        image = self._prepare(frame.image)
        height, width = image.shape[:2]

        # ---- perception ---------------------------------------------------
        perception = self.perception.process(image, frame.index, frame.timestamp)
        metrics = perception.metrics

        # ---- risk ---------------------------------------------------------
        t0 = time.perf_counter()
        try:
            scene_risk = self.risk_engine.evaluate(perception.objects, width, height)
        except Exception as exc:
            logger.error("Risk evaluation failed on frame %d: %s", frame.index, exc)
            from .core.types import SceneRisk
            scene_risk = SceneRisk(
                level=RiskLevel.HIGH,
                reasons=["risk engine error - defaulting to cautious state"],
                perception_confidence=0.0,
            )
        metrics.risk_ms = (time.perf_counter() - t0) * 1000.0

        world = WorldState(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            frame_width=width,
            frame_height=height,
            objects=perception.objects,
            scene_risk=scene_risk,
            ego_path_polygon=self.ego_path.polygon,
            drivable_ratio=perception.drivable.drivable_ratio,
            drivable_available=perception.drivable.available,
            metrics=metrics,
            dt=perception.dt,
        )

        # ---- decision -----------------------------------------------------
        t0 = time.perf_counter()
        try:
            recommendation = self.decision_engine.decide(world)
        except Exception as exc:
            logger.error("Decision engine failed on frame %d: %s", frame.index, exc)
            recommendation = DrivingRecommendation(
                action=ActionType.STOP,
                target_speed_kmh=0.0,
                risk=RiskLevel.CRITICAL,
                reason=(
                    "Decision engine error. Defaulting to STOP because the "
                    "system cannot currently assess the scene."
                ),
                confidence=0.0,
                frame_index=frame.index,
            )
        metrics.decision_ms = (time.perf_counter() - t0) * 1000.0

        # ---- events -------------------------------------------------------
        self.event_logger.update(world, recommendation)

        # ---- render -------------------------------------------------------
        metrics.total_ms = self._last_total_ms
        metrics.fps = self._last_fps

        t0 = time.perf_counter()
        rendered: Optional[np.ndarray] = None
        if self.config.visualization.enabled:
            try:
                rendered = self.renderer.render(
                    image, world, recommendation, metrics,
                    drivable_mask=perception.drivable.mask,
                    status_lines=self.perception.status_lines,
                    extra=self._extra_panel(source),
                )
            except Exception as exc:
                logger.error("Rendering failed on frame %d: %s", frame.index, exc)
                rendered = image
        metrics.render_ms = (time.perf_counter() - t0) * 1000.0

        metrics.total_ms = (time.perf_counter() - loop_start) * 1000.0
        metrics.fps = self._smoothed_fps(metrics.total_ms)
        self._last_total_ms = metrics.total_ms
        self._last_fps = metrics.fps

        self.stats.record(metrics, recommendation)
        self._last_render = rendered
        return rendered, world, recommendation

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Resize to the configured processing width."""
        target = self.config.app.process_width
        height, width = image.shape[:2]
        if width <= target:
            return image
        scale = target / width
        return cv2.resize(
            image, (target, int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _smoothed_fps(self, total_ms: float) -> float:
        """Rolling FPS over the last 15 frames, so the readout is stable."""
        if total_ms <= 0.0:
            return 0.0
        self._fps_window.append(1000.0 / total_ms)
        if len(self._fps_window) > 15:
            del self._fps_window[0]
        return sum(self._fps_window) / len(self._fps_window)

    def _extra_panel(self, source: Optional[InputSource]) -> dict[str, str]:
        info: dict[str, str] = {}
        if isinstance(source, VideoSource):
            progress = source.progress
            if progress is not None:
                info["Progress"] = f"{progress * 100:.0f}%"
            if source.total_frames:
                info["Frames"] = f"{source.total_frames}"
        info["Replans"] = str(self.event_logger.counters["replans"])
        if self._paused:
            info["State"] = "PAUSED"
        return info

    # ---- keyboard ---------------------------------------------------------

    def _handle_key(self, key: int, rendered: Optional[np.ndarray]) -> bool:
        """Process one keypress. Returns True when the loop should break."""
        if key in (ord("q"), ord("Q"), 27):
            logger.info("Quit requested.")
            self._quit = True
            return True

        if key in (ord("r"), ord("R")):
            self.perception.reset()
            self.decision_engine.reset()
            logger.info("Tracking and decision state reset.")
        elif key in (ord("s"), ord("S")):
            image = rendered if rendered is not None else self._last_render
            if image is not None:
                self.recorder.save_screenshot(image, "frame")
        elif key in (ord("d"), ord("D")):
            state = self.renderer.toggle_details()
            logger.info("Detection details: %s", "on" if state else "off")
        elif key in (ord("m"), ord("M")):
            state = self.renderer.toggle_dashboard()
            logger.info("Dashboard panel: %s", "on" if state else "off")
        elif key in (ord("t"), ord("T")):
            state = self.renderer.toggle_trajectories()
            logger.info("Trajectories: %s", "on" if state else "off")
        elif key in (ord("a"), ord("A")):
            state = self.renderer.toggle_drivable()
            logger.info("Drivable-area overlay: %s", "on" if state else "off")
        elif key == ord(" "):
            self._paused = not self._paused
            logger.info("%s", "Paused." if self._paused else "Resumed.")
        elif key in (ord("h"), ord("H")):
            logger.info("%s", HELP_TEXT)

        return False

    # ---- shutdown ---------------------------------------------------------

    def _shutdown(self) -> None:
        self.event_logger.log_session_end(self.stats.frames)
        summary = self.stats.to_dict({
            "events": self.event_logger.counters,
            "detector": self.perception.detector.info.name,
            "model": self.perception.detector.info.model_path,
            "pothole_mode": self.perception.pothole_detector.mode,
            "config": self.config.source_path,
            "notice": (
                "Prototype driver-assistance / autonomous-driving research "
                "system. Not intended for direct control of a real vehicle."
            ),
        })
        session_dir = self.recorder.finish(self.event_logger, summary)

        self.perception.release()
        if not self.headless:
            cv2.destroyAllWindows()

        self._print_summary(summary, session_dir)

    @staticmethod
    def _print_summary(summary: dict[str, Any], session_dir: Optional[Path]) -> None:
        logger.info("=" * 62)
        logger.info("SESSION SUMMARY")
        logger.info("  Frames processed : %s", summary["frames_processed"])
        logger.info("  Duration         : %.1f s", summary["duration_seconds"])
        logger.info("  Average FPS      : %.2f", summary["average_fps"])
        logger.info("  Avg inference    : %.1f ms", summary["average_inference_ms"])
        logger.info("  Avg total/frame  : %.1f ms", summary["average_total_ms"])
        logger.info("  Max objects      : %s", summary["max_simultaneous_objects"])
        if summary["recommendation_distribution"]:
            logger.info("  Recommendations  :")
            total = max(1, summary["frames_processed"])
            for action, count in sorted(
                summary["recommendation_distribution"].items(),
                key=lambda kv: -kv[1],
            ):
                logger.info("      %-12s %5d  (%4.1f%%)",
                            action, count, 100.0 * count / total)
        events = summary.get("events", {})
        if events:
            logger.info("  Events           : %s", ", ".join(
                f"{k}={v}" for k, v in events.items() if v
            ) or "none")
        if session_dir:
            logger.info("  Output           : %s", session_dir)
        logger.info("=" * 62)
