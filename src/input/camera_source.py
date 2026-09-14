"""Live camera / webcam input.

Designed to survive the messy realities of USB cameras: a device that needs a
moment to warm up, occasional dropped frames, and disconnection mid-session.
A single failed frame is skipped; only sustained failure is reported as a
disconnection.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from .base import Frame, InputSource, SourceError

logger = logging.getLogger(__name__)

__all__ = ["CameraSource"]


class CameraSource(InputSource):
    """Webcam or attached camera, read through OpenCV.

    Args:
        camera_id: Device index, as understood by ``cv2.VideoCapture``.
        width: Requested capture width (the driver may ignore it).
        height: Requested capture height.
        warmup_frames: Frames discarded after opening, to let auto-exposure settle.
        max_consecutive_failures: Failed reads tolerated before declaring the
            camera disconnected.
    """

    mode = "camera"

    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        warmup_frames: int = 3,
        max_consecutive_failures: int = 30,
    ) -> None:
        self.camera_id = camera_id
        self.requested_width = width
        self.requested_height = height
        self.warmup_frames = warmup_frames
        self.max_consecutive_failures = max_consecutive_failures

        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._failures = 0
        self._total_dropped = 0

    def open(self) -> None:
        logger.info("Opening camera device %d ...", self.camera_id)
        backends = (
            [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
        )

        cap = None
        for backend in backends:
            candidate = cv2.VideoCapture(self.camera_id, backend)
            if candidate.isOpened():
                cap = candidate
                logger.debug("Camera opened with backend %s", backend)
                break
            candidate.release()

        if cap is None:
            raise SourceError(
                f"Camera {self.camera_id} could not be opened.\n"
                "  - Check that a camera is connected and not in use by another app.\n"
                "  - Try a different index, e.g. --camera-id 1\n"
                "  - On Windows, check Settings > Privacy > Camera access."
            )

        if self.requested_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
        if self.requested_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))

        self._cap = cap

        for _ in range(max(0, self.warmup_frames)):
            cap.read()

        ok, _ = cap.read()
        if not ok:
            self.release()
            raise SourceError(
                f"Camera {self.camera_id} opened but returned no frames. "
                "The device may be in use by another application."
            )

        res = self.resolution
        logger.info(
            "Camera %d ready (%s).",
            self.camera_id,
            f"{res[0]}x{res[1]}" if res else "unknown resolution",
        )

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise SourceError("CameraSource.read() called before open().")

        while self._failures < self.max_consecutive_failures:
            ok, image = self._cap.read()
            if ok and image is not None and image.size > 0:
                self._failures = 0
                now = time.time()
                frame = Frame(
                    image=image,
                    index=self._index,
                    timestamp=now,
                    source_time=now,
                )
                self._index += 1
                return frame

            self._failures += 1
            self._total_dropped += 1
            if self._failures == 1:
                logger.debug("Dropped a camera frame; continuing.")
            time.sleep(0.005)

        logger.error(
            "Camera %d appears disconnected after %d consecutive failed reads "
            "(%d frames dropped in total).",
            self.camera_id, self._failures, self._total_dropped,
        )
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(
                "Camera %d released after %d frames (%d dropped).",
                self.camera_id, self._index, self._total_dropped,
            )

    @property
    def native_fps(self) -> Optional[float]:
        if self._cap is None:
            return None
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return fps if 1.0 < fps < 240.0 else None

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        if self._cap is None:
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None

    @property
    def is_live(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"camera:{self.camera_id}"
