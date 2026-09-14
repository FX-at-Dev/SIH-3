"""Video-file input.

Handles MP4, AVI, MOV, MKV and anything else the local OpenCV/FFmpeg build can
decode. Corrupt frames in the middle of a file are skipped rather than treated
as end-of-stream, because a partially damaged recording is still useful.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from .base import Frame, InputSource, SourceError

logger = logging.getLogger(__name__)

__all__ = ["SUPPORTED_VIDEO_EXTENSIONS", "VideoSource"]

SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".webm",
})


class VideoSource(InputSource):
    """Frame-by-frame reader for a recorded video file.

    Args:
        path: Path to the video file.
        loop: Restart from frame 0 at end-of-file (useful for demo loops).
        start_frame: Frame index to seek to before reading.
        max_skip: Consecutive decode failures tolerated before stopping.
    """

    mode = "video"

    def __init__(
        self,
        path: str | Path,
        loop: bool = False,
        start_frame: int = 0,
        max_skip: int = 30,
    ) -> None:
        self.path = Path(path)
        self.loop = loop
        self.start_frame = max(0, start_frame)
        self.max_skip = max_skip

        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._total_frames: Optional[int] = None
        self._fps: Optional[float] = None
        self._skipped = 0

    def open(self) -> None:
        if not self.path.exists():
            raise SourceError(f"Video file not found: {self.path}")
        if not self.path.is_file():
            raise SourceError(f"Not a file: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            logger.warning(
                "Extension '%s' is outside the tested set (%s). Attempting to "
                "decode anyway.",
                suffix, ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS)),
            )

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"Could not open video: {self.path}\n"
                "  The file may be corrupt, or this OpenCV build may lack the "
                "required codec. Try re-encoding to H.264 MP4:\n"
                f'    ffmpeg -i "{self.path}" -c:v libx264 -pix_fmt yuv420p out.mp4'
            )

        self._cap = cap

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._total_frames = count if count > 0 else None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if 0.0 < fps < 240.0 else None

        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.start_frame))
            self._index = self.start_frame

        res = self.resolution
        logger.info(
            "Opened video %s (%s, %s frames, %.2f fps).",
            self.path.name,
            f"{res[0]}x{res[1]}" if res else "unknown size",
            self._total_frames if self._total_frames else "unknown",
            self._fps or 0.0,
        )

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise SourceError("VideoSource.read() called before open().")

        skips = 0
        while skips <= self.max_skip:
            ok, image = self._cap.read()

            if ok and image is not None and image.size > 0:
                source_time = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                is_last = (
                    self._total_frames is not None
                    and self._index >= self._total_frames - 1
                    and not self.loop
                )
                frame = Frame(
                    image=image,
                    index=self._index,
                    timestamp=source_time if source_time > 0 else self._index / (self._fps or 30.0),
                    source_time=source_time,
                    is_last=is_last,
                )
                self._index += 1
                return frame

            pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            at_end = (
                self._total_frames is not None and pos >= self._total_frames
            ) or pos <= 0

            if at_end:
                if self.loop:
                    logger.debug("Looping video back to frame 0.")
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    self._index = 0
                    skips += 1
                    continue
                logger.info(
                    "End of video after %d frames (%d skipped).",
                    self._index, self._skipped,
                )
                return None

            skips += 1
            self._skipped += 1
            self._index += 1
            logger.debug("Skipping undecodable frame at index %d.", pos)

        logger.warning(
            "Stopping: %d consecutive frames failed to decode in %s.",
            skips, self.path.name,
        )
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def total_frames(self) -> Optional[int]:
        return self._total_frames

    @property
    def native_fps(self) -> Optional[float]:
        return self._fps

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        if self._cap is None:
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None

    @property
    def progress(self) -> Optional[float]:
        """Playback position as a 0-1 fraction, when the length is known."""
        if not self._total_frames:
            return None
        return min(1.0, self._index / self._total_frames)

    @property
    def name(self) -> str:
        return f"video:{self.path.name}"
