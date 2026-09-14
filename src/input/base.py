"""Input source abstraction.

The application loop reads frames through :class:`InputSource` and never learns
whether they came from a webcam, a video file, or a single still image. Adding
an RTSP stream or a ROS bag later means implementing this interface only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Frame", "InputSource", "SourceError"]


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed unrecoverably."""


@dataclass(slots=True)
class Frame:
    """One frame plus the metadata the pipeline needs to reason about time."""

    image: np.ndarray
    index: int
    timestamp: float
    source_time: float = 0.0
    is_last: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class InputSource(ABC):
    """Common interface for every frame provider."""

    mode: str = "unknown"

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource. Raises :class:`SourceError`."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` when the source is exhausted.

        Implementations should return ``None`` only at genuine end-of-stream.
        A single corrupt frame should be skipped internally, not treated as
        the end of the source.
        """

    @abstractmethod
    def release(self) -> None:
        """Release the underlying resource. Must be safe to call twice."""

    # ---- optional metadata, overridden where it is knowable ---------------

    @property
    def total_frames(self) -> Optional[int]:
        """Frame count when known (video files), else ``None``."""
        return None

    @property
    def native_fps(self) -> Optional[float]:
        """Source frame rate when known, else ``None``."""
        return None

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        """``(width, height)`` when known, else ``None``."""
        return None

    @property
    def is_live(self) -> bool:
        """True for real-time sources that must not be throttled to a clock."""
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ---- context manager + iteration --------------------------------------

    def __enter__(self) -> "InputSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
