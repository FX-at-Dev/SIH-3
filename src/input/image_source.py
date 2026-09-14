"""Still-image input.

Yields a single frame (or, in directory mode, one frame per image file). The
application handles the single-frame case by analysing it and holding the
result on screen rather than looping.

Tracking and velocity are meaningless for one still frame; the pipeline detects
this via ``is_single_shot`` and reports distances and risk without motion terms.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Frame, InputSource, SourceError

logger = logging.getLogger(__name__)

__all__ = ["SUPPORTED_IMAGE_EXTENSIONS", "ImageSource"]

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif",
})


class ImageSource(InputSource):
    """Reads one image file, or every image in a directory.

    Args:
        path: Image file, or a directory of images.
        recursive: Recurse into sub-directories when ``path`` is a directory.
    """

    mode = "image"

    def __init__(self, path: str | Path, recursive: bool = False) -> None:
        self.path = Path(path)
        self.recursive = recursive
        self._files: list[Path] = []
        self._cursor = 0
        self._index = 0

    def open(self) -> None:
        if not self.path.exists():
            raise SourceError(f"Image path not found: {self.path}")

        if self.path.is_dir():
            pattern = "**/*" if self.recursive else "*"
            self._files = sorted(
                p for p in self.path.glob(pattern)
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            if not self._files:
                raise SourceError(
                    f"No supported images in {self.path}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                )
            logger.info("Found %d image(s) in %s.", len(self._files), self.path)
        else:
            suffix = self.path.suffix.lower()
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                raise SourceError(
                    f"Unsupported image format '{suffix}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                )
            self._files = [self.path]

        self._cursor = 0

    def read(self) -> Optional[Frame]:
        while self._cursor < len(self._files):
            file_path = self._files[self._cursor]
            self._cursor += 1

            image = self._imread_unicode(file_path)
            if image is None:
                logger.warning("Could not decode image, skipping: %s", file_path)
                continue

            frame = Frame(
                image=image,
                index=self._index,
                timestamp=time.time(),
                source_time=0.0,
                is_last=self._cursor >= len(self._files),
            )
            frame_path_note = file_path.name
            logger.info(
                "Loaded image %d/%d: %s (%dx%d)",
                self._cursor, len(self._files), frame_path_note,
                frame.width, frame.height,
            )
            self._index += 1
            self._current_file = file_path
            return frame

        return None

    @staticmethod
    def _imread_unicode(path: Path) -> Optional[np.ndarray]:
        """Read an image, tolerating non-ASCII paths.

        ``cv2.imread`` fails silently on Windows for paths containing non-ASCII
        characters, so decode from bytes instead.
        """
        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
            if buffer.size == 0:
                return None
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return image if image is not None and image.size > 0 else None
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    def release(self) -> None:
        self._files = []
        self._cursor = 0

    @property
    def total_frames(self) -> Optional[int]:
        return len(self._files) if self._files else None

    @property
    def is_single_shot(self) -> bool:
        """True when this source yields exactly one frame."""
        return len(self._files) == 1

    @property
    def current_file(self) -> Optional[Path]:
        """The file most recently returned by :meth:`read`."""
        return getattr(self, "_current_file", None)

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    @property
    def name(self) -> str:
        return f"image:{self.path.name}"
