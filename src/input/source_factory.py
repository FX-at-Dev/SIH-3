"""Factory that builds the right :class:`InputSource` for the CLI arguments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .base import InputSource, SourceError
from .camera_source import CameraSource
from .image_source import SUPPORTED_IMAGE_EXTENSIONS, ImageSource
from .video_source import SUPPORTED_VIDEO_EXTENSIONS, VideoSource

logger = logging.getLogger(__name__)

__all__ = ["create_source", "infer_source_type"]


def infer_source_type(input_path: str | Path) -> str:
    """Guess ``"video"`` or ``"image"`` from a path's extension."""
    suffix = Path(input_path).suffix.lower()
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if Path(input_path).is_dir():
        return "image"
    raise SourceError(
        f"Cannot infer source type from '{input_path}'. "
        "Pass --source video or --source image explicitly."
    )


def create_source(
    source_type: str,
    input_path: Optional[str | Path] = None,
    camera_id: int = 0,
    **kwargs: Any,
) -> InputSource:
    """Create an input source.

    Args:
        source_type: ``"camera"``, ``"video"``, or ``"image"``.
        input_path: Required for video and image modes.
        camera_id: Device index for camera mode.
        **kwargs: Forwarded to the concrete source (``loop``, ``recursive``, ...).

    Raises:
        SourceError: For an unknown type or a missing required path.
    """
    source_type = source_type.lower().strip()

    if source_type == "camera":
        return CameraSource(
            camera_id=camera_id,
            width=kwargs.get("width"),
            height=kwargs.get("height"),
        )

    if source_type in ("video", "image"):
        if input_path is None:
            raise SourceError(
                f"--source {source_type} requires --input <path>."
            )
        path = Path(input_path)
        if not path.exists():
            raise SourceError(f"Input path does not exist: {path}")

        if source_type == "video":
            return VideoSource(
                path,
                loop=bool(kwargs.get("loop", False)),
                start_frame=int(kwargs.get("start_frame", 0)),
            )
        return ImageSource(path, recursive=bool(kwargs.get("recursive", False)))

    raise SourceError(
        f"Unknown source type '{source_type}'. Expected: camera, video, image."
    )
