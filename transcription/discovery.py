"""Recursive media discovery below a source folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "MEDIA_EXTENSIONS",
    "MEDIA_NAME_FILTER",
    "DiscoveredMedia",
    "discover_media",
    "is_supported_media",
    "iter_media",
]

#: Containers the bundled FFmpeg can demux and decode audio from.
#:
#: MXF, AVI, MTS, M2TS, and WMV were added for editorial work: MXF is what
#: most broadcast and Avid workflows hand over, and the AVCHD pair turn up
#: straight off camera cards. Everything here is decode-only as far as this
#: application is concerned, so adding a container costs nothing but the
#: recognition.
MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".aac", ".aif", ".aiff", ".avi", ".flac", ".m2ts", ".m4a", ".m4v",
        ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".mts", ".mxf",
        ".ogg", ".opus", ".wav", ".webm", ".wma", ".wmv",
    }
)

#: The same list as a file-dialog filter, built once so the picker and the
#: drop handler can never disagree about what is supported.
MEDIA_NAME_FILTER = "Media files ({})".format(
    " ".join(f"*{suffix}" for suffix in sorted(MEDIA_EXTENSIONS))
)


@dataclass(frozen=True)
class DiscoveredMedia:
    """One media file found below the source root."""

    path: Path
    source_root: Path

    @property
    def relative_path(self) -> Path:
        """Path of the file relative to the source root."""
        return self.path.relative_to(self.source_root)

    @property
    def relative_folder(self) -> Path:
        """Folder holding the file, relative to the source root."""
        return self.relative_path.parent

    @property
    def relative_folder_label(self) -> str:
        """Human readable relative folder, ``.`` shown as the root marker."""
        folder = str(self.relative_folder)
        return "/" if folder == "." else folder


def is_supported_media(path: Path) -> bool:
    """Return True when the suffix is one of the supported media extensions."""
    return path.suffix.lower() in MEDIA_EXTENSIONS


def _is_hidden(path: Path, source_root: Path) -> bool:
    """Return True when any path component below the root starts with a dot.

    This keeps macOS AppleDouble sidecars (``._clip.mov``) and hidden support
    folders out of the queue.
    """
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return path.name.startswith(".")
    return any(part.startswith(".") for part in relative.parts)


def iter_media(
    source_root: Path,
    excluded_roots: Iterable[Path] = (),
    skip_hidden: bool = True,
) -> Iterator[Path]:
    """Yield supported media files below ``source_root`` in sorted order."""
    source_root = Path(source_root)
    excluded = [Path(root) for root in excluded_roots]

    candidates = [source_root] if source_root.is_file() else sorted(source_root.rglob("*"))
    hidden_root = source_root.parent if source_root.is_file() else source_root
    for path in candidates:
        if not path.is_file() or not is_supported_media(path):
            continue
        if skip_hidden and _is_hidden(path, hidden_root):
            continue
        if any(root == path or root in path.parents for root in excluded):
            continue
        yield path


def discover_media(
    source_root: Path,
    excluded_roots: Iterable[Path] = (),
    skip_hidden: bool = True,
) -> list[DiscoveredMedia]:
    """Return every supported media file below ``source_root``.

    ``excluded_roots`` normally holds the ``Transcription`` output folder so a
    previous run's transcripts, or media stored beneath the output tree, are
    never queued.
    """
    source_root = Path(source_root)
    relative_root = source_root.parent if source_root.is_file() else source_root
    return [
        DiscoveredMedia(path=path, source_root=relative_root)
        for path in iter_media(source_root, excluded_roots, skip_hidden)
    ]
