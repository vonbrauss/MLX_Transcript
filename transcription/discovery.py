"""Recursive media discovery below a source folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "MEDIA_EXTENSIONS",
    "MEDIA_NAME_FILTER",
    "NON_MEDIA_EXTENSIONS",
    "EXCLUDED_FILENAMES",
    "DiscoveredMedia",
    "discover_media",
    "is_probe_candidate",
    "is_supported_media",
    "iter_media",
    "skip_reason_for",
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

#: The file-dialog filter. The well-known extensions come first as a
#: convenience, and All Files is offered second because a clip with an
#: unusual extension is still perfectly transcribable.
MEDIA_NAME_FILTER = "Media files ({});;All files (*)".format(
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


#: Filenames that are never media, whatever else is in the folder. These are
#: the sidecars macOS and Windows scatter around.
EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {".ds_store", "thumbs.db", "desktop.ini", ".localized", "icon\r"}
)

#: Extensions that cannot carry a decodable audio stream, so probing them
#: would only cost time. This list is the optional fast path and nothing
#: more: it is deliberately limited to formats that are definitively not
#: media, because a wrong entry here silently drops a real clip. Anything not
#: named here is probed, including a file with no extension at all.
NON_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Documents and text
        ".pdf", ".txt", ".md", ".rtf", ".doc", ".docx", ".pages",
        ".xls", ".xlsx", ".numbers", ".csv", ".tsv", ".ppt", ".pptx", ".key",
        # Images
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic",
        ".heif", ".webp", ".psd", ".ai", ".eps", ".svg", ".icns", ".ico",
        # Subtitles and captions, which are our own output as well
        ".srt", ".vtt", ".scc", ".ass", ".ssa", ".sub", ".stl", ".itt",
        # Editorial project and exchange files
        ".edl", ".aaf", ".fcpxml", ".xml", ".drp", ".prproj", ".veg", ".als",
        ".logicx", ".ptx", ".sesx",
        # Waveform and thumbnail caches
        ".pkf", ".wfm", ".sfk", ".asd", ".pek", ".cfa",
        # Archives, disk images and code
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".dmg", ".iso",
        ".pkg", ".exe", ".app", ".py", ".js", ".json", ".yaml", ".yml",
        ".html", ".htm", ".css", ".log", ".plist", ".db", ".sqlite",
    }
)


def is_supported_media(path: Path) -> bool:
    """Return True when the suffix is one of the well-known media extensions.

    This is the fast path, not the rule. A file that fails this test is still
    offered to ffprobe, because the only reliable answer to "can this be
    transcribed" is whether it holds an audio stream.
    """
    return path.suffix.lower() in MEDIA_EXTENSIONS


def is_probe_candidate(path: Path) -> bool:
    """Return True when this file is worth asking ffprobe about.

    Everything normal and non-hidden is a candidate. Only the sidecars and
    the formats that definitively cannot hold audio are turned away, so a
    clip with an unusual extension, or none, still reaches the probe.
    """
    if path.name.lower() in EXCLUDED_FILENAMES:
        return False
    if is_supported_media(path):
        return True
    return path.suffix.lower() not in NON_MEDIA_EXTENSIONS


def skip_reason_for(path: Path) -> str:
    """Explain, for the skipped-files summary, why a file was never probed."""
    if path.name.lower() in EXCLUDED_FILENAMES:
        return "system file"
    suffix = path.suffix.lower()
    if suffix in NON_MEDIA_EXTENSIONS:
        return f"{suffix} is not a media file"
    return "not a media file"


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
    rejected: list[tuple[Path, str]] | None = None,
) -> Iterator[Path]:
    """Yield every file below ``source_root`` worth probing, in sorted order.

    ``rejected`` collects ``(path, reason)`` for files turned away by the
    cheap filter, so the window can say what it skipped. Hidden files,
    AppleDouble sidecars and system files are not collected: naming every
    ``.DS_Store`` would bury the entries that matter.
    """
    source_root = Path(source_root)
    excluded = [Path(root) for root in excluded_roots]

    candidates = [source_root] if source_root.is_file() else sorted(source_root.rglob("*"))
    hidden_root = source_root.parent if source_root.is_file() else source_root
    for path in candidates:
        if not path.is_file():
            continue
        if skip_hidden and _is_hidden(path, hidden_root):
            continue
        if not is_probe_candidate(path):
            reason = skip_reason_for(path)
            if rejected is not None and reason != "system file":
                rejected.append((path, reason))
            continue
        if any(root == path or root in path.parents for root in excluded):
            continue
        yield path


def discover_media(
    source_root: Path,
    excluded_roots: Iterable[Path] = (),
    skip_hidden: bool = True,
    rejected: list[tuple[Path, str]] | None = None,
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
        for path in iter_media(source_root, excluded_roots, skip_hidden, rejected)
    ]
