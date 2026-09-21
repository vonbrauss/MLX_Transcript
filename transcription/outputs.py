"""Output tree layout, transcript formatting, and atomic transcript writing.

Two parallel trees are produced beneath the chosen output parent::

    <output parent>/Transcription/ScriptSync/<relative folder>/<prefix> ScriptSync.txt
    <output parent>/Transcription/Timecoded/<relative folder>/<prefix> timecoded.txt

Both trees mirror the folder hierarchy found below the source root. Nothing
else is written: no JSON, no subtitle files, and no leftover temporary files.
"""

from __future__ import annotations

import os
import re
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from .speakers import scriptsync_speaker_label
from .timecode import ZERO_TIMECODE, TimecodeConverter, TimecodeError

__all__ = [
    "TRANSCRIPTION_FOLDER",
    "SCRIPTSYNC_FOLDER",
    "TIMECODED_FOLDER",
    "SCRIPTSYNC_WRAP_WIDTH",
    "SCRIPTSYNC_PARAGRAPH_GAP",
    "FALLBACK_FRAME_RATE",
    "OutputFormat",
    "NameStyle",
    "FolderLayout",
    "OutputOptions",
    "OutputRoots",
    "TranscriptPaths",
    "atomic_write_text",
    "build_output_paths",
    "clean_scriptsync_text",
    "existing_outputs",
    "make_timecoded_text",
    "output_roots",
    "transcript_prefix",
    "write_scriptsync",
    "write_timecoded",
    "write_srt",
    "write_vtt",
]

TRANSCRIPTION_FOLDER = "Transcription"
SCRIPTSYNC_FOLDER = "ScriptSync"
TIMECODED_FOLDER = "Timecoded"
SUBTITLES_FOLDER = "Subtitles"
SCRIPTSYNC_WRAP_WIDTH = 64

#: Counting base used when a clip reports no usable frame rate at all.
FALLBACK_FRAME_RATE = Fraction(24, 1)

#: Silence in seconds between two segments that starts a new paragraph.
SCRIPTSYNC_PARAGRAPH_GAP = 2.0

# Avid reads the ScriptSync prefix as "<enclosing folder> EM DASH <clip name>".
_PREFIX_SEPARATOR = "—"


class OutputFormat(str, Enum):
    SCRIPTSYNC = "scriptsync"
    TIMECODED = "timecoded"
    SRT = "srt"
    VTT = "vtt"


class NameStyle(str, Enum):
    ORIGINAL = "original"
    TYPE_SUFFIX = "type_suffix"
    CUSTOM_SUFFIX = "custom_suffix"

    @property
    def label(self) -> str:
        return {
            NameStyle.ORIGINAL: "Original media name",
            NameStyle.TYPE_SUFFIX: "Original name + transcript type",
            NameStyle.CUSTOM_SUFFIX: "Custom suffix",
        }[self]


class FolderLayout(str, Enum):
    TREE = "tree"
    FLAT = "flat"

    @property
    def label(self) -> str:
        return {
            FolderLayout.TREE: "Keep source folder structure",
            FolderLayout.FLAT: "Put all transcripts together",
        }[self]


@dataclass(frozen=True)
class OutputOptions:
    name_style: NameStyle = NameStyle.ORIGINAL
    custom_suffix: str = "_transcript"
    folder_layout: FolderLayout = FolderLayout.TREE
    formats: tuple[OutputFormat, ...] = (
        OutputFormat.SCRIPTSYNC,
        OutputFormat.TIMECODED,
    )

    def normalized_suffix(self) -> str:
        suffix = re.sub(r'[\\/:*?"<>|]', "_", self.custom_suffix.strip())
        return suffix or "_transcript"


@dataclass(frozen=True)
class OutputRoots:
    """The three folders created beneath the chosen output parent."""

    parent: Path
    transcription: Path
    scriptsync: Path
    timecoded: Path
    subtitles: Path

    def create(self, formats: Sequence[OutputFormat] | None = None) -> None:
        """Create only the roots needed by the selected output formats."""
        if formats is None:
            formats = (OutputFormat.SCRIPTSYNC, OutputFormat.TIMECODED)
        if OutputFormat.SCRIPTSYNC in formats:
            self.scriptsync.mkdir(parents=True, exist_ok=True)
        if OutputFormat.TIMECODED in formats:
            self.timecoded.mkdir(parents=True, exist_ok=True)
        if OutputFormat.SRT in formats or OutputFormat.VTT in formats:
            self.subtitles.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TranscriptPaths:
    """Where the two transcripts for one clip belong."""

    scriptsync: Path
    timecoded: Path
    srt: Path | None = None
    vtt: Path | None = None
    formats: tuple[OutputFormat, ...] = (
        OutputFormat.SCRIPTSYNC,
        OutputFormat.TIMECODED,
    )

    def as_tuple(self) -> tuple[Path, ...]:
        choices = {
            OutputFormat.SCRIPTSYNC: self.scriptsync,
            OutputFormat.TIMECODED: self.timecoded,
            OutputFormat.SRT: self.srt,
            OutputFormat.VTT: self.vtt,
        }
        return tuple(choices[item] for item in self.formats if choices[item] is not None)


def output_roots(output_parent: Path) -> OutputRoots:
    """Return the Transcription, ScriptSync, and Timecoded roots."""
    parent = Path(output_parent)
    transcription = parent / TRANSCRIPTION_FOLDER
    return OutputRoots(
        parent=parent,
        transcription=transcription,
        scriptsync=transcription / SCRIPTSYNC_FOLDER,
        timecoded=transcription / TIMECODED_FOLDER,
        subtitles=transcription / SUBTITLES_FOLDER,
    )


def transcript_prefix(source: Path) -> str:
    """Return ``<enclosing folder> — <clip name>`` for a media file.

    Media sitting directly at the source root therefore takes the source
    folder's own name as the enclosing folder.
    """
    return f"{Path(source).parent.name} {_PREFIX_SEPARATOR} {Path(source).stem}"


def build_output_paths(
    source: Path,
    source_root: Path,
    output_parent: Path,
    options: OutputOptions | None = None,
    duplicate_number: int = 1,
    root_label: str | None = None,
) -> TranscriptPaths:
    """Return the ScriptSync and Timecoded paths for one media file."""
    source = Path(source)
    legacy = options is None
    options = options or OutputOptions(name_style=NameStyle.TYPE_SUFFIX)
    roots = output_roots(output_parent)
    relative_parent = (
        source.relative_to(Path(source_root)).parent
        if options.folder_layout is FolderLayout.TREE
        else Path()
    )
    if root_label and options.folder_layout is FolderLayout.TREE:
        relative_parent = Path(root_label) / relative_parent
    base = transcript_prefix(source) if legacy else source.stem
    if duplicate_number > 1:
        base = f"{base} ({duplicate_number})"

    def filename(kind: OutputFormat, extension: str) -> str:
        if legacy:
            labels = {OutputFormat.SCRIPTSYNC: " ScriptSync", OutputFormat.TIMECODED: " timecoded"}
            return f"{base}{labels.get(kind, '')}.{extension}"
        if options.name_style is NameStyle.TYPE_SUFFIX:
            label = {
                OutputFormat.SCRIPTSYNC: "ScriptSync",
                OutputFormat.TIMECODED: "Timecoded",
                OutputFormat.SRT: "Subtitles",
                OutputFormat.VTT: "WebVTT",
            }[kind]
            return f"{base} {_PREFIX_SEPARATOR} {label}.{extension}"
        if options.name_style is NameStyle.CUSTOM_SUFFIX:
            return f"{base}{options.normalized_suffix()}.{extension}"
        return f"{base}.{extension}"

    return TranscriptPaths(
        scriptsync=roots.scriptsync / relative_parent / filename(OutputFormat.SCRIPTSYNC, "txt"),
        timecoded=roots.timecoded / relative_parent / filename(OutputFormat.TIMECODED, "txt"),
        srt=roots.subtitles / relative_parent / filename(OutputFormat.SRT, "srt"),
        vtt=roots.subtitles / relative_parent / filename(OutputFormat.VTT, "vtt"),
        formats=options.formats,
    )


def existing_outputs(paths: TranscriptPaths) -> list[Path]:
    """Return whichever of the two transcripts already exist on disk."""
    return [path for path in paths.as_tuple() if path.exists()]


# ------------------------------------------------------------------ writing


def atomic_write_text(
    path: Path,
    text: str,
    encoding: str = "utf-8",
) -> Path:
    """Write ``text`` to ``path`` without ever leaving a half-written file.

    The content goes to a temporary file in the destination folder, is flushed
    to disk, and only then replaces the target in one filesystem operation. A
    failure part way through leaves any previous transcript untouched and
    removes the temporary file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline="",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------- scriptsync


@dataclass
class _Paragraph:
    """One block of ScriptSync text and the speaker it belongs to."""

    speaker: str
    pieces: list[str]


def _segment_paragraphs(
    segments: Sequence[dict],
    paragraph_gap: float,
) -> list[_Paragraph]:
    """Group segment text into paragraphs separated by long silences.

    A new paragraph starts when the gap between the previous segment's ``end``
    and this segment's ``start`` reaches ``paragraph_gap``, and always when the
    speaker changes. Segments without an ``end`` time, which is how a caller
    asks for one continuous block, never start a new paragraph on their own.
    """
    paragraphs: list[_Paragraph] = []
    current: _Paragraph | None = None
    previous_end: float | None = None

    for segment in segments:
        spoken = str(segment.get("text", "")).strip()
        speaker = str(segment.get("speaker", "") or "").strip()
        try:
            start = float(segment.get("start"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            start = None  # type: ignore[assignment]

        silence_break = (
            current is not None
            and previous_end is not None
            and start is not None
            and (start - previous_end) >= paragraph_gap
        )
        speaker_break = current is not None and speaker != current.speaker

        if silence_break or speaker_break:
            if current is not None and current.pieces:
                paragraphs.append(current)
            current = None

        if current is None:
            current = _Paragraph(speaker=speaker, pieces=[])

        if spoken:
            current.pieces.append(spoken)

        try:
            previous_end = float(segment.get("end"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            previous_end = None

    if current is not None and current.pieces:
        paragraphs.append(current)
    return paragraphs


def _to_scriptsync_ascii(text: str) -> str:
    """Strip dashes and non-ASCII characters and collapse whitespace.

    Wording is never changed: only the characters Avid cannot read are
    removed, and runs of whitespace become single spaces.
    """
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def clean_scriptsync_text(
    segments: Iterable[dict],
    paragraph_gap: float = SCRIPTSYNC_PARAGRAPH_GAP,
    include_speakers: bool = False,
) -> str:
    """Return Avid ScriptSync text: plain ASCII, no timestamps, no hyphens.

    Lines are hard wrapped at about 64 characters and use CRLF endings, which
    is what Avid's script integration expects. Long silences between segments
    become blank lines so the text reads in paragraphs.

    With ``include_speakers`` the speaker's name is written above their text as
    an uppercase ASCII heading. The heading is not repeated while the same
    speaker keeps talking across paragraphs.
    """
    paragraphs = _segment_paragraphs(list(segments), paragraph_gap)

    blocks: list[str] = []
    last_heading: str | None = None
    for paragraph in paragraphs:
        text = _to_scriptsync_ascii(" ".join(paragraph.pieces))
        if not text:
            continue
        lines = textwrap.wrap(
            text,
            width=SCRIPTSYNC_WRAP_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
        body = "\r\n".join(lines)

        if include_speakers:
            heading = scriptsync_speaker_label(paragraph.speaker)
            if heading and heading != last_heading:
                body = f"{heading}:\r\n{body}"
                last_heading = heading
            elif not heading:
                last_heading = None
        blocks.append(body)

    if not blocks:
        return "\r\n"
    return "\r\n\r\n".join(blocks) + "\r\n"


def write_scriptsync(
    path: Path,
    segments: Iterable[dict],
    paragraph_gap: float = SCRIPTSYNC_PARAGRAPH_GAP,
    include_speakers: bool = False,
) -> Path:
    """Atomically write the ScriptSync transcript."""
    return atomic_write_text(
        path,
        clean_scriptsync_text(segments, paragraph_gap, include_speakers),
        encoding="ascii",
    )


# ---------------------------------------------------------------- timecoded


def make_timecoded_text(
    source: Path,
    segments: Sequence[dict],
    rate: Fraction,
    source_timecode: str = ZERO_TIMECODE,
    missing_timecode: bool = False,
) -> str:
    """Return the timecoded transcript body for one clip.

    A ``source_timecode`` the converter cannot read is treated exactly like a
    missing one: counting starts from ``00:00:00:00`` and the header says so.
    A single odd file never costs the caller its transcript.
    """
    source = Path(source)
    try:
        converter = TimecodeConverter(rate, source_timecode or ZERO_TIMECODE)
    except TimecodeError:
        missing_timecode = True
        try:
            converter = TimecodeConverter(rate, ZERO_TIMECODE)
        except TimecodeError:
            # An unusable frame rate as well. Count at the audio default so the
            # transcript is still written with readable, if nominal, stamps.
            converter = TimecodeConverter(FALLBACK_FRAME_RATE, ZERO_TIMECODE)
    lines = [
        f"FILE: {source.name}",
        f"START TIMECODE: {converter.source_timecode}",
        f"FRAME RATE: {converter.rate.numerator}/{converter.rate.denominator}",
    ]
    if missing_timecode:
        lines.append("NOTE: No embedded timecode found; 00:00:00:00 was used.")
    lines.append("")

    for segment in segments:
        timestamp = converter.at_offset(float(segment.get("start", 0)))
        speaker = str(segment.get("speaker", "") or "").strip()
        spoken = " ".join(str(segment.get("text", "")).split())
        if speaker:
            lines.append(f"{timestamp}  {speaker}: {spoken}")
        else:
            lines.append(f"{timestamp}  {spoken}")
    return "\n".join(lines) + "\n"


def write_timecoded(
    path: Path,
    source: Path,
    segments: Sequence[dict],
    rate: Fraction,
    source_timecode: str = ZERO_TIMECODE,
    missing_timecode: bool = False,
) -> Path:
    """Atomically write the timecoded transcript."""
    return atomic_write_text(
        path,
        make_timecoded_text(
            source, segments, rate, source_timecode, missing_timecode
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------- subtitles


def _subtitle_time(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def _subtitle_text(segment: dict, include_speakers: bool = True) -> str:
    spoken = " ".join(str(segment.get("text", "")).split())
    speaker = str(segment.get("speaker", "") or "").strip()
    return f"{speaker}: {spoken}" if speaker and include_speakers else spoken


def make_srt_text(segments: Sequence[dict], include_speakers: bool = True) -> str:
    """Return standard SubRip subtitles using Whisper segment timing."""
    blocks: list[str] = []
    for number, segment in enumerate(segments, start=1):
        start = _subtitle_time(float(segment.get("start", 0)))
        end = _subtitle_time(max(float(segment.get("end", 0)), float(segment.get("start", 0))))
        blocks.append(f"{number}\n{start} --> {end}\n{_subtitle_text(segment, include_speakers)}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, segments: Sequence[dict], include_speakers: bool = True) -> Path:
    return atomic_write_text(path, make_srt_text(segments, include_speakers), encoding="utf-8")


def make_vtt_text(segments: Sequence[dict], include_speakers: bool = True) -> str:
    """Return WebVTT subtitles using Whisper segment timing."""
    blocks = ["WEBVTT", ""]
    for segment in segments:
        start = _subtitle_time(float(segment.get("start", 0)), ".")
        end = _subtitle_time(
            max(float(segment.get("end", 0)), float(segment.get("start", 0))), "."
        )
        blocks.extend((f"{start} --> {end}", _subtitle_text(segment, include_speakers), ""))
    return "\n".join(blocks)


def write_vtt(path: Path, segments: Sequence[dict], include_speakers: bool = True) -> Path:
    return atomic_write_text(path, make_vtt_text(segments, include_speakers), encoding="utf-8")
