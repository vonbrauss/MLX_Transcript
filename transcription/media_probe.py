"""ffprobe inspection: duration, embedded timecode, and frame rate."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .timecode import ZERO_TIMECODE, TimecodeError, parse_timecode
from app.runtime import bundled_binary

__all__ = [
    "DEFAULT_AUDIO_FRAME_RATE",
    "FFPROBE",
    "usable_timecode",
    "MediaInfo",
    "MediaProbeError",
    "ffprobe_available",
    "format_duration",
    "parse_probe_payload",
    "probe_media",
]

FFPROBE = "ffprobe"

# Audio-only clips carry no frame rate. Timecoded transcripts still need a
# counting base, so this rate is used and the transcript header says so.
DEFAULT_AUDIO_FRAME_RATE = Fraction(24, 1)

_UNUSABLE_RATES = {"", "0/0", "0:0", "N/A"}


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot be run or its output cannot be read."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe reports about one media file."""

    path: Path
    duration_seconds: float | None = None
    frame_rate: Fraction = DEFAULT_AUDIO_FRAME_RATE
    start_timecode: str = ZERO_TIMECODE
    has_embedded_timecode: bool = False
    has_video: bool = False

    @property
    def missing_timecode(self) -> bool:
        """True when no embedded timecode was found and zero is assumed."""
        return not self.has_embedded_timecode

    @property
    def duration_label(self) -> str:
        """Duration formatted as ``HH:MM:SS``, or ``--:--:--`` when unknown."""
        return format_duration(self.duration_seconds)


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds as ``HH:MM:SS``, truncating the remainder."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ffprobe_available(program: str | None = None) -> bool:
    """Return True when ffprobe is bundled or can be found on PATH."""
    resolved = bundled_binary(program or FFPROBE)
    return Path(resolved).is_file() or resolved != (program or FFPROBE)


def _parse_rate(stream: dict[str, Any]) -> Fraction | None:
    """Return a usable frame rate from a video stream, or None."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        text = str(stream.get(key) or "").strip()
        if text in _UNUSABLE_RATES:
            continue
        try:
            rate = Fraction(text)
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def usable_timecode(value: Any) -> str | None:
    """Return a timecode string the converter can actually read, or None.

    Cameras and transcoders write all kinds of things into a ``timecode`` tag.
    Anything the converter cannot parse is treated exactly like no timecode at
    all, so the transcript falls back to ``00:00:00:00`` and says so rather
    than failing the clip.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parse_timecode(text)
    except TimecodeError:
        return None
    return text


def _parse_duration(payload: dict[str, Any], video: dict[str, Any] | None) -> float | None:
    """Return the clip duration in seconds from the format or stream entries."""
    candidates: list[Any] = [payload.get("format", {}).get("duration")]
    if video is not None:
        candidates.append(video.get("duration"))
    for stream in payload.get("streams", []) or []:
        candidates.append(stream.get("duration"))

    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def parse_probe_payload(payload: dict[str, Any], path: Path) -> MediaInfo:
    """Turn decoded ffprobe JSON into a :class:`MediaInfo`.

    Kept separate from the subprocess call so it can be tested without ffprobe.
    """
    streams = payload.get("streams", []) or []
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )

    rate = _parse_rate(video) if video is not None else None
    # Only a timecode the converter can read counts as an embedded timecode.
    # An unreadable tag is treated as absent, which is what puts the
    # "no embedded timecode" note into the transcript instead of failing.
    candidates = [stream.get("tags", {}).get("timecode") for stream in streams]
    candidates.append(payload.get("format", {}).get("tags", {}).get("timecode"))
    timecode = next(
        (usable for usable in map(usable_timecode, candidates) if usable),
        None,
    )

    return MediaInfo(
        path=Path(path),
        duration_seconds=_parse_duration(payload, video),
        frame_rate=rate or DEFAULT_AUDIO_FRAME_RATE,
        start_timecode=str(timecode) if timecode else ZERO_TIMECODE,
        has_embedded_timecode=bool(timecode),
        has_video=video is not None and rate is not None,
    )


def probe_media(
    source: Path,
    program: str | None = None,
    timeout: float = 60.0,
) -> MediaInfo:
    """Run ffprobe against one media file and return what it reports."""
    source = Path(source)
    program = bundled_binary(program or FFPROBE)
    command = [
        program,
        "-v", "error",
        "-show_entries",
        "format=duration:format_tags=timecode:"
        "stream=codec_type,duration,avg_frame_rate,r_frame_rate:"
        "stream_tags=timecode",
        "-of", "json",
        str(source),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise MediaProbeError(f"ffprobe was not found: {program}") from error
    except subprocess.TimeoutExpired as error:
        raise MediaProbeError(f"ffprobe timed out on {source.name}") from error
    except subprocess.CalledProcessError as error:
        message = (error.stderr or "").strip() or "ffprobe reported an error"
        raise MediaProbeError(f"{source.name}: {message}") from error

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as error:
        raise MediaProbeError(f"{source.name}: unreadable ffprobe output") from error

    return parse_probe_payload(payload, source)
