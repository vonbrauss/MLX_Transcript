"""Source-timecode arithmetic for drop-frame and non-drop-frame media.

The math here is ported verbatim from ``reference/avid_transcribe_folders.py``
so that transcripts produced by the application match the ones produced by the
original command line script frame for frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "TimecodeError",
    "ParsedTimecode",
    "TimecodeConverter",
    "ZERO_TIMECODE",
    "nominal_frame_rate",
    "parse_timecode",
]

ZERO_TIMECODE = "00:00:00:00"

_TIMECODE_PATTERN = re.compile(r"^\d{1,3}[:;]\d{1,2}[:;]\d{1,2}[:;]\d{1,3}$")


class TimecodeError(ValueError):
    """Raised when a timecode string cannot be understood."""


@dataclass(frozen=True)
class ParsedTimecode:
    """A timecode string split into its parts."""

    hours: int
    minutes: int
    seconds: int
    frames: int
    drop_frame: bool

    @property
    def separator(self) -> str:
        """Return ``;`` for drop-frame timecode and ``:`` otherwise."""
        return ";" if self.drop_frame else ":"


def nominal_frame_rate(rate: Fraction) -> int:
    """Return the integer frame rate used for timecode counting.

    29.97 counts as 30, 23.976 counts as 24, and so on.
    """
    nominal = round(float(rate))
    if nominal <= 0:
        raise TimecodeError(f"Unusable frame rate: {rate}")
    return nominal


def parse_timecode(source_timecode: str) -> ParsedTimecode:
    """Split ``HH:MM:SS:FF`` or ``HH:MM:SS;FF`` into its numeric parts."""
    text = (source_timecode or "").strip()
    if not _TIMECODE_PATTERN.match(text):
        raise TimecodeError(f"Unrecognized timecode: {source_timecode}")

    drop_frame = ";" in text
    parts = text.replace(";", ":").split(":")
    if len(parts) != 4:
        raise TimecodeError(f"Unrecognized timecode: {source_timecode}")

    hours, minutes, seconds, frames = (int(part) for part in parts)
    return ParsedTimecode(hours, minutes, seconds, frames, drop_frame)


class TimecodeConverter:
    """Convert transcript offsets in seconds into source timecode."""

    def __init__(self, rate: Fraction, source_timecode: str = ZERO_TIMECODE) -> None:
        self.rate = Fraction(rate)
        self.source_timecode = source_timecode or ZERO_TIMECODE
        self.parsed = parse_timecode(self.source_timecode)
        self.nominal_fps = nominal_frame_rate(self.rate)
        self.drop_frame = self.parsed.drop_frame
        self.separator = self.parsed.separator
        self.drop_frames = round(self.nominal_fps * 0.066666) if self.drop_frame else 0
        self.start_frame = self._start_frame()

    def _start_frame(self) -> int:
        """Return the absolute frame number the media starts on."""
        parsed = self.parsed
        total_minutes = parsed.hours * 60 + parsed.minutes
        start_frame = (
            (parsed.hours * 3600 + parsed.minutes * 60 + parsed.seconds)
            * self.nominal_fps
            + parsed.frames
        )
        if self.drop_frame:
            start_frame -= self.drop_frames * (total_minutes - total_minutes // 10)
        return start_frame

    def frames_at(self, offset_seconds: float) -> int:
        """Return the absolute frame number for an offset into the media."""
        return self.start_frame + round(float(offset_seconds) * float(self.rate))

    def frames_to_timecode(self, frame_number: int) -> str:
        """Format an absolute frame number as a timecode string."""
        if self.drop_frame:
            frames_per_10_minutes = round(float(self.rate) * 600)
            frames_per_minute = round(float(self.rate) * 60)
            blocks = frame_number // frames_per_10_minutes
            remainder = frame_number % frames_per_10_minutes
            frame_number += self.drop_frames * 9 * blocks
            if remainder > self.drop_frames:
                frame_number += self.drop_frames * (
                    (remainder - self.drop_frames) // frames_per_minute
                )

        frame_number %= self.nominal_fps * 60 * 60 * 24
        frames = frame_number % self.nominal_fps
        total_seconds = frame_number // self.nominal_fps
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60
        return (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            f"{self.separator}{frames:02d}"
        )

    def at_offset(self, offset_seconds: float) -> str:
        """Return the source timecode for an offset in seconds."""
        return self.frames_to_timecode(self.frames_at(offset_seconds))

    def __call__(self, offset_seconds: float) -> str:
        return self.at_offset(offset_seconds)
