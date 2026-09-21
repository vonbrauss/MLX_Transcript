"""Speaker-attributed transcript data model.

Everything here is plain Python: no Qt, no audio library, no model. The review
dialog, the transcript writers, and the tests all work against these types.

Two label spaces are kept apart on purpose:

``identifier``
    What the diarization backend emitted, such as ``speaker_00``. It never
    changes, so renaming and merging stay stable.
``display_name``
    What the user sees and what lands in the transcripts, starting at
    ``Speaker 1`` and renameable to anything.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "SpeakerNameError",
    "SpeakerTurn",
    "Word",
    "Speaker",
    "AttributedSegment",
    "SpeakerTranscript",
    "clean_speaker_name",
    "default_display_name",
    "scriptsync_speaker_label",
]

# Control characters, line breaks, and the Unicode separators a pasted name
# might drag in. Everything here collapses to a single space.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")

_MAX_NAME_LENGTH = 120


class SpeakerNameError(ValueError):
    """Raised when a speaker name cannot be used as given."""


def clean_speaker_name(raw: str) -> str:
    """Return a usable speaker name, or raise :class:`SpeakerNameError`.

    Surrounding whitespace is trimmed, line breaks and control characters are
    removed, and runs of whitespace collapse. Unicode is preserved: only the
    ScriptSync label is reduced to ASCII, and only at write time.
    """
    text = _CONTROL_CHARACTERS.sub(" ", str(raw or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise SpeakerNameError("A speaker name cannot be blank.")
    if len(text) > _MAX_NAME_LENGTH:
        raise SpeakerNameError(
            f"A speaker name cannot be longer than {_MAX_NAME_LENGTH} characters."
        )
    return text


def default_display_name(position: int) -> str:
    """Return ``Speaker 1``, ``Speaker 2``, and so on for a zero-based index."""
    return f"Speaker {position + 1}"


def scriptsync_speaker_label(name: str) -> str:
    """Return the ASCII, uppercase heading Avid ScriptSync can read.

    The same character rules the ScriptSync body already uses are applied, so a
    Unicode name never reaches the ASCII file. A name that reduces to nothing
    falls back to its unaccented shape being empty, in which case the caller
    should treat the speaker as unlabeled.
    """
    text = str(name or "")
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


@dataclass(frozen=True)
class SpeakerTurn:
    """One interval the diarization backend attributed to one speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap_with(self, start: float, end: float) -> float:
        """Return the number of seconds this turn shares with a span."""
        return max(0.0, min(self.end, end) - max(self.start, start))

    def distance_to(self, start: float, end: float) -> float:
        """Return the silence in seconds between this turn and a span."""
        if self.overlap_with(start, end) > 0:
            return 0.0
        if end <= self.start:
            return self.start - end
        return start - self.end


@dataclass(frozen=True)
class Word:
    """One Whisper word with its timestamps."""

    start: float
    end: float
    text: str

    @classmethod
    def from_payload(cls, payload: dict) -> "Word | None":
        """Build a word from Whisper's dictionary, or None when unusable."""
        text = str(payload.get("word", payload.get("text", ""))).strip()
        try:
            start = float(payload["start"])
            end = float(payload["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not text:
            return None
        return cls(start=start, end=max(end, start), text=text)


@dataclass
class Speaker:
    """One detected speaker and the name the user sees."""

    identifier: str
    display_name: str

    @classmethod
    def numbered(cls, identifier: str, position: int) -> "Speaker":
        return cls(identifier=identifier, display_name=default_display_name(position))


@dataclass
class AttributedSegment:
    """A run of words from one speaker.

    ``overlap_identifiers`` keeps every speaker the backend placed inside this
    span, primary first. The plain-text formats use only the primary speaker;
    the field exists so a richer export can show overlap later without
    re-running diarization.
    """

    start: float
    end: float
    text: str
    speaker_id: str | None = None
    overlap_identifiers: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def has_overlap(self) -> bool:
        return len(self.overlap_identifiers) > 1


@dataclass
class SpeakerTranscript:
    """Speaker-attributed segments plus the speaker list the user edits."""

    segments: list[AttributedSegment] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)

    # ----------------------------------------------------------- construction

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[AttributedSegment],
        identifiers: Iterable[str] | None = None,
    ) -> "SpeakerTranscript":
        """Build a transcript, numbering speakers in order of first appearance."""
        segments = list(segments)
        ordered: list[str] = []
        for segment in segments:
            if segment.speaker_id and segment.speaker_id not in ordered:
                ordered.append(segment.speaker_id)
        for identifier in identifiers or ():
            if identifier not in ordered:
                ordered.append(identifier)
        return cls(
            segments=segments,
            speakers=[
                Speaker.numbered(identifier, position)
                for position, identifier in enumerate(ordered)
            ],
        )

    # ------------------------------------------------------------- inspection

    @property
    def has_speakers(self) -> bool:
        return bool(self.speakers)

    @property
    def identifiers(self) -> list[str]:
        return [speaker.identifier for speaker in self.speakers]

    def speaker(self, identifier: str) -> Speaker | None:
        return next(
            (item for item in self.speakers if item.identifier == identifier), None
        )

    def display_name(self, identifier: str | None) -> str:
        """Return the name for an identifier, or an empty string when unknown."""
        if identifier is None:
            return ""
        found = self.speaker(identifier)
        return found.display_name if found else ""

    def segments_for(self, identifier: str) -> list[AttributedSegment]:
        return [
            segment for segment in self.segments if segment.speaker_id == identifier
        ]

    def segment_count(self, identifier: str) -> int:
        return len(self.segments_for(identifier))

    def speaking_duration(self, identifier: str) -> float:
        return sum(segment.duration for segment in self.segments_for(identifier))

    def samples(self, identifier: str, count: int = 3, length: int = 90) -> list[str]:
        """Return a few short lines of what this speaker said, for review."""
        chosen: list[str] = []
        for segment in self.segments_for(identifier):
            text = " ".join(segment.text.split())
            if not text:
                continue
            chosen.append(text if len(text) <= length else text[: length - 1] + "…")
            if len(chosen) >= count:
                break
        return chosen

    @property
    def unattributed_count(self) -> int:
        return sum(1 for segment in self.segments if segment.speaker_id is None)

    # ---------------------------------------------------------------- editing

    def rename(self, identifier: str, new_name: str, allow_duplicate: bool = False) -> str:
        """Rename one speaker, returning the cleaned name that was stored."""
        speaker = self.speaker(identifier)
        if speaker is None:
            raise SpeakerNameError(f"No such speaker: {identifier}")
        cleaned = clean_speaker_name(new_name)
        if not allow_duplicate and self.duplicates_of(cleaned, identifier):
            raise SpeakerNameError(
                f"Another speaker is already named {cleaned!r}."
            )
        speaker.display_name = cleaned
        return cleaned

    def duplicates_of(self, name: str, ignoring: str | None = None) -> list[str]:
        """Return identifiers already using this name, case-insensitively."""
        target = name.strip().casefold()
        return [
            speaker.identifier
            for speaker in self.speakers
            if speaker.identifier != ignoring
            and speaker.display_name.strip().casefold() == target
        ]

    def reset_names(self) -> None:
        """Put every speaker back to Speaker 1, Speaker 2, and so on."""
        for position, speaker in enumerate(self.speakers):
            speaker.display_name = default_display_name(position)

    def merge(self, source_id: str, target_id: str) -> None:
        """Fold one detected speaker into another, keeping the target's name."""
        if source_id == target_id:
            return
        if self.speaker(source_id) is None or self.speaker(target_id) is None:
            raise SpeakerNameError("Both speakers must exist to merge them.")

        for segment in self.segments:
            if segment.speaker_id == source_id:
                segment.speaker_id = target_id
            if source_id in segment.overlap_identifiers:
                merged: list[str] = []
                for identifier in segment.overlap_identifiers:
                    replacement = target_id if identifier == source_id else identifier
                    if replacement not in merged:
                        merged.append(replacement)
                segment.overlap_identifiers = tuple(merged)

        self.speakers = [
            speaker for speaker in self.speakers if speaker.identifier != source_id
        ]

    def reassign(self, index: int, identifier: str | None) -> None:
        """Move one segment to a different speaker."""
        if not 0 <= index < len(self.segments):
            raise IndexError(f"No segment at position {index}")
        if identifier is not None and self.speaker(identifier) is None:
            raise SpeakerNameError(f"No such speaker: {identifier}")
        self.segments[index].speaker_id = identifier

    # ---------------------------------------------------------------- export

    def as_output_segments(self, include_speakers: bool = True) -> list[dict]:
        """Return the segment dictionaries the transcript writers consume."""
        rows: list[dict] = []
        for segment in self.segments:
            row: dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            if include_speakers:
                name = self.display_name(segment.speaker_id)
                if name:
                    row["speaker"] = name
            rows.append(row)
        return rows
