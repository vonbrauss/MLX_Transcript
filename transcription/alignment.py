"""Align Whisper word timestamps with diarization intervals.

The rules, all of them documented here because they decide what an editor
reads in the transcript:

Assignment
    Each word goes to the speaker turn it shares the most time with. A tie goes
    to the turn that starts earlier, so the result is stable.

Nearest speaker
    A word that overlaps no turn at all, which happens on a short interjection
    or at a boundary, goes to the closest turn within
    :data:`NEAREST_SPEAKER_TOLERANCE` seconds. Beyond that it stays
    unattributed rather than being guessed at.

Splitting
    A Whisper segment is split wherever its words change speaker. A long
    segment is never handed to one speaker when the word timestamps disagree.

Merging
    Consecutive words keep sharing a segment while the speaker is unchanged,
    the order is chronological, and the silence between them stays under
    :data:`MERGE_GAP_SECONDS`.

Overlap
    Overlapped speech is recorded in ``overlap_identifiers``, primary first.
    The primary speaker is the one with the most overlapped time across the
    segment's words, which is the same rule used per word, so the plain-text
    formats stay readable. Nothing about the overlap is lost.

Wording is never altered. Only whitespace between joined words is normalized.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .speakers import AttributedSegment, SpeakerTranscript, SpeakerTurn, Word

__all__ = [
    "NEAREST_SPEAKER_TOLERANCE",
    "MERGE_GAP_SECONDS",
    "assign_word",
    "segment_words",
    "attribute_transcript",
]

#: A word with no overlap may borrow the speaker of a turn this close.
NEAREST_SPEAKER_TOLERANCE = 0.75

#: Silence at or above this many seconds starts a new display segment.
MERGE_GAP_SECONDS = 1.0


def segment_words(segment: dict) -> list[Word]:
    """Return the words of one Whisper segment.

    When word timestamps are missing, which happens if they were switched off
    or the backend dropped them, the whole segment becomes a single span so it
    can still be attributed as a unit.
    """
    words: list[Word] = []
    for payload in segment.get("words") or ():
        if isinstance(payload, dict):
            word = Word.from_payload(payload)
            if word is not None:
                words.append(word)
    if words:
        return words

    text = " ".join(str(segment.get("text", "")).split())
    if not text:
        return []
    try:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
    except (TypeError, ValueError):
        return []
    return [Word(start=start, end=max(end, start), text=text)]


def assign_word(
    word: Word,
    turns: Sequence[SpeakerTurn],
    tolerance: float = NEAREST_SPEAKER_TOLERANCE,
) -> tuple[str | None, tuple[str, ...]]:
    """Return the primary speaker for a word and every speaker overlapping it."""
    if not turns:
        return None, ()

    overlaps: list[tuple[float, float, str]] = []
    for turn in turns:
        shared = turn.overlap_with(word.start, word.end)
        if shared > 0:
            overlaps.append((shared, turn.start, turn.speaker))

    if overlaps:
        overlaps.sort(key=lambda row: (-row[0], row[1]))
        ordered: list[str] = []
        for _shared, _start, speaker in overlaps:
            if speaker not in ordered:
                ordered.append(speaker)
        return ordered[0], tuple(ordered)

    nearest: tuple[float, float, str] | None = None
    for turn in turns:
        distance = turn.distance_to(word.start, word.end)
        if distance <= tolerance:
            candidate = (distance, turn.start, turn.speaker)
            if nearest is None or candidate < nearest:
                nearest = candidate
    if nearest is not None:
        return nearest[2], (nearest[2],)
    return None, ()


def _merge_runs(
    assigned: Sequence[tuple[Word, str | None, tuple[str, ...]]],
    merge_gap: float,
) -> list[AttributedSegment]:
    """Group consecutive words into display segments."""
    segments: list[AttributedSegment] = []
    current: list[Word] = []
    current_speaker: str | None = None
    overlap_totals: dict[str, float] = {}

    def flush() -> None:
        if not current:
            return
        ordered = sorted(overlap_totals, key=lambda key: (-overlap_totals[key], key))
        if current_speaker is not None:
            ordered = [current_speaker] + [
                key for key in ordered if key != current_speaker
            ]
        segments.append(
            AttributedSegment(
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(word.text for word in current),
                speaker_id=current_speaker,
                overlap_identifiers=tuple(ordered),
            )
        )

    previous_end: float | None = None
    for word, speaker, overlaps in assigned:
        gap = 0.0 if previous_end is None else word.start - previous_end
        breaks = (
            current
            and (
                speaker != current_speaker
                or gap >= merge_gap
                or word.start < current[-1].start
            )
        )
        if breaks:
            flush()
            current = []
            overlap_totals = {}

        if not current:
            current_speaker = speaker
        current.append(word)
        for identifier in overlaps:
            overlap_totals[identifier] = overlap_totals.get(identifier, 0.0) + max(
                word.end - word.start, 0.0
            )
        previous_end = word.end

    flush()
    return segments


def attribute_transcript(
    segments: Iterable[dict],
    turns: Sequence[SpeakerTurn],
    tolerance: float = NEAREST_SPEAKER_TOLERANCE,
    merge_gap: float = MERGE_GAP_SECONDS,
) -> SpeakerTranscript:
    """Turn Whisper segments plus diarization turns into an attributed transcript.

    With no turns at all, which is what an empty diarization result or a
    music-only clip produces, every segment comes back unattributed and the
    transcript writers fall back to their unlabeled format.
    """
    turns = sorted(turns, key=lambda turn: (turn.start, turn.end))

    assigned: list[tuple[Word, str | None, tuple[str, ...]]] = []
    for segment in segments:
        for word in segment_words(segment):
            speaker, overlaps = assign_word(word, turns, tolerance)
            assigned.append((word, speaker, overlaps))

    # Only speakers that actually carry words are listed. A turn the aligner
    # never matched would otherwise show up in review with nothing to review.
    return SpeakerTranscript.from_segments(_merge_runs(assigned, merge_gap))
