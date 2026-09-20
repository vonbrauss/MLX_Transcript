"""Queue and job data structures shared by the UI and the workers.

These are plain dataclasses and enums with no Qt imports so they stay testable
and reusable by the transcription layer. The policy enums live in
``transcription.pipeline`` and are re-exported here, which is where the rest of
the application has always imported them from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from transcription.media_probe import MediaInfo, format_duration
from transcription.outputs import TranscriptPaths
from transcription.pipeline import (
    BatchSummary,
    ConflictChoice,
    ExistingFilePolicy,
    ItemOutcome,
    ItemStage,
    ReviewDecision,
    ReviewMode,
)

__all__ = [
    "BatchSummary",
    "ConflictChoice",
    "ExistingFilePolicy",
    "ItemOutcome",
    "ItemStage",
    "ReviewDecision",
    "ReviewMode",
    "QueueStatus",
    "QueueItem",
    "SpeakerLabel",
]


class QueueStatus(str, Enum):
    """Lifecycle of one clip in the queue.

    The values match :class:`~transcription.pipeline.ItemStage` and
    :class:`~transcription.pipeline.ItemOutcome`, so the worker can map a
    pipeline event straight onto a queue status.
    """

    PENDING = "pending"
    PROBING = "probing"
    READY = "ready"
    WAITING = "waiting"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    DETECTING_SPEAKERS = "detecting_speakers"
    AWAITING_REVIEW = "awaiting_review"
    WRITING = "writing"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def label(self) -> str:
        return {
            QueueStatus.PENDING: "Queued",
            QueueStatus.PROBING: "Reading",
            QueueStatus.READY: "Ready",
            QueueStatus.WAITING: "Waiting",
            QueueStatus.LOADING_MODEL: "Loading Model",
            QueueStatus.TRANSCRIBING: "Transcribing",
            QueueStatus.DETECTING_SPEAKERS: "Detecting Speakers",
            QueueStatus.AWAITING_REVIEW: "Waiting for Speaker Review",
            QueueStatus.WRITING: "Writing Files",
            QueueStatus.COMPLETED: "Completed",
            QueueStatus.SKIPPED: "Skipped",
            QueueStatus.FAILED: "Failed",
            QueueStatus.CANCELLED: "Cancelled",
        }[self]

    @property
    def is_finished(self) -> bool:
        return self in {
            QueueStatus.COMPLETED,
            QueueStatus.SKIPPED,
            QueueStatus.FAILED,
            QueueStatus.CANCELLED,
        }

    @property
    def is_running(self) -> bool:
        return self in {
            QueueStatus.WAITING,
            QueueStatus.LOADING_MODEL,
            QueueStatus.TRANSCRIBING,
            QueueStatus.DETECTING_SPEAKERS,
            QueueStatus.AWAITING_REVIEW,
            QueueStatus.WRITING,
        }

    @classmethod
    def from_stage(cls, stage: ItemStage) -> "QueueStatus":
        """Map a pipeline stage onto a queue status."""
        return cls(stage.value)

    @classmethod
    def from_outcome(cls, outcome: ItemOutcome) -> "QueueStatus":
        """Map a pipeline outcome onto a queue status."""
        return cls(outcome.value)


@dataclass
class SpeakerLabel:
    """One diarized speaker, ready for manual renaming in milestone 3."""

    identifier: str
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.identifier


@dataclass
class QueueItem:
    """One media file and everything the UI shows about it."""

    source: Path
    source_root: Path
    relative_folder: Path = field(default_factory=lambda: Path("."))
    status: QueueStatus = QueueStatus.PENDING
    message: str = ""
    media: MediaInfo | None = None
    outputs: TranscriptPaths | None = None
    speakers: list[SpeakerLabel] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def relative_folder_label(self) -> str:
        folder = str(self.relative_folder)
        return "/" if folder == "." else folder

    @property
    def duration_seconds(self) -> float | None:
        return self.media.duration_seconds if self.media else None

    @property
    def duration_label(self) -> str:
        return format_duration(self.duration_seconds)

    @property
    def status_label(self) -> str:
        if self.message:
            return f"{self.status.label}: {self.message}"
        return self.status.label

    def reset_for_batch(self) -> None:
        """Put a finished item back into the queue for a new batch."""
        self.status = QueueStatus.READY if self.media else QueueStatus.FAILED
        if self.status is QueueStatus.READY:
            self.message = ""
