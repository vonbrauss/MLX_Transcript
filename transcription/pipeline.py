"""Batch orchestration: conflicts, diarization, review, cancellation, summary.

This module holds the whole batch policy and deliberately imports nothing from
Qt, so the pipeline can be driven from a worker thread, from a script, or from
the tests with stand-in engines.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .alignment import attribute_transcript
from .diarization import (
    DiarizationBackend,
    DiarizationError,
    DiarizationOptions,
    NullDiarizer,
)
from .engine import TranscriptionEngine, TranscriptionError
from .media_probe import MediaInfo, MediaProbeError, probe_media
from .outputs import (
    FolderLayout,
    OutputOptions,
    TranscriptPaths,
    build_output_paths,
    existing_outputs,
    output_roots,
    write_scriptsync,
    write_timecoded,
    write_srt,
    write_vtt,
)
from .speakers import SpeakerTranscript

__all__ = [
    "ConflictChoice",
    "ExistingFilePolicy",
    "ReviewMode",
    "ReviewDecision",
    "ItemStage",
    "ItemOutcome",
    "BatchJob",
    "BatchSummary",
    "BatchProcessor",
    "UNEXPECTED_ITEM_ERROR",
    "OUTPUT_FOLDER_ERROR",
]

logger = logging.getLogger(__name__)

#: Shown on the queue row when a clip fails for a reason we did not anticipate.
UNEXPECTED_ITEM_ERROR = "Unexpected error"

#: Shown on every queue row when the output tree could not be created at all.
OUTPUT_FOLDER_ERROR = "Could not create the output folder"


class ExistingFilePolicy(str, Enum):
    """What to do when a transcript already exists."""

    ASK = "ask"
    SKIP = "skip"
    OVERWRITE = "overwrite"

    @property
    def label(self) -> str:
        return {
            ExistingFilePolicy.ASK: "Ask every time",
            ExistingFilePolicy.SKIP: "Skip existing",
            ExistingFilePolicy.OVERWRITE: "Overwrite all",
        }[self]


class ConflictChoice(str, Enum):
    """The four answers offered by the conflict dialog."""

    OVERWRITE_THIS = "overwrite_this"
    OVERWRITE_ALL = "overwrite_all"
    SKIP_THIS = "skip_this"
    CANCEL_BATCH = "cancel_batch"


class ReviewMode(str, Enum):
    """Whether detected speakers are reviewed before the transcripts are written."""

    EVERY_FILE = "every_file"
    AUTOMATIC = "automatic"

    @property
    def label(self) -> str:
        return {
            ReviewMode.EVERY_FILE: "Review speakers for every file",
            ReviewMode.AUTOMATIC: "Use automatic labels without review",
        }[self]


class ReviewDecision(str, Enum):
    """What the speaker review dialog decided for one clip."""

    CONTINUE = "continue"
    EXPORT_WITHOUT_SPEAKERS = "export_without_speakers"
    CANCEL_BATCH = "cancel_batch"


class ItemStage(str, Enum):
    """Progress of one clip. Values match the queue status values in the UI."""

    WAITING = "waiting"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    DETECTING_SPEAKERS = "detecting_speakers"
    AWAITING_REVIEW = "awaiting_review"
    WRITING = "writing"


class ItemOutcome(str, Enum):
    """How one clip ended."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """One clip queued for transcription."""

    source: Path
    media: MediaInfo | None = None
    source_root: Path | None = None
    output_root_label: str | None = None

    def __post_init__(self) -> None:
        self.source = Path(self.source)
        if self.source_root is not None:
            self.source_root = Path(self.source_root)


@dataclass
class BatchSummary:
    """Counts reported when a batch finishes or is cancelled."""

    completed: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: int = 0
    was_cancelled: bool = False
    transcription_folder: Path | None = None
    failures: list[tuple[Path, str]] = field(default_factory=list)
    warnings: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.completed + self.skipped + self.failed + self.cancelled

    @property
    def diarization_warnings(self) -> int:
        return len(self.warnings)

    def as_sentence(self) -> str:
        prefix = "Cancelled" if self.was_cancelled else "Finished"
        sentence = (
            f"{prefix}. {self.completed} completed, {self.skipped} skipped, "
            f"{self.failed} failed, {self.cancelled} cancelled."
        )
        if self.warnings:
            sentence += f" {len(self.warnings)} exported without speaker labels."
        return sentence


StageCallback = Callable[[int, BatchJob, ItemStage], None]
OutcomeCallback = Callable[[int, BatchJob, ItemOutcome, str], None]
ProgressCallback = Callable[[int, int], None]
ConflictResolver = Callable[[BatchJob, Sequence[Path]], ConflictChoice]
ReviewResolver = Callable[[BatchJob, SpeakerTranscript], ReviewDecision]


class BatchProcessor:
    """Run a queue of clips through the engine and write both transcripts.

    One failure never stops the queue: the clip is recorded as failed with a
    readable message and the run moves on. Diarization is softer still, because
    a good transcript should never be thrown away over a speaker pass that did
    not work: the clip is exported without labels and the reason lands in the
    batch summary as a warning.

    Cancellation is checked between clips and again around the review step, so
    a clip that has already been transcribed still gets its transcripts written
    and nothing half-finished is left behind.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        source_root: Path,
        output_parent: Path,
        policy: ExistingFilePolicy = ExistingFilePolicy.ASK,
        conflict_resolver: ConflictResolver | None = None,
        on_stage: StageCallback | None = None,
        on_outcome: OutcomeCallback | None = None,
        on_progress: ProgressCallback | None = None,
        probe: Callable[[Path], MediaInfo] | None = None,
        diarizer: DiarizationBackend | None = None,
        diarization: DiarizationOptions | None = None,
        include_speakers_in_scriptsync: bool = False,
        include_speakers_in_subtitles: bool = True,
        review_mode: ReviewMode = ReviewMode.EVERY_FILE,
        review_resolver: ReviewResolver | None = None,
        output_options: OutputOptions | None = None,
    ) -> None:
        self.engine = engine
        self.source_root = Path(source_root)
        self.output_parent = Path(output_parent)
        self.policy = policy
        self.conflict_resolver = conflict_resolver
        self.on_stage = on_stage
        self.on_outcome = on_outcome
        self.on_progress = on_progress
        self.probe = probe or probe_media

        self.diarization = diarization or DiarizationOptions()
        self.diarizer = diarizer or NullDiarizer()
        self.include_speakers_in_scriptsync = include_speakers_in_scriptsync
        self.include_speakers_in_subtitles = include_speakers_in_subtitles
        self.review_mode = review_mode
        self.review_resolver = review_resolver
        self.output_options = output_options
        self._duplicate_numbers: dict[Path, int] = {}

        self._cancel = threading.Event()
        # "Overwrite all" is a decision for one batch only. It is never
        # persisted and it resets with every new processor.
        self._overwrite_all = policy is ExistingFilePolicy.OVERWRITE

    # --------------------------------------------------------------- control

    def cancel(self) -> None:
        """Ask the batch to stop after the current clip finishes safely."""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def overwrite_all(self) -> bool:
        """True once the user has chosen to overwrite the rest of this batch."""
        return self._overwrite_all

    @property
    def speakers_enabled(self) -> bool:
        return bool(self.diarization.enabled)

    # ------------------------------------------------------------------ run

    def run(self, jobs: Iterable[BatchJob]) -> BatchSummary:
        """Process every job in order and return the batch summary.

        This method is the batch's outer boundary and it always returns a
        summary. No single clip can end the run: a failure that was never
        anticipated is recorded against that clip and the queue carries on,
        and an output tree that cannot be created fails every clip with the
        same readable reason rather than escaping into the worker thread.
        """
        jobs = list(jobs)
        self._prepare_duplicate_numbers(jobs)
        roots = output_roots(self.output_parent)
        summary = BatchSummary(transcription_folder=roots.transcription)
        total = len(jobs)

        if total:
            try:
                roots.create(
                    self.output_options.formats
                    if self.output_options is not None
                    else None
                )
            except OSError as error:
                return self._fail_whole_batch(
                    summary, jobs, f"{OUTPUT_FOLDER_ERROR}: {error}"
                )

        for index, job in enumerate(jobs):
            if self._cancel.is_set():
                self._finish(summary, index, job, ItemOutcome.CANCELLED, "")
                self._report_progress(index + 1, total)
                continue

            self._stage(index, job, ItemStage.WAITING)
            try:
                outcome, message, warning = self._process(index, job)
                if warning:
                    summary.warnings.append((job.source, warning))
            except _CancelBatch:
                self._cancel.set()
                self._finish(summary, index, job, ItemOutcome.CANCELLED, "")
                self._report_progress(index + 1, total)
                continue
            except Exception as error:  # noqa: BLE001 - the batch must survive
                logger.exception(
                    "Unexpected failure while processing %s", job.source
                )
                outcome = ItemOutcome.FAILED
                message = (
                    f"{UNEXPECTED_ITEM_ERROR} "
                    f"({type(error).__name__}): {error}"
                )

            self._finish(summary, index, job, outcome, message)
            self._report_progress(index + 1, total)

        summary.was_cancelled = self._cancel.is_set()
        return summary

    def _fail_whole_batch(
        self,
        summary: BatchSummary,
        jobs: Sequence[BatchJob],
        message: str,
    ) -> BatchSummary:
        """Record one shared failure against every job and return the summary.

        Used when nothing can be written at all, for example a destination on
        a read-only volume. Every queue row gets the same reason, so the user
        sees what went wrong instead of a batch that never reports back.
        """
        logger.error("%s", message)
        total = len(jobs)
        for index, job in enumerate(jobs):
            self._finish(summary, index, job, ItemOutcome.FAILED, message)
            self._report_progress(index + 1, total)
        summary.was_cancelled = self._cancel.is_set()
        return summary

    # ------------------------------------------------------------ one clip

    def _process(self, index: int, job: BatchJob) -> tuple[ItemOutcome, str, str]:
        """Handle one clip and return its outcome, message, and any warning."""
        paths = self._paths_for(job)

        targets = self._resolve_conflict(job, paths)
        if not targets:
            return ItemOutcome.SKIPPED, "Transcripts already exist", ""

        try:
            media = job.media or self.probe(job.source)
            job.media = media
        except MediaProbeError as error:
            return ItemOutcome.FAILED, str(error), ""

        stage = (
            ItemStage.TRANSCRIBING
            if self.engine.is_loaded
            else ItemStage.LOADING_MODEL
        )
        self._stage(index, job, stage)
        try:
            if stage is ItemStage.LOADING_MODEL:
                self.engine.load()
                self._stage(index, job, ItemStage.TRANSCRIBING)
            result = self.engine.transcribe(job.source)
        except TranscriptionError as error:
            return ItemOutcome.FAILED, str(error), ""
        except Exception as error:  # a stand-in engine may raise anything
            return ItemOutcome.FAILED, f"{job.source.name}: {error}", ""

        segments: Sequence[dict] = result.segments
        include_speakers = False
        message = ""
        warning = ""

        if self.speakers_enabled:
            transcript, warning = self._detect_speakers(index, job, result.segments)
            if warning:
                message = warning
            elif transcript is not None:
                decision = self._review(index, job, transcript)
                if decision is ReviewDecision.CANCEL_BATCH:
                    raise _CancelBatch()
                include_speakers = decision is ReviewDecision.CONTINUE
                if not include_speakers:
                    message = "Exported without speaker labels"
                segments = transcript.as_output_segments(
                    include_speakers=include_speakers
                )

        self._stage(index, job, ItemStage.WRITING)
        try:
            self._write(job, paths, targets, segments, media, include_speakers)
        except OSError as error:
            return ItemOutcome.FAILED, f"Could not write transcripts: {error}", ""

        return ItemOutcome.COMPLETED, message, warning

    def _paths_for(self, job: BatchJob) -> TranscriptPaths:
        """Return the output paths for one clip.

        A clip that does not sit under the source root it was queued with
        cannot be mirrored into the output tree. Rather than failing the clip,
        its own folder becomes the root, so the transcript still lands
        somewhere sensible.
        """
        root = job.source_root or self.source_root
        duplicate = self._duplicate_numbers.get(job.source, 1)
        try:
            return build_output_paths(
                job.source,
                root,
                self.output_parent,
                self.output_options,
                duplicate,
                job.output_root_label,
            )
        except ValueError:
            logger.warning(
                "%s is not below %s; using its own folder as the source root",
                job.source,
                root,
            )
            return build_output_paths(
                job.source,
                job.source.parent,
                self.output_parent,
                self.output_options,
                duplicate,
                job.output_root_label,
            )

    # ------------------------------------------------------------- speakers

    def _detect_speakers(
        self,
        index: int,
        job: BatchJob,
        segments: Sequence[dict],
    ) -> tuple[SpeakerTranscript | None, str]:
        """Diarize and align one clip.

        Returns the attributed transcript, or ``None`` plus a warning when the
        speaker pass could not produce anything usable. A warning is never a
        failure: the transcript itself is already good and gets exported.
        """
        self._stage(index, job, ItemStage.DETECTING_SPEAKERS)
        try:
            turns = self.diarizer.diarize(
                job.source,
                self.diarization,
                cancelled=self._cancel.is_set,
            )
        except DiarizationError as error:
            return None, f"Speaker detection failed: {error}"
        except Exception as error:
            return None, f"Speaker detection failed: {error}"

        if not turns:
            return None, "No speakers were detected"

        transcript = attribute_transcript(
            segments,
            turns,
            tolerance=self.diarization.nearest_tolerance,
            merge_gap=self.diarization.merge_gap,
        )
        if not transcript.has_speakers:
            return None, "No speakers could be matched to the transcript"
        return transcript, ""

    def _review(
        self,
        index: int,
        job: BatchJob,
        transcript: SpeakerTranscript,
    ) -> ReviewDecision:
        """Let the user rename and correct speakers before anything is written."""
        if self.review_mode is not ReviewMode.EVERY_FILE:
            return ReviewDecision.CONTINUE
        if self.review_resolver is None:
            return ReviewDecision.CONTINUE
        if self._cancel.is_set():
            return ReviewDecision.CANCEL_BATCH

        self._stage(index, job, ItemStage.AWAITING_REVIEW)
        return self.review_resolver(job, transcript)

    # -------------------------------------------------------------- writing

    def _write(
        self,
        job: BatchJob,
        paths: TranscriptPaths,
        targets: set[Path],
        segments: Sequence[dict],
        media: MediaInfo,
        include_speakers: bool,
    ) -> None:
        """Write whichever of the two transcripts this clip is allowed to."""
        if paths.scriptsync in targets:
            write_scriptsync(
                paths.scriptsync,
                segments,
                include_speakers=(
                    include_speakers and self.include_speakers_in_scriptsync
                ),
            )
        if paths.timecoded in targets:
            write_timecoded(
                paths.timecoded,
                job.source,
                segments,
                media.frame_rate,
                media.start_timecode,
                media.missing_timecode,
            )
        if paths.srt is not None and paths.srt in targets:
            write_srt(paths.srt, segments, include_speakers and self.include_speakers_in_subtitles)
        if paths.vtt is not None and paths.vtt in targets:
            write_vtt(paths.vtt, segments, include_speakers and self.include_speakers_in_subtitles)

    def _prepare_duplicate_numbers(self, jobs: Sequence[BatchJob]) -> None:
        """Number same-named sources when all outputs share one flat folder."""
        self._duplicate_numbers = {}
        if self.output_options is None or self.output_options.folder_layout is not FolderLayout.FLAT:
            return
        seen: dict[str, int] = {}
        for job in jobs:
            key = job.source.stem.casefold()
            seen[key] = seen.get(key, 0) + 1
            self._duplicate_numbers[job.source] = seen[key]

    # ------------------------------------------------------------- conflicts

    def _resolve_conflict(
        self,
        job: BatchJob,
        paths: TranscriptPaths,
    ) -> set[Path]:
        """Return the transcripts this clip may write.

        An empty set means skip the clip entirely. Raises :class:`_CancelBatch`
        when the user cancels from the conflict dialog.
        """
        existing = existing_outputs(paths)
        if not existing or self._overwrite_all:
            return set(paths.as_tuple())

        if self.policy is ExistingFilePolicy.OVERWRITE:
            return set(paths.as_tuple())

        if self.policy is ExistingFilePolicy.SKIP:
            return self._missing_only(paths, existing)

        choice = ConflictChoice.SKIP_THIS
        if self.conflict_resolver is not None:
            choice = self.conflict_resolver(job, list(existing))

        if choice is ConflictChoice.CANCEL_BATCH:
            raise _CancelBatch()
        if choice is ConflictChoice.OVERWRITE_ALL:
            self._overwrite_all = True
            return set(paths.as_tuple())
        if choice is ConflictChoice.OVERWRITE_THIS:
            return set(paths.as_tuple())
        return self._missing_only(paths, existing)

    @staticmethod
    def _missing_only(paths: TranscriptPaths, existing: Sequence[Path]) -> set[Path]:
        """Keep existing transcripts and create only the missing one."""
        return {path for path in paths.as_tuple() if path not in set(existing)}

    # ------------------------------------------------------------- reporting

    def _stage(self, index: int, job: BatchJob, stage: ItemStage) -> None:
        if self.on_stage is not None:
            self.on_stage(index, job, stage)

    def _finish(
        self,
        summary: BatchSummary,
        index: int,
        job: BatchJob,
        outcome: ItemOutcome,
        message: str,
    ) -> None:
        if outcome is ItemOutcome.COMPLETED:
            summary.completed += 1
        elif outcome is ItemOutcome.SKIPPED:
            summary.skipped += 1
        elif outcome is ItemOutcome.FAILED:
            summary.failed += 1
            summary.failures.append((job.source, message))
        else:
            summary.cancelled += 1

        if self.on_outcome is not None:
            self.on_outcome(index, job, outcome, message)

    def _report_progress(self, done: int, total: int) -> None:
        if self.on_progress is not None:
            self.on_progress(done, total)


class _CancelBatch(Exception):
    """Internal signal that the user cancelled from a dialog."""
