"""Background workers. The UI thread never touches the filesystem directly.

Each worker is a plain :class:`QObject` moved onto a :class:`QThread` by the
window, so long scans and long transcription batches leave the interface
responsive.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QObject, QThread, Signal

from app.models import (
    ConflictChoice,
    QueueItem,
    QueueStatus,
    ReviewDecision,
    source_identity,
)
from transcription.diarization import DiarizationBackend, DiarizationOptions
from transcription.discovery import discover_media, is_supported_media
from transcription.engine import TranscriptionEngine
from transcription.media_probe import MediaProbeError, probe_media
from transcription.model_cache import (
    DEFAULT_BUNDLE,
    DownloadCancelled,
    ModelBundle,
    ModelCache,
    ModelCacheError,
)
from transcription.outputs import OutputOptions, build_output_paths, output_roots
from transcription.pipeline import (
    BatchJob,
    BatchProcessor,
    BatchSummary,
    ExistingFilePolicy,
    ItemOutcome,
    ItemStage,
    ReviewMode,
    UNEXPECTED_ITEM_ERROR,
)
from transcription.speakers import SpeakerTranscript

__all__ = [
    "ModelDownloadWorker",
    "ScanWorker",
    "TranscriptionWorker",
    "run_worker_on_thread",
]

logger = logging.getLogger(__name__)


class ScanWorker(QObject):
    """Find media below a source folder and read each duration with ffprobe."""

    started_scan = Signal()
    skipped_files = Signal(list)
    found_files = Signal(int)
    item_ready = Signal(int, object)
    progress = Signal(int, int)
    failed = Signal(str)
    finished = Signal(list)

    def __init__(
        self,
        source_root: Path | Sequence[Path],
        output_parent: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_roots = (
            [Path(item) for item in source_root]
            if not isinstance(source_root, (str, Path))
            else [Path(source_root)]
        )
        self.output_parent = Path(output_parent) if output_parent else None
        #: ``(path, reason)`` for everything this scan turned away, so the
        #: window can show a concise summary instead of dropping files
        #: silently.
        self.skipped: list[tuple[Path, str]] = []
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the scan to stop at the next file boundary."""
        self._cancelled = True

    def run(self) -> None:
        """Scan, probe, and emit one :class:`QueueItem` per media file.

        ``finished`` is emitted on every path, including an unexpected
        failure, because the window uses it to retire the scan thread and to
        put its controls back.
        """
        items: list[QueueItem] = []
        try:
            items = self._scan()
        except Exception as error:  # noqa: BLE001 - the window must be told
            logger.exception("The media scan failed")
            self.failed.emit(f"The media scan could not finish: {error}")
        finally:
            # Emitted before finished, so the window has the skipped list when
            # it builds the queue summary.
            self.skipped_files.emit(list(self.skipped))
            self.finished.emit(items)

    def _scan(self) -> list[QueueItem]:
        """Do the scanning work and return everything that was queued.

        Whether a file can be transcribed is decided by ffprobe finding an
        audio stream, not by its filename. The extension list is only a fast
        path that keeps obvious non-media out of the probe queue.
        """
        self.started_scan.emit()
        self.skipped = []
        items: list[QueueItem] = []

        excluded: list[Path] = []
        if self.output_parent is not None:
            excluded.append(output_roots(self.output_parent).transcription)

        # Each discovered path is resolved exactly once, here, and the result
        # travels on the QueueItem. The window then de-duplicates by that
        # stored identity rather than resolving the whole queue again.
        discovered: list[tuple[object, Path]] = []
        try:
            seen: set[Path] = set()
            for root in self.source_roots:
                if not root.exists():
                    self.failed.emit(f"Source does not exist: {root}")
                    continue
                for found in discover_media(
                    root, excluded_roots=excluded, rejected=self.skipped
                ):
                    identity = source_identity(found.path)
                    if identity not in seen:
                        seen.add(identity)
                        discovered.append((found, identity))
        except OSError as error:
            self.failed.emit(f"Could not read the source folder: {error}")
            return items

        total = len(discovered)
        self.found_files.emit(total)

        root_identities: dict[Path, Path] = {}
        for index, (found, identity) in enumerate(discovered):
            if self._cancelled:
                break

            root_identity = root_identities.get(found.source_root)
            if root_identity is None:
                root_identity = source_identity(found.source_root)
                root_identities[found.source_root] = root_identity

            item = QueueItem(
                source=found.path,
                source_root=found.source_root,
                relative_folder=found.relative_folder,
                status=QueueStatus.PROBING,
                identity=identity,
                root_identity=root_identity,
            )
            if self.output_parent is not None:
                item.outputs = build_output_paths(
                    found.path, found.source_root, self.output_parent
                )

            expected_media = is_supported_media(found.path)
            try:
                media = probe_media(found.path)
            except MediaProbeError as error:
                if expected_media:
                    # A file whose extension says it is media stays visible in
                    # the queue as a failure, because the user meant to queue
                    # it and needs to see that it could not be read.
                    item.status = QueueStatus.FAILED
                    item.message = str(error)
                else:
                    self.skipped.append(
                        (found.path, self._probe_failure_reason(error))
                    )
                    self.progress.emit(index + 1, total)
                    continue
            else:
                if not self._has_usable_audio(media, expected_media):
                    self.skipped.append((found.path, "no audio stream"))
                    self.progress.emit(index + 1, total)
                    continue
                item.media = media
                item.status = QueueStatus.READY

            items.append(item)
            self.item_ready.emit(index, item)
            self.progress.emit(index + 1, total)

        return items

    @staticmethod
    def _has_usable_audio(media, expected_media: bool) -> bool:
        """Decide whether a probed file carries audio worth transcribing.

        ``stream_count`` of zero means ffprobe told us nothing about the
        streams rather than that there is no audio, so a file whose extension
        says it is media is given the benefit of the doubt instead of being
        dropped on a reading we cannot trust.
        """
        if media.has_audio:
            return True
        if media.stream_count == 0:
            return expected_media
        return False

    @staticmethod
    def _probe_failure_reason(error: Exception) -> str:
        """Turn a probe failure into a short reason for the skipped summary."""
        detail = str(error).strip()
        if ": " in detail:
            detail = detail.split(": ", 1)[1]
        detail = " ".join(detail.split())
        if len(detail) > 120:
            detail = detail[:119] + "\u2026"
        return f"could not be read ({detail})" if detail else "could not be read"


class ModelDownloadWorker(QObject):
    """Fetch the diarization models on a thread, with progress and cancel.

    The download used to happen inside the batch, where it looked like a
    transcription that had stalled. Doing it up front behind a progress dialog
    means the user can see what is happening, and stop it.
    """

    progress = Signal(str, int, int)
    finished = Signal(bool, str)

    def __init__(
        self,
        cache: ModelCache,
        bundle: ModelBundle = DEFAULT_BUNDLE,
        repair: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.cache = cache
        self.bundle = bundle
        self.repair = repair
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the transfer to stop at the next chunk boundary."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        """Download or repair the bundle and report how it went."""
        succeeded = False
        message = ""
        try:
            fetch = self.cache.repair if self.repair else self.cache.ensure
            fetch(
                self.bundle,
                progress=self.progress.emit,
                cancelled=self.is_cancelled,
            )
            succeeded = True
        except DownloadCancelled:
            message = "Download cancelled."
        except ModelCacheError as error:
            message = str(error)
        except Exception as error:  # noqa: BLE001 - the dialog must be told
            logger.exception("The model download failed")
            message = f"{type(error).__name__}: {error}"
        finally:
            self.finished.emit(succeeded, message)


class TranscriptionWorker(QObject):
    """Run a :class:`BatchProcessor` on a background thread.

    Conflict questions travel to the main thread as a signal; this worker then
    blocks on an event until the window hands back the user's choice. That
    keeps every dialog on the UI thread without the pipeline knowing about Qt.
    """

    stage_changed = Signal(int, object)
    item_finished = Signal(int, object, str)
    progress = Signal(int, int)
    conflict = Signal(int, object, object)
    review = Signal(int, object, object)
    finished = Signal(object)

    #: How often a blocked worker re-checks whether the batch was cancelled
    #: while it waits for a dialog answer. This is what stops a cancel that
    #: lands between the pipeline's check and the wait from deadlocking.
    ANSWER_POLL_SECONDS = 0.2

    def __init__(
        self,
        engine: TranscriptionEngine,
        jobs: Sequence[BatchJob],
        rows: Sequence[int],
        source_root: Path | None,
        output_parent: Path,
        policy: ExistingFilePolicy = ExistingFilePolicy.ASK,
        diarizer: DiarizationBackend | None = None,
        diarization: DiarizationOptions | None = None,
        include_speakers_in_scriptsync: bool = False,
        include_speakers_in_subtitles: bool = True,
        review_mode: ReviewMode = ReviewMode.EVERY_FILE,
        output_options: OutputOptions | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.jobs = list(jobs)
        self.rows = list(rows)
        self._answered = threading.Event()
        self._choice: ConflictChoice = ConflictChoice.SKIP_THIS
        self._reviewed = threading.Event()
        self._decision: ReviewDecision = ReviewDecision.CONTINUE

        self.processor = BatchProcessor(
            engine=engine,
            source_root=source_root or Path.cwd(),
            output_parent=output_parent,
            policy=policy,
            conflict_resolver=self._ask_about_conflict,
            on_stage=self._emit_stage,
            on_outcome=self._emit_outcome,
            on_progress=self.progress.emit,
            diarizer=diarizer,
            diarization=diarization,
            include_speakers_in_scriptsync=include_speakers_in_scriptsync,
            include_speakers_in_subtitles=include_speakers_in_subtitles,
            review_mode=review_mode,
            review_resolver=self._ask_about_speakers,
            output_options=output_options,
        )

    # --------------------------------------------------------------- control

    def cancel(self) -> None:
        """Stop the batch after the clip being processed finishes safely."""
        self.processor.cancel()
        # Unblock any pending dialog so the batch can wind down.
        if not self._answered.is_set():
            self._choice = ConflictChoice.CANCEL_BATCH
            self._answered.set()
        if not self._reviewed.is_set():
            self._decision = ReviewDecision.CANCEL_BATCH
            self._reviewed.set()

    def provide_conflict_choice(self, choice: ConflictChoice) -> None:
        """Hand the user's dialog answer back to the waiting worker."""
        self._choice = choice
        self._answered.set()

    def provide_review_decision(self, decision: ReviewDecision) -> None:
        """Hand the speaker review answer back to the waiting worker."""
        self._decision = decision
        self._reviewed.set()

    def run(self) -> None:
        """Process the batch and emit the summary.

        ``finished`` is emitted exactly once on every path. The window uses it
        to quit the thread, release the sleep assertion, and re-enable its
        controls, so a worker that stopped without emitting would leave the
        application permanently stuck in its running state.
        """
        summary: BatchSummary
        try:
            summary = self.processor.run(self.jobs)
        except BaseException as error:  # noqa: BLE001 - the window must be told
            logger.exception("The transcription batch failed")
            summary = self._crash_summary(error)
        finally:
            self.finished.emit(summary)

    def _crash_summary(self, error: BaseException) -> BatchSummary:
        """Describe a batch that stopped for a reason the pipeline did not catch."""
        message = f"{UNEXPECTED_ITEM_ERROR} ({type(error).__name__}): {error}"
        summary = BatchSummary(failed=len(self.jobs))
        summary.failures = [(job.source, message) for job in self.jobs]
        return summary

    # -------------------------------------------------------------- plumbing

    def _row_for(self, index: int) -> int:
        return self.rows[index] if index < len(self.rows) else index

    def _emit_stage(self, index: int, _job: BatchJob, stage: ItemStage) -> None:
        self.stage_changed.emit(self._row_for(index), QueueStatus.from_stage(stage))

    def _emit_outcome(
        self,
        index: int,
        _job: BatchJob,
        outcome: ItemOutcome,
        message: str,
    ) -> None:
        self.item_finished.emit(
            self._row_for(index), QueueStatus.from_outcome(outcome), message
        )

    def _index_of(self, job: BatchJob) -> int:
        return next(
            (
                position
                for position, candidate in enumerate(self.jobs)
                if candidate is job
            ),
            0,
        )

    def _await_answer(self, event: threading.Event) -> bool:
        """Wait for a dialog answer, giving up if the batch is cancelled.

        The wait is polled rather than unbounded. Cancelling sets the event so
        an already-waiting worker wakes at once, but a cancel that arrives
        just before the event is cleared would otherwise leave this thread
        waiting for an answer nobody is going to give.
        """
        while not event.wait(self.ANSWER_POLL_SECONDS):
            if self.processor.cancelled:
                return False
        return True

    def _ask_about_conflict(
        self,
        job: BatchJob,
        existing: Sequence[Path],
    ) -> ConflictChoice:
        """Ask the window what to do and block until it answers."""
        self._answered.clear()
        self._choice = ConflictChoice.SKIP_THIS
        self.conflict.emit(self._row_for(self._index_of(job)), job.source, list(existing))
        if not self._await_answer(self._answered):
            return ConflictChoice.CANCEL_BATCH
        return self._choice

    def _ask_about_speakers(
        self,
        job: BatchJob,
        transcript: SpeakerTranscript,
    ) -> ReviewDecision:
        """Show the speaker review dialog on the UI thread and wait for it.

        The transcript object itself is handed over, so every rename, merge,
        and reassignment the user makes is already applied when this returns.
        """
        self._reviewed.clear()
        self._decision = ReviewDecision.CONTINUE
        self.review.emit(self._row_for(self._index_of(job)), job, transcript)
        if not self._await_answer(self._reviewed):
            return ReviewDecision.CANCEL_BATCH
        return self._decision


def run_worker_on_thread(worker: QObject, thread: QThread) -> None:
    """Move ``worker`` onto ``thread`` and start it when the thread starts."""
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[attr-defined]
    worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
    thread.start()
