"""Background workers. The UI thread never touches the filesystem directly.

Each worker is a plain :class:`QObject` moved onto a :class:`QThread` by the
window, so long scans and long transcription batches leave the interface
responsive.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QObject, QThread, Signal

from app.models import ConflictChoice, QueueItem, QueueStatus, ReviewDecision
from transcription.diarization import DiarizationBackend, DiarizationOptions
from transcription.discovery import discover_media
from transcription.engine import TranscriptionEngine
from transcription.media_probe import MediaProbeError, probe_media
from transcription.outputs import OutputOptions, build_output_paths, output_roots
from transcription.pipeline import (
    BatchJob,
    BatchProcessor,
    ExistingFilePolicy,
    ItemOutcome,
    ItemStage,
    ReviewMode,
)
from transcription.speakers import SpeakerTranscript

__all__ = ["ScanWorker", "TranscriptionWorker", "run_worker_on_thread"]


class ScanWorker(QObject):
    """Find media below a source folder and read each duration with ffprobe."""

    started_scan = Signal()
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
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the scan to stop at the next file boundary."""
        self._cancelled = True

    def run(self) -> None:
        """Scan, probe, and emit one :class:`QueueItem` per media file."""
        self.started_scan.emit()
        items: list[QueueItem] = []

        excluded: list[Path] = []
        if self.output_parent is not None:
            excluded.append(output_roots(self.output_parent).transcription)

        try:
            discovered = []
            seen: set[Path] = set()
            for root in self.source_roots:
                if not root.exists():
                    self.failed.emit(f"Source does not exist: {root}")
                    continue
                for found in discover_media(root, excluded_roots=excluded):
                    try:
                        identity = found.path.resolve()
                    except OSError:
                        identity = found.path
                    if identity not in seen:
                        seen.add(identity)
                        discovered.append(found)
        except OSError as error:
            self.failed.emit(f"Could not read the source folder: {error}")
            self.finished.emit(items)
            return

        total = len(discovered)
        self.found_files.emit(total)

        for index, found in enumerate(discovered):
            if self._cancelled:
                break

            item = QueueItem(
                source=found.path,
                source_root=found.source_root,
                relative_folder=found.relative_folder,
                status=QueueStatus.PROBING,
            )
            if self.output_parent is not None:
                item.outputs = build_output_paths(
                    found.path, found.source_root, self.output_parent
                )

            try:
                item.media = probe_media(found.path)
                item.status = QueueStatus.READY
            except MediaProbeError as error:
                item.status = QueueStatus.FAILED
                item.message = str(error)

            items.append(item)
            self.item_ready.emit(index, item)
            self.progress.emit(index + 1, total)

        self.finished.emit(items)


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
        """Process the batch and emit the summary."""
        summary = self.processor.run(self.jobs)
        self.finished.emit(summary)

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

    def _ask_about_conflict(
        self,
        job: BatchJob,
        existing: Sequence[Path],
    ) -> ConflictChoice:
        """Ask the window what to do and block until it answers."""
        self._answered.clear()
        self._choice = ConflictChoice.SKIP_THIS
        self.conflict.emit(self._row_for(self._index_of(job)), job.source, list(existing))
        self._answered.wait()
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
        self._reviewed.wait()
        return self._decision


def run_worker_on_thread(worker: QObject, thread: QThread) -> None:
    """Move ``worker`` onto ``thread`` and start it when the thread starts."""
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[attr-defined]
    worker.finished.connect(thread.quit)  # type: ignore[attr-defined]
    thread.start()
