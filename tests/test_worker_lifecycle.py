"""The workers always report back, and never wait forever for an answer.

The window drives everything off the workers' completion signals: quitting
the thread, releasing the sleep assertion, and putting its controls back. A
worker that stopped without emitting left the application stuck in a running
state that only a relaunch cleared, which is what these pin down.
"""

from __future__ import annotations

import os
import threading
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from app.models import ConflictChoice, ReviewDecision  # noqa: E402
from app.workers import ScanWorker, TranscriptionWorker  # noqa: E402
from transcription.engine import TranscriptionEngine  # noqa: E402
from transcription.media_probe import MediaInfo  # noqa: E402
from transcription.pipeline import (  # noqa: E402
    UNEXPECTED_ITEM_ERROR,
    BatchJob,
    BatchSummary,
)
from transcription.speakers import SpeakerTranscript  # noqa: E402


@pytest.fixture
def stub_probe(monkeypatch):
    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(
            path=path, duration_seconds=1.0, frame_rate=Fraction(24, 1)
        ),
    )


def build_worker(tmp_path: Path, jobs: list[BatchJob]) -> TranscriptionWorker:
    engine = TranscriptionEngine(
        transcribe_fn=lambda path, **kwargs: {"segments": [], "language": "en"}
    )
    return TranscriptionWorker(
        engine=engine,
        jobs=jobs,
        rows=list(range(len(jobs))),
        source_root=tmp_path,
        output_parent=tmp_path / "out",
    )


# ------------------------------------------------------ the completion signal


def test_the_transcription_worker_emits_finished_after_a_crash(tmp_path):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    summaries: list[BatchSummary] = []
    worker.finished.connect(summaries.append)

    def explode(_jobs):
        raise RuntimeError("the pipeline gave up")

    worker.processor.run = explode
    worker.run()

    assert len(summaries) == 1
    assert summaries[0].failed == 1
    path, message = summaries[0].failures[0]
    assert path == clip
    assert UNEXPECTED_ITEM_ERROR in message
    assert "the pipeline gave up" in message


def test_the_transcription_worker_emits_finished_exactly_once(tmp_path):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    seen: list[object] = []
    worker.finished.connect(seen.append)

    worker.run()

    assert len(seen) == 1


def test_a_crash_summary_names_every_queued_clip(tmp_path):
    made = []
    for index in range(3):
        clip = tmp_path / f"clip{index}.mov"
        clip.write_bytes(b"0")
        made.append(clip)
    worker = build_worker(tmp_path, [BatchJob(source=clip) for clip in made])
    worker.processor.run = lambda _jobs: (_ for _ in ()).throw(OSError("gone"))
    summaries: list[BatchSummary] = []
    worker.finished.connect(summaries.append)

    worker.run()

    assert [path for path, _message in summaries[0].failures] == made


def test_the_scan_worker_emits_finished_after_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.workers.discover_media",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scan blew up")),
    )
    worker = ScanWorker([tmp_path])
    finished: list[list] = []
    failures: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert finished == [[]]
    assert failures and "scan blew up" in failures[0]


def test_the_scan_worker_emits_finished_when_a_source_is_unreadable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "app.workers.discover_media",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    worker = ScanWorker([tmp_path])
    finished: list[list] = []
    worker.finished.connect(finished.append)

    worker.run()

    assert finished == [[]]


def test_the_scan_worker_records_one_identity_per_item(tmp_path, stub_probe):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = ScanWorker([tmp_path])
    finished: list[list] = []
    worker.finished.connect(finished.append)

    worker.run()

    (item,) = finished[0]
    assert item.identity == clip.resolve()
    assert item.root_identity == tmp_path.resolve()


# ----------------------------------------------------------- bounded waiting


def test_cancelling_releases_a_worker_waiting_on_a_conflict(tmp_path):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    answered: list[ConflictChoice] = []

    def ask():
        answered.append(worker._ask_about_conflict(worker.jobs[0], []))

    asking = threading.Thread(target=ask, daemon=True)
    asking.start()
    worker.cancel()
    asking.join(timeout=5)

    assert not asking.is_alive()
    assert answered == [ConflictChoice.CANCEL_BATCH]


def test_a_cancel_that_lands_before_the_wait_does_not_deadlock(tmp_path):
    """The race the review flagged: cancel arriving just before the clear."""
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    worker.cancel()  # sets the events, then the ask clears them again
    answered: list[ConflictChoice] = []

    asking = threading.Thread(
        target=lambda: answered.append(
            worker._ask_about_conflict(worker.jobs[0], [])
        ),
        daemon=True,
    )
    asking.start()
    asking.join(timeout=5)

    assert not asking.is_alive()
    assert answered == [ConflictChoice.CANCEL_BATCH]


def test_a_cancel_before_a_speaker_review_does_not_deadlock(tmp_path):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    worker.cancel()
    decided: list[ReviewDecision] = []

    asking = threading.Thread(
        target=lambda: decided.append(
            worker._ask_about_speakers(worker.jobs[0], SpeakerTranscript())
        ),
        daemon=True,
    )
    asking.start()
    asking.join(timeout=5)

    assert not asking.is_alive()
    assert decided == [ReviewDecision.CANCEL_BATCH]


def test_an_answer_still_reaches_a_waiting_worker(tmp_path):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    worker = build_worker(tmp_path, [BatchJob(source=clip)])
    answered: list[ConflictChoice] = []
    asked = threading.Event()
    # A direct connection, because this worker was never moved onto a thread
    # with an event loop to deliver a queued one.
    worker.conflict.connect(
        lambda *args: asked.set(), Qt.ConnectionType.DirectConnection
    )

    asking = threading.Thread(
        target=lambda: answered.append(
            worker._ask_about_conflict(worker.jobs[0], [])
        ),
        daemon=True,
    )
    asking.start()
    assert asked.wait(5)
    worker.provide_conflict_choice(ConflictChoice.OVERWRITE_ALL)
    asking.join(timeout=5)

    assert answered == [ConflictChoice.OVERWRITE_ALL]
