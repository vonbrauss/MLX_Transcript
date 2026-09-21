"""Closing the window must never destroy a thread that is still running.

The review found that closing during a batch blocked the main thread for
thirty seconds and then tore down a live QThread, which Qt answers with an
abort. The window now asks the batch to stop, shows that it is finishing the
current file, and closes itself when the worker reports back.
"""

from __future__ import annotations

import os
import threading
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.main_window import FINISHING_TEXT, MainWindow  # noqa: E402
from app.models import ConflictChoice, QueueItem, QueueStatus  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription.engine import TranscriptionEngine  # noqa: E402
from transcription.media_probe import MediaInfo  # noqa: E402


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture(autouse=True)
def quiet_summary(monkeypatch):
    """The batch summary is modal; this module is about the shutdown path."""
    monkeypatch.setattr(MainWindow, "_show_batch_summary", lambda self, summary: None)


def queued(window: MainWindow, clip: Path) -> None:
    item = QueueItem(
        source=clip,
        source_root=clip.parent,
        status=QueueStatus.READY,
        media=MediaInfo(
            path=clip, duration_seconds=1.0, frame_rate=Fraction(24, 1)
        ),
    )
    window.items.append(item)
    window._append_row(item)
    window._queued_identities.add(item.resolved_identity)


def pump(application: QApplication, seconds: float = 0.4) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)


def wait_until(application: QApplication, predicate, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def start_blocking_batch(
    window: MainWindow, application: QApplication, tmp_path: Path
) -> threading.Event:
    """Start a real batch on a real thread that parks inside the engine."""
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    queued(window, clip)
    window.output_field.setText(str(tmp_path / "out"))

    release = threading.Event()

    def slow_transcribe(path, **kwargs):
        release.wait(20)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}], "language": "en"}

    window.engine_factory = lambda: TranscriptionEngine(transcribe_fn=slow_transcribe)
    window._confirm_whisper_download = lambda: True
    window._start_transcription()
    assert wait_until(application, lambda: window.is_transcribing)
    return release


# ------------------------------------------------------- closing mid-batch


def test_closing_during_a_batch_does_not_close_immediately(
    application, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Close),
    )
    window = MainWindow(AppSettings())
    window.show()
    release = start_blocking_batch(window, application, tmp_path)
    try:
        window.close()
        pump(application)

        assert window.isVisible(), "the window closed while a clip was in flight"
        assert window.current_file_label.text() == FINISHING_TEXT
        assert window._closing is True
    finally:
        release.set()
        wait_until(application, lambda: not window.is_transcribing)
        pump(application)


def test_the_window_closes_once_the_worker_reports_back(
    application, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Close),
    )
    window = MainWindow(AppSettings())
    window.show()
    release = start_blocking_batch(window, application, tmp_path)

    window.close()
    pump(application)
    assert window.isVisible()
    release.set()

    assert wait_until(application, lambda: not window.isVisible())
    assert not window.is_transcribing


def test_declining_the_prompt_keeps_the_batch_running(
    application, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Cancel),
    )
    window = MainWindow(AppSettings())
    window.show()
    release = start_blocking_batch(window, application, tmp_path)
    try:
        window.close()
        pump(application)

        assert window.isVisible()
        assert window._closing is False
        assert window.is_transcribing
    finally:
        release.set()
        wait_until(application, lambda: not window.is_transcribing)
        pump(application)


def test_closing_never_blocks_the_main_thread_on_a_running_clip(
    application, tmp_path, monkeypatch
):
    """The old close path waited thirty seconds here before giving up."""
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Close),
    )
    window = MainWindow(AppSettings())
    release = start_blocking_batch(window, application, tmp_path)
    try:
        started = time.monotonic()
        window.close()
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, f"close() blocked for {elapsed:.1f}s"
    finally:
        release.set()
        wait_until(application, lambda: not window.is_transcribing)
        pump(application)


def test_closing_an_idle_window_still_works(application, tmp_path):
    window = MainWindow(AppSettings())
    window.show()
    assert window.isVisible()

    window.close()

    assert not window.isVisible()


# ------------------------------------------------------- thread lifecycle


def test_a_finished_scan_thread_is_retired(application, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(
            path=path, duration_seconds=1.0, frame_rate=Fraction(24, 1)
        ),
    )
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "a.mov").write_bytes(b"0")

    window = MainWindow(AppSettings())
    try:
        window._queue_sources([folder])
        assert wait_until(
            application,
            lambda: window._scan_thread is None
            or not window._scan_thread.isRunning(),
        )
        pump(application)

        assert window._live_threads == []
    finally:
        window.close()


def test_a_finished_batch_thread_is_retired(application, tmp_path, monkeypatch):
    window = MainWindow(AppSettings())
    try:
        release = start_blocking_batch(window, application, tmp_path)
        release.set()
        assert wait_until(application, lambda: not window.is_transcribing)
        pump(application)

        assert window._live_threads == []
    finally:
        window.close()


def test_the_sleep_assertion_is_released_when_a_batch_ends(
    application, tmp_path
):
    window = MainWindow(AppSettings())
    try:
        release = start_blocking_batch(window, application, tmp_path)
        release.set()
        assert wait_until(application, lambda: not window.is_transcribing)

        assert window._sleep_blocker.active is False
    finally:
        window.close()


# ---------------------------------------------- dialog handlers never raise


def test_a_conflict_dialog_that_fails_does_not_raise_through_the_event_loop(
    application, tmp_path, monkeypatch
):
    window = MainWindow(AppSettings())
    try:
        release = start_blocking_batch(window, application, tmp_path)
        monkeypatch.setattr(
            "app.main_window.ConflictDialog",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no display")),
        )
        worker = window._batch_worker
        assert worker is not None

        # Must return rather than raise, and the worker must still get an answer.
        window._on_conflict(0, tmp_path / "clip.mov", [])

        assert worker._choice is ConflictChoice.CANCEL_BATCH
        assert worker._answered.is_set()
    finally:
        release.set()
        wait_until(application, lambda: not window.is_transcribing)
        window.close()


def test_a_review_dialog_that_fails_does_not_raise_through_the_event_loop(
    application, tmp_path, monkeypatch
):
    window = MainWindow(AppSettings())
    try:
        release = start_blocking_batch(window, application, tmp_path)
        monkeypatch.setattr(
            "app.main_window.SpeakerReviewDialog",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no display")),
        )
        worker = window._batch_worker
        assert worker is not None

        window._on_review(0, worker.jobs[0], object())

        assert worker._reviewed.is_set()
    finally:
        release.set()
        wait_until(application, lambda: not window.is_transcribing)
        window.close()
