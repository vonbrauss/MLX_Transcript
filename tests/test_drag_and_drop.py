"""Finder drops onto the Media workspace and onto the Queue table.

Drag and drop is a headline feature and had no automated coverage at all. A
real Cocoa drag cannot be synthesised here, so these drive the window's event
filter with the same events Qt delivers during one: a DragEnter carrying a
``text/uri-list``, then a Drop. That is the part this application owns. The
handover from AppKit still wants a manual check on a real Mac.
"""

from __future__ import annotations

import os
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QMimeData, QPointF, QUrl, Qt  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription.media_probe import MediaInfo  # noqa: E402


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture(autouse=True)
def stub_probe(monkeypatch):
    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(
            path=path, duration_seconds=3.0, frame_rate=Fraction(24, 1)
        ),
    )


@pytest.fixture(autouse=True)
def quiet_dialogs(monkeypatch):
    seen: list[tuple[str, str]] = []
    for name in ("warning", "information"):
        monkeypatch.setattr(
            f"app.main_window.QMessageBox.{name}",
            staticmethod(
                lambda parent, title, text, *args, **kwargs: seen.append((title, text))
            ),
        )
    return seen


@pytest.fixture
def window(application):
    built = MainWindow(AppSettings())
    yield built
    built.close()


def urls_for(paths: list[Path]) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def drop_targets(window: MainWindow) -> dict[str, object]:
    return {
        "media drop zone": window.drop_target,
        "inline queue table": window.media_queue_table,
        "queue table": window.queue_table,
        "queue viewport": window.queue_drop_viewport,
    }


def send_drag_enter(window: MainWindow, target, mime: QMimeData) -> QDragEnterEvent:
    event = QDragEnterEvent(
        QPointF(10, 10).toPoint(),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.eventFilter(target, event)
    return event


def send_drop(window: MainWindow, target, mime: QMimeData) -> QDropEvent:
    event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.eventFilter(target, event)
    return event


def drain(window: MainWindow, application: QApplication, seconds: float = 8.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        thread = window._scan_thread
        if thread is None or not thread.isRunning():
            application.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError("the scan did not finish")


def clip(folder: Path, name: str = "clip.mov") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    made = folder / name
    made.write_bytes(b"0")
    return made


# ----------------------------------------------------------- accepting a drag


@pytest.mark.parametrize("target_name", ["media drop zone", "inline queue table", "queue table", "queue viewport"])
def test_a_file_drag_is_accepted_on_every_target(window, tmp_path, target_name):
    target = drop_targets(window)[target_name]
    event = send_drag_enter(window, target, urls_for([clip(tmp_path)]))

    assert event.isAccepted()
    assert target.property("dropActive") is True


@pytest.mark.parametrize("target_name", ["media drop zone", "inline queue table", "queue table", "queue viewport"])
def test_a_drag_without_local_files_is_not_accepted(window, target_name):
    target = drop_targets(window)[target_name]
    mime = QMimeData()
    mime.setText("just some text")

    event = send_drag_enter(window, target, mime)

    assert not event.isAccepted()


def test_a_remote_url_is_not_accepted(window):
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.com/clip.mov")])

    event = send_drag_enter(window, window.drop_target, mime)

    assert not event.isAccepted()


def test_leaving_clears_the_highlight(window, tmp_path):
    target = window.drop_target
    send_drag_enter(window, target, urls_for([clip(tmp_path)]))
    assert target.property("dropActive") is True

    leave = QDragLeaveEvent()
    window.eventFilter(target, leave)

    assert target.property("dropActive") is False


def test_dropping_clears_the_highlight(window, application, tmp_path):
    target = window.drop_target
    mime = urls_for([clip(tmp_path)])
    send_drag_enter(window, target, mime)
    send_drop(window, target, mime)
    drain(window, application)

    assert target.property("dropActive") is False


# ------------------------------------------------------------ queueing a drop


@pytest.mark.parametrize("target_name", ["media drop zone", "inline queue table", "queue table", "queue viewport"])
def test_dropping_a_file_queues_it_on_every_target(
    window, application, tmp_path, target_name
):
    target = drop_targets(window)[target_name]
    made = clip(tmp_path / target_name.replace(" ", "_"))
    mime = urls_for([made])

    send_drag_enter(window, target, mime)
    event = send_drop(window, target, mime)
    drain(window, application)

    assert event.isAccepted()
    assert [item.name for item in window.items] == ["clip.mov"]


def test_dropping_a_folder_queues_everything_below_it(window, application, tmp_path):
    folder = tmp_path / "Dailies"
    clip(folder / "Day1", "a.mov")
    clip(folder / "Day2", "b.wav")
    mime = urls_for([folder])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert sorted(item.name for item in window.items) == ["a.mov", "b.wav"]


def test_dropping_a_mix_of_files_and_folders_queues_both(
    window, application, tmp_path
):
    folder = tmp_path / "Dailies"
    clip(folder, "inside.mov")
    loose = clip(tmp_path / "loose", "outside.wav")
    mime = urls_for([folder, loose])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert sorted(item.name for item in window.items) == ["inside.mov", "outside.wav"]


def test_dropping_onto_the_queue_appends_to_an_existing_queue(
    window, application, tmp_path
):
    first = clip(tmp_path / "one", "a.mov")
    second = clip(tmp_path / "two", "b.mov")

    for target, path in ((window.drop_target, first), (window.queue_table, second)):
        mime = urls_for([path])
        send_drag_enter(window, target, mime)
        send_drop(window, target, mime)
        drain(window, application)

    assert sorted(item.name for item in window.items) == ["a.mov", "b.mov"]


def test_a_drop_is_ignored_while_a_batch_runs(window, application, tmp_path):
    made = clip(tmp_path)
    mime = urls_for([made])

    window._batch_thread = None
    original = MainWindow.is_transcribing
    try:
        MainWindow.is_transcribing = property(lambda self: True)
        send_drag_enter(window, window.drop_target, mime)
        send_drop(window, window.drop_target, mime)
        drain(window, application)
    finally:
        MainWindow.is_transcribing = original

    assert window.items == []


# ------------------------------------------------- mixed supported and not


def test_dropping_an_unsupported_file_reports_it(
    window, application, tmp_path, quiet_dialogs
):
    notes = tmp_path / "notes.pdf"
    notes.write_bytes(b"%PDF")
    mime = urls_for([notes])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert window.items == []
    assert "not a supported media file" in window.statusBar().currentMessage()
    assert any("Not supported media" in title for title, _text in quiet_dialogs)


def test_a_mixed_drop_queues_the_media_and_names_the_rest(
    window, application, tmp_path, quiet_dialogs
):
    good = clip(tmp_path / "media", "take.mov")
    notes = tmp_path / "notes.pdf"
    notes.write_bytes(b"%PDF")
    sheet = tmp_path / "budget.xlsx"
    sheet.write_bytes(b"PK")
    mime = urls_for([good, notes, sheet])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert [item.name for item in window.items] == ["take.mov"]
    # The scan's own summary lands in the status bar afterwards, so the
    # durable report of what was skipped is the dialog.
    reports = [text for title, text in quiet_dialogs if title == "Not supported media"]
    assert reports, "the skipped files were never reported"
    assert "2 file(s) were skipped" in reports[0]
    assert ".pdf" in reports[0] and ".xlsx" in reports[0]
    assert "notes.pdf" in reports[0] and "budget.xlsx" in reports[0]


def test_an_unsupported_drop_does_not_claim_the_files_were_duplicates(
    window, application, tmp_path
):
    notes = tmp_path / "notes.pdf"
    notes.write_bytes(b"%PDF")
    mime = urls_for([notes])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert "already in the queue" not in window.statusBar().currentMessage()


def test_a_folder_with_no_media_says_which_folder(window, application, tmp_path):
    empty = tmp_path / "Paperwork"
    empty.mkdir()
    (empty / "call-sheet.pdf").write_bytes(b"%PDF")
    mime = urls_for([empty])

    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert window.items == []
    message = window.statusBar().currentMessage()
    # The wording changed when the audio-stream rule replaced the extension
    # whitelist: the folder was readable, it simply held nothing with audio.
    assert "No audio was found in Paperwork" in message
    # And the PDF is accounted for rather than dropped silently.
    assert "1 file skipped" in message
    assert "call-sheet.pdf" in window.queue_summary.toolTip()


# -------------------------------------------------------- the event filter

def test_unrelated_widgets_are_left_alone(window, tmp_path):
    mime = urls_for([clip(tmp_path)])
    event = send_drag_enter(window, window.start_button, mime)

    assert not event.isAccepted()


def test_an_unrelated_event_type_passes_through(window):
    event = QEvent(QEvent.Type.Show)

    assert window.eventFilter(window.drop_target, event) is False
