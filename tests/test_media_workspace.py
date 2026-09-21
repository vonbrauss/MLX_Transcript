"""The Media workspace: the rename, the inline queue, and row reordering.

The Folders page became Media, and its large lower area now does two jobs.
Empty, it is the drop target it always was. Loaded, it is the queue itself,
inline, showing the same rows as the dedicated Queue page. These cover the
parts of that the application owns: the visible wording, which state the
workspace shows, that both views read one list, and that dragging a row
moves the authoritative list rather than just the pixels.

A real Cocoa drag cannot be synthesised here, so the drag tests drive the
window's event filter with the events Qt delivers during one. The handover
from AppKit still wants a manual check on a real Mac.
"""

from __future__ import annotations

import os
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import (  # noqa: E402
    QByteArray,
    QEvent,
    QMimeData,
    QObject,
    QPointF,
    QUrl,
    Qt,
    Signal,
)
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app import main_window as main_window_module  # noqa: E402
from app.main_window import (  # noqa: E402
    INTERNAL_ROW_MIME,
    MEDIA_DROP_OVERLAY_TEXT,
    MEDIA_EMPTY_HINT,
    MEDIA_EMPTY_TITLE,
    MainWindow,
)
from app import queue_panel as queue_panel_module  # noqa: E402
from app.models import QueueItem, QueueStatus  # noqa: E402
from app.queue_panel import QueueTable  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription.media_probe import MediaInfo  # noqa: E402
from transcription.pipeline import BatchSummary  # noqa: E402


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
    for name in ("warning", "information", "question"):
        monkeypatch.setattr(
            f"app.main_window.QMessageBox.{name}",
            staticmethod(
                lambda parent, title, text, *args, **kwargs: seen.append(
                    (title, text)
                )
            ),
        )
    return seen


@pytest.fixture
def window(application, tmp_path):
    built = MainWindow(AppSettings(output_parent=str(tmp_path / "Out")))
    yield built
    built.close()


# ------------------------------------------------------------------ helpers


def clip(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    made = folder / name
    made.write_bytes(b"0")
    return made


def urls_for(paths: list[Path]) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def internal_row_mime() -> QMimeData:
    """The payload Qt gives an item view dragging its own rows."""
    mime = QMimeData()
    mime.setData(INTERNAL_ROW_MIME, QByteArray(b"row"))
    return mime


def send_drag_enter(
    window, target, mime, point=(10, 10), action=Qt.DropAction.CopyAction
) -> QDragEnterEvent:
    event = QDragEnterEvent(
        QPointF(*point).toPoint(),
        action,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.eventFilter(target, event)
    return event


def send_drop(
    window, target, mime, point=(10, 10), action=Qt.DropAction.CopyAction
) -> QDropEvent:
    event = QDropEvent(
        QPointF(*point),
        action,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.eventFilter(target, event)
    return event


def drain(window, application, seconds: float = 8.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        thread = window._scan_thread
        if thread is None or not thread.isRunning():
            application.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError("the scan did not finish")


def queue_clips(window, application, folder: Path, names: list[str]) -> None:
    """Put named clips into the queue through the real drop path."""
    made = [clip(folder, name) for name in names]
    mime = urls_for(made)
    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)


def fake_item(source: Path, root: Path) -> QueueItem:
    """A ready queue item, without going through a scan."""
    return QueueItem(
        source=source,
        source_root=root,
        status=QueueStatus.READY,
        media=MediaInfo(
            path=source, duration_seconds=1.0, frame_rate=Fraction(24, 1)
        ),
    )


def load_queue(window, items: list[QueueItem]) -> None:
    for item in items:
        window._add_item(item)
    window._refresh_queue_summary()
    window._update_actions()


def names(window) -> list[str]:
    return [item.name for item in window.items]


def column(panel, index: int = 0) -> list[str]:
    return [
        panel.table.item(row, index).text()
        for row in range(panel.table.rowCount())
    ]


# ------------------------------------------------------------- the rename


def test_the_navigation_tab_is_called_media(window):
    assert window.section_tabs.tabText(0) == "Media"
    assert "Folders" not in [
        window.section_tabs.tabText(index)
        for index in range(window.section_tabs.count())
    ]


def test_the_media_page_heading_and_guidance_name_media(window):
    window._select_section("folders")
    page = window.section_stack.currentWidget()
    text = " ".join(
        label.text() for label in page.findChildren(type(window.app_title_label))
    )
    assert "Choose your media" in text


def test_the_help_text_points_at_media_not_folders(window):
    help_text = window.help_text.toPlainText()
    assert "Open Media" in help_text
    assert "Open Folders" not in help_text


def test_the_empty_queue_summary_points_at_media(application, tmp_path):
    built = MainWindow(AppSettings())
    try:
        assert "in Media" in built.queue_summary.text()
        built._refresh_queue_summary()
        assert "add them in Media" in built.queue_summary.text()
        assert "Folders" not in built.queue_summary.text()
    finally:
        built.close()


def test_the_destination_hint_names_media(application):
    built = MainWindow(AppSettings())
    try:
        built.items.append(
            fake_item(Path("/tmp/a.mov"), Path("/tmp"))
        )
        built._update_actions()
        assert "in Media" in built.start_button.toolTip()
    finally:
        built.close()


def test_the_drop_target_tooltip_names_the_media_queue(window):
    assert "Media queue" in window.drop_target.toolTip()


def test_the_section_key_is_unchanged_for_stored_settings(window):
    """Existing saved settings say "folders"; they must still open Media."""
    window._select_section("folders")
    assert window.section_stack.currentWidget() is window.media_group
    assert window._collect_settings().selected_section == "folders"


# ------------------------------------------------------- the empty state


def test_the_empty_media_workspace_keeps_the_drop_zone(window):
    assert not window.items
    assert window.media_stack.currentWidget() is window.media_empty_state
    assert window.drop_target.property("mode") == "empty"
    assert window.drop_title_label.text() == MEDIA_EMPTY_TITLE
    assert window.drop_hint_label.text() == MEDIA_EMPTY_HINT
    assert (
        window.drop_title_label.alignment() & Qt.AlignmentFlag.AlignHCenter
    )
    assert window.drop_target.minimumHeight() >= 150


def test_the_selection_buttons_stay_on_the_media_page(window):
    assert window.source_button.text() == "Choose Folder…"
    assert window.source_file_button.text() == "Choose File…"
    assert window.source_button.isEnabled()
    assert window.source_file_button.isEnabled()


# ------------------------------------------------------- the loaded state


def test_queued_media_replaces_the_empty_state_with_the_queue(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    assert window.media_queue_is_loaded
    assert window.media_stack.currentWidget() is window.media_queue_page
    assert window.drop_target.property("mode") == "queue"
    assert sorted(column(window.media_queue_panel)) == ["a.mov", "b.mov"]


def test_the_inline_queue_carries_the_same_controls_as_the_queue_page(window):
    panel = window.media_queue_panel
    assert [
        panel.table.horizontalHeaderItem(index).text()
        for index in range(panel.table.columnCount())
    ] == ["File", "Folder", "Duration", "Status"]
    assert panel.remove_button.text() == "Remove Selected"
    assert panel.clear_button.text() == "Clear Queue"
    assert panel.reveal_button.text() == "Reveal Source"
    assert panel.summary is not window.queue_panel.summary


def test_the_inline_queue_fills_the_workspace(window, application, tmp_path):
    window.resize(1180, 800)
    window.show()
    window._select_section("folders")
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])
    application.processEvents()

    assert window.media_queue_table.height() > 100
    assert (
        window.media_queue_table.width()
        > window.drop_target.width() - 60
    )


def test_clearing_the_queue_returns_the_empty_state(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])
    assert window.media_queue_is_loaded

    window._clear_queue()

    assert not window.media_queue_is_loaded
    assert window.media_stack.currentWidget() is window.media_empty_state
    assert window.drop_target.property("mode") == "empty"


# ------------------------------------------------------ the two views agree


def test_both_views_show_the_same_rows(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    assert column(window.media_queue_panel) == column(window.queue_panel)
    assert column(window.media_queue_panel) == names(window)


def test_a_status_change_reaches_both_views(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])

    window._set_item_status(0, QueueStatus.TRANSCRIBING)

    assert window.media_queue_panel.table.item(0, 3).text() == "Transcribing"
    assert window.queue_panel.table.item(0, 3).text() == "Transcribing"


def test_the_summary_is_the_same_in_both_views(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    assert window.media_queue_panel.summary.text() == window.queue_summary.text()
    assert "2 media file(s)" in window.queue_summary.text()


def test_selecting_in_one_view_selects_in_the_other(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    window.media_queue_panel.table.selectRow(1)
    application.processEvents()

    assert window.queue_panel.selected_rows() == [1]
    assert window.media_queue_panel.remove_button.isEnabled()
    assert window.remove_selected_button.isEnabled()


def test_removing_from_the_inline_queue_empties_both_views(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    window.media_queue_panel.table.selectRow(0)
    application.processEvents()
    window.media_queue_panel.remove_button.click()

    assert names(window) == ["b.mov"]
    assert column(window.media_queue_panel) == ["b.mov"]
    assert column(window.queue_panel) == ["b.mov"]


def test_there_is_only_one_queue(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])
    assert window.media_queue_panel.table is not window.queue_panel.table
    assert window.media_queue_panel.table.rowCount() == len(window.items)
    assert window.queue_panel.table.rowCount() == len(window.items)


# ---------------------------------------------------------------- reorder


def test_moving_a_row_updates_the_authoritative_list(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov"])
    assert names(window) == ["a.mov", "b.mov", "c.mov"]

    assert window._move_queue_rows([2], 0) is True

    assert names(window) == ["c.mov", "a.mov", "b.mov"]


def test_a_move_reaches_both_views_immediately(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov"])

    window._move_queue_rows([0], 3)

    assert names(window) == ["b.mov", "c.mov", "a.mov"]
    assert column(window.media_queue_panel) == ["b.mov", "c.mov", "a.mov"]
    assert column(window.queue_panel) == ["b.mov", "c.mov", "a.mov"]


def test_a_move_that_changes_nothing_reports_nothing(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])

    assert window._move_queue_rows([0], 0) is False
    assert window._move_queue_rows([0], 1) is False
    assert names(window) == ["a.mov", "b.mov"]


def test_several_rows_move_together_and_stay_selected(
    window, application, tmp_path
):
    queue_clips(
        window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov", "d.mov"]
    )

    window._move_queue_rows([0, 1], 4)

    assert names(window) == ["c.mov", "d.mov", "a.mov", "b.mov"]
    assert window.queue_panel.selected_rows() == [2, 3]
    assert window.media_queue_panel.selected_rows() == [2, 3]


def test_an_internal_row_drag_moves_the_queue(window, application, tmp_path):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov"])
    window.show()
    application.processEvents()
    panel = window.media_queue_panel
    panel.table.selectRow(2)
    application.processEvents()

    mime = internal_row_mime()
    send_drag_enter(window, panel.table.viewport(), mime, point=(10, 0))
    send_drop(window, panel.table.viewport(), mime, point=(10, 0))

    assert names(window) == ["c.mov", "a.mov", "b.mov"]


def test_an_internal_row_drag_shows_an_insertion_indicator(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])
    window.show()
    application.processEvents()
    panel = window.media_queue_panel
    panel.table.selectRow(0)

    mime = internal_row_mime()
    event = send_drag_enter(window, panel.table.viewport(), mime, point=(10, 0))

    assert event.isAccepted()
    assert panel.insertion_visible
    assert panel.insertion_row == 0

    send_drop(window, panel.table.viewport(), mime, point=(10, 0))
    assert not panel.insertion_visible


def test_an_internal_row_drag_never_shows_the_external_overlay(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])
    panel = window.media_queue_panel
    panel.table.selectRow(0)

    mime = internal_row_mime()
    send_drag_enter(window, panel.table.viewport(), mime)

    assert not window.media_drop_overlay_visible
    assert panel.table.property("dropActive") is not True


def test_an_internal_row_drag_takes_down_a_showing_overlay(
    window, application, tmp_path
):
    """A Finder drag that turns into a row move must not leave the sheet up."""
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])
    send_drag_enter(
        window,
        window.media_queue_table.viewport(),
        urls_for([clip(tmp_path / "New", "d.mov")]),
    )
    assert window.media_drop_overlay_visible

    window.media_queue_panel.table.selectRow(0)
    send_drag_enter(
        window, window.media_queue_table.viewport(), internal_row_mime()
    )

    assert not window.media_drop_overlay_visible
    assert (
        window.media_queue_table.viewport().property("dropActive") is False
    )


def test_reordering_is_refused_while_a_batch_runs(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])
    original = MainWindow.is_transcribing
    try:
        MainWindow.is_transcribing = property(lambda self: True)
        window._update_actions()

        assert window.media_queue_panel.reorder_enabled is False
        assert window.queue_panel.reorder_enabled is False
        assert not window.media_queue_panel.table.dragEnabled()
        assert window._move_queue_rows([1], 0) is False

        mime = internal_row_mime()
        event = send_drag_enter(
            window, window.media_queue_panel.table.viewport(), mime
        )
        assert not event.isAccepted()
        assert not window.media_queue_panel.insertion_visible
    finally:
        MainWindow.is_transcribing = original

    assert names(window) == ["a.mov", "b.mov"]
    window._update_actions()
    assert window.media_queue_panel.reorder_enabled is True


def test_a_reorder_preserves_duplicate_detection(
    window, application, tmp_path
):
    folder = tmp_path / "Day1"
    queue_clips(window, application, folder, ["a.mov", "b.mov"])
    window._move_queue_rows([1], 0)

    again = urls_for([folder / "a.mov"])
    send_drag_enter(window, window.drop_target, again)
    send_drop(window, window.drop_target, again)
    drain(window, application)

    assert sorted(names(window)) == ["a.mov", "b.mov"]
    assert "already in the queue" in window.statusBar().currentMessage()


def test_a_reorder_keeps_source_roots_and_folder_labels(
    window, application, tmp_path
):
    """Same-named roots keep the labels they were first given."""
    first = tmp_path / "one" / "Dailies"
    second = tmp_path / "two" / "Dailies"
    queue_clips(window, application, first, ["a.mov"])
    queue_clips(window, application, second, ["b.mov"])

    before = {
        item.name: item.extras["queue_root_label"] for item in window.items
    }
    assert set(before.values()) == {"Dailies", "Dailies (2)"}
    roots = {item.name: item.source_root for item in window.items}

    window._move_queue_rows([1], 0)

    after = {
        item.name: item.extras["queue_root_label"] for item in window.items
    }
    assert after == before
    assert {item.name: item.source_root for item in window.items} == roots

    labels = window._batch_root_labels(list(range(len(window.items))))
    by_name = {
        item.name: labels[item.resolved_root_identity] for item in window.items
    }
    assert by_name == before


def test_an_external_drop_appends_after_a_reorder(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov"])
    window._move_queue_rows([1], 0)
    assert names(window) == ["b.mov", "a.mov"]

    queue_clips(window, application, tmp_path / "Day2", ["c.mov"])

    assert names(window) == ["b.mov", "a.mov", "c.mov"]


# ------------------------------------------------------ the batch's order


class RecordingWorker(QObject):
    """Stands in for the transcription worker and records the jobs it got."""

    stage_changed = Signal(int, object)
    item_finished = Signal(int, object, str)
    progress = Signal(int, int)
    conflict = Signal(int, object, object)
    review = Signal(int, object, object)
    finished = Signal(object)

    captured: list[list] = []

    def __init__(self, **kwargs):
        super().__init__()
        RecordingWorker.captured.append(list(kwargs["jobs"]))

    def run(self) -> None:
        self.finished.emit(BatchSummary())

    def cancel(self) -> None:
        pass


def test_transcription_runs_in_the_reordered_order(
    window, application, tmp_path, monkeypatch
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov"])
    window._move_queue_rows([2], 0)
    assert names(window) == ["c.mov", "a.mov", "b.mov"]

    RecordingWorker.captured.clear()
    monkeypatch.setattr(main_window_module, "TranscriptionWorker", RecordingWorker)
    monkeypatch.setattr(main_window_module, "model_is_cached", lambda _model: True)
    monkeypatch.setattr(MainWindow, "_show_batch_summary", lambda self, summary: None)

    window._start_transcription()
    deadline = time.monotonic() + 8.0
    while window.is_transcribing and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    application.processEvents()

    assert RecordingWorker.captured, "the batch never reached the worker"
    assert [job.source.name for job in RecordingWorker.captured[0]] == [
        "c.mov",
        "a.mov",
        "b.mov",
    ]


# --------------------------------------------------------- the drag overlay


@pytest.mark.parametrize("loaded", [False, True])
def test_a_finder_drag_shows_the_overlay(window, application, tmp_path, loaded):
    if loaded:
        queue_clips(window, application, tmp_path / "Day1", ["a.mov"])
    assert not window.media_drop_overlay_visible

    mime = urls_for([clip(tmp_path / "New", "d.mov")])
    event = send_drag_enter(window, window.drop_target, mime)

    assert event.isAccepted()
    assert window.media_drop_overlay_visible
    assert window.media_drop_overlay_label.text() == MEDIA_DROP_OVERLAY_TEXT


def test_the_overlay_covers_the_workspace_and_leaves_the_queue_underneath(
    window, application, tmp_path
):
    window.resize(1180, 800)
    window.show()
    window._select_section("folders")
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])
    application.processEvents()

    send_drag_enter(
        window, window.drop_target, urls_for([clip(tmp_path / "New", "d.mov")])
    )
    application.processEvents()

    assert window.media_drop_overlay.geometry() == window.drop_target.rect()
    # The queue is still there behind the sheet rather than replaced by it.
    assert window.media_stack.currentWidget() is window.media_queue_page
    assert window.media_queue_panel.table.rowCount() == 1


def test_a_drag_over_the_inline_queue_shows_the_overlay(
    window, application, tmp_path
):
    queue_clips(window, application, tmp_path / "Day1", ["a.mov"])

    mime = urls_for([clip(tmp_path / "New", "d.mov")])
    send_drag_enter(window, window.media_queue_table.viewport(), mime)

    assert window.media_drop_overlay_visible


def test_the_overlay_clears_on_drag_leave(window, tmp_path):
    mime = urls_for([clip(tmp_path / "New", "d.mov")])
    send_drag_enter(window, window.drop_target, mime)
    assert window.media_drop_overlay_visible

    window.eventFilter(window.drop_target, QDragLeaveEvent())

    assert not window.media_drop_overlay_visible


def test_the_overlay_clears_after_a_drop(window, application, tmp_path):
    made = clip(tmp_path / "New", "d.mov")
    mime = urls_for([made])
    send_drag_enter(window, window.drop_target, mime)
    send_drop(window, window.drop_target, mime)
    drain(window, application)

    assert not window.media_drop_overlay_visible
    assert names(window) == ["d.mov"]


def test_a_drag_with_no_local_files_shows_no_overlay(window):
    mime = QMimeData()
    mime.setText("just some text")

    event = send_drag_enter(window, window.drop_target, mime)

    assert not event.isAccepted()
    assert not window.media_drop_overlay_visible


def test_a_drop_on_the_dedicated_queue_still_queues(
    window, application, tmp_path
):
    made = clip(tmp_path / "New", "d.mov")
    mime = urls_for([made])
    send_drag_enter(window, window.queue_table, mime)
    send_drop(window, window.queue_table, mime)
    drain(window, application)

    assert names(window) == ["d.mov"]


def test_an_unrelated_event_still_passes_through(window):
    assert window.eventFilter(window.drop_target, QEvent(QEvent.Type.Show)) is False


# ------------------------------------------------- reordering never loses a row

"""Regression cover for a dragged row disappearing from the queue.

``QAbstractItemView.startDrag`` finishes with ``if (drag->exec(...) ==
Qt::MoveAction) d->clearOrRemove();``, and ``clearOrRemove`` removes the
selected rows from the model. The drop had already been handled and the queue
already redrawn by then, so a clip dragged to a new position was reordered
correctly and then deleted from the table it came from. It needed a real
mouse, because ``drag->exec`` cannot return ``MoveAction`` without one, which
is why nothing here caught it.

These pin the two halves of the fix: the table never calls up to that branch,
and the reorder itself is one validated permutation of ``items`` that either
happens completely or not at all.
"""


@pytest.fixture
def queued(window, application, tmp_path):
    """A four-file queue, added the way a person adds one."""
    queue_clips(
        window, application, tmp_path / "Day1", ["a.mov", "b.mov", "c.mov", "d.mov"]
    )
    assert names(window) == ["a.mov", "b.mov", "c.mov", "d.mov"]
    return window


def assert_queue_is_intact(window, expected_names: list[str]) -> None:
    """Every item present exactly once, in order, in the list and both views."""
    assert names(window) == expected_names
    # Exactly once: identity, not equality, so two items that merely look
    # alike cannot stand in for each other.
    identities = [id(item) for item in window.items]
    assert len(set(identities)) == len(identities) == len(expected_names)
    assert column(window.media_queue_panel) == expected_names
    assert column(window.queue_panel) == expected_names
    assert window.media_queue_panel.table.rowCount() == len(expected_names)
    assert window.queue_panel.table.rowCount() == len(expected_names)


# ---------------------------------------------------------- every direction


def test_the_first_row_moves_to_last(queued):
    assert queued._move_queue_rows([0], 4) is True

    assert_queue_is_intact(queued, ["b.mov", "c.mov", "d.mov", "a.mov"])


def test_the_last_row_moves_to_first(queued):
    assert queued._move_queue_rows([3], 0) is True

    assert_queue_is_intact(queued, ["d.mov", "a.mov", "b.mov", "c.mov"])


def test_a_middle_row_moves_up(queued):
    assert queued._move_queue_rows([2], 1) is True

    assert_queue_is_intact(queued, ["a.mov", "c.mov", "b.mov", "d.mov"])


def test_a_middle_row_moves_down(queued):
    assert queued._move_queue_rows([1], 3) is True

    assert_queue_is_intact(queued, ["a.mov", "c.mov", "b.mov", "d.mov"])


def test_a_row_moves_to_the_very_top(queued):
    assert queued._move_queue_rows([2], 0) is True

    assert_queue_is_intact(queued, ["c.mov", "a.mov", "b.mov", "d.mov"])


def test_a_row_moves_between_two_others(queued):
    assert queued._move_queue_rows([0], 3) is True

    assert_queue_is_intact(queued, ["b.mov", "c.mov", "a.mov", "d.mov"])


def test_a_row_moves_below_the_final_row(queued):
    """The destination equal to the queue length means past the last row."""
    assert queued._move_queue_rows([1], len(queued.items)) is True

    assert_queue_is_intact(queued, ["a.mov", "c.mov", "d.mov", "b.mov"])


def test_moving_one_row_preserves_every_other_row(queued):
    before = {item.name: item for item in queued.items}

    queued._move_queue_rows([1], 4)

    assert {item.name: item for item in queued.items} == before
    assert_queue_is_intact(queued, ["a.mov", "c.mov", "d.mov", "b.mov"])


def test_every_item_survives_a_run_of_reorders(queued):
    """Whatever order they end in, the same four objects have to be there."""
    original = {id(item) for item in queued.items}

    for rows, destination in (([0], 4), ([3], 0), ([1], 3), ([2], 1), ([0], 2)):
        queued._move_queue_rows(rows, destination)
        assert {id(item) for item in queued.items} == original
        assert len(queued.items) == 4
        assert column(queued.media_queue_panel) == names(queued)
        assert column(queued.queue_panel) == names(queued)


# ------------------------------------------------------------ invalid drops


@pytest.mark.parametrize(
    "rows, destination",
    [
        ([], 0),              # nothing selected
        ([9], 0),             # a row that is not in the queue
        ([-1], 0),            # a row before the first
        ([0], -1),            # a destination before the first
        ([0], 99),            # a destination past the end
        ([0, 9], 1),          # one good row and one that is not
        (["x"], 0),           # not an index at all
        ([0], None),          # no destination at all
    ],
)
def test_an_invalid_drop_leaves_the_queue_unchanged(queued, rows, destination):
    before = list(queued.items)

    assert queued._move_queue_rows(rows, destination) is False

    assert queued.items == before
    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


def test_a_move_onto_its_own_position_changes_nothing(queued):
    before = list(queued.items)

    assert queued._move_queue_rows([1], 1) is False
    assert queued._move_queue_rows([1], 2) is False

    assert queued.items == before
    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


def test_a_reorder_keeps_status_source_and_duration(queued):
    before = {
        item.name: (item.status, item.source, item.duration_label)
        for item in queued.items
    }
    queued._set_item_status(0, QueueStatus.COMPLETED)
    completed = queued.items[0]

    queued._move_queue_rows([0], 4)

    assert completed.status is QueueStatus.COMPLETED
    assert queued.items[-1] is completed
    for item in queued.items:
        status, source, duration = before[item.name]
        assert item.source == source
        assert item.duration_label == duration
        if item is not completed:
            assert item.status is status
    row = queued.items.index(completed)
    assert queued.media_queue_panel.table.item(row, 3).text() == "Completed"
    assert queued.queue_panel.table.item(row, 3).text() == "Completed"


# --------------------------------------------- the table never edits itself


class RecordingDrag:
    """Stands in for QDrag so a drag can be started without a mouse."""

    started: list[tuple] = []

    def __init__(self, parent=None):
        self._mime = None

    def setMimeData(self, data):  # noqa: N802 - Qt naming
        self._mime = data

    def exec(self, *actions):
        RecordingDrag.started.append(actions)
        return Qt.DropAction.CopyAction

    exec_ = exec


def test_the_queue_table_does_not_use_qts_row_removing_drag():
    """The exact branch that deleted the dragged row must be unreachable."""
    from PySide6.QtWidgets import QTableWidget

    assert QueueTable.startDrag is not QTableWidget.startDrag


def test_starting_a_drag_removes_nothing(queued, monkeypatch):
    from PySide6.QtWidgets import QTableWidget

    panel = queued.media_queue_panel
    # Fail here rather than hand a real drag loop to an offscreen test.
    assert type(panel.table).startDrag is not QTableWidget.startDrag

    RecordingDrag.started.clear()
    monkeypatch.setattr(queue_panel_module, "QDrag", RecordingDrag)
    panel.table.selectRow(1)

    panel.table.startDrag(Qt.DropAction.MoveAction)

    assert RecordingDrag.started, "no drag was started"
    # Copy on both counts: a Move result is Qt's cue to delete the source rows.
    for actions in RecordingDrag.started:
        assert Qt.DropAction.MoveAction not in actions
    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


def test_a_drag_that_ends_reconciles_the_views(queued):
    """The moment Qt used to edit rows is the moment the views are checked."""
    panel = queued.media_queue_panel
    # Simulate a stray row removal, which is what the old code suffered.
    panel.table.removeRow(2)
    assert panel.table.rowCount() == 3

    panel.table.drag_finished.emit()

    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


def test_a_dropped_row_does_not_disappear(queued, application):
    """The reported bug, driven through the real drop path."""
    queued.show()
    application.processEvents()
    panel = queued.media_queue_panel
    panel.table.selectRow(0)
    application.processEvents()

    mime = internal_row_mime()
    bottom = panel.table.viewport().height()
    proposed = Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
    send_drag_enter(
        queued, panel.table.viewport(), mime, point=(10, bottom), action=proposed
    )
    send_drop(
        queued, panel.table.viewport(), mime, point=(10, bottom), action=proposed
    )
    # Qt removed the source rows at exactly this point. Now it cannot, and
    # the views are reconciled against the queue instead.
    panel.table.drag_finished.emit()

    assert_queue_is_intact(queued, ["b.mov", "c.mov", "d.mov", "a.mov"])


def test_the_drag_this_table_starts_can_never_report_a_move(
    queued, monkeypatch
):
    """Move is the only result Qt acts on, so it is never offered.

    ``drag->exec`` returns one of the actions it was given. This table offers
    Copy alone, so the ``== Qt::MoveAction`` branch that removed the source
    rows cannot be reached whatever the drop does.

    Asserting on a synthetic ``QDropEvent`` instead would prove nothing:
    ``setDropAction`` is ignored unless the action is among the event's
    possible ones, and whenever Copy is possible Qt already proposes it, so
    accepting the proposal and setting Copy are indistinguishable there. The
    guarantee lives at the drag's source, which is where this looks.
    """
    RecordingDrag.started.clear()
    monkeypatch.setattr(queue_panel_module, "QDrag", RecordingDrag)
    queued.media_queue_panel.table.selectRow(0)

    queued.media_queue_panel.table.startDrag(
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
    )

    assert RecordingDrag.started, "no drag was started"
    for actions in RecordingDrag.started:
        assert Qt.DropAction.MoveAction not in actions
        assert Qt.DropAction.CopyAction in actions


def test_an_internal_drop_is_accepted_and_keeps_the_queue_whole(
    queued, application
):
    """Whatever action Qt proposes, the drop reorders and loses nothing."""
    queued.show()
    application.processEvents()
    panel = queued.media_queue_panel
    panel.table.selectRow(0)

    mime = internal_row_mime()
    proposed = Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
    send_drag_enter(
        queued, panel.table.viewport(), mime, point=(10, 0), action=proposed
    )
    event = send_drop(
        queued, panel.table.viewport(), mime, point=(10, 0), action=proposed
    )

    assert event.isAccepted()
    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


# ------------------------------------------ the rest of the queue's behaviour


def test_a_finder_drop_appends_without_disturbing_the_order(
    queued, application, tmp_path
):
    queued._move_queue_rows([3], 0)
    assert names(queued) == ["d.mov", "a.mov", "b.mov", "c.mov"]

    queue_clips(queued, application, tmp_path / "Day2", ["e.mov"])

    assert_queue_is_intact(
        queued, ["d.mov", "a.mov", "b.mov", "c.mov", "e.mov"]
    )


def test_reordering_stays_disabled_while_processing(queued):
    original = MainWindow.is_transcribing
    try:
        MainWindow.is_transcribing = property(lambda self: True)
        queued._update_actions()

        assert queued._move_queue_rows([0], 4) is False
    finally:
        MainWindow.is_transcribing = original

    assert_queue_is_intact(queued, ["a.mov", "b.mov", "c.mov", "d.mov"])


def test_processing_follows_the_reordered_queue(
    queued, application, monkeypatch
):
    """The batch reads items, so the revised order is the order it runs."""
    queued._move_queue_rows([3], 0)
    queued._move_queue_rows([3], 1)
    assert names(queued) == ["d.mov", "c.mov", "a.mov", "b.mov"]

    RecordingWorker.captured.clear()
    monkeypatch.setattr(main_window_module, "TranscriptionWorker", RecordingWorker)
    monkeypatch.setattr(main_window_module, "model_is_cached", lambda _model: True)
    monkeypatch.setattr(MainWindow, "_show_batch_summary", lambda self, summary: None)

    queued._start_transcription()
    deadline = time.monotonic() + 8.0
    while queued.is_transcribing and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    application.processEvents()

    assert RecordingWorker.captured, "the batch never reached the worker"
    assert [job.source.name for job in RecordingWorker.captured[0]] == [
        "d.mov",
        "c.mov",
        "a.mov",
        "b.mov",
    ]
