"""Queue bookkeeping: duplicates, missing sources, labels, and scale.

The release readiness review found that adding a file rebuilt and re-resolved
the whole queue, so a folder of dailies froze the window for over a minute.
The fix is an identity set the window maintains as items arrive, and the
scaling test below is what stops that regressing.
"""

from __future__ import annotations

import os
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.models import QueueItem, QueueStatus, source_identity  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from transcription.media_probe import MediaInfo  # noqa: E402


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture(autouse=True)
def stub_probe(monkeypatch):
    """Keep ffprobe out of it; this module is about queue bookkeeping."""
    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(
            path=path, duration_seconds=2.0, frame_rate=Fraction(24, 1)
        ),
    )


@pytest.fixture(autouse=True)
def silence_dialogs(monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.main_window.QMessageBox.warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: seen.append(
            (title, text)
        )),
    )
    return seen


@pytest.fixture
def window(application):
    built = MainWindow(AppSettings())
    yield built
    built.close()


def drain(window: MainWindow, application: QApplication, seconds: float = 8.0) -> None:
    """Run the scan thread to completion."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        thread = window._scan_thread
        if thread is None or not thread.isRunning():
            application.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError("the scan did not finish")


def make_clips(folder: Path, names: list[str]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names:
        clip = folder / name
        clip.write_bytes(b"0")
        made.append(clip)
    return made


def queued_names(window: MainWindow) -> list[str]:
    return [item.name for item in window.items]


def folder_column(window: MainWindow) -> list[str]:
    return [
        window.queue_table.item(row, 1).text()
        for row in range(window.queue_table.rowCount())
    ]


# ------------------------------------------------------------ S3: missing sources


def test_a_missing_source_is_not_reported_as_a_duplicate(
    window, application, tmp_path, silence_dialogs
):
    window._queue_sources([tmp_path / "gone.mov"])
    drain(window, application)

    assert window.items == []
    assert "could not be found" in window.statusBar().currentMessage()
    assert "already in the queue" not in window.statusBar().currentMessage()
    assert silence_dialogs and "not available" in silence_dialogs[0][0].lower()


def test_several_missing_sources_are_counted(
    window, application, tmp_path, silence_dialogs
):
    window._queue_sources([tmp_path / "a.mov", tmp_path / "b.mov"])
    drain(window, application)

    assert "2 item(s) could not be found" in window.statusBar().currentMessage()


def test_a_real_duplicate_still_says_duplicate(window, application, tmp_path):
    (clip,) = make_clips(tmp_path / "media", ["clip.mov"])
    window._queue_sources([clip])
    drain(window, application)

    window._queue_sources([clip])
    drain(window, application)

    assert queued_names(window) == ["clip.mov"]
    assert "already in the queue" in window.statusBar().currentMessage()


def test_a_missing_source_alongside_a_good_one_still_queues_the_good_one(
    window, application, tmp_path, silence_dialogs
):
    (clip,) = make_clips(tmp_path / "media", ["clip.mov"])
    window._queue_sources([clip, tmp_path / "gone.mov"])
    drain(window, application)

    assert queued_names(window) == ["clip.mov"]
    assert silence_dialogs


# --------------------------------------------------- S4: labels after the scan


def test_mixed_roots_are_labelled_as_soon_as_the_scan_finishes(
    window, application, tmp_path
):
    make_clips(tmp_path / "ProjectA" / "Dailies", ["clip.mov"])
    make_clips(tmp_path / "ProjectB" / "Dailies", ["clip.mov"])

    window._queue_sources(
        [tmp_path / "ProjectA" / "Dailies", tmp_path / "ProjectB" / "Dailies"]
    )
    drain(window, application)

    assert folder_column(window) == ["Dailies", "Dailies (2)"]


def test_one_root_needs_no_label(window, application, tmp_path):
    make_clips(tmp_path / "Dailies", ["clip.mov"])

    window._queue_sources([tmp_path / "Dailies"])
    drain(window, application)

    assert folder_column(window) == ["/"]


def test_a_second_root_relabels_the_rows_already_queued(
    window, application, tmp_path
):
    make_clips(tmp_path / "ProjectA" / "Dailies", ["a.mov"])
    make_clips(tmp_path / "ProjectB" / "Dailies", ["b.mov"])

    window._queue_sources([tmp_path / "ProjectA" / "Dailies"])
    drain(window, application)
    assert folder_column(window) == ["/"]

    window._queue_sources([tmp_path / "ProjectB" / "Dailies"])
    drain(window, application)
    assert folder_column(window) == ["Dailies", "Dailies (2)"]


# ------------------------------------------------- S5: already-covered sources


def test_re_dropping_a_queued_folder_does_not_rescan(window, application, tmp_path):
    folder = tmp_path / "Dailies"
    make_clips(folder, ["a.mov", "b.mov"])
    window._queue_sources([folder])
    drain(window, application)

    window._queue_sources([folder])

    assert window._scan_thread is None or not window._scan_thread.isRunning()
    assert len(window.items) == 2


def test_a_subfolder_of_a_queued_folder_does_not_rescan(
    window, application, tmp_path
):
    folder = tmp_path / "Dailies"
    make_clips(folder / "Day1", ["a.mov"])
    window._queue_sources([folder])
    drain(window, application)

    window._queue_sources([folder / "Day1"])

    assert window._scan_thread is None or not window._scan_thread.isRunning()
    assert len(window.items) == 1


def test_a_file_inside_a_queued_folder_does_not_rescan(
    window, application, tmp_path
):
    folder = tmp_path / "Dailies"
    (clip,) = make_clips(folder, ["a.mov"])
    window._queue_sources([folder])
    drain(window, application)

    window._queue_sources([clip])

    assert window._scan_thread is None or not window._scan_thread.isRunning()
    assert len(window.items) == 1


def test_a_parent_of_a_queued_folder_is_still_scanned(window, application, tmp_path):
    parent = tmp_path / "Media"
    make_clips(parent / "Dailies", ["a.mov"])
    make_clips(parent / "Pickups", ["b.mov"])

    window._queue_sources([parent / "Dailies"])
    drain(window, application)
    window._queue_sources([parent])
    drain(window, application)

    assert sorted(queued_names(window)) == ["a.mov", "b.mov"]


def test_removing_items_gives_their_sources_back(window, application, tmp_path):
    folder = tmp_path / "Dailies"
    make_clips(folder, ["a.mov"])
    window._queue_sources([folder])
    drain(window, application)

    window.queue_table.selectRow(0)
    window._remove_selected_items()
    assert window.items == []

    window._queue_sources([folder])
    drain(window, application)

    assert queued_names(window) == ["a.mov"]


def test_clearing_the_queue_gives_every_source_back(window, application, tmp_path):
    folder = tmp_path / "Dailies"
    make_clips(folder, ["a.mov", "b.mov"])
    window._queue_sources([folder])
    drain(window, application)

    window._clear_queue()
    assert window._queued_identities == set()
    assert window._queued_roots == []

    window._queue_sources([folder])
    drain(window, application)

    assert len(window.items) == 2


def test_duplicate_files_from_overlapping_drops_are_still_ignored(
    window, application, tmp_path
):
    folder = tmp_path / "Dailies"
    make_clips(folder / "Day1", ["a.mov"])
    make_clips(folder / "Day2", ["a.mov"])

    # Both roots in one drop, the subfolder covered by the parent.
    window._queue_sources([folder, folder / "Day1"])
    drain(window, application)

    assert len(window.items) == 2
    assert sorted(str(item.relative_folder) for item in window.items) == [
        "Day1",
        "Day2",
    ]


# ------------------------------------------------------ B4: identity handling


def test_an_item_resolves_its_source_only_once(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"0")
    item = QueueItem(source=clip, source_root=tmp_path)

    calls = {"count": 0}
    original = Path.resolve

    def counting(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting)

    first = item.resolved_identity
    second = item.resolved_identity

    assert first == second
    assert calls["count"] == 1


def test_an_unresolvable_source_still_has_a_stable_identity(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    monkeypatch.setattr(
        Path, "resolve", lambda self, *a, **k: (_ for _ in ()).throw(OSError("gone"))
    )

    assert source_identity(clip) == clip.absolute()


def test_queue_bookkeeping_does_not_resolve_the_whole_queue_per_item(
    window, application, tmp_path, monkeypatch
):
    """The heart of B4: adding item N must not touch the N-1 already queued."""
    folder = tmp_path / "Dailies"
    make_clips(folder, [f"c{index:03d}.mov" for index in range(40)])
    window._queue_sources([folder])
    drain(window, application)
    assert len(window.items) == 40

    calls = {"count": 0}
    original = Path.resolve

    def counting(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting)

    extra = tmp_path / "Extra"
    (clip,) = make_clips(extra, ["late.mov"])
    window._add_item(
        QueueItem(source=clip, source_root=extra, status=QueueStatus.READY)
    )

    assert len(window.items) == 41
    # One resolve for the new item, and nothing at all for the existing forty.
    assert calls["count"] <= 2


def test_queue_bookkeeping_scales_about_linearly(window, application, tmp_path):
    """A quadratic queue took 27 seconds for 1600 files. Linear takes under one."""
    timings: dict[int, float] = {}
    for count in (400, 1600):
        folder = tmp_path / f"batch{count}"
        make_clips(folder, [f"c{index:05d}.mov" for index in range(count)])
        fresh = MainWindow(AppSettings())
        try:
            started = time.monotonic()
            fresh._queue_sources([folder])
            drain(fresh, application, seconds=120)
            timings[count] = time.monotonic() - started
            assert len(fresh.items) == count
        finally:
            fresh.close()

    # Four times the files must not cost anything like sixteen times the work.
    # The quadratic version measured a ratio near 16; linear sits near 4.
    ratio = timings[1600] / max(timings[400], 1e-3)
    assert ratio < 8, f"queue bookkeeping looks superlinear: {timings}"
    # An absolute ceiling as well, far above the measured time but far below
    # the 27 seconds the quadratic version needed.
    assert timings[1600] < 10, f"queueing 1600 files took {timings[1600]:.1f}s"
