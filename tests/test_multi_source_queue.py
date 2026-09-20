"""Mixed folders and files are scanned into one de-duplicated queue."""

from __future__ import annotations

from fractions import Fraction

from app.workers import ScanWorker
from transcription.media_probe import MediaInfo


def test_scan_worker_merges_multiple_roots_without_duplicate_files(tmp_path, monkeypatch):
    first_root = tmp_path / "Interviews"
    second_root = tmp_path / "Media Day"
    first = first_root / "Day 1" / "A.mov"
    second = second_root / "Chicago" / "B.wav"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(
            path=path, duration_seconds=1.0, frame_rate=Fraction(24)
        ),
    )
    worker = ScanWorker([first_root, second_root, first])
    finished: list[list] = []
    worker.finished.connect(finished.append)

    worker.run()

    assert len(finished) == 1
    items = finished[0]
    assert [item.source for item in items] == [first, second]
    assert items[0].source_root == first_root
    assert items[1].source_root == second_root
