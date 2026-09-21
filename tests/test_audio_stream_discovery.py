"""A file is media because it holds audio, not because of its name.

MLX Transcript accepts any local media file FFmpeg can read that contains an
audio stream, whatever the filename. The extension list survives only as a
fast path that keeps obvious non-media out of the probe queue, and it must
never be the thing that turns a real clip away.

ffprobe is stubbed here so the decision logic is tested without FFmpeg. The
stub reads a marker from the file, which lets one folder hold a clip with
audio, a video-only clip, a corrupt file and a document at the same time.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from app.workers import ScanWorker  # noqa: E402
from transcription.discovery import (  # noqa: E402
    MEDIA_EXTENSIONS,
    NON_MEDIA_EXTENSIONS,
    discover_media,
    is_probe_candidate,
    is_supported_media,
)
from transcription.media_probe import MediaInfo, MediaProbeError  # noqa: E402

AUDIO = b"AUDIO"
VIDEO_ONLY = b"VIDEOONLY"
CORRUPT = b"CORRUPT"


def write(folder: Path, name: str, marker: bytes = AUDIO) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(marker + b"\x00" * 16)
    return path


def fake_probe(path: Path) -> MediaInfo:
    """Stand in for ffprobe, deciding from a marker in the file."""
    data = Path(path).read_bytes()
    if data.startswith(AUDIO):
        return MediaInfo(
            path=Path(path),
            duration_seconds=5.0,
            frame_rate=Fraction(24, 1),
            stream_count=2,
            has_audio=True,
            audio_codec="aac",
        )
    if data.startswith(VIDEO_ONLY):
        return MediaInfo(
            path=Path(path),
            duration_seconds=5.0,
            frame_rate=Fraction(24, 1),
            stream_count=1,
            has_audio=False,
        )
    raise MediaProbeError(
        f"{Path(path).name}: Invalid data found when processing input"
    )


@pytest.fixture(autouse=True)
def stub_probe(monkeypatch):
    monkeypatch.setattr("app.workers.probe_media", fake_probe)


def scan(*roots: Path) -> tuple[list, list[tuple[Path, str]]]:
    """Run one scan synchronously and return its items and skipped list."""
    worker = ScanWorker(list(roots))
    finished: list[list] = []
    skipped: list[list] = []
    worker.finished.connect(finished.append)
    worker.skipped_files.connect(skipped.append)
    worker.run()
    assert len(finished) == 1
    assert len(skipped) == 1
    return finished[0], [(Path(p), r) for p, r in skipped[0]]


def names(items) -> list[str]:
    return sorted(item.name for item in items)


def reason_for(skipped: list[tuple[Path, str]], name: str) -> str:
    for path, reason in skipped:
        if path.name == name:
            return reason
    raise AssertionError(f"{name} was not in the skipped list: {skipped}")


# ------------------------------------------------- accepted because of audio


def test_an_unfamiliar_extension_with_audio_is_queued(tmp_path):
    write(tmp_path, "interview.zoomrec")

    items, skipped = scan(tmp_path)

    assert names(items) == ["interview.zoomrec"]
    assert skipped == []


def test_a_file_with_no_extension_at_all_is_queued(tmp_path):
    write(tmp_path, "A001C003_240101")

    items, skipped = scan(tmp_path)

    assert names(items) == ["A001C003_240101"]
    assert skipped == []


@pytest.mark.parametrize(
    "name",
    ["take.r3d", "take.braw", "take.dv", "take.wma9", "take.MOV.bak", "take.0001"],
)
def test_uncommon_extensions_are_queued_when_they_hold_audio(tmp_path, name):
    write(tmp_path, name)

    items, skipped = scan(tmp_path)

    assert names(items) == [name]
    assert not is_supported_media(Path(name)) or name.endswith(".mov")


def test_a_known_extension_still_works(tmp_path):
    write(tmp_path, "clip.mov")

    items, _skipped = scan(tmp_path)

    assert names(items) == ["clip.mov"]


def test_the_probed_audio_codec_reaches_the_queue_item(tmp_path):
    write(tmp_path, "clip.strange")

    items, _skipped = scan(tmp_path)

    assert items[0].media is not None
    assert items[0].media.has_audio is True
    assert items[0].media.audio_codec == "aac"


# ------------------------------------------------- skipped, with a reason


def test_a_video_with_no_audio_is_skipped_and_explained(tmp_path):
    write(tmp_path, "plate.mov", VIDEO_ONLY)

    items, skipped = scan(tmp_path)

    assert items == []
    assert reason_for(skipped, "plate.mov") == "no audio stream"


def test_a_video_only_file_with_an_odd_extension_is_also_skipped(tmp_path):
    write(tmp_path, "plate.imgseq", VIDEO_ONLY)

    items, skipped = scan(tmp_path)

    assert items == []
    assert reason_for(skipped, "plate.imgseq") == "no audio stream"


def test_a_pdf_is_skipped_without_probing(tmp_path):
    write(tmp_path, "call-sheet.pdf", b"%PDF-1.7")

    items, skipped = scan(tmp_path)

    assert items == []
    assert ".pdf is not a media file" == reason_for(skipped, "call-sheet.pdf")


def test_a_text_file_is_skipped_without_probing(tmp_path):
    write(tmp_path, "notes.txt", b"just notes")

    items, skipped = scan(tmp_path)

    assert items == []
    assert ".txt is not a media file" == reason_for(skipped, "notes.txt")


def test_a_corrupt_file_with_an_odd_extension_is_skipped_with_the_reason(tmp_path):
    write(tmp_path, "broken.capture", CORRUPT)

    items, skipped = scan(tmp_path)

    assert items == []
    reason = reason_for(skipped, "broken.capture")
    assert reason.startswith("could not be read")
    assert "Invalid data" in reason


def test_a_corrupt_file_that_looks_like_media_stays_visible_as_failed(tmp_path):
    """A .mov the user meant to queue must not vanish into a summary."""
    write(tmp_path, "broken.mov", CORRUPT)

    items, skipped = scan(tmp_path)

    assert names(items) == ["broken.mov"]
    assert items[0].status.value == "failed"
    assert "Invalid data" in items[0].message
    assert skipped == []


def test_a_probe_reason_is_kept_short(tmp_path):
    long = "x" * 400
    path = write(tmp_path, "broken.capture", CORRUPT)

    worker = ScanWorker([tmp_path])
    reason = worker._probe_failure_reason(MediaProbeError(f"{path.name}: {long}"))

    assert len(reason) < 160
    # Truncated inside the parenthesis, so the reason still reads as one clause.
    assert "…)" in reason
    assert reason.startswith("could not be read (")


# --------------------------------------------------------- exclusions stay


def test_hidden_files_and_appledouble_sidecars_are_never_reported(tmp_path):
    write(tmp_path, "clip.mov")
    write(tmp_path, "._clip.mov")
    write(tmp_path, ".hidden.wav")
    write(tmp_path, ".DS_Store", b"\x00")

    items, skipped = scan(tmp_path)

    assert names(items) == ["clip.mov"]
    # Silence is right for sidecars: naming every one would bury the rest.
    assert skipped == []


def test_a_system_file_is_skipped_silently(tmp_path):
    write(tmp_path, "clip.mov")
    write(tmp_path, "Thumbs.db", b"\x00")

    items, skipped = scan(tmp_path)

    assert names(items) == ["clip.mov"]
    assert [path.name for path, _ in skipped] == []


def test_directories_are_never_queued(tmp_path):
    (tmp_path / "Dailies.mov").mkdir()
    write(tmp_path, "clip.mov")

    items, _skipped = scan(tmp_path)

    assert names(items) == ["clip.mov"]


def test_the_output_tree_is_still_excluded(tmp_path):
    write(tmp_path, "clip.mov")
    write(tmp_path / "Transcription" / "ScriptSync", "clip.txt", b"a transcript")

    worker = ScanWorker([tmp_path], output_parent=tmp_path)
    finished: list[list] = []
    worker.finished.connect(finished.append)
    worker.run()

    assert names(finished[0]) == ["clip.mov"]


# ------------------------------------------------------------ a mixed folder


def test_a_mixed_folder_queues_the_audio_and_accounts_for_the_rest(tmp_path):
    folder = tmp_path / "Shoot Day 1"
    write(folder, "A001.mov")
    write(folder, "A002.zoomrec")
    write(folder, "A003_no_extension")
    write(folder / "Audio", "boom.wav")
    write(folder, "plate.mov", VIDEO_ONLY)
    write(folder, "broken.capture", CORRUPT)
    write(folder, "call-sheet.pdf", b"%PDF")
    write(folder, "budget.xlsx", b"PK")
    write(folder, "titles.srt", b"1\n")
    write(folder, ".DS_Store", b"\x00")
    write(folder, "._A001.mov")

    items, skipped = scan(folder)

    assert names(items) == [
        "A001.mov",
        "A002.zoomrec",
        "A003_no_extension",
        "boom.wav",
    ]
    assert sorted(path.name for path, _ in skipped) == [
        "broken.capture",
        "budget.xlsx",
        "call-sheet.pdf",
        "plate.mov",
        "titles.srt",
    ]
    assert reason_for(skipped, "plate.mov") == "no audio stream"
    assert reason_for(skipped, "call-sheet.pdf") == ".pdf is not a media file"


def test_progress_counts_every_candidate_including_the_skipped(tmp_path):
    write(tmp_path, "good.mov")
    write(tmp_path, "plate.mov", VIDEO_ONLY)
    write(tmp_path, "broken.capture", CORRUPT)

    worker = ScanWorker([tmp_path])
    seen: list[tuple[int, int]] = []
    worker.progress.connect(lambda done, total: seen.append((done, total)))
    worker.run()

    assert seen == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------- the cheap filter is a fast path


def test_every_known_media_extension_is_a_candidate():
    for suffix in MEDIA_EXTENSIONS:
        assert is_probe_candidate(Path(f"clip{suffix}"))


def test_the_two_lists_never_overlap():
    assert not (MEDIA_EXTENSIONS & NON_MEDIA_EXTENSIONS)


@pytest.mark.parametrize(
    "name",
    ["clip", "clip.", "clip.weird", "clip.MXF", "clip.a", "A001C003.R3D"],
)
def test_anything_not_definitively_non_media_is_probed(name):
    assert is_probe_candidate(Path(name))


@pytest.mark.parametrize(
    "name",
    ["a.pdf", "a.PDF", "a.txt", "a.jpg", "a.srt", "a.zip", "a.edl", "a.xml"],
)
def test_definitely_non_media_is_not_probed(name):
    assert not is_probe_candidate(Path(name))


def test_discovery_collects_rejections_for_the_caller(tmp_path):
    write(tmp_path, "clip.mov")
    write(tmp_path, "notes.pdf", b"%PDF")
    rejected: list[tuple[Path, str]] = []

    found = discover_media(tmp_path, rejected=rejected)

    assert [item.path.name for item in found] == ["clip.mov"]
    assert [(path.name, reason) for path, reason in rejected] == [
        ("notes.pdf", ".pdf is not a media file")
    ]


def test_discovery_without_a_collector_still_works(tmp_path):
    write(tmp_path, "clip.mov")
    write(tmp_path, "notes.pdf", b"%PDF")

    found = discover_media(tmp_path)

    assert [item.path.name for item in found] == ["clip.mov"]


# ------------------------------------------- an unreadable stream count reading


def test_a_file_ffprobe_says_nothing_about_is_trusted_by_extension(
    tmp_path, monkeypatch
):
    """stream_count of zero means "no information", not "no audio"."""
    monkeypatch.setattr(
        "app.workers.probe_media",
        lambda path: MediaInfo(path=Path(path), duration_seconds=1.0),
    )
    write(tmp_path, "clip.mov")
    write(tmp_path, "mystery.unknown")

    items, skipped = scan(tmp_path)

    # The .mov gets the benefit of the doubt; the unknown one does not.
    assert names(items) == ["clip.mov"]
    assert reason_for(skipped, "mystery.unknown") == "no audio stream"
