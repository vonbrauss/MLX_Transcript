"""Atomic transcript writing and ScriptSync paragraph breaks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transcription.outputs import (
    SCRIPTSYNC_PARAGRAPH_GAP,
    atomic_write_text,
    clean_scriptsync_text,
    write_scriptsync,
    write_timecoded,
)


def temp_leftovers(folder: Path) -> list[Path]:
    """Any temporary file the writer might have left behind."""
    return [path for path in folder.rglob("*") if path.suffix == ".tmp"]


# ----------------------------------------------------------- atomic writing


def test_atomic_write_creates_the_file_and_parent_folders(tmp_path):
    target = tmp_path / "Transcription" / "ScriptSync" / "Day 01" / "clip.txt"
    atomic_write_text(target, "hello", encoding="ascii")
    assert target.read_text(encoding="ascii") == "hello"


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "clip.txt"
    atomic_write_text(target, "hello")
    assert temp_leftovers(tmp_path) == []
    assert [path.name for path in tmp_path.iterdir()] == ["clip.txt"]


def test_atomic_write_replaces_an_existing_file(tmp_path):
    target = tmp_path / "clip.txt"
    target.write_text("OLD", encoding="utf-8")
    atomic_write_text(target, "NEW")
    assert target.read_text(encoding="utf-8") == "NEW"


def test_a_failed_write_keeps_the_previous_file_and_cleans_up(tmp_path):
    target = tmp_path / "clip.txt"
    target.write_text("OLD", encoding="ascii")

    # An accented character cannot be encoded as ASCII, so the write fails
    # after the temporary file has already been created.
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, "café", encoding="ascii")

    assert target.read_text(encoding="ascii") == "OLD"
    assert temp_leftovers(tmp_path) == []


def test_temporary_file_is_written_beside_the_target(tmp_path, monkeypatch):
    """The temp file must share the destination folder so os.replace is atomic."""
    import os as real_os

    from transcription import outputs

    seen: list[tuple[Path, Path]] = []
    original_replace = real_os.replace

    def spy(source, destination):
        seen.append((Path(source), Path(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(outputs.os, "replace", spy)

    target = tmp_path / "nested" / "clip.txt"
    atomic_write_text(target, "content")

    temporary, destination = seen[0]
    assert temporary.parent == target.parent
    assert destination == target
    assert not temporary.exists()
    assert target.read_text(encoding="utf-8") == "content"


def test_write_scriptsync_is_atomic(tmp_path):
    target = tmp_path / "Day 01" / "clip.txt"
    write_scriptsync(target, [{"start": 0.0, "text": "hello there"}])
    assert target.read_text(encoding="ascii").strip() == "hello there"
    assert temp_leftovers(tmp_path) == []


def test_write_timecoded_is_atomic(tmp_path):
    target = tmp_path / "Day 01" / "clip.txt"
    write_timecoded(
        target,
        Path("/Source/Day 01/A001.mov"),
        [{"start": 0.0, "text": "Take one."}],
        Fraction(25, 1),
        "10:00:00:00",
    )
    assert "10:00:00:00  Take one." in target.read_text(encoding="utf-8")
    assert temp_leftovers(tmp_path) == []


def test_crlf_survives_the_atomic_write(tmp_path):
    target = tmp_path / "clip.txt"
    write_scriptsync(target, [{"start": 0.0, "text": "word " * 40}])
    raw = target.read_bytes()
    assert b"\r\n" in raw
    assert b"\r\r\n" not in raw


# -------------------------------------------------------- paragraph breaks


def test_a_long_silence_starts_a_new_paragraph():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "First thought."},
        {"start": 6.0, "end": 8.0, "text": "Second thought."},
    ]
    text = clean_scriptsync_text(segments)
    assert text == "First thought.\r\n\r\nSecond thought.\r\n"


def test_a_short_gap_keeps_one_paragraph():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "First thought."},
        {"start": 2.4, "end": 4.0, "text": "Second thought."},
    ]
    text = clean_scriptsync_text(segments)
    assert text == "First thought. Second thought.\r\n"


def test_the_paragraph_gap_is_tunable():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "First thought."},
        {"start": 3.0, "end": 4.0, "text": "Second thought."},
    ]
    assert "\r\n\r\n" not in clean_scriptsync_text(segments)
    assert "\r\n\r\n" in clean_scriptsync_text(segments, paragraph_gap=0.5)


def test_segments_without_end_times_stay_in_one_paragraph():
    segments = [
        {"start": 0.0, "text": "First thought."},
        {"start": 90.0, "text": "Second thought."},
    ]
    assert "\r\n\r\n" not in clean_scriptsync_text(segments)


def test_the_default_paragraph_gap_is_two_seconds():
    assert SCRIPTSYNC_PARAGRAPH_GAP == 2.0


def test_empty_segments_produce_an_empty_file(tmp_path):
    target = tmp_path / "clip.txt"
    write_scriptsync(target, [])
    assert target.read_bytes() == b"\r\n"


def test_paragraphs_are_wrapped_independently():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "word " * 30},
        {"start": 10.0, "end": 11.0, "text": "other " * 30},
    ]
    text = clean_scriptsync_text(segments)
    assert "\r\n\r\n" in text
    for line in text.split("\r\n"):
        assert len(line) <= 64


def test_wording_is_preserved_across_paragraphs():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "We ran it twice."},
        {"start": 10.0, "end": 11.0, "text": "Then we reset."},
    ]
    text = clean_scriptsync_text(segments)
    assert "We ran it twice." in text
    assert "Then we reset." in text
