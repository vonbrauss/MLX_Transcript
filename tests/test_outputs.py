"""Output path construction, ScriptSync cleanup, and timecoded formatting."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from transcription.outputs import (
    SCRIPTSYNC_WRAP_WIDTH,
    build_output_paths,
    clean_scriptsync_text,
    existing_outputs,
    make_timecoded_text,
    output_roots,
    transcript_prefix,
    write_scriptsync,
    write_timecoded,
)

NTSC_30 = Fraction(30000, 1001)


# ------------------------------------------------------------------- layout


def test_output_roots_sit_below_the_chosen_parent():
    roots = output_roots(Path("/Volumes/Work/Output"))
    assert roots.transcription == Path("/Volumes/Work/Output/Transcription")
    assert roots.scriptsync == Path("/Volumes/Work/Output/Transcription/ScriptSync")
    assert roots.timecoded == Path("/Volumes/Work/Output/Transcription/Timecoded")


def test_transcript_prefix_uses_the_enclosing_folder():
    source = Path("/Source/Media day sound/Atlanta Dream.mov")
    assert transcript_prefix(source) == "Media day sound — Atlanta Dream"


def test_build_output_paths_mirrors_the_source_hierarchy():
    source = Path("/Source/Media day sound/Atlanta Dream.mov")
    paths = build_output_paths(source, Path("/Source"), Path("/Output"))
    assert paths.scriptsync == Path(
        "/Output/Transcription/ScriptSync/Media day sound/"
        "Media day sound — Atlanta Dream ScriptSync.txt"
    )
    assert paths.timecoded == Path(
        "/Output/Transcription/Timecoded/Media day sound/"
        "Media day sound — Atlanta Dream timecoded.txt"
    )


def test_build_output_paths_handles_deeply_nested_media():
    source = Path("/Source/Day 01/Cam A/Card 02/A001_C003.mov")
    paths = build_output_paths(source, Path("/Source"), Path("/Output"))
    assert paths.scriptsync.parent == Path(
        "/Output/Transcription/ScriptSync/Day 01/Cam A/Card 02"
    )
    assert paths.timecoded.name == "Card 02 — A001_C003 timecoded.txt"


def test_build_output_paths_handles_media_at_the_source_root():
    source = Path("/Source/Interview.wav")
    paths = build_output_paths(source, Path("/Source"), Path("/Output"))
    assert paths.scriptsync == Path(
        "/Output/Transcription/ScriptSync/Source — Interview ScriptSync.txt"
    )


def test_source_and_output_can_be_the_same_folder():
    source = Path("/Source/Day 01/A001.mov")
    paths = build_output_paths(source, Path("/Source"), Path("/Source"))
    assert paths.timecoded.is_relative_to(Path("/Source/Transcription"))


def test_existing_outputs_reports_only_what_is_on_disk(tmp_path):
    source_root = tmp_path / "Source" / "Day 01"
    source_root.mkdir(parents=True)
    source = source_root / "A001.mov"
    source.touch()

    paths = build_output_paths(source, tmp_path / "Source", tmp_path / "Output")
    assert existing_outputs(paths) == []

    paths.scriptsync.parent.mkdir(parents=True, exist_ok=True)
    paths.scriptsync.write_text("existing", encoding="ascii")
    assert existing_outputs(paths) == [paths.scriptsync]


# --------------------------------------------------------------- scriptsync


def test_scriptsync_joins_segments_and_strips_timestamps():
    segments = [
        {"start": 0.0, "text": " First line. "},
        {"start": 2.0, "text": "Second line."},
    ]
    text = clean_scriptsync_text(segments)
    assert "First line. Second line." in text.replace("\r\n", " ")
    assert "00:00" not in text


def test_scriptsync_removes_hyphens_and_dashes():
    segments = [{"text": "state of the art — well known – co-operate"}]
    text = clean_scriptsync_text(segments)
    assert "-" not in text
    assert "—" not in text
    assert "–" not in text
    assert "co operate" in text.replace("\r\n", " ")


def test_scriptsync_output_is_pure_ascii():
    segments = [{"text": "Café naïve “quoted” … resumé"}]
    text = clean_scriptsync_text(segments)
    text.encode("ascii")  # raises if any character survived as non-ASCII
    assert "Cafe" in text
    assert "naive" in text


def test_scriptsync_wraps_at_about_sixty_four_characters():
    segments = [{"text": "word " * 120}]
    text = clean_scriptsync_text(segments)
    lines = text.split("\r\n")
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= SCRIPTSYNC_WRAP_WIDTH


def test_scriptsync_uses_crlf_endings_and_a_trailing_newline():
    segments = [{"text": "word " * 40}]
    text = clean_scriptsync_text(segments)
    assert text.endswith("\r\n")
    assert "\n" not in text.replace("\r\n", "")


def test_scriptsync_collapses_runs_of_whitespace():
    segments = [{"text": "one   two\n\nthree\tfour"}]
    text = clean_scriptsync_text(segments)
    assert text.strip() == "one two three four"


def test_write_scriptsync_creates_parent_folders(tmp_path):
    target = tmp_path / "Transcription" / "ScriptSync" / "Day 01" / "clip.txt"
    write_scriptsync(target, [{"text": "hello there"}])
    assert target.read_text(encoding="ascii").strip() == "hello there"


# ---------------------------------------------------------------- timecoded


def test_timecoded_header_reports_the_source_timecode_and_rate():
    text = make_timecoded_text(
        Path("/Source/Day 01/A001.mov"),
        [],
        NTSC_30,
        "01:00:00;00",
    )
    assert "FILE: A001.mov" in text
    assert "START TIMECODE: 01:00:00;00" in text
    assert "FRAME RATE: 30000/1001" in text


def test_timecoded_notes_a_missing_source_timecode():
    text = make_timecoded_text(
        Path("/Source/Interview.wav"),
        [],
        Fraction(24, 1),
        "00:00:00:00",
        missing_timecode=True,
    )
    assert "NOTE: No embedded timecode found" in text


def test_timecoded_lines_carry_source_timecode_stamps():
    segments = [
        {"start": 0.0, "text": "Rolling."},
        {"start": 5.0, "text": "And  action."},
    ]
    text = make_timecoded_text(
        Path("/Source/A001.mov"), segments, Fraction(24, 1), "01:00:00:00"
    )
    lines = text.splitlines()
    assert lines[-2] == "01:00:00:00  Rolling."
    assert lines[-1] == "01:00:05:00  And action."


def test_timecoded_includes_speaker_labels_when_present():
    segments = [{"start": 0.0, "text": "Hello.", "speaker": "Speaker 1"}]
    text = make_timecoded_text(
        Path("/Source/A001.mov"), segments, Fraction(24, 1), "00:00:00:00"
    )
    assert text.splitlines()[-1] == "00:00:00:00  Speaker 1: Hello."


def test_write_timecoded_creates_parent_folders(tmp_path):
    target = tmp_path / "Transcription" / "Timecoded" / "Day 01" / "clip.txt"
    write_timecoded(
        target,
        Path("/Source/Day 01/A001.mov"),
        [{"start": 0.0, "text": "Take one."}],
        Fraction(25, 1),
        "10:00:00:00",
    )
    assert "10:00:00:00  Take one." in target.read_text(encoding="utf-8")
