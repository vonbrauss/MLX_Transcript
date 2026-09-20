"""User-selectable output names, layouts, and subtitle formats."""

from fractions import Fraction
from pathlib import Path

from transcription.engine import TranscriptionEngine
from transcription.media_probe import MediaInfo
from transcription.outputs import (
    FolderLayout,
    NameStyle,
    OutputFormat,
    OutputOptions,
    build_output_paths,
    make_srt_text,
    make_vtt_text,
)
from transcription.pipeline import BatchJob, BatchProcessor


SEGMENTS = [
    {"start": 1.25, "end": 3.5, "text": " Hello there. ", "speaker": "Coach"},
]


def test_default_names_use_only_the_original_media_stem():
    paths = build_output_paths(
        Path("/Source/Interviews/Player A.mov"),
        Path("/Source"),
        Path("/Output"),
        OutputOptions(),
    )
    assert paths.scriptsync == Path(
        "/Output/Transcription/ScriptSync/Interviews/Player A.txt"
    )
    assert paths.timecoded == Path(
        "/Output/Transcription/Timecoded/Interviews/Player A.txt"
    )


def test_type_suffix_and_all_formats():
    options = OutputOptions(
        name_style=NameStyle.TYPE_SUFFIX,
        formats=tuple(OutputFormat),
    )
    paths = build_output_paths(Path("/Source/A.mov"), Path("/Source"), Path("/Out"), options)
    assert paths.scriptsync.name == "A — ScriptSync.txt"
    assert paths.timecoded.name == "A — Timecoded.txt"
    assert paths.srt.name == "A — Subtitles.srt"
    assert paths.vtt.name == "A — WebVTT.vtt"
    assert len(paths.as_tuple()) == 4


def test_custom_suffix_is_sanitized_and_flat_layout_discards_source_tree():
    options = OutputOptions(
        name_style=NameStyle.CUSTOM_SUFFIX,
        custom_suffix="_review/copy",
        folder_layout=FolderLayout.FLAT,
        formats=(OutputFormat.SRT,),
    )
    paths = build_output_paths(
        Path("/Source/Day 1/A.mov"), Path("/Source"), Path("/Out"), options
    )
    assert paths.as_tuple() == (
        Path("/Out/Transcription/Subtitles/A_review_copy.srt"),
    )


def test_mixed_source_roots_gain_separate_output_folders():
    paths = build_output_paths(
        Path("/Volumes/Drive A/Day 1/A.mov"),
        Path("/Volumes/Drive A"),
        Path("/Output"),
        OutputOptions(),
        root_label="Drive A",
    )
    assert paths.scriptsync == Path(
        "/Output/Transcription/ScriptSync/Drive A/Day 1/A.txt"
    )


def test_srt_and_webvtt_use_segment_timing_and_speaker_names():
    assert "00:00:01,250 --> 00:00:03,500\nCoach: Hello there." in make_srt_text(SEGMENTS)
    assert make_vtt_text(SEGMENTS).startswith("WEBVTT\n\n")
    assert "00:00:01.250 --> 00:00:03.500\nCoach: Hello there." in make_vtt_text(SEGMENTS)


def test_flat_layout_numbers_duplicate_media_names_and_writes_selected_formats(tmp_path):
    source = tmp_path / "Media"
    first = source / "Day 1" / "Interview.mov"
    second = source / "Day 2" / "Interview.wav"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    engine = TranscriptionEngine(
        transcribe_fn=lambda _audio, **_kwargs: {"segments": SEGMENTS}
    )
    info = lambda path: MediaInfo(path=path, duration_seconds=4, frame_rate=Fraction(24))
    options = OutputOptions(
        folder_layout=FolderLayout.FLAT,
        formats=(OutputFormat.SRT, OutputFormat.VTT),
    )
    processor = BatchProcessor(
        engine,
        source,
        tmp_path / "Output",
        output_options=options,
        probe=lambda path: info(path),
    )
    summary = processor.run(
        [BatchJob(first, info(first)), BatchJob(second, info(second))]
    )

    subtitles = tmp_path / "Output" / "Transcription" / "Subtitles"
    assert (subtitles / "Interview.srt").exists()
    assert (subtitles / "Interview.vtt").exists()
    assert (subtitles / "Interview (2).srt").exists()
    assert (subtitles / "Interview (2).vtt").exists()
    assert summary.completed == 2
