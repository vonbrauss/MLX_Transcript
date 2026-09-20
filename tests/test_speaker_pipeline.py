"""Speaker detection through the batch pipeline, end to end with stand-ins."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transcription.diarization import (
    DiarizationError,
    DiarizationOptions,
    SpeakerCountMode,
)
from transcription.engine import TranscriptionEngine
from transcription.media_probe import MediaInfo
from transcription.outputs import output_roots
from transcription.pipeline import (
    BatchJob,
    BatchProcessor,
    ConflictChoice,
    ExistingFilePolicy,
    ItemOutcome,
    ItemStage,
    ReviewDecision,
    ReviewMode,
)
from transcription.speakers import SpeakerTurn

NTSC_30 = Fraction(30000, 1001)

SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.0,
        "text": "Welcome, everyone.",
        "words": [
            {"word": "Welcome,", "start": 0.2, "end": 0.9},
            {"word": "everyone.", "start": 1.0, "end": 1.8},
        ],
    },
    {
        "start": 6.0,
        "end": 8.0,
        "text": "Can you discuss the fourth quarter?",
        "words": [
            {"word": "Can", "start": 6.1, "end": 6.3},
            {"word": "you", "start": 6.4, "end": 6.6},
            {"word": "discuss", "start": 6.7, "end": 7.1},
            {"word": "the", "start": 7.2, "end": 7.3},
            {"word": "fourth", "start": 7.4, "end": 7.7},
            {"word": "quarter?", "start": 7.8, "end": 8.0},
        ],
    },
    {
        "start": 12.0,
        "end": 14.0,
        "text": "We defended with more discipline.",
        "words": [
            {"word": "We", "start": 12.1, "end": 12.3},
            {"word": "defended", "start": 12.4, "end": 12.9},
            {"word": "with", "start": 13.0, "end": 13.2},
            {"word": "more", "start": 13.3, "end": 13.5},
            {"word": "discipline.", "start": 13.6, "end": 14.0},
        ],
    },
]

TURNS = [
    SpeakerTurn(0.0, 3.0, "speaker_00"),
    SpeakerTurn(5.5, 9.0, "speaker_01"),
    SpeakerTurn(11.5, 15.0, "speaker_00"),
]


# ------------------------------------------------------------------ helpers


class FakeDiarizer:
    """Returns fixed turns, or raises, and records how it was called."""

    def __init__(self, turns=None, error: Exception | None = None) -> None:
        self.turns = TURNS if turns is None else turns
        self.error = error
        self.calls: list[tuple[Path, DiarizationOptions]] = []

    def diarize(self, source, options, progress=None, cancelled=None):
        self.calls.append((Path(source), options))
        if self.error is not None:
            raise self.error
        return list(self.turns)


def engine(segments=None) -> TranscriptionEngine:
    payload = {"segments": [dict(row) for row in (segments or SEGMENTS)]}
    return TranscriptionEngine(transcribe_fn=lambda audio, **kwargs: payload)


def media(path: Path, timecode="00:00:00:00", rate=Fraction(24, 1), embedded=False):
    return MediaInfo(
        path=path,
        duration_seconds=20.0,
        frame_rate=rate,
        start_timecode=timecode,
        has_embedded_timecode=embedded,
        has_video=True,
    )


def build_source(tmp_path: Path, relatives: list[str]) -> Path:
    source = tmp_path / "Media"
    for relative in relatives:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return source


def jobs_for(source: Path, relatives: list[str], **kwargs) -> list[BatchJob]:
    return [
        BatchJob(source=source / relative, media=media(source / relative, **kwargs))
        for relative in relatives
    ]


def make_processor(tmp_path: Path, source: Path, **kwargs) -> BatchProcessor:
    kwargs.setdefault("engine", engine())
    kwargs.setdefault("probe", lambda path: media(path))
    kwargs.setdefault("diarization", DiarizationOptions(enabled=True))
    kwargs.setdefault("diarizer", FakeDiarizer())
    kwargs.setdefault("include_speakers_in_scriptsync", True)
    kwargs.setdefault("review_mode", ReviewMode.AUTOMATIC)
    return BatchProcessor(
        source_root=source, output_parent=tmp_path / "Projects", **kwargs
    )


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def outputs(tmp_path: Path, stem: str = "A") -> tuple[Path, Path]:
    roots = output_roots(tmp_path / "Projects")
    return (
        roots.scriptsync / f"Media — {stem} ScriptSync.txt",
        roots.timecoded / f"Media — {stem} timecoded.txt",
    )


# ------------------------------------------- detection off is unchanged


def test_detection_off_produces_byte_identical_milestone_two_output(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    make_processor(
        tmp_path,
        source,
        diarization=DiarizationOptions(enabled=False),
        diarizer=FakeDiarizer(),
    ).run(jobs_for(source, relatives))
    plain_scriptsync, plain_timecoded = (path.read_bytes() for path in outputs(tmp_path))

    # A second run with no diarization configured at all must match exactly.
    for path in outputs(tmp_path):
        path.unlink()
    BatchProcessor(
        engine=engine(),
        source_root=source,
        output_parent=tmp_path / "Projects",
        probe=lambda path: media(path),
    ).run(jobs_for(source, relatives))

    assert outputs(tmp_path)[0].read_bytes() == plain_scriptsync
    assert outputs(tmp_path)[1].read_bytes() == plain_timecoded


def test_detection_off_never_calls_the_diarizer(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    diarizer = FakeDiarizer()
    make_processor(
        tmp_path,
        source,
        diarization=DiarizationOptions(enabled=False),
        diarizer=diarizer,
    ).run(jobs_for(source, relatives))
    assert diarizer.calls == []


def test_detection_off_writes_no_speaker_labels(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(
        tmp_path, source, diarization=DiarizationOptions(enabled=False)
    ).run(jobs_for(source, relatives))

    scriptsync, timecoded = outputs(tmp_path)
    assert "SPEAKER" not in read_text(scriptsync).upper()
    assert ":" not in read_text(timecoded).split("\n")[-2].split("  ", 1)[1]


# ----------------------------------------------------------- detection on


def test_speakers_reach_both_transcripts(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source).run(jobs_for(source, relatives))

    scriptsync, timecoded = outputs(tmp_path)
    assert "SPEAKER 1:" in read_text(scriptsync)
    assert "SPEAKER 2:" in read_text(scriptsync)
    assert "Speaker 1: Welcome, everyone." in read_text(timecoded)
    assert "Speaker 2: Can you discuss the fourth quarter?" in read_text(timecoded)


def test_the_diarizer_receives_the_chosen_options(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    diarizer = FakeDiarizer()
    options = DiarizationOptions(
        enabled=True, count_mode=SpeakerCountMode.EXACT, exact_speakers=3
    )
    make_processor(
        tmp_path, source, diarizer=diarizer, diarization=options
    ).run(jobs_for(source, relatives))

    assert diarizer.calls[0][1].as_backend_kwargs()["num_clusters"] == 3


def test_stages_include_detection(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    stages: list[ItemStage] = []
    make_processor(
        tmp_path, source, on_stage=lambda index, job, stage: stages.append(stage)
    ).run(jobs_for(source, relatives))

    assert ItemStage.DETECTING_SPEAKERS in stages
    assert stages.index(ItemStage.DETECTING_SPEAKERS) < stages.index(ItemStage.WRITING)
    assert ItemStage.AWAITING_REVIEW not in stages  # automatic mode


def test_review_mode_adds_the_waiting_stage(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    stages: list[ItemStage] = []
    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=lambda job, transcript: ReviewDecision.CONTINUE,
        on_stage=lambda index, job, stage: stages.append(stage),
    ).run(jobs_for(source, relatives))

    assert ItemStage.AWAITING_REVIEW in stages


def test_scriptsync_headings_are_off_when_the_option_is_off(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(
        tmp_path, source, include_speakers_in_scriptsync=False
    ).run(jobs_for(source, relatives))

    scriptsync, timecoded = outputs(tmp_path)
    assert "SPEAKER 1:" not in read_text(scriptsync)
    # Timecoded always carries speakers while detection is on.
    assert "Speaker 1:" in read_text(timecoded)


def test_scriptsync_keeps_crlf_endings_with_speakers(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source).run(jobs_for(source, relatives))

    raw = outputs(tmp_path)[0].read_bytes()
    assert b"\r\n" in raw
    assert b"\r\r\n" not in raw
    assert raw.endswith(b"\r\n")
    assert b"SPEAKER 1:\r\n" in raw


def test_a_speaker_heading_is_not_repeated_back_to_back(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    turns = [SpeakerTurn(0.0, 30.0, "speaker_00")]
    make_processor(tmp_path, source, diarizer=FakeDiarizer(turns)).run(
        jobs_for(source, relatives)
    )

    text = read_text(outputs(tmp_path)[0])
    assert text.count("SPEAKER 1:") == 1
    assert text.count("\r\n\r\n") >= 1  # paragraphs still break on silence


def test_speaker_names_survive_a_review_rename(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    def review(job, transcript):
        transcript.rename("speaker_00", "Coach Smith")
        transcript.rename("speaker_01", "Reporter")
        return ReviewDecision.CONTINUE

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    ).run(jobs_for(source, relatives))

    assert "COACH SMITH:" in read_text(outputs(tmp_path)[0])
    assert "Coach Smith: Welcome, everyone." in read_text(outputs(tmp_path)[1])
    assert "Reporter: Can you discuss" in read_text(outputs(tmp_path)[1])


def test_a_unicode_name_reaches_the_timecoded_file_intact(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    def review(job, transcript):
        transcript.rename("speaker_00", "Renée Müller")
        return ReviewDecision.CONTINUE

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    ).run(jobs_for(source, relatives))

    assert "Renée Müller: Welcome, everyone." in read_text(outputs(tmp_path)[1])
    # ScriptSync must stay ASCII, so the same name arrives unaccented.
    assert "RENEE MULLER:" in outputs(tmp_path)[0].read_bytes().decode("ascii")


def test_merging_in_review_collapses_the_labels(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    def review(job, transcript):
        transcript.merge("speaker_01", "speaker_00")
        transcript.rename("speaker_00", "Everyone")
        return ReviewDecision.CONTINUE

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    ).run(jobs_for(source, relatives))

    text = read_text(outputs(tmp_path)[1])
    assert "Speaker 2" not in text
    assert text.count("Everyone:") == 3


def test_reassigning_in_review_moves_one_line(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    def review(job, transcript):
        transcript.reassign(1, "speaker_00")
        return ReviewDecision.CONTINUE

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    ).run(jobs_for(source, relatives))

    assert "Speaker 2" not in read_text(outputs(tmp_path)[1])


# ------------------------------------------------------------- timecode


def test_speaker_lines_use_the_embedded_source_timecode(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    jobs = jobs_for(source, relatives, timecode="01:00:00:00", embedded=True)
    make_processor(tmp_path, source).run(jobs)

    text = read_text(outputs(tmp_path)[1])
    assert "START TIMECODE: 01:00:00:00" in text
    # The first word starts at 0.2s, which is 4 frames at 24 fps.
    assert "01:00:00:05  Speaker 1: Welcome, everyone." in text


def test_drop_frame_speaker_lines_keep_the_semicolon(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    jobs = jobs_for(source, relatives, timecode="00:59:59;28", rate=NTSC_30, embedded=True)
    make_processor(tmp_path, source).run(jobs)

    lines = read_text(outputs(tmp_path)[1]).splitlines()
    speaker_lines = [line for line in lines if "Speaker" in line and ";" in line]
    assert speaker_lines
    # 00:59:59;28 plus the first word's 0.2s rolls across the minute, and
    # drop-frame counting skips two frames doing it.
    assert speaker_lines[0] == "01:00:00;04  Speaker 1: Welcome, everyone."
    assert all(";" in line.split("  ")[0] for line in speaker_lines)


def test_missing_timecode_still_discloses_the_fallback(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source).run(jobs_for(source, relatives))

    text = read_text(outputs(tmp_path)[1])
    assert "NOTE: No embedded timecode found; 00:00:00:00 was used." in text
    assert "Speaker 1:" in text


# -------------------------------------------------------------- fallbacks


def test_a_diarization_failure_still_exports_the_transcript(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(
        tmp_path, source, diarizer=FakeDiarizer(error=DiarizationError("no audio"))
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert summary.failed == 0
    scriptsync, timecoded = outputs(tmp_path)
    assert scriptsync.exists() and timecoded.exists()
    assert "SPEAKER" not in read_text(scriptsync).upper()


def test_a_diarization_failure_is_recorded_as_a_warning(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(
        tmp_path, source, diarizer=FakeDiarizer(error=DiarizationError("no audio"))
    ).run(jobs_for(source, relatives))

    assert summary.diarization_warnings == 1
    path, message = summary.warnings[0]
    assert path.name == "A.mov"
    assert "no audio" in message
    assert "exported without speaker labels" in summary.as_sentence()


def test_an_unexpected_backend_crash_is_also_a_warning(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(
        tmp_path, source, diarizer=FakeDiarizer(error=MemoryError("out of memory"))
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert summary.diarization_warnings == 1


def test_an_empty_diarization_result_falls_back_cleanly(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(tmp_path, source, diarizer=FakeDiarizer([])).run(
        jobs_for(source, relatives)
    )

    assert summary.completed == 1
    assert "No speakers were detected" in summary.warnings[0][1]
    assert "SPEAKER" not in read_text(outputs(tmp_path)[0]).upper()


def test_export_without_labels_from_review(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=lambda job, transcript: ReviewDecision.EXPORT_WITHOUT_SPEAKERS,
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert "SPEAKER" not in read_text(outputs(tmp_path)[0]).upper()
    assert "Speaker 1:" not in read_text(outputs(tmp_path)[1])


def test_cancelling_from_review_stops_the_batch(tmp_path):
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=lambda job, transcript: ReviewDecision.CANCEL_BATCH,
    ).run(jobs_for(source, relatives))

    assert summary.cancelled == 2
    assert summary.completed == 0
    assert summary.was_cancelled is True
    assert list((tmp_path / "Projects").rglob("*.txt")) == []


def test_cancelling_mid_batch_keeps_finished_speaker_transcripts(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov"]
    source = build_source(tmp_path, relatives)
    processor: BatchProcessor

    def review(job, transcript):
        if job.source.name == "A.mov":
            processor.cancel()
        return ReviewDecision.CONTINUE

    processor = make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    )
    summary = processor.run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert summary.cancelled == 2
    assert outputs(tmp_path, "A")[0].exists()
    assert not outputs(tmp_path, "B")[0].exists()


def test_review_is_skipped_in_automatic_mode(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    asked: list[Path] = []

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.AUTOMATIC,
        review_resolver=lambda job, transcript: asked.append(job.source)
        or ReviewDecision.CONTINUE,
    ).run(jobs_for(source, relatives))

    assert asked == []
    assert "SPEAKER 1:" in read_text(outputs(tmp_path)[0])


def test_names_do_not_leak_between_files(tmp_path):
    """Speaker 1 in one recording is not Speaker 1 in the next."""
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)
    seen: list[list[str]] = []

    def review(job, transcript):
        seen.append([speaker.display_name for speaker in transcript.speakers])
        if job.source.name == "A.mov":
            transcript.rename("speaker_00", "Coach Smith")
        return ReviewDecision.CONTINUE

    make_processor(
        tmp_path,
        source,
        review_mode=ReviewMode.EVERY_FILE,
        review_resolver=review,
    ).run(jobs_for(source, relatives))

    assert seen[1] == ["Speaker 1", "Speaker 2"]
    assert "Coach Smith" not in read_text(outputs(tmp_path, "B")[1])


# ------------------------------------------------------- unchanged policies


def test_only_two_files_are_written_per_clip(tmp_path):
    relatives = ["A.mov", "B.wav"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source).run(jobs_for(source, relatives))

    produced = [path for path in (tmp_path / "Projects").rglob("*") if path.is_file()]
    assert len(produced) == 4
    assert {path.suffix.lower() for path in produced} == {".txt"}


def test_no_rttm_json_or_embedding_side_products(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source).run(jobs_for(source, relatives))

    suffixes = {
        path.suffix.lower()
        for path in (tmp_path / "Projects").rglob("*")
        if path.is_file()
    }
    for unwanted in (".rttm", ".json", ".srt", ".vtt", ".csv", ".npy", ".wav", ".tmp"):
        assert unwanted not in suffixes


def test_the_conflict_policy_still_applies_with_speakers(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    scriptsync, timecoded = outputs(tmp_path)
    for path in (scriptsync, timecoded):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("OLD", encoding="ascii")

    summary = make_processor(
        tmp_path,
        source,
        conflict_resolver=lambda job, existing: ConflictChoice.SKIP_THIS,
    ).run(jobs_for(source, relatives))

    assert summary.skipped == 1
    assert scriptsync.read_text(encoding="ascii") == "OLD"


def test_overwrite_all_still_latches_with_speakers(tmp_path):
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)
    for stem in ("A", "B"):
        for path in outputs(tmp_path, stem):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("OLD", encoding="ascii")

    asked: list[Path] = []
    processor = make_processor(
        tmp_path,
        source,
        conflict_resolver=lambda job, existing: asked.append(job.source)
        or ConflictChoice.OVERWRITE_ALL,
    )
    summary = processor.run(jobs_for(source, relatives))

    assert len(asked) == 1
    assert summary.completed == 2
    assert "SPEAKER 1:" in read_text(outputs(tmp_path, "B")[0])


def test_a_transcription_failure_is_still_a_failure_not_a_warning(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)

    def explode(audio, **kwargs):
        raise RuntimeError("decode failed")

    summary = make_processor(
        tmp_path, source, engine=TranscriptionEngine(transcribe_fn=explode)
    ).run(jobs_for(source, relatives))

    assert summary.failed == 1
    assert summary.completed == 0
    assert summary.diarization_warnings == 0


def test_summary_counts_survive_a_mixed_speaker_batch(tmp_path):
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)

    class Flaky(FakeDiarizer):
        def diarize(self, source, options, progress=None, cancelled=None):
            if Path(source).name == "B.mov":
                raise DiarizationError("model gave up")
            return list(TURNS)

    summary = make_processor(tmp_path, source, diarizer=Flaky()).run(
        jobs_for(source, relatives)
    )

    assert (summary.completed, summary.failed, summary.skipped) == (2, 0, 0)
    assert summary.diarization_warnings == 1
    assert "SPEAKER 1:" in read_text(outputs(tmp_path, "A")[0])
    assert "SPEAKER" not in read_text(outputs(tmp_path, "B")[0]).upper()


@pytest.mark.parametrize("policy", list(ExistingFilePolicy))
def test_every_policy_runs_with_speakers_enabled(tmp_path, policy):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(tmp_path, source, policy=policy).run(
        jobs_for(source, relatives)
    )
    assert summary.completed == 1
