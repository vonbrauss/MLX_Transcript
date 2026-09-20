"""Batch processing: success, failure, cancellation, and conflict policy.

The transcription engine is always a stand-in. No weights are downloaded and
no real media is decoded.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transcription.engine import TranscriptionEngine, TranscriptionError
from transcription.media_probe import MediaInfo
from transcription.outputs import OutputOptions, build_output_paths, output_roots
from transcription.pipeline import (
    BatchJob,
    BatchProcessor,
    ConflictChoice,
    ExistingFilePolicy,
    ItemOutcome,
    ItemStage,
)

NTSC_30 = Fraction(30000, 1001)

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "Rolling on the interview."},
    {"start": 2.5, "end": 4.0, "text": "Here we go."},
]


# ------------------------------------------------------------------ helpers


def fake_engine(segments=None, error: Exception | None = None):
    """An engine whose transcription callable is already injected."""

    def transcribe(audio, **kwargs):
        if error is not None:
            raise error
        return {"segments": list(segments if segments is not None else SEGMENTS)}

    return TranscriptionEngine(transcribe_fn=transcribe)


class LazyFakeEngine(TranscriptionEngine):
    """An engine that reports itself unloaded until ``load`` is called."""

    def __init__(self, segments=None) -> None:
        super().__init__(transcribe_fn=None)
        self.load_calls = 0
        self._segments = segments if segments is not None else SEGMENTS

    def load(self):
        if self._transcribe_fn is None:
            self.load_calls += 1
            self._transcribe_fn = lambda audio, **kwargs: {
                "segments": list(self._segments)
            }
        return self._transcribe_fn


def media(
    path: Path,
    timecode: str = "00:00:00:00",
    rate: Fraction = Fraction(24, 1),
    embedded: bool = False,
) -> MediaInfo:
    return MediaInfo(
        path=path,
        duration_seconds=10.0,
        frame_rate=rate,
        start_timecode=timecode,
        has_embedded_timecode=embedded,
        has_video=True,
    )


def build_source(tmp_path: Path, relative_paths: list[str]) -> Path:
    source = tmp_path / "Media"
    for relative in relative_paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return source


def jobs_for(source: Path, relative_paths: list[str], **kwargs) -> list[BatchJob]:
    return [
        BatchJob(source=source / relative, media=media(source / relative, **kwargs))
        for relative in relative_paths
    ]


def never_probe(path: Path) -> MediaInfo:
    raise AssertionError("the pipeline should reuse the media info it was given")


def make_processor(tmp_path: Path, source: Path, engine, **kwargs) -> BatchProcessor:
    kwargs.setdefault("probe", never_probe)
    return BatchProcessor(
        engine=engine,
        source_root=source,
        output_parent=tmp_path / "Projects",
        **kwargs,
    )


# ------------------------------------------------------------------ success


def test_batch_writes_both_transcripts_into_mirrored_trees(tmp_path):
    relatives = ["Interviews/Player A.mov", "Press Conferences/Coach.mov"]
    source = build_source(tmp_path, relatives)
    processor = make_processor(tmp_path, source, fake_engine())

    summary = processor.run(jobs_for(source, relatives))

    roots = output_roots(tmp_path / "Projects")
    assert (
        roots.scriptsync / "Interviews" / "Interviews — Player A ScriptSync.txt"
    ).exists()
    assert (
        roots.timecoded / "Interviews" / "Interviews — Player A timecoded.txt"
    ).exists()
    assert (
        roots.scriptsync
        / "Press Conferences"
        / "Press Conferences — Coach ScriptSync.txt"
    ).exists()
    assert (
        roots.timecoded
        / "Press Conferences"
        / "Press Conferences — Coach timecoded.txt"
    ).exists()
    assert summary.completed == 2


def test_batch_uses_each_job_source_root_for_mixed_folders(tmp_path):
    first_root = tmp_path / "Interviews"
    second_root = tmp_path / "Media Day"
    first = first_root / "Day 1" / "A.mov"
    second = second_root / "Chicago" / "B.mov"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.touch()
    processor = make_processor(
        tmp_path, first_root, fake_engine(), output_options=OutputOptions()
    )

    summary = processor.run([
        BatchJob(first, media(first), first_root, "Interviews"),
        BatchJob(second, media(second), second_root, "Media Day"),
    ])

    output = tmp_path / "Projects" / "Transcription" / "ScriptSync"
    assert (output / "Interviews" / "Day 1" / "A.txt").exists()
    assert (output / "Media Day" / "Chicago" / "B.txt").exists()
    assert summary.completed == 2
    assert summary.failed == 0


def test_media_at_the_source_root_uses_the_source_folder_name(tmp_path):
    relatives = ["Player A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source, fake_engine()).run(jobs_for(source, relatives))

    roots = output_roots(tmp_path / "Projects")
    assert (roots.scriptsync / "Media — Player A ScriptSync.txt").exists()


def test_batch_writes_exactly_two_files_per_clip(tmp_path):
    relatives = ["Interviews/Player A.mov"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source, fake_engine()).run(jobs_for(source, relatives))

    produced = sorted(
        path for path in (tmp_path / "Projects").rglob("*") if path.is_file()
    )
    assert len(produced) == 2
    assert {path.suffix for path in produced} == {".txt"}


def test_no_json_or_subtitle_side_products_are_written(tmp_path):
    relatives = ["Interviews/Player A.mov", "Interviews/Player B.wav"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source, fake_engine()).run(jobs_for(source, relatives))

    suffixes = {
        path.suffix.lower()
        for path in (tmp_path / "Projects").rglob("*")
        if path.is_file()
    }
    assert suffixes == {".txt"}
    for unwanted in (".json", ".srt", ".vtt", ".tsv", ".tmp", ".wav"):
        assert unwanted not in suffixes


def test_stages_are_reported_in_order(tmp_path):
    relatives = ["Interviews/Player A.mov"]
    source = build_source(tmp_path, relatives)
    stages: list[ItemStage] = []
    processor = make_processor(
        tmp_path,
        source,
        LazyFakeEngine(),
        on_stage=lambda index, job, stage: stages.append(stage),
    )
    processor.run(jobs_for(source, relatives))

    assert stages == [
        ItemStage.WAITING,
        ItemStage.LOADING_MODEL,
        ItemStage.TRANSCRIBING,
        ItemStage.WRITING,
    ]


def test_the_model_loads_once_for_the_whole_batch(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov"]
    source = build_source(tmp_path, relatives)
    engine = LazyFakeEngine()
    make_processor(tmp_path, source, engine).run(jobs_for(source, relatives))
    assert engine.load_calls == 1


def test_progress_is_reported_for_every_clip(tmp_path):
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)
    progress: list[tuple[int, int]] = []
    processor = make_processor(
        tmp_path, source, fake_engine(), on_progress=lambda done, total: progress.append((done, total))
    )
    processor.run(jobs_for(source, relatives))
    assert progress == [(1, 2), (2, 2)]


# ------------------------------------------------------------------ failure


def test_one_failure_does_not_stop_the_rest_of_the_queue(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov"]
    source = build_source(tmp_path, relatives)

    def transcribe(audio, **kwargs):
        if audio.endswith("B.mov"):
            raise RuntimeError("corrupt audio stream")
        return {"segments": list(SEGMENTS)}

    outcomes: list[ItemOutcome] = []
    processor = make_processor(
        tmp_path,
        source,
        TranscriptionEngine(transcribe_fn=transcribe),
        on_outcome=lambda index, job, outcome, message: outcomes.append(outcome),
    )
    summary = processor.run(jobs_for(source, relatives))

    assert outcomes == [
        ItemOutcome.COMPLETED,
        ItemOutcome.FAILED,
        ItemOutcome.COMPLETED,
    ]
    assert summary.completed == 2
    assert summary.failed == 1
    roots = output_roots(tmp_path / "Projects")
    assert (roots.scriptsync / "Media — C ScriptSync.txt").exists()
    assert not (roots.scriptsync / "Media — B ScriptSync.txt").exists()


def test_a_failure_records_a_readable_message(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    engine = fake_engine(error=TranscriptionError("A.mov: no audio track"))
    summary = make_processor(tmp_path, source, engine).run(
        jobs_for(source, relatives)
    )

    assert summary.failed == 1
    failed_path, message = summary.failures[0]
    assert failed_path.name == "A.mov"
    assert "no audio track" in message


# ------------------------------------------------------------- cancellation


def test_cancelling_leaves_finished_transcripts_and_marks_the_rest(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov"]
    source = build_source(tmp_path, relatives)
    processor: BatchProcessor

    def transcribe(audio, **kwargs):
        if audio.endswith("A.mov"):
            processor.cancel()
        return {"segments": list(SEGMENTS)}

    outcomes: list[ItemOutcome] = []
    processor = make_processor(
        tmp_path,
        source,
        TranscriptionEngine(transcribe_fn=transcribe),
        on_outcome=lambda index, job, outcome, message: outcomes.append(outcome),
    )
    summary = processor.run(jobs_for(source, relatives))

    assert outcomes == [
        ItemOutcome.COMPLETED,
        ItemOutcome.CANCELLED,
        ItemOutcome.CANCELLED,
    ]
    assert summary.completed == 1
    assert summary.cancelled == 2
    assert summary.was_cancelled is True

    roots = output_roots(tmp_path / "Projects")
    assert (roots.scriptsync / "Media — A ScriptSync.txt").exists()
    assert (roots.timecoded / "Media — A timecoded.txt").exists()
    assert not (roots.scriptsync / "Media — B ScriptSync.txt").exists()


def test_cancelling_before_the_batch_starts_processes_nothing(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    processor = make_processor(tmp_path, source, fake_engine())
    processor.cancel()
    summary = processor.run(jobs_for(source, relatives))

    assert summary.cancelled == 1
    assert summary.completed == 0
    assert list((tmp_path / "Projects").rglob("*.txt")) == []


# ---------------------------------------------------------------- conflicts


def seed_existing(tmp_path: Path, source: Path, relative: str) -> tuple[Path, Path]:
    """Pre-create both transcripts for one clip with recognizable content."""
    paths = build_output_paths(source / relative, source, tmp_path / "Projects")
    for path in paths.as_tuple():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("OLD", encoding="ascii")
    return paths.as_tuple()


def test_conflict_skip_keeps_the_existing_transcripts(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    scriptsync, timecoded = seed_existing(tmp_path, source, "A.mov")

    summary = make_processor(
        tmp_path,
        source,
        fake_engine(),
        conflict_resolver=lambda job, existing: ConflictChoice.SKIP_THIS,
    ).run(jobs_for(source, relatives))

    assert summary.skipped == 1
    assert scriptsync.read_text(encoding="ascii") == "OLD"
    assert timecoded.read_text(encoding="utf-8") == "OLD"


def test_conflict_overwrite_this_replaces_both_outputs(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    scriptsync, timecoded = seed_existing(tmp_path, source, "A.mov")

    summary = make_processor(
        tmp_path,
        source,
        fake_engine(),
        conflict_resolver=lambda job, existing: ConflictChoice.OVERWRITE_THIS,
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert "Rolling on the interview" in scriptsync.read_text(encoding="ascii")
    assert "FILE: A.mov" in timecoded.read_text(encoding="utf-8")


def test_conflict_cancel_batch_stops_everything(tmp_path):
    relatives = ["A.mov", "B.mov"]
    source = build_source(tmp_path, relatives)
    scriptsync, _ = seed_existing(tmp_path, source, "A.mov")

    summary = make_processor(
        tmp_path,
        source,
        fake_engine(),
        conflict_resolver=lambda job, existing: ConflictChoice.CANCEL_BATCH,
    ).run(jobs_for(source, relatives))

    assert summary.cancelled == 2
    assert summary.completed == 0
    assert summary.was_cancelled is True
    assert scriptsync.read_text(encoding="ascii") == "OLD"


def test_conflict_overwrite_all_asks_once_for_the_whole_batch(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov"]
    source = build_source(tmp_path, relatives)
    for relative in relatives:
        seed_existing(tmp_path, source, relative)

    asked: list[Path] = []

    def resolver(job, existing):
        asked.append(job.source)
        return ConflictChoice.OVERWRITE_ALL

    processor = make_processor(tmp_path, source, fake_engine(), conflict_resolver=resolver)
    summary = processor.run(jobs_for(source, relatives))

    assert len(asked) == 1
    assert processor.overwrite_all is True
    assert summary.completed == 3
    roots = output_roots(tmp_path / "Projects")
    for name in ("A", "B", "C"):
        text = (roots.scriptsync / f"Media — {name} ScriptSync.txt").read_text(
            encoding="ascii"
        )
        assert "Rolling on the interview" in text


def test_overwrite_all_resets_for_the_next_batch(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    seed_existing(tmp_path, source, "A.mov")

    first = make_processor(
        tmp_path,
        source,
        fake_engine(),
        conflict_resolver=lambda job, existing: ConflictChoice.OVERWRITE_ALL,
    )
    first.run(jobs_for(source, relatives))
    assert first.overwrite_all is True

    second = make_processor(tmp_path, source, fake_engine())
    assert second.overwrite_all is False


def test_skip_policy_never_asks_and_creates_only_the_missing_output(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    paths = build_output_paths(source / "A.mov", source, tmp_path / "Projects")
    paths.scriptsync.parent.mkdir(parents=True, exist_ok=True)
    paths.scriptsync.write_text("OLD", encoding="ascii")

    def resolver(job, existing):
        raise AssertionError("Skip existing must not raise a dialog")

    summary = make_processor(
        tmp_path,
        source,
        fake_engine(),
        policy=ExistingFilePolicy.SKIP,
        conflict_resolver=resolver,
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert paths.scriptsync.read_text(encoding="ascii") == "OLD"
    assert paths.timecoded.exists()


def test_overwrite_policy_never_asks(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    scriptsync, _ = seed_existing(tmp_path, source, "A.mov")

    def resolver(job, existing):
        raise AssertionError("Overwrite all must not raise a dialog")

    summary = make_processor(
        tmp_path,
        source,
        fake_engine(),
        policy=ExistingFilePolicy.OVERWRITE,
        conflict_resolver=resolver,
    ).run(jobs_for(source, relatives))

    assert summary.completed == 1
    assert scriptsync.read_text(encoding="ascii") != "OLD"


# ---------------------------------------------------------------- timecode


def test_timecoded_output_starts_from_the_embedded_timecode(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    jobs = jobs_for(source, relatives, timecode="01:00:00:00", embedded=True)
    make_processor(tmp_path, source, fake_engine()).run(jobs)

    roots = output_roots(tmp_path / "Projects")
    text = (roots.timecoded / "Media — A timecoded.txt").read_text(
        encoding="utf-8"
    )
    assert "START TIMECODE: 01:00:00:00" in text
    assert "01:00:00:00  Rolling on the interview." in text
    assert "01:00:02:12  Here we go." in text
    assert "NOTE: No embedded timecode" not in text


def test_drop_frame_media_produces_drop_frame_stamps(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    jobs = jobs_for(
        source, relatives, timecode="00:59:59;28", rate=NTSC_30, embedded=True
    )
    make_processor(tmp_path, source, fake_engine()).run(jobs)

    roots = output_roots(tmp_path / "Projects")
    lines = (
        (roots.timecoded / "Media — A timecoded.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "START TIMECODE: 00:59:59;28" in lines[1]
    assert lines[-2].startswith("00:59:59;28")
    assert ";" in lines[-1]
    assert lines[-1].startswith("01:00:02")


def test_missing_timecode_falls_back_and_says_so(tmp_path):
    relatives = ["Interview.wav"]
    source = build_source(tmp_path, relatives)
    make_processor(tmp_path, source, fake_engine()).run(jobs_for(source, relatives))

    roots = output_roots(tmp_path / "Projects")
    text = (roots.timecoded / "Media — Interview timecoded.txt").read_text(
        encoding="utf-8"
    )
    assert "START TIMECODE: 00:00:00:00" in text
    assert "NOTE: No embedded timecode found; 00:00:00:00 was used." in text
    assert "00:00:00:00  Rolling on the interview." in text


def test_scriptsync_output_has_no_timestamps(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    jobs = jobs_for(source, relatives, timecode="01:00:00:00", embedded=True)
    make_processor(tmp_path, source, fake_engine()).run(jobs)

    roots = output_roots(tmp_path / "Projects")
    # Read the bytes: reading as text would translate the CRLF endings away.
    raw = (roots.scriptsync / "Media — A ScriptSync.txt").read_bytes()
    text = raw.decode("ascii")
    assert "01:00:00" not in text
    assert text.endswith("\r\n")
    assert "Rolling on the interview. Here we go." in text.replace("\r\n", " ")


# ------------------------------------------------------------------ probing


def test_media_is_probed_when_the_job_has_none(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    calls: list[Path] = []

    def probe(path: Path) -> MediaInfo:
        calls.append(path)
        return media(path, timecode="02:00:00:00", embedded=True)

    processor = BatchProcessor(
        engine=fake_engine(),
        source_root=source,
        output_parent=tmp_path / "Projects",
        probe=probe,
    )
    summary = processor.run([BatchJob(source=source / "A.mov")])

    assert calls == [source / "A.mov"]
    assert summary.completed == 1


# ------------------------------------------------------------------ summary


def test_summary_counts_every_outcome(tmp_path):
    relatives = ["A.mov", "B.mov", "C.mov", "D.mov"]
    source = build_source(tmp_path, relatives)
    seed_existing(tmp_path, source, "B.mov")
    processor: BatchProcessor

    def transcribe(audio, **kwargs):
        if audio.endswith("C.mov"):
            raise RuntimeError("decode failed")
        if audio.endswith("A.mov"):
            return {"segments": list(SEGMENTS)}
        return {"segments": list(SEGMENTS)}

    def resolver(job, existing):
        return ConflictChoice.SKIP_THIS

    processor = make_processor(
        tmp_path,
        source,
        TranscriptionEngine(transcribe_fn=transcribe),
        conflict_resolver=resolver,
    )

    original_process = processor._process

    def process(index, job):
        outcome = original_process(index, job)
        if job.source.name == "C.mov":
            processor.cancel()
        return outcome

    processor._process = process
    summary = processor.run(jobs_for(source, relatives))

    assert (summary.completed, summary.skipped, summary.failed, summary.cancelled) == (
        1,
        1,
        1,
        1,
    )
    assert summary.total == 4
    assert summary.transcription_folder == output_roots(
        tmp_path / "Projects"
    ).transcription


def test_summary_sentence_reads_cleanly():
    from transcription.pipeline import BatchSummary

    summary = BatchSummary(completed=3, skipped=1, failed=2, cancelled=0)
    assert summary.as_sentence() == (
        "Finished. 3 completed, 1 skipped, 2 failed, 0 cancelled."
    )
    summary.was_cancelled = True
    assert summary.as_sentence().startswith("Cancelled.")


def test_summary_carries_the_output_folder(tmp_path):
    relatives = ["A.mov"]
    source = build_source(tmp_path, relatives)
    summary = make_processor(tmp_path, source, fake_engine()).run(
        jobs_for(source, relatives)
    )
    assert summary.transcription_folder == tmp_path / "Projects" / "Transcription"


@pytest.mark.parametrize("policy", list(ExistingFilePolicy))
def test_an_empty_queue_is_harmless(tmp_path, policy):
    source = build_source(tmp_path, [])
    summary = make_processor(tmp_path, source, fake_engine(), policy=policy).run([])
    assert summary.total == 0
    assert summary.was_cancelled is False
