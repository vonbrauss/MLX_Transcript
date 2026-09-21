"""One clip can never take the batch, or the application, down with it.

These cover the three failures found in the release readiness review:

B1
    An exception nobody anticipated escaping into the worker thread, which
    left the window permanently convinced a batch was still running.
B2
    An output tree that cannot be created, for example a destination on a
    read-only volume.
B3
    An embedded timecode the converter cannot read.

The shape every one of them checks is the same: ``run`` returns a summary,
``finished`` is emitted, and the rest of the queue is still processed.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transcription import outputs
from transcription.engine import TranscriptionEngine
from transcription.media_probe import MediaInfo, parse_probe_payload
from transcription.outputs import OutputFormat, OutputOptions
from transcription.pipeline import (
    OUTPUT_FOLDER_ERROR,
    UNEXPECTED_ITEM_ERROR,
    BatchJob,
    BatchProcessor,
    ItemOutcome,
)
from transcription.timecode import TimecodeError


def media(path: Path, timecode: str = "00:00:00:00", embedded: bool = False) -> MediaInfo:
    return MediaInfo(
        path=path,
        duration_seconds=4.0,
        frame_rate=Fraction(24, 1),
        start_timecode=timecode,
        has_embedded_timecode=embedded,
    )


def engine_returning_one_segment() -> TranscriptionEngine:
    return TranscriptionEngine(
        transcribe_fn=lambda path, **kwargs: {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
        }
    )


def clips(root: Path, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(count):
        clip = root / f"clip{index}.mov"
        clip.write_bytes(b"0")
        made.append(clip)
    return made


def processor(source_root: Path, output_parent: Path, **kwargs) -> BatchProcessor:
    kwargs.setdefault("engine", engine_returning_one_segment())
    kwargs.setdefault("probe", lambda path: media(path))
    kwargs.setdefault("output_options", OutputOptions())
    return BatchProcessor(
        source_root=source_root, output_parent=output_parent, **kwargs
    )


# ------------------------------------------------------- B1: unexpected errors


def test_an_unexpected_writer_failure_fails_only_that_clip(tmp_path, monkeypatch):
    source = tmp_path / "src"
    first, second = clips(source, 2)
    outcomes: list[tuple[Path, ItemOutcome]] = []

    def explode(path, *args, **kwargs):
        if Path(path).stem.startswith(first.stem):
            raise ValueError("something nobody planned for")
        return outputs.atomic_write_text(path, "ok")

    monkeypatch.setattr("transcription.pipeline.write_scriptsync", explode)

    batch = processor(source, tmp_path / "out")
    batch.on_outcome = lambda index, job, outcome, message: outcomes.append(
        (job.source, outcome)
    )
    summary = batch.run([BatchJob(source=first), BatchJob(source=second)])

    assert summary.failed == 1
    assert summary.completed == 1
    assert dict(outcomes)[first] is ItemOutcome.FAILED
    assert dict(outcomes)[second] is ItemOutcome.COMPLETED


def test_an_unexpected_failure_records_a_readable_message(tmp_path, monkeypatch):
    source = tmp_path / "src"
    (only,) = clips(source, 1)
    monkeypatch.setattr(
        "transcription.pipeline.write_scriptsync",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    summary = processor(source, tmp_path / "out").run([BatchJob(source=only)])

    assert summary.failed == 1
    path, message = summary.failures[0]
    assert path == only
    assert UNEXPECTED_ITEM_ERROR in message
    assert "ValueError" in message
    assert "boom" in message


def test_an_unexpected_failure_never_escapes_run(tmp_path, monkeypatch):
    source = tmp_path / "src"
    (only,) = clips(source, 1)
    monkeypatch.setattr(
        "transcription.pipeline.write_timecoded",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    # The assertion is simply that this returns rather than raises.
    summary = processor(source, tmp_path / "out").run([BatchJob(source=only)])
    assert summary.total == 1


def test_a_clip_outside_its_source_root_still_gets_a_transcript(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    stray = tmp_path / "elsewhere" / "stray.mov"
    stray.parent.mkdir()
    stray.write_bytes(b"0")

    summary = processor(source, tmp_path / "out").run(
        [BatchJob(source=stray, source_root=source)]
    )

    assert summary.completed == 1
    assert summary.failed == 0


# ------------------------------------------------------ B2: output tree failure


def test_an_uncreatable_output_tree_fails_the_batch_without_raising(tmp_path):
    source = tmp_path / "src"
    made = clips(source, 3)
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("this is a file")

    summary = processor(source, blocker / "dest").run(
        [BatchJob(source=clip) for clip in made]
    )

    assert summary.failed == 3
    assert summary.completed == 0
    assert summary.total == 3
    assert all(OUTPUT_FOLDER_ERROR in message for _path, message in summary.failures)


def test_an_uncreatable_output_tree_still_reports_progress(tmp_path):
    source = tmp_path / "src"
    made = clips(source, 2)
    blocker = tmp_path / "blocked"
    blocker.write_text("file")
    seen: list[tuple[int, int]] = []

    batch = processor(source, blocker / "dest")
    batch.on_progress = lambda done, total: seen.append((done, total))
    batch.run([BatchJob(source=clip) for clip in made])

    assert seen == [(1, 2), (2, 2)]


def test_an_empty_queue_does_not_touch_the_output_tree(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("file")

    summary = processor(tmp_path, blocker / "dest").run([])

    assert summary.total == 0


# ------------------------------------------------------- B3: malformed timecode


@pytest.mark.parametrize(
    "value",
    ["1:00:00", "not a timecode", "01:00:00:00:00", "", "  ", "99:99"],
)
def test_an_unreadable_timecode_tag_counts_as_no_timecode(value):
    info = parse_probe_payload(
        {"streams": [{"codec_type": "video", "tags": {"timecode": value}}]},
        Path("clip.mov"),
    )

    assert info.has_embedded_timecode is False
    assert info.missing_timecode is True
    assert info.start_timecode == "00:00:00:00"


def test_a_readable_timecode_tag_is_still_used():
    info = parse_probe_payload(
        {"streams": [{"codec_type": "video", "tags": {"timecode": "01:00:00:00"}}]},
        Path("clip.mov"),
    )

    assert info.has_embedded_timecode is True
    assert info.start_timecode == "01:00:00:00"


def test_a_usable_tag_is_preferred_over_an_unusable_one():
    info = parse_probe_payload(
        {
            "streams": [
                {"codec_type": "video", "tags": {"timecode": "garbage"}},
                {"codec_type": "data", "tags": {"timecode": "10:00:00:00"}},
            ]
        },
        Path("clip.mov"),
    )

    assert info.start_timecode == "10:00:00:00"


def test_make_timecoded_text_falls_back_instead_of_raising():
    text = outputs.make_timecoded_text(
        Path("clip.mov"),
        [{"start": 0.0, "end": 1.0, "text": "hello"}],
        Fraction(24, 1),
        source_timecode="1:00:00",
        missing_timecode=False,
    )

    assert "START TIMECODE: 00:00:00:00" in text
    assert "NOTE: No embedded timecode found" in text
    assert "hello" in text


def test_make_timecoded_text_survives_an_unusable_frame_rate():
    text = outputs.make_timecoded_text(
        Path("clip.mov"),
        [{"start": 0.0, "end": 1.0, "text": "hello"}],
        Fraction(0, 1),
        source_timecode="00:00:00:00",
    )

    assert "NOTE: No embedded timecode found" in text
    assert "hello" in text


def test_a_clip_with_a_malformed_timecode_still_completes(tmp_path):
    source = tmp_path / "src"
    (only,) = clips(source, 1)
    output = tmp_path / "out"

    summary = processor(
        source,
        output,
        probe=lambda path: media(path, timecode="1:00:00", embedded=True),
    ).run([BatchJob(source=only)])

    assert summary.completed == 1
    written = list((output / "Transcription" / "Timecoded").rglob("*.txt"))
    assert len(written) == 1
    assert "NOTE: No embedded timecode found" in written[0].read_text()


def test_the_converter_itself_still_rejects_a_bad_timecode():
    # The fallback belongs to the writer. The converter stays strict, which is
    # what lets the probe decide that a tag is unusable in the first place.
    with pytest.raises(TimecodeError):
        outputs.TimecodeConverter(Fraction(24, 1), "1:00:00")


# ------------------------------------------------- the batch keeps its promises


def test_the_rest_of_the_queue_runs_after_an_unexpected_failure(tmp_path, monkeypatch):
    source = tmp_path / "src"
    made = clips(source, 4)
    output = tmp_path / "out"
    calls = {"count": 0}
    real = outputs.write_scriptsync

    def flaky(path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise ZeroDivisionError("unlucky")
        return real(path, *args, **kwargs)

    monkeypatch.setattr("transcription.pipeline.write_scriptsync", flaky)

    summary = processor(
        source, output, output_options=OutputOptions(formats=(OutputFormat.SCRIPTSYNC,))
    ).run([BatchJob(source=clip) for clip in made])

    assert summary.completed == 3
    assert summary.failed == 1
    assert summary.total == 4
