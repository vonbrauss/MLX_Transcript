"""Diarization option construction, lazy loading, and result normalization."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from transcription.diarization import (
    DiarizationError,
    DiarizationOptions,
    NullDiarizer,
    SherpaOnnxDiarizer,
    SpeakerCountMode,
    backend_available,
    normalize_turns,
)
from transcription.model_cache import DEFAULT_BUNDLE, ModelCache, ModelCacheError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeSegment:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker


class FakePipeline:
    """Stands in for sherpa-onnx's OfflineSpeakerDiarization."""

    def __init__(self, segments=None, error=None):
        self.segments = segments or []
        self.error = error
        self.calls = 0

    def process(self, samples, callback=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if callback is not None:
            callback(1, 2)
            callback(2, 2)
        return FakeResult(self.segments)


class FakeResult:
    def __init__(self, segments):
        self._segments = segments

    def sort_by_start_time(self):
        return sorted(self._segments, key=lambda item: item.start)


def ready_cache(tmp_path: Path) -> ModelCache:
    store = ModelCache(root=tmp_path / "models", downloader=lambda *a, **k: None)
    store.root.mkdir(parents=True)
    for path in store.paths_for(DEFAULT_BUNDLE):
        path.write_bytes(b"model")
    return store


def diarizer(tmp_path: Path, pipeline: FakePipeline, samples=b"audio") -> SherpaOnnxDiarizer:
    return SherpaOnnxDiarizer(
        cache=ready_cache(tmp_path),
        build_fn=lambda options, paths: pipeline,
        decode_fn=lambda source: samples,
    )


# ------------------------------------------------------- option construction


def test_automatic_asks_the_backend_to_decide():
    options = DiarizationOptions(count_mode=SpeakerCountMode.AUTOMATIC)
    kwargs = options.as_backend_kwargs()
    assert kwargs["num_clusters"] == -1
    assert kwargs["threshold"] == pytest.approx(0.5)


def test_an_exact_count_pins_the_cluster_number():
    options = DiarizationOptions(count_mode=SpeakerCountMode.EXACT, exact_speakers=3)
    kwargs = options.as_backend_kwargs()
    assert kwargs["num_clusters"] == 3
    assert "threshold" not in kwargs


def test_an_exact_count_is_never_below_one():
    options = DiarizationOptions(count_mode=SpeakerCountMode.EXACT, exact_speakers=0)
    assert options.as_backend_kwargs()["num_clusters"] == 1


def test_the_automatic_threshold_is_configurable():
    options = DiarizationOptions(clustering_threshold=0.8)
    assert options.as_backend_kwargs()["threshold"] == pytest.approx(0.8)


def test_timing_and_alignment_defaults_are_explicit():
    options = DiarizationOptions()
    assert options.min_duration_on == pytest.approx(0.3)
    assert options.min_duration_off == pytest.approx(0.5)
    assert options.nearest_tolerance == pytest.approx(0.75)
    assert options.merge_gap == pytest.approx(1.0)


def test_minimum_and_maximum_are_deliberately_absent():
    """FastClusteringConfig has no min/max, so the options must not pretend."""
    options = DiarizationOptions()
    assert not hasattr(options, "min_speakers")
    assert not hasattr(options, "max_speakers")
    assert set(options.as_backend_kwargs()) <= {"num_clusters", "threshold"}


def test_detection_is_off_by_default():
    assert DiarizationOptions().enabled is False


def test_the_count_modes_are_labelled_for_the_picker():
    assert SpeakerCountMode.AUTOMATIC.label == "Automatic"
    assert SpeakerCountMode.EXACT.label == "Exact number"


# ------------------------------------------------------------- lazy loading


def test_importing_diarization_does_not_import_sherpa_onnx():
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "import transcription.diarization;"
        "print('sherpa_onnx' in sys.modules)" % PROJECT_ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_importing_the_whole_app_loads_no_model():
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "import transcription.pipeline, transcription.alignment, transcription.speakers;"
        "print('sherpa_onnx' in sys.modules, 'mlx_whisper' in sys.modules)"
        % PROJECT_ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_backend_availability_is_checked_without_importing():
    assert isinstance(backend_available(), bool)


def test_a_diarizer_is_not_loaded_until_asked(tmp_path):
    assert SherpaOnnxDiarizer(cache=ready_cache(tmp_path)).is_loaded is False


def test_the_pipeline_is_built_once_for_the_same_options(tmp_path):
    built: list[int] = []
    engine = SherpaOnnxDiarizer(
        cache=ready_cache(tmp_path),
        build_fn=lambda options, paths: built.append(1) or FakePipeline(),
        decode_fn=lambda source: b"audio",
    )
    options = DiarizationOptions(enabled=True)
    engine.diarize(Path("a.mov"), options)
    engine.diarize(Path("b.mov"), options)
    assert len(built) == 1


def test_changing_the_speaker_count_rebuilds_the_pipeline(tmp_path):
    built: list[int] = []
    engine = SherpaOnnxDiarizer(
        cache=ready_cache(tmp_path),
        build_fn=lambda options, paths: built.append(1) or FakePipeline(),
        decode_fn=lambda source: b"audio",
    )
    engine.diarize(Path("a.mov"), DiarizationOptions(enabled=True))
    engine.diarize(
        Path("b.mov"),
        DiarizationOptions(enabled=True, count_mode=SpeakerCountMode.EXACT),
    )
    assert len(built) == 2


def test_changing_a_backend_timing_value_rebuilds_the_pipeline(tmp_path):
    built: list[int] = []
    engine = SherpaOnnxDiarizer(
        cache=ready_cache(tmp_path),
        build_fn=lambda options, paths: built.append(1) or FakePipeline(),
        decode_fn=lambda source: b"audio",
    )
    engine.diarize(Path("a.mov"), DiarizationOptions(enabled=True))
    engine.diarize(
        Path("b.mov"),
        DiarizationOptions(enabled=True, min_duration_on=0.6),
    )
    assert len(built) == 2


def test_a_missing_model_becomes_a_diarization_error(tmp_path):
    store = ModelCache(
        root=tmp_path / "models",
        downloader=lambda *a, **k: (_ for _ in ()).throw(ModelCacheError("offline")),
    )
    engine = SherpaOnnxDiarizer(cache=store, build_fn=lambda options, paths: FakePipeline())
    with pytest.raises(DiarizationError) as error:
        engine.diarize(Path("a.mov"), DiarizationOptions(enabled=True))
    assert "offline" in str(error.value)


def test_a_build_failure_becomes_a_diarization_error(tmp_path):
    engine = SherpaOnnxDiarizer(
        cache=ready_cache(tmp_path),
        build_fn=lambda options, paths: (_ for _ in ()).throw(RuntimeError("bad model")),
    )
    with pytest.raises(DiarizationError) as error:
        engine.diarize(Path("a.mov"), DiarizationOptions(enabled=True))
    assert "bad model" in str(error.value)


# -------------------------------------------------------------- diarizing


def test_turns_come_back_sorted_and_typed(tmp_path):
    pipeline = FakePipeline(
        [
            FakeSegment(5.0, 9.0, "speaker_01"),
            FakeSegment(0.0, 4.5, "speaker_00"),
        ]
    )
    turns = diarizer(tmp_path, pipeline).diarize(
        Path("a.mov"), DiarizationOptions(enabled=True)
    )
    assert [turn.speaker for turn in turns] == ["speaker_00", "speaker_01"]
    assert turns[0].start == 0.0


def test_a_backend_failure_becomes_a_diarization_error(tmp_path):
    pipeline = FakePipeline(error=RuntimeError("segfault-ish"))
    with pytest.raises(DiarizationError) as error:
        diarizer(tmp_path, pipeline).diarize(
            Path("a.mov"), DiarizationOptions(enabled=True)
        )
    assert "segfault-ish" in str(error.value)


def test_cancelling_before_processing_raises(tmp_path):
    pipeline = FakePipeline([FakeSegment(0.0, 1.0, "speaker_00")])
    with pytest.raises(DiarizationError):
        diarizer(tmp_path, pipeline).diarize(
            Path("a.mov"), DiarizationOptions(enabled=True), cancelled=lambda: True
        )
    assert pipeline.calls == 0


def test_progress_is_forwarded_as_a_fraction(tmp_path):
    seen: list[float] = []
    pipeline = FakePipeline([FakeSegment(0.0, 1.0, "speaker_00")])
    diarizer(tmp_path, pipeline).diarize(
        Path("a.mov"), DiarizationOptions(enabled=True), progress=seen.append
    )
    assert seen == [0.5, 1.0]


# -------------------------------------------------------------- normalizing


def test_zero_length_and_broken_segments_are_dropped():
    turns = normalize_turns(
        [
            FakeSegment(1.0, 1.0, "speaker_00"),
            FakeSegment(0.0, 2.0, "speaker_00"),
            object(),
        ]
    )
    assert len(turns) == 1


def test_numeric_speaker_labels_are_formatted():
    turns = normalize_turns([FakeSegment(0.0, 1.0, 3)])
    assert turns[0].speaker == "speaker_03"


def test_an_empty_result_gives_no_turns():
    assert normalize_turns([]) == []
    assert normalize_turns(None) == []


# ------------------------------------------------------------ null diarizer


def test_the_null_diarizer_returns_nothing():
    assert NullDiarizer().diarize(Path("a.mov"), DiarizationOptions()) == []
