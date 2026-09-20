"""Engine option construction, lazy loading, and result shaping.

Nothing here downloads weights or imports MLX: the transcription callable is
injected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from transcription.engine import (
    AUTO_DETECT_LANGUAGE,
    DEFAULT_MODEL,
    LARGE_V3_MODEL,
    MODEL_CHOICES,
    TURBO_MODEL,
    TranscriptionEngine,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)

CLIP = Path("/Source/Day 01/A001.mov")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingTranscriber:
    """Stand-in for ``mlx_whisper.transcribe`` that records its arguments."""

    def __init__(self, payload=None):
        self.calls: list[tuple[str, dict]] = []
        self.payload = payload if payload is not None else {"segments": []}

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


# ---------------------------------------------------------------- defaults


def test_large_v3_is_the_default_model():
    assert DEFAULT_MODEL == LARGE_V3_MODEL == "mlx-community/whisper-large-v3-mlx"
    assert DEFAULT_MODEL in MODEL_CHOICES.values()


def test_both_models_remain_selectable():
    assert set(MODEL_CHOICES.values()) == {
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/whisper-large-v3-mlx",
    }


# ------------------------------------------------------- option construction


def test_options_carry_the_anti_hallucination_settings():
    kwargs = TranscriptionOptions().as_whisper_kwargs()
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["compression_ratio_threshold"] == 2.4
    assert kwargs["logprob_threshold"] == -1.0
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["word_timestamps"] is True


def test_options_transcribe_rather_than_translate():
    assert TranscriptionOptions().as_whisper_kwargs()["task"] == "transcribe"


def test_options_pass_the_selected_model_repository():
    options = TranscriptionOptions(model="mlx-community/whisper-large-v3-mlx")
    kwargs = options.as_whisper_kwargs()
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"


def test_options_pass_the_hallucination_silence_threshold():
    options = TranscriptionOptions(hallucination_silence_threshold=2.5)
    assert options.as_whisper_kwargs()["hallucination_silence_threshold"] == 2.5


def test_explicit_language_is_passed_through():
    kwargs = TranscriptionOptions(language="es").as_whisper_kwargs()
    assert kwargs["language"] == "es"


@pytest.mark.parametrize("language", [AUTO_DETECT_LANGUAGE, "", "   ", None])
def test_auto_detect_omits_the_language_argument(language):
    options = TranscriptionOptions(language=language or "")
    kwargs = options.as_whisper_kwargs()
    assert "language" not in kwargs
    assert options.auto_detect_language is True


def test_advanced_thresholds_flow_into_the_kwargs():
    options = TranscriptionOptions(
        no_speech_threshold=0.45,
        logprob_threshold=-0.8,
        hallucination_silence_threshold=3.0,
    )
    kwargs = options.as_whisper_kwargs()
    assert kwargs["no_speech_threshold"] == 0.45
    assert kwargs["logprob_threshold"] == -0.8
    assert kwargs["hallucination_silence_threshold"] == 3.0


# -------------------------------------------------------------- lazy loading


def test_importing_the_engine_does_not_import_mlx_whisper():
    """A fresh interpreter importing the engine must not pull MLX in."""
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "import transcription.engine;"
        "print('mlx_whisper' in sys.modules)" % PROJECT_ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_engine_is_not_loaded_until_asked():
    engine = TranscriptionEngine()
    assert engine.is_loaded is False


def test_injected_callable_counts_as_loaded():
    engine = TranscriptionEngine(transcribe_fn=RecordingTranscriber())
    assert engine.is_loaded is True
    assert engine.load() is engine._transcribe_fn


# ------------------------------------------------------------- transcription


def test_transcribe_calls_whisper_with_the_path_and_options():
    transcriber = RecordingTranscriber(
        {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}], "language": "en"}
    )
    engine = TranscriptionEngine(
        TranscriptionOptions(language="en"), transcribe_fn=transcriber
    )
    result = engine.transcribe(CLIP)

    audio, kwargs = transcriber.calls[0]
    assert audio == str(CLIP)
    assert kwargs["language"] == "en"
    assert isinstance(result, TranscriptionResult)
    assert result.segments[0]["text"] == "Hello."
    assert result.language == "en"


def test_transcribe_keeps_wording_exactly_as_returned():
    spoken = "Um, we we ran the play, uh, twice."
    transcriber = RecordingTranscriber({"segments": [{"start": 0.0, "text": spoken}]})
    engine = TranscriptionEngine(transcribe_fn=transcriber)
    assert engine.transcribe(CLIP).segments[0]["text"] == spoken


def test_transcribe_wraps_backend_failures():
    transcriber = RecordingTranscriber(RuntimeError("model blew up"))
    engine = TranscriptionEngine(transcribe_fn=transcriber)
    with pytest.raises(TranscriptionError) as error:
        engine.transcribe(CLIP)
    assert "A001.mov" in str(error.value)
    assert "model blew up" in str(error.value)


def test_transcribe_rejects_an_unexpected_payload():
    engine = TranscriptionEngine(transcribe_fn=lambda audio, **kwargs: "nonsense")
    with pytest.raises(TranscriptionError):
        engine.transcribe(CLIP)


def test_missing_segments_produce_an_empty_result():
    engine = TranscriptionEngine(transcribe_fn=lambda audio, **kwargs: {})
    result = engine.transcribe(CLIP)
    assert result.segments == []


def test_model_is_loaded_once_per_batch():
    transcriber = RecordingTranscriber()
    engine = TranscriptionEngine(transcribe_fn=transcriber)
    engine.transcribe(CLIP)
    engine.transcribe(CLIP)
    assert len(transcriber.calls) == 2
    assert engine.load() is transcriber
