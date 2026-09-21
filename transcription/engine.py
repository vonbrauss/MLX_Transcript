"""MLX Whisper transcription engine.

``mlx_whisper`` is imported lazily inside :meth:`TranscriptionEngine.transcribe`
rather than at module import time, so the application shell, the folder scan,
and the whole test suite run without model weights or MLX installed. The first
call pays for the model download and load; every later call in the same batch
reuses the process-resident model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "DEFAULT_MODEL",
    "LARGE_V3_MODEL",
    "TURBO_MODEL",
    "MODEL_CHOICES",
    "AUTO_DETECT_LANGUAGE",
    "model_cache_root",
    "model_is_cached",
    "TranscriptionError",
    "TranscriptionOptions",
    "TranscriptionResult",
    "TranscriptionEngine",
]

LARGE_V3_MODEL = "mlx-community/whisper-large-v3-mlx"
TURBO_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL = LARGE_V3_MODEL

#: Label shown in the model picker mapped to the MLX model repository.
MODEL_CHOICES: dict[str, str] = {
    "Whisper Large v3 Turbo": TURBO_MODEL,
    "Whisper Large v3": LARGE_V3_MODEL,
}

#: Sentinel language value meaning "let Whisper decide".
AUTO_DETECT_LANGUAGE = ""

TranscribeCallable = Callable[..., dict[str, Any]]


def model_cache_root() -> Path:
    """Return the folder Hugging Face keeps downloaded model weights in.

    Resolved from the same environment variables ``huggingface_hub`` reads, so
    the path shown to the user is the path the weights actually land in. This
    is deliberately computed without importing the library, which would pull
    the whole transcription stack in just to answer a question about a folder.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "huggingface" / "hub"


def model_is_cached(repository: str, root: Path | None = None) -> bool:
    """True when a model repository already has weights on this Mac.

    Used to decide whether the first-download disclosure is worth showing. A
    partially fetched repository counts as not cached, so the disclosure is
    shown again rather than the download silently resuming behind a progress
    bar that cannot describe it.
    """
    folder = (root or model_cache_root()) / (
        "models--" + str(repository).replace("/", "--")
    )
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    for revision in snapshots.iterdir():
        if not revision.is_dir():
            continue
        if any(item.is_file() or item.is_symlink() for item in revision.iterdir()):
            return True
    return False


class TranscriptionError(RuntimeError):
    """Raised when one clip cannot be transcribed."""


@dataclass(frozen=True)
class TranscriptionOptions:
    """Everything the engine needs beyond the media path itself.

    The thresholds are the anti-hallucination controls. ``temperature`` is held
    at zero so decoding is deterministic, ``condition_on_previous_text`` is off
    so one bad segment cannot poison the rest of a clip, and word timestamps
    stay on because the hallucination silence threshold depends on them.
    """

    model: str = DEFAULT_MODEL
    language: str = "en"
    temperature: float = 0.0
    hallucination_silence_threshold: float = 1.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    task: str = "transcribe"
    detect_speakers: bool = False

    @property
    def auto_detect_language(self) -> bool:
        """True when Whisper should detect the language itself."""
        return not (self.language or "").strip()

    def as_whisper_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for ``mlx_whisper.transcribe``.

        ``language`` is omitted entirely for Auto Detect. Passing an empty
        string instead would be read as a language code and fail.
        """
        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self.model,
            "task": self.task,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "word_timestamps": self.word_timestamps,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
        }
        if not self.auto_detect_language:
            kwargs["language"] = self.language.strip()
        return kwargs


@dataclass
class TranscriptionResult:
    """Raw transcript segments for one clip.

    ``segments`` keeps the shape MLX Whisper returns: dictionaries with
    ``start``, ``end``, ``text``, and, when word timestamps are on, ``words``.
    Milestone 3 adds a ``speaker`` key per segment and fills :attr:`speakers`
    with renameable labels.
    """

    source: Path
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    speakers: list[str] = field(default_factory=list)

    @property
    def has_speakers(self) -> bool:
        return bool(self.speakers)

    def rename_speaker(self, old_label: str, new_label: str) -> None:
        """Rename one speaker label across every segment."""
        self.speakers = [
            new_label if label == old_label else label for label in self.speakers
        ]
        for segment in self.segments:
            if segment.get("speaker") == old_label:
                segment["speaker"] = new_label


class TranscriptionEngine:
    """Holds the loaded MLX Whisper model for the duration of a batch.

    ``transcribe_fn`` exists so tests and the headless harness can inject a
    stand-in. Left as None, the real ``mlx_whisper.transcribe`` is imported on
    first use.
    """

    def __init__(
        self,
        options: TranscriptionOptions | None = None,
        transcribe_fn: TranscribeCallable | None = None,
    ) -> None:
        self.options = options or TranscriptionOptions()
        self._transcribe_fn = transcribe_fn

    @property
    def is_loaded(self) -> bool:
        """True once the transcription callable has been resolved."""
        return self._transcribe_fn is not None

    def load(self) -> TranscribeCallable:
        """Import MLX Whisper once and keep the callable for the batch.

        The model weights themselves are fetched by the first ``transcribe``
        call, which is why the UI shows a loading status around it.
        """
        if self._transcribe_fn is None:
            try:
                import mlx_whisper  # noqa: PLC0415 - deliberately lazy
            except ImportError as error:  # pragma: no cover - environment
                raise TranscriptionError(
                    f"The transcription engine could not load: {error}"
                ) from error
            self._transcribe_fn = mlx_whisper.transcribe
        return self._transcribe_fn

    def transcribe(self, source: Path) -> TranscriptionResult:
        """Transcribe one media file in this process.

        Nothing is written to disk here: MLX Whisper is called directly rather
        than through its command line, so no JSON, SRT, VTT, TSV, or temporary
        audio files are produced.
        """
        source = Path(source)
        transcribe_fn = self.load()
        kwargs = self.options.as_whisper_kwargs()

        try:
            payload = transcribe_fn(str(source), **kwargs)
        except TranscriptionError:
            raise
        except Exception as error:
            raise TranscriptionError(f"{source.name}: {error}") from error

        if not isinstance(payload, dict):
            raise TranscriptionError(
                f"{source.name}: unexpected transcription result"
            )

        segments = list(payload.get("segments") or [])
        language = str(payload.get("language") or self.options.language or "")
        return TranscriptionResult(
            source=source,
            segments=segments,
            language=language,
        )
