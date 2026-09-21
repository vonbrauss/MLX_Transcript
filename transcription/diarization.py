"""Local speaker diarization.

The backend is sherpa-onnx running pyannote segmentation 3.0 with WeSpeaker
embeddings through ONNX Runtime. It is CPU-only, needs no PyTorch and no CUDA,
and nothing leaves the Mac. ``sherpa_onnx`` is imported lazily inside
:meth:`SherpaOnnxDiarizer.load`, so launching the application and running the
tests never loads a model or even touches the package.

Audio is decoded to 16 kHz mono through an ffmpeg pipe straight into memory.
No extracted audio file is ever written, least of all into the output tree.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .model_cache import DEFAULT_BUNDLE, ModelBundle, ModelCache, ModelCacheError
from .speakers import SpeakerTurn
from app.runtime import bundled_binary

__all__ = [
    "SpeakerCountMode",
    "DiarizationOptions",
    "DiarizationError",
    "DiarizationBackend",
    "SherpaOnnxDiarizer",
    "NullDiarizer",
    "backend_available",
    "decode_mono_16k",
    "SAMPLE_RATE",
]

SAMPLE_RATE = 16_000

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class DiarizationError(RuntimeError):
    """Raised when diarization cannot run or fails for one clip."""


class SpeakerCountMode(str, Enum):
    """How many speakers the backend should look for."""

    AUTOMATIC = "automatic"
    EXACT = "exact"

    @property
    def label(self) -> str:
        return {
            SpeakerCountMode.AUTOMATIC: "Automatic",
            SpeakerCountMode.EXACT: "Exact number",
        }[self]


@dataclass(frozen=True)
class DiarizationOptions:
    """Everything the diarizer needs beyond the media path.

    sherpa-onnx clusters with ``FastClusteringConfig``, which takes either a
    fixed cluster count or a distance threshold. It has no minimum or maximum
    speaker constraint, so those controls are deliberately absent rather than
    faked: ``AUTOMATIC`` tunes :attr:`clustering_threshold` and ``EXACT`` pins
    :attr:`exact_speakers`.
    """

    enabled: bool = False
    count_mode: SpeakerCountMode = SpeakerCountMode.AUTOMATIC
    exact_speakers: int = 2
    clustering_threshold: float = 0.5
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5
    nearest_tolerance: float = 0.75
    merge_gap: float = 1.0
    threads: int = 4

    @property
    def num_clusters(self) -> int:
        """The cluster count sherpa-onnx wants, with -1 meaning automatic."""
        if self.count_mode is SpeakerCountMode.EXACT:
            return max(1, int(self.exact_speakers))
        return -1

    def as_backend_kwargs(self) -> dict[str, Any]:
        """The clustering settings, kept small and explicit for testing."""
        kwargs: dict[str, Any] = {"num_clusters": self.num_clusters}
        if self.count_mode is SpeakerCountMode.AUTOMATIC:
            kwargs["threshold"] = float(self.clustering_threshold)
        return kwargs


class DiarizationBackend(Protocol):
    """What the pipeline needs from any diarization implementation."""

    def diarize(
        self,
        source: Path,
        options: DiarizationOptions,
        progress: ProgressCallback | None = ...,
        cancelled: CancelCheck | None = ...,
    ) -> list[SpeakerTurn]:
        ...


class NullDiarizer:
    """Returns no turns. Used when speaker detection is switched off."""

    def diarize(
        self,
        source: Path,
        options: DiarizationOptions,
        progress: ProgressCallback | None = None,
        cancelled: CancelCheck | None = None,
    ) -> list[SpeakerTurn]:
        return []


def backend_available() -> bool:
    """True when sherpa-onnx can be imported, without importing it here."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("sherpa_onnx") is not None


def decode_mono_16k(
    source: Path,
    ffmpeg: str | None = None,
    timeout: float = 1800.0,
):
    """Decode any supported media to a 16 kHz mono float array in memory.

    Returns a numpy array of float32 samples in the range -1 to 1. Nothing is
    written to disk: ffmpeg streams raw samples over a pipe.
    """
    import numpy  # noqa: PLC0415 - already a transitive dependency

    ffmpeg = bundled_binary(ffmpeg or "ffmpeg")
    command = [
        ffmpeg,
        "-nostdin",
        "-v", "error",
        "-i", str(source),
        "-map", "a:0?",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise DiarizationError(f"ffmpeg was not found: {ffmpeg}") from error
    except subprocess.TimeoutExpired as error:
        raise DiarizationError(f"ffmpeg timed out on {Path(source).name}") from error
    except subprocess.CalledProcessError as error:
        message = (error.stderr or b"").decode("utf-8", "ignore").strip()
        raise DiarizationError(
            f"{Path(source).name}: could not decode audio. {message}".strip()
        ) from error

    if not result.stdout:
        raise DiarizationError(f"{Path(source).name}: no audio track was found.")
    return numpy.frombuffer(result.stdout, dtype=numpy.float32)


class SherpaOnnxDiarizer:
    """sherpa-onnx offline speaker diarization, loaded on first use.

    ``build_fn`` and ``decode_fn`` are injection points: the tests hand in
    stand-ins so no model is ever downloaded or loaded.
    """

    def __init__(
        self,
        cache: ModelCache | None = None,
        bundle: ModelBundle = DEFAULT_BUNDLE,
        build_fn: Callable[..., Any] | None = None,
        decode_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.cache = cache or ModelCache()
        self.bundle = bundle
        self._build_fn = build_fn
        self._decode_fn = decode_fn or decode_mono_16k
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[Any, ...] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # --------------------------------------------------------------- loading

    def ensure_models(
        self,
        progress: Callable[[str, int, int], None] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> list[Path]:
        """Download the models if they are not cached yet."""
        return self.cache.ensure(self.bundle, progress=progress, cancelled=cancelled)

    def _default_build(self, options: DiarizationOptions, paths: Sequence[Path]) -> Any:
        """Build the real sherpa-onnx pipeline. Imported only when called."""
        import sherpa_onnx  # noqa: PLC0415 - deliberately lazy

        segmentation, embedding = paths[0], paths[1]
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation)
                ),
                num_threads=options.threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding),
                num_threads=options.threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(**options.as_backend_kwargs()),
            min_duration_on=options.min_duration_on,
            min_duration_off=options.min_duration_off,
        )
        if not config.validate():
            raise DiarizationError(
                "The diarization model files could not be validated."
            )
        return sherpa_onnx.OfflineSpeakerDiarization(config)

    def load(self, options: DiarizationOptions) -> Any:
        """Return the pipeline, rebuilding it when the clustering changes."""
        key = tuple(sorted(options.as_backend_kwargs().items())) + (
            options.min_duration_on,
            options.min_duration_off,
            options.threads,
        )
        if self._pipeline is not None and self._pipeline_key == key:
            return self._pipeline

        try:
            paths = self.ensure_models()
        except ModelCacheError as error:
            raise DiarizationError(str(error)) from error

        build = self._build_fn or self._default_build
        try:
            self._pipeline = build(options, paths)
        except DiarizationError:
            raise
        except ImportError as error:
            raise DiarizationError(
                "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            ) from error
        except Exception as error:
            raise DiarizationError(f"Could not load the diarization model: {error}") from error

        self._pipeline_key = key
        return self._pipeline

    # ------------------------------------------------------------ diarizing

    def diarize(
        self,
        source: Path,
        options: DiarizationOptions,
        progress: ProgressCallback | None = None,
        cancelled: CancelCheck | None = None,
    ) -> list[SpeakerTurn]:
        """Return the speaker turns for one clip, sorted by start time."""
        source = Path(source)
        pipeline = self.load(options)

        samples = self._decode_fn(source)
        if cancelled is not None and cancelled():
            raise DiarizationError("Cancelled before diarization started.")

        try:
            if progress is not None:
                result = pipeline.process(samples, callback=_ProgressAdapter(progress))
            else:
                result = pipeline.process(samples)
        except Exception as error:
            raise DiarizationError(f"{source.name}: {error}") from error

        return normalize_turns(result)


class _ProgressAdapter:
    """sherpa-onnx hands the callback processed and total chunk counts."""

    def __init__(self, progress: ProgressCallback) -> None:
        self._progress = progress

    def __call__(self, processed: int, total: int, *_extra: object) -> int:
        if total:
            self._progress(min(1.0, float(processed) / float(total)))
        return 0


def normalize_turns(result: Any) -> list[SpeakerTurn]:
    """Turn a backend result into sorted :class:`SpeakerTurn` objects."""
    raw = result
    sort_by_start_time = getattr(result, "sort_by_start_time", None)
    if callable(sort_by_start_time):
        raw = sort_by_start_time()

    turns: list[SpeakerTurn] = []
    for segment in raw or ():
        try:
            start = float(getattr(segment, "start"))
            end = float(getattr(segment, "end"))
            speaker = getattr(segment, "speaker")
        except (AttributeError, TypeError, ValueError):
            continue
        label = speaker if isinstance(speaker, str) else f"speaker_{int(speaker):02d}"
        if end > start:
            turns.append(SpeakerTurn(start=start, end=end, speaker=label))

    turns.sort(key=lambda turn: (turn.start, turn.end))
    return turns
