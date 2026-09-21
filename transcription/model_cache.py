"""Diarization model acquisition and local cache management.

Models live in the per-user application support folder, never in the project
and never in the output tree::

    ~/Library/Application Support/MLX Transcript/models/

Downloads come from the model's official public host. Both assets used here
are ungated: no account, no access token, and nothing is ever uploaded. Once a
file is on disk it is reused offline forever.

The network call sits behind an injectable ``downloader`` so tests exercise
every state without touching the network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

__all__ = [
    "APPLICATION_FOLDER_NAME",
    "ModelState",
    "ModelAsset",
    "ModelBundle",
    "ModelCacheError",
    "ModelIntegrityError",
    "DownloadCancelled",
    "ModelCache",
    "ModelProblem",
    "cleanup_partials",
    "default_cache_root",
    "file_digest",
    "SEGMENTATION_ASSET",
    "EMBEDDING_ASSET",
    "DEFAULT_BUNDLE",
]

logger = logging.getLogger(__name__)

APPLICATION_FOLDER_NAME = "MLX Transcript"

#: Read size used when hashing a cached model file.
_DIGEST_CHUNK = 1 << 20

ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]


class ModelState(str, Enum):
    """What the interface shows about the diarization model."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"

    @property
    def label(self) -> str:
        return {
            ModelState.NOT_DOWNLOADED: "Not downloaded",
            ModelState.DOWNLOADING: "Downloading",
            ModelState.READY: "Ready",
            ModelState.UNAVAILABLE: "Unavailable",
            ModelState.FAILED: "Failed",
        }[self]


class ModelCacheError(RuntimeError):
    """Raised when a model cannot be fetched or stored."""


class DownloadCancelled(ModelCacheError):
    """Raised when the user cancels a download in progress."""


class ModelIntegrityError(ModelCacheError):
    """Raised when a downloaded file is not the file it was supposed to be."""


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file on disk, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelAsset:
    """One file that has to be on disk before diarization can run.

    ``sha256`` pins the exact file. When it is set the download is checked
    against both the digest and the byte count before it is promoted into the
    cache, so a truncated or substituted file is never accepted. An empty
    digest means the asset is not pinned yet: the transfer is still checked
    against the server's ``Content-Length``, but nothing stronger.
    """

    name: str
    filename: str
    url: str
    size_bytes: int
    license_name: str
    source: str
    sha256: str = ""

    @property
    def size_megabytes(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def is_pinned(self) -> bool:
        """True when this asset's exact contents are known in advance."""
        return bool(self.sha256)

    def check(self, path: Path) -> None:
        """Raise :class:`ModelIntegrityError` unless ``path`` is this asset.

        An unpinned asset is only checked for being present and non-empty,
        because there is nothing trustworthy to compare it against.
        """
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise ModelIntegrityError(
                f"{self.name}: the downloaded file could not be read ({error})"
            ) from error

        if actual_size == 0:
            raise ModelIntegrityError(f"{self.name}: the downloaded file is empty")

        if not self.is_pinned:
            return

        if actual_size != self.size_bytes:
            raise ModelIntegrityError(
                f"{self.name}: expected {self.size_bytes} bytes but got "
                f"{actual_size}. The download was incomplete."
            )

        actual_digest = file_digest(path)
        if actual_digest.lower() != self.sha256.lower():
            raise ModelIntegrityError(
                f"{self.name}: the downloaded file does not match its expected "
                f"SHA-256 checksum."
            )


@dataclass(frozen=True)
class ModelProblem:
    """One cached file that is missing or is not what it should be."""

    asset: ModelAsset
    reason: str

    def __str__(self) -> str:
        return f"{self.asset.name}: {self.reason}"


@dataclass(frozen=True)
class ModelBundle:
    """The set of assets one diarization backend needs."""

    identifier: str
    display_name: str
    assets: tuple[ModelAsset, ...]

    @property
    def total_bytes(self) -> int:
        return sum(asset.size_bytes for asset in self.assets)

    @property
    def total_megabytes(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def licenses(self) -> tuple[str, ...]:
        seen: list[str] = []
        for asset in self.assets:
            if asset.license_name not in seen:
                seen.append(asset.license_name)
        return tuple(seen)

    def disclosure(self) -> str:
        """The sentence shown before any download starts."""
        licenses = ", ".join(self.licenses)
        return (
            f"{self.display_name} needs a one-time download of about "
            f"{self.total_megabytes:.0f} MB ({licenses}). No account or access "
            f"token is required, nothing is uploaded, and the files are reused "
            f"offline afterwards."
        )


SEGMENTATION_ASSET = ModelAsset(
    name="Speaker segmentation",
    filename="sherpa-onnx-pyannote-segmentation-3-0.onnx",
    url=(
        "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
        "/resolve/main/model.onnx"
    ),
    size_bytes=5_992_913,
    license_name="MIT",
    source="pyannote/segmentation-3.0, ONNX export by csukuangfj",
    # Recorded with scripts/record_model_digests.py. An empty value means the
    # asset is not pinned and only the transfer length is checked.
    sha256="",
)

EMBEDDING_ASSET = ModelAsset(
    name="Speaker embedding",
    filename="wespeaker_en_voxceleb_resnet34_LM.onnx",
    url=(
        "https://huggingface.co/csukuangfj/speaker-embedding-models"
        "/resolve/main/wespeaker_en_voxceleb_resnet34_LM.onnx"
    ),
    size_bytes=26_530_550,
    license_name="Apache-2.0",
    source="WeSpeaker VoxCeleb ResNet34-LM",
    sha256="",
)

DEFAULT_BUNDLE = ModelBundle(
    identifier="sherpa-onnx-pyannote-wespeaker",
    display_name="pyannote segmentation 3.0 with WeSpeaker embeddings",
    assets=(SEGMENTATION_ASSET, EMBEDDING_ASSET),
)


def default_cache_root() -> Path:
    """Return the per-user folder where model files are kept."""
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / APPLICATION_FOLDER_NAME
            / "models"
        )
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "mlx-transcript" / "models"


def _urllib_downloader(
    url: str,
    destination: Path,
    progress: Callable[[int, int], None] | None,
    cancelled: CancelCheck | None,
) -> None:
    """Fetch one file with the standard library, resumable only by retrying."""
    import urllib.error  # noqa: PLC0415 - kept out of import time
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, headers={"User-Agent": "MLX-Transcript"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with destination.open("wb") as handle:
                while True:
                    if cancelled is not None and cancelled():
                        raise DownloadCancelled("Download cancelled.")
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
                handle.flush()
                os.fsync(handle.fileno())
            if total and done != total:
                # A connection that closes early still ends the read loop
                # cleanly, so the short transfer has to be caught here.
                raise ModelIntegrityError(
                    f"The download stopped early: {done} of {total} bytes."
                )
    except DownloadCancelled:
        raise
    except ModelIntegrityError:
        raise
    except urllib.error.URLError as error:
        raise ModelCacheError(f"Could not reach the model host: {error.reason}") from error
    except OSError as error:
        raise ModelCacheError(f"Could not write the model file: {error}") from error


class ModelCache:
    """Knows where model files live, whether they are there, and how to get them."""

    def __init__(
        self,
        root: Path | None = None,
        downloader: Callable[..., None] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_cache_root()
        self._downloader = downloader or _urllib_downloader
        self.last_error: str = ""

    # ------------------------------------------------------------------ paths

    def path_for(self, asset: ModelAsset) -> Path:
        return self.root / asset.filename

    def paths_for(self, bundle: ModelBundle = DEFAULT_BUNDLE) -> list[Path]:
        return [self.path_for(asset) for asset in bundle.assets]

    def missing(self, bundle: ModelBundle = DEFAULT_BUNDLE) -> list[ModelAsset]:
        """Return the assets that still have to be downloaded."""
        return [
            asset
            for asset in bundle.assets
            if not self._is_present(self.path_for(asset))
        ]

    @staticmethod
    def _is_present(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    # ------------------------------------------------------------------ state

    def state(self, bundle: ModelBundle = DEFAULT_BUNDLE) -> ModelState:
        """Report whether the bundle is ready, missing, or in a failed state."""
        if self.last_error:
            return ModelState.FAILED
        return ModelState.READY if not self.missing(bundle) else ModelState.NOT_DOWNLOADED

    def clear_error(self) -> None:
        self.last_error = ""

    # --------------------------------------------------------------- fetching

    def ensure(
        self,
        bundle: ModelBundle = DEFAULT_BUNDLE,
        progress: ProgressCallback | None = None,
        cancelled: CancelCheck | None = None,
    ) -> list[Path]:
        """Make sure every asset is on disk and return the local paths.

        Each file is written to a temporary name in the cache folder and moved
        into place only once it is complete, so a cancelled or failed download
        never leaves a half-written model behind.
        """
        self.clear_error()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.last_error = f"Could not create the model folder: {error}"
            raise ModelCacheError(self.last_error) from error

        # A download that was killed part way leaves a .part file behind.
        # Clearing them first keeps the cache folder from growing forever and
        # makes a retry start from a clean slate.
        cleanup_partials(self.root)

        for asset in bundle.assets:
            target = self.path_for(asset)
            if self._is_present(target):
                continue
            if cancelled is not None and cancelled():
                raise DownloadCancelled("Download cancelled.")

            handle = tempfile.NamedTemporaryFile(
                dir=self.root,
                prefix=f".{asset.filename}.",
                suffix=".part",
                delete=False,
            )
            handle.close()
            temporary = Path(handle.name)

            def report(done: int, total: int, asset=asset) -> None:
                if progress is not None:
                    progress(asset.name, done, total or asset.size_bytes)

            try:
                self._downloader(asset.url, temporary, report, cancelled)
                # Verified before it is promoted, never after. A file that
                # reaches its final name is a file the application will trust
                # offline from then on.
                asset.check(temporary)
                os.replace(temporary, target)
            except DownloadCancelled:
                temporary.unlink(missing_ok=True)
                raise
            except ModelCacheError as error:
                temporary.unlink(missing_ok=True)
                self.last_error = str(error)
                raise
            except Exception as error:
                temporary.unlink(missing_ok=True)
                self.last_error = f"{asset.name}: {error}"
                raise ModelCacheError(self.last_error) from error

        return self.paths_for(bundle)

    # ------------------------------------------------------------ verifying

    def verify(self, bundle: ModelBundle = DEFAULT_BUNDLE) -> list[ModelProblem]:
        """Check every cached file and report what is wrong with it.

        This is what the Check or Repair action runs. It answers the one
        question the status line cannot: the files are there, but are they the
        right files? A model that was truncated by an interrupted download
        looks present and fails only when diarization tries to load it.
        """
        problems: list[ModelProblem] = []
        for asset in bundle.assets:
            path = self.path_for(asset)
            if not path.exists():
                problems.append(ModelProblem(asset, "not downloaded"))
                continue
            try:
                asset.check(path)
            except ModelIntegrityError as error:
                problems.append(ModelProblem(asset, str(error).split(": ", 1)[-1]))
            except OSError as error:
                problems.append(ModelProblem(asset, f"could not be read ({error})"))
        return problems

    def discard(self, problems: Sequence[ModelProblem]) -> int:
        """Delete the cached files behind a set of problems. Returns the count."""
        removed = 0
        for problem in problems:
            path = self.path_for(problem.asset)
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError as error:
                logger.warning("Could not remove %s: %s", path, error)
        return removed

    def repair(
        self,
        bundle: ModelBundle = DEFAULT_BUNDLE,
        progress: ProgressCallback | None = None,
        cancelled: CancelCheck | None = None,
    ) -> list[Path]:
        """Throw away anything that does not verify, then fetch it again."""
        self.discard(self.verify(bundle))
        return self.ensure(bundle, progress=progress, cancelled=cancelled)

    # --------------------------------------------------------------- clean up

    def remove(self, bundle: ModelBundle = DEFAULT_BUNDLE) -> None:
        """Delete the cached files for a bundle. Only ever called by the user."""
        for path in self.paths_for(bundle):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def cache_size_bytes(self) -> int:
        """Return how much disk the cache is using."""
        if not self.root.is_dir():
            return 0
        return sum(
            path.stat().st_size for path in self.root.rglob("*") if path.is_file()
        )

    def describe(self) -> str:
        """One line for the interface: where the models are kept."""
        return str(self.root)


def cleanup_partials(root: Path) -> int:
    """Remove leftover .part files, for example after a crash."""
    removed = 0
    if not Path(root).is_dir():
        return removed
    for path in Path(root).glob("*.part"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
