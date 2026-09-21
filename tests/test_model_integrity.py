"""A downloaded model is verified before it is trusted offline forever.

A truncated download used to be promoted into the cache, reported as Ready,
and then fail on every clip with an ONNX error the user could do nothing
about. These cover the length and digest checks, the repair path, and the
containers list that decides what the application will even look at.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from transcription.discovery import (
    MEDIA_EXTENSIONS,
    MEDIA_NAME_FILTER,
    is_supported_media,
)
from transcription.model_cache import (
    DEFAULT_BUNDLE,
    EMBEDDING_ASSET,
    SEGMENTATION_ASSET,
    ModelAsset,
    ModelBundle,
    ModelCache,
    ModelCacheError,
    ModelIntegrityError,
    ModelState,
    file_digest,
)

PAYLOAD = b"a pretend onnx model"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()

PINNED = ModelAsset(
    name="Pinned asset",
    filename="pinned.onnx",
    url="https://example.invalid/pinned.onnx",
    size_bytes=len(PAYLOAD),
    license_name="MIT",
    source="test",
    sha256=DIGEST,
)
PINNED_BUNDLE = ModelBundle(
    identifier="pinned", display_name="Pinned bundle", assets=(PINNED,)
)


class Downloader:
    """Writes whatever content the test asks for."""

    def __init__(self, content: bytes = PAYLOAD) -> None:
        self.content = content
        self.calls = 0

    def __call__(self, url, destination, progress, cancelled):
        self.calls += 1
        Path(destination).write_bytes(self.content)


def cache(tmp_path: Path, downloader=None) -> ModelCache:
    return ModelCache(root=tmp_path / "models", downloader=downloader or Downloader())


# --------------------------------------------------------------- the checks


def test_a_matching_download_is_promoted(tmp_path):
    store = cache(tmp_path)
    (path,) = store.ensure(PINNED_BUNDLE)

    assert path.read_bytes() == PAYLOAD
    assert store.state(PINNED_BUNDLE) is ModelState.READY


def test_a_truncated_download_is_never_promoted(tmp_path):
    store = cache(tmp_path, Downloader(PAYLOAD[:5]))

    with pytest.raises(ModelCacheError):
        store.ensure(PINNED_BUNDLE)

    assert not store.path_for(PINNED).exists()
    assert store.state(PINNED_BUNDLE) is ModelState.FAILED


def test_a_truncated_download_says_the_size_was_wrong(tmp_path):
    store = cache(tmp_path, Downloader(PAYLOAD[:5]))

    with pytest.raises(ModelCacheError) as caught:
        store.ensure(PINNED_BUNDLE)

    assert "incomplete" in str(caught.value)


def test_a_substituted_file_of_the_right_length_is_rejected(tmp_path):
    store = cache(tmp_path, Downloader(b"X" * len(PAYLOAD)))

    with pytest.raises(ModelCacheError) as caught:
        store.ensure(PINNED_BUNDLE)

    assert "SHA-256" in str(caught.value)
    assert not store.path_for(PINNED).exists()


def test_an_empty_download_is_rejected(tmp_path):
    store = cache(tmp_path, Downloader(b""))

    with pytest.raises(ModelCacheError):
        store.ensure(PINNED_BUNDLE)


def test_an_unpinned_asset_only_has_to_be_non_empty(tmp_path):
    unpinned = replace(PINNED, sha256="")
    bundle = ModelBundle(identifier="loose", display_name="Loose", assets=(unpinned,))
    store = cache(tmp_path, Downloader(b"anything at all"))

    (path,) = store.ensure(bundle)

    assert path.exists()


def test_the_shipped_assets_declare_their_provenance():
    for asset in DEFAULT_BUNDLE.assets:
        assert asset.url.startswith("https://huggingface.co/")
        assert asset.license_name
        assert asset.source
        # sha256 may be empty until recorded, but the field has to exist.
        assert isinstance(asset.sha256, str)


def test_file_digest_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(PAYLOAD)

    assert file_digest(path) == DIGEST


def test_check_reports_a_missing_file(tmp_path):
    with pytest.raises(ModelIntegrityError):
        PINNED.check(tmp_path / "absent.onnx")


# -------------------------------------------------------------- verify and repair


def test_verify_is_quiet_when_everything_matches(tmp_path):
    store = cache(tmp_path)
    store.ensure(PINNED_BUNDLE)

    assert store.verify(PINNED_BUNDLE) == []


def test_verify_reports_a_missing_model(tmp_path):
    store = cache(tmp_path)

    problems = store.verify(PINNED_BUNDLE)

    assert len(problems) == 1
    assert "not downloaded" in problems[0].reason


def test_verify_catches_a_file_that_was_damaged_after_download(tmp_path):
    store = cache(tmp_path)
    (path,) = store.ensure(PINNED_BUNDLE)
    path.write_bytes(b"corrupted beyond recognition")

    problems = store.verify(PINNED_BUNDLE)

    assert len(problems) == 1
    assert str(problems[0]).startswith("Pinned asset")


def test_repair_replaces_a_damaged_file(tmp_path):
    downloader = Downloader()
    store = cache(tmp_path, downloader)
    (path,) = store.ensure(PINNED_BUNDLE)
    path.write_bytes(b"junk")
    assert store.verify(PINNED_BUNDLE)

    store.repair(PINNED_BUNDLE)

    assert path.read_bytes() == PAYLOAD
    assert store.verify(PINNED_BUNDLE) == []
    assert downloader.calls == 2


def test_repair_leaves_a_healthy_cache_alone(tmp_path):
    downloader = Downloader()
    store = cache(tmp_path, downloader)
    store.ensure(PINNED_BUNDLE)

    store.repair(PINNED_BUNDLE)

    assert downloader.calls == 1


def test_discard_removes_only_the_problem_files(tmp_path):
    store = cache(tmp_path)
    store.ensure(PINNED_BUNDLE)
    path = store.path_for(PINNED)
    path.write_bytes(b"junk")

    removed = store.discard(store.verify(PINNED_BUNDLE))

    assert removed == 1
    assert not path.exists()


def test_stale_partials_are_cleared_before_a_download(tmp_path):
    store = cache(tmp_path)
    store.root.mkdir(parents=True)
    stale = store.root / ".pinned.onnx.abc123.part"
    stale.write_bytes(b"left over from a crash")

    store.ensure(PINNED_BUNDLE)

    assert not stale.exists()
    assert store.path_for(PINNED).exists()


# ---------------------------------------------------------- media containers


@pytest.mark.parametrize(
    "extension",
    [".mxf", ".avi", ".mts", ".m2ts", ".wmv"],
)
def test_the_editorial_containers_are_recognised(extension):
    assert extension in MEDIA_EXTENSIONS
    assert is_supported_media(Path(f"clip{extension}"))


@pytest.mark.parametrize(
    "extension",
    [".mxf", ".MXF", ".Avi", ".MTS", ".m2ts", ".WMV", ".mov", ".WAV"],
)
def test_container_matching_ignores_case(extension):
    assert is_supported_media(Path(f"clip{extension}"))


@pytest.mark.parametrize("extension", [".pdf", ".xlsx", ".txt", ".jpg", ".zip", ""])
def test_non_media_is_still_rejected(extension):
    assert not is_supported_media(Path(f"file{extension}"))


def test_the_original_containers_are_all_still_supported():
    for extension in (
        ".aac", ".aif", ".aiff", ".flac", ".m4a", ".m4v", ".mkv", ".mov",
        ".mp3", ".mp4", ".mpeg", ".mpg", ".ogg", ".opus", ".wav", ".webm",
        ".wma",
    ):
        assert extension in MEDIA_EXTENSIONS


def test_the_file_dialog_filter_lists_every_container():
    for extension in MEDIA_EXTENSIONS:
        assert f"*{extension}" in MEDIA_NAME_FILTER


def test_the_shipped_bundle_still_holds_both_assets():
    assert DEFAULT_BUNDLE.assets == (SEGMENTATION_ASSET, EMBEDDING_ASSET)
