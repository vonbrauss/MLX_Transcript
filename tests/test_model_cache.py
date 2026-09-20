"""Model cache state, download success, cancellation, and failure.

The downloader is always injected, so nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription.model_cache import (
    APPLICATION_FOLDER_NAME,
    DEFAULT_BUNDLE,
    EMBEDDING_ASSET,
    SEGMENTATION_ASSET,
    DownloadCancelled,
    ModelCache,
    ModelCacheError,
    ModelState,
    cleanup_partials,
    default_cache_root,
)


class FakeDownloader:
    """Writes a placeholder file, or fails, or reports cancellation."""

    def __init__(self, error: Exception | None = None, content: bytes = b"model") -> None:
        self.calls: list[tuple[str, Path]] = []
        self.error = error
        self.content = content

    def __call__(self, url, destination, progress, cancelled):
        self.calls.append((url, Path(destination)))
        if cancelled is not None and cancelled():
            raise DownloadCancelled("Download cancelled.")
        if self.error is not None:
            raise self.error
        if progress is not None:
            progress(len(self.content), len(self.content))
        Path(destination).write_bytes(self.content)


def cache(tmp_path: Path, downloader=None) -> ModelCache:
    return ModelCache(root=tmp_path / "models", downloader=downloader or FakeDownloader())


# ------------------------------------------------------------------ location


def test_the_cache_lives_in_a_per_user_folder(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    root = default_cache_root()
    assert "Library/Application Support" in str(root)
    assert APPLICATION_FOLDER_NAME in str(root)
    assert root.name == "models"


def test_the_cache_is_never_inside_the_project(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert "MLX_Transcript" not in str(default_cache_root())


def test_a_custom_root_is_respected(tmp_path):
    assert cache(tmp_path).root == tmp_path / "models"


def test_paths_are_named_after_the_assets(tmp_path):
    paths = cache(tmp_path).paths_for(DEFAULT_BUNDLE)
    assert [path.name for path in paths] == [
        SEGMENTATION_ASSET.filename,
        EMBEDDING_ASSET.filename,
    ]


# -------------------------------------------------------------------- state


def test_a_fresh_cache_reports_not_downloaded(tmp_path):
    assert cache(tmp_path).state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED


def test_a_full_cache_reports_ready(tmp_path):
    store = cache(tmp_path)
    store.ensure(DEFAULT_BUNDLE)
    assert store.state(DEFAULT_BUNDLE) is ModelState.READY
    assert store.missing(DEFAULT_BUNDLE) == []


def test_a_partially_filled_cache_is_still_not_downloaded(tmp_path):
    store = cache(tmp_path)
    store.root.mkdir(parents=True)
    store.path_for(SEGMENTATION_ASSET).write_bytes(b"model")
    assert store.state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED
    assert store.missing(DEFAULT_BUNDLE) == [EMBEDDING_ASSET]


def test_an_empty_file_does_not_count_as_downloaded(tmp_path):
    store = cache(tmp_path)
    store.root.mkdir(parents=True)
    for path in store.paths_for(DEFAULT_BUNDLE):
        path.write_bytes(b"")
    assert store.state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED


def test_state_labels_are_the_ones_the_interface_shows():
    assert ModelState.NOT_DOWNLOADED.label == "Not downloaded"
    assert ModelState.DOWNLOADING.label == "Downloading"
    assert ModelState.READY.label == "Ready"
    assert ModelState.UNAVAILABLE.label == "Unavailable"
    assert ModelState.FAILED.label == "Failed"


# ---------------------------------------------------------------- disclosure


def test_the_disclosure_states_the_size_and_licenses():
    text = DEFAULT_BUNDLE.disclosure()
    assert "31 MB" in text
    assert "MIT" in text and "Apache-2.0" in text
    assert "No account or access token is required" in text
    assert "nothing is uploaded" in text


def test_the_bundle_totals_match_the_assets():
    assert DEFAULT_BUNDLE.total_bytes == (
        SEGMENTATION_ASSET.size_bytes + EMBEDDING_ASSET.size_bytes
    )
    assert 31 < DEFAULT_BUNDLE.total_megabytes < 32


def test_the_models_come_from_their_official_public_host():
    for asset in DEFAULT_BUNDLE.assets:
        assert asset.url.startswith("https://huggingface.co/")


# ----------------------------------------------------------------- download


def test_a_successful_download_stores_every_asset(tmp_path):
    downloader = FakeDownloader()
    store = cache(tmp_path, downloader)
    paths = store.ensure(DEFAULT_BUNDLE)

    assert len(downloader.calls) == 2
    assert all(path.is_file() for path in paths)
    assert store.state(DEFAULT_BUNDLE) is ModelState.READY


def test_a_second_run_reuses_the_cache_offline(tmp_path):
    downloader = FakeDownloader()
    store = cache(tmp_path, downloader)
    store.ensure(DEFAULT_BUNDLE)
    store.ensure(DEFAULT_BUNDLE)
    assert len(downloader.calls) == 2  # not four


def test_progress_is_reported_per_asset(tmp_path):
    seen: list[tuple[str, int, int]] = []
    store = cache(tmp_path)
    store.ensure(DEFAULT_BUNDLE, progress=lambda name, done, total: seen.append((name, done, total)))
    assert {row[0] for row in seen} == {
        SEGMENTATION_ASSET.name,
        EMBEDDING_ASSET.name,
    }


def test_a_cancelled_download_leaves_nothing_behind(tmp_path):
    store = cache(tmp_path)
    with pytest.raises(DownloadCancelled):
        store.ensure(DEFAULT_BUNDLE, cancelled=lambda: True)

    assert store.state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED
    assert list(store.root.glob("*.part")) == []
    assert not store.path_for(SEGMENTATION_ASSET).exists()


def test_a_failed_download_reports_and_cleans_up(tmp_path):
    downloader = FakeDownloader(error=ModelCacheError("Could not reach the model host"))
    store = cache(tmp_path, downloader)

    with pytest.raises(ModelCacheError):
        store.ensure(DEFAULT_BUNDLE)

    assert store.state(DEFAULT_BUNDLE) is ModelState.FAILED
    assert "Could not reach" in store.last_error
    assert list(store.root.glob("*.part")) == []


def test_an_unexpected_error_is_wrapped(tmp_path):
    store = cache(tmp_path, FakeDownloader(error=ValueError("boom")))
    with pytest.raises(ModelCacheError):
        store.ensure(DEFAULT_BUNDLE)
    assert "boom" in store.last_error


def test_clearing_the_error_restores_the_state(tmp_path):
    store = cache(tmp_path, FakeDownloader(error=ValueError("boom")))
    with pytest.raises(ModelCacheError):
        store.ensure(DEFAULT_BUNDLE)
    store.clear_error()
    assert store.state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED


def test_a_half_downloaded_file_is_never_promoted(tmp_path):
    """The real download writes a .part file and replaces only when complete."""

    def half_way(url, destination, progress, cancelled):
        Path(destination).write_bytes(b"half")
        raise ModelCacheError("connection reset")

    store = cache(tmp_path, half_way)
    with pytest.raises(ModelCacheError):
        store.ensure(DEFAULT_BUNDLE)
    assert not store.path_for(SEGMENTATION_ASSET).exists()
    assert list(store.root.glob("*.part")) == []


# ------------------------------------------------------------------ clean up


def test_cleanup_removes_stray_part_files(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "a.part").write_bytes(b"x")
    (root / "b.part").write_bytes(b"x")
    (root / "keep.onnx").write_bytes(b"x")

    assert cleanup_partials(root) == 2
    assert (root / "keep.onnx").exists()


def test_cleanup_on_a_missing_folder_is_harmless(tmp_path):
    assert cleanup_partials(tmp_path / "nope") == 0


def test_removing_the_bundle_clears_the_cache(tmp_path):
    store = cache(tmp_path)
    store.ensure(DEFAULT_BUNDLE)
    store.remove(DEFAULT_BUNDLE)
    assert store.state(DEFAULT_BUNDLE) is ModelState.NOT_DOWNLOADED


def test_cache_size_counts_the_stored_files(tmp_path):
    store = cache(tmp_path)
    assert store.cache_size_bytes() == 0
    store.ensure(DEFAULT_BUNDLE)
    assert store.cache_size_bytes() == len(b"model") * 2
