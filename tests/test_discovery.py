"""Recursive media discovery and output-tree exclusion."""

from __future__ import annotations

from pathlib import Path

from transcription.discovery import (
    MEDIA_EXTENSIONS,
    discover_media,
    is_supported_media,
)
from transcription.outputs import output_roots


def build_tree(root: Path, relative_paths: list[str]) -> None:
    """Create empty files at each relative path below ``root``."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_supported_extensions_cover_editorial_media():
    for suffix in (".mov", ".mp4", ".m4v", ".wav", ".aif", ".m4a", ".mp3"):
        assert suffix in MEDIA_EXTENSIONS
    assert is_supported_media(Path("clip.MOV"))
    assert not is_supported_media(Path("notes.txt"))


def test_discovery_finds_media_recursively(tmp_path):
    source = tmp_path / "Source"
    build_tree(
        source,
        [
            "Media day sound/Atlanta Dream.mov",
            "Media day sound/B roll.mp4",
            "Interviews/Day 02/Coach.wav",
            "notes.txt",
        ],
    )

    found = discover_media(source)
    names = [item.path.name for item in found]
    assert names == ["Coach.wav", "Atlanta Dream.mov", "B roll.mp4"]
    assert "notes.txt" not in names


def test_discovery_reports_relative_folders(tmp_path):
    source = tmp_path / "Source"
    build_tree(source, ["Day 01/A001.mov", "root_clip.wav"])

    by_name = {item.path.name: item for item in discover_media(source)}
    assert by_name["A001.mov"].relative_folder == Path("Day 01")
    assert by_name["A001.mov"].relative_folder_label == "Day 01"
    assert by_name["root_clip.wav"].relative_folder == Path(".")
    assert by_name["root_clip.wav"].relative_folder_label == "/"


def test_discovery_excludes_the_transcription_tree(tmp_path):
    source = tmp_path / "Source"
    build_tree(source, ["Day 01/A001.mov"])
    roots = output_roots(source)
    build_tree(source, ["Transcription/ScriptSync/Day 01/stray.mov"])

    found = discover_media(source, excluded_roots=[roots.transcription])
    assert [item.path.name for item in found] == ["A001.mov"]


def test_discovery_skips_appledouble_sidecars(tmp_path):
    source = tmp_path / "Source"
    build_tree(source, ["Day 01/A001.mov", "Day 01/._A001.mov", ".Trash/old.mov"])

    found = discover_media(source)
    assert [item.path.name for item in found] == ["A001.mov"]


def test_discovery_can_keep_hidden_files_when_asked(tmp_path):
    source = tmp_path / "Source"
    build_tree(source, ["Day 01/A001.mov", "Day 01/._A001.mov"])

    found = discover_media(source, skip_hidden=False)
    assert len(found) == 2


def test_discovery_returns_nothing_for_an_empty_folder(tmp_path):
    source = tmp_path / "Source"
    source.mkdir()
    assert discover_media(source) == []


def test_discovery_accepts_one_media_file(tmp_path):
    source = tmp_path / "Interview.mov"
    source.touch()

    found = discover_media(source)

    assert [item.path for item in found] == [source]
    assert found[0].source_root == tmp_path
    assert found[0].relative_folder == Path(".")


def test_discovery_rejects_one_unsupported_file(tmp_path):
    source = tmp_path / "notes.txt"
    source.touch()
    assert discover_media(source) == []
