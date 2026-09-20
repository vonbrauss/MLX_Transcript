"""The macOS Qt plugin repair that runs before PySide6 is imported.

This Mac keeps acquiring the hidden flag on the Qt plugin tree inside the
virtual environment, which makes Qt report that it cannot find the "cocoa"
platform plugin. ``main._repair_macos_qt_plugin_flags`` clears that flag.

The filesystem flags are mocked so these tests run anywhere, and nothing here
reads or writes a real virtual environment.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

import main


UF_HIDDEN = getattr(stat, "UF_HIDDEN", 0x8000)


class StatWithFlags:
    """A real stat result with the ``st_flags`` this platform may not have."""

    def __init__(self, result, flags: int) -> None:
        self._result = result
        self.st_flags = flags

    def __getattr__(self, name):
        return getattr(self._result, name)


class FakeFilesystem:
    """Tracks flags per path and records every chflags call."""

    def __init__(self, flags: dict[Path, int]) -> None:
        self.flags = dict(flags)
        self.changed: list[tuple[Path, int]] = []
        self.denied: set[Path] = set()
        self._real_stat = Path.stat

    def stat(self, path: Path, **kwargs) -> StatWithFlags:
        if path in self.denied:
            raise PermissionError("nope")
        return StatWithFlags(
            self._real_stat(path, **kwargs), self.flags.get(path, 0)
        )

    def chflags(self, path, value: int) -> None:
        path = Path(path)
        if path in self.denied:
            raise PermissionError("nope")
        self.flags[path] = value
        self.changed.append((path, value))


@pytest.fixture
def plugin_tree(tmp_path, monkeypatch):
    """Build a fake PySide6 Qt plugin tree with hidden files."""
    package = tmp_path / "PySide6"
    plugins = package / "Qt" / "plugins"
    platforms = plugins / "platforms"
    platforms.mkdir(parents=True)
    cocoa = platforms / "libqcocoa.dylib"
    cocoa.write_bytes(b"dylib")

    class Spec:
        origin = str(package / "__init__.py")

    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: Spec())
    monkeypatch.setattr(main.sys, "platform", "darwin")
    monkeypatch.setattr(stat, "UF_HIDDEN", UF_HIDDEN, raising=False)

    filesystem = FakeFilesystem(
        {plugins: UF_HIDDEN, platforms: UF_HIDDEN, cocoa: UF_HIDDEN}
    )
    monkeypatch.setattr(
        Path, "stat", lambda self, **kwargs: filesystem.stat(self, **kwargs)
    )
    monkeypatch.setattr(main.os, "chflags", filesystem.chflags, raising=False)

    return filesystem, plugins, platforms, cocoa


def test_the_repair_function_is_still_present():
    assert callable(main._repair_macos_qt_plugin_flags)


def test_the_repair_runs_before_pyside6_is_imported():
    """The call must sit above the PySide6 import in the source file."""
    source = Path(main.__file__).read_text(encoding="utf-8")
    call = source.index("\n_repair_macos_qt_plugin_flags()")
    qt_import = source.index("from PySide6")
    assert call < qt_import


def test_no_qt_plugin_path_environment_workaround_was_added():
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "QT_PLUGIN_PATH" not in source
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in source


def test_the_hidden_flag_is_cleared_across_the_whole_tree(plugin_tree):
    filesystem, plugins, platforms, cocoa = plugin_tree

    main._repair_macos_qt_plugin_flags()

    changed = {path for path, _value in filesystem.changed}
    assert plugins in changed
    assert platforms in changed
    assert cocoa in changed
    for path in (plugins, platforms, cocoa):
        assert not filesystem.flags[path] & UF_HIDDEN


def test_other_flags_are_left_alone(plugin_tree, monkeypatch):
    filesystem, plugins, _platforms, cocoa = plugin_tree
    filesystem.flags[cocoa] = UF_HIDDEN | stat.UF_IMMUTABLE

    main._repair_macos_qt_plugin_flags()

    assert filesystem.flags[cocoa] == stat.UF_IMMUTABLE


def test_files_that_are_not_hidden_are_not_touched(plugin_tree):
    filesystem, plugins, platforms, cocoa = plugin_tree
    filesystem.flags = {plugins: 0, platforms: 0, cocoa: 0}

    main._repair_macos_qt_plugin_flags()

    assert filesystem.changed == []


def test_a_denied_file_does_not_stop_the_rest(plugin_tree):
    filesystem, plugins, platforms, cocoa = plugin_tree
    filesystem.denied.add(platforms)

    main._repair_macos_qt_plugin_flags()

    changed = {path for path, _value in filesystem.changed}
    assert cocoa in changed
    assert platforms not in changed


def test_nothing_happens_off_macos(monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(main.sys, "platform", "linux")
    monkeypatch.setattr(
        main.importlib.util, "find_spec", lambda name: calls.append(name)
    )
    main._repair_macos_qt_plugin_flags()
    assert calls == []


def test_a_missing_pyside6_is_handled(monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "darwin")
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: None)
    main._repair_macos_qt_plugin_flags()  # must not raise


def test_a_missing_plugin_folder_is_handled(tmp_path, monkeypatch):
    class Spec:
        origin = str(tmp_path / "PySide6" / "__init__.py")

    monkeypatch.setattr(main.sys, "platform", "darwin")
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: Spec())
    main._repair_macos_qt_plugin_flags()  # must not raise


def test_the_repair_never_touches_anything_outside_the_plugin_tree(plugin_tree):
    filesystem, plugins, _platforms, _cocoa = plugin_tree
    main._repair_macos_qt_plugin_flags()
    for path, _value in filesystem.changed:
        assert plugins == path or plugins in path.parents
