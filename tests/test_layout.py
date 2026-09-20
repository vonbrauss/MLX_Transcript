"""Window layout: alignment, footer visibility, and a queue that shrinks.

These run against a real widget tree on the offscreen platform, so they
measure geometry rather than asserting that a line of layout code exists.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QFormLayout,
    QFileDialog,
    QGridLayout,
    QLabel,
    QSizePolicy,
)

from app.collapsible import CollapsibleSection  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from app.theme import build_style_sheet, chevron_path  # noqa: E402

DEFAULT_SIZE = (1100, 820)
SHORT_SIZE = (1000, 560)

EXPECTED_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


def test_the_header_contains_the_product_name(window):
    assert window.brand_mark.text() == "MLX"
    assert window.app_title_label.text() == "MLX Transcript"


def test_the_launch_size_is_wide_and_compact(application):
    built = MainWindow(AppSettings())
    try:
        assert built.size().width() == 1180
        assert built.size().height() == 720
        assert built.minimumHeight() == 560
    finally:
        built.close()


def test_navigation_is_one_horizontal_row(window):
    assert window.section_tabs.count() == 7
    assert [window.section_tabs.tabText(index) for index in range(7)] == [
        "Folders",
        "Transcription",
        "Output",
        "Speaker Detection",
        "Advanced Settings",
        "Queue",
        "Help",
    ]


def test_selecting_a_menu_shows_only_its_page(window, application):
    window._select_section("speakers")
    application.processEvents()
    assert window.section_stack.currentWidget() is window.speaker_group
    assert window.speaker_group.isVisibleTo(window)
    assert not window.folders_group.isVisibleTo(window)


def test_the_selected_menu_is_collected_for_the_next_launch(window):
    window._select_section("queue")
    assert window._collect_settings().selected_section == "queue"


def test_each_launch_starts_on_folders(application):
    built = MainWindow(AppSettings(selected_section="queue"))
    try:
        assert built.section_stack.currentWidget() is built.folders_group
    finally:
        built.close()


def test_the_help_workspace_contains_quick_start_instructions(window):
    window._select_section("help")
    assert window.section_stack.currentWidget() is window.help_group
    assert "Getting started" in window.help_text.toPlainText()


@pytest.fixture(scope="module")
def application():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def window(application, tmp_path):
    built = MainWindow(AppSettings(output_parent=str(tmp_path / "Projects")))
    built.resize(*DEFAULT_SIZE)
    built.show()
    application.processEvents()
    yield built
    built.close()


# ------------------------------------------------------------------ alignment


def test_every_section_aligns_its_form_labels_left_and_centred(window):
    checked = 0
    for section in window.sections:
        layout = section._content.layout()
        for form in _forms(layout):
            assert form.labelAlignment() == EXPECTED_ALIGNMENT
            checked += 1
    assert checked, "no form layouts were found to check"


def test_grid_labels_are_aligned_rather_than_stretched(window):
    """A grid stretches a label over the row unless told otherwise."""
    layout = window.folders_group._content.layout()
    assert isinstance(layout, QGridLayout)

    captions = 0
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if not isinstance(widget, QLabel):
            continue
        if widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding:
            # A value label has to fill its cell to show a long path, and an
            # aligned layout item is pinned to its size hint instead.
            assert not item.alignment()
            continue
        assert item.alignment() == EXPECTED_ALIGNMENT
        captions += 1
    assert captions, "the folders section has no caption labels"


def test_sections_share_one_spacing_scale(window):
    for section in window.sections:
        layout = section._content.layout()
        assert layout.contentsMargins().left() == 14
        assert layout.contentsMargins().right() == 14
        for form in _forms(layout):
            assert form.horizontalSpacing() == CollapsibleSection.HORIZONTAL_SPACING
            assert form.verticalSpacing() == CollapsibleSection.VERTICAL_SPACING


def _forms(layout):
    """Yield every QFormLayout nested anywhere inside a layout."""
    if layout is None:
        return
    if isinstance(layout, QFormLayout):
        yield layout
    for index in range(layout.count()):
        child = layout.itemAt(index).layout()
        if child is not None:
            yield from _forms(child)


# ------------------------------------------------------------- the footer


def test_the_footer_stays_visible_at_the_default_size(window):
    assert window.footer.isVisible()
    assert window.footer.y() + window.footer.height() <= window.height() + 1


def test_the_footer_stays_visible_when_the_window_is_short(window, application):
    window.resize(*SHORT_SIZE)
    application.processEvents()

    assert window.footer.isVisible()
    bottom = window.footer.y() + window.footer.height()
    assert bottom <= window.height() + 1, (
        f"footer bottom {bottom} fell past the window height {window.height()}"
    )


def test_the_start_button_stays_reachable_when_the_window_is_short(
    window, application
):
    window.resize(*SHORT_SIZE)
    application.processEvents()

    position = window.start_button.mapTo(window, window.start_button.rect().topLeft())
    assert 0 <= position.y() < window.height()


# -------------------------------------------------------------- the queue


def test_the_queue_can_shrink_rather_than_push_the_footer_away(window, application):
    window.resize(*DEFAULT_SIZE)
    application.processEvents()
    tall = window.queue_table.height()

    window.resize(*SHORT_SIZE)
    application.processEvents()
    short = window.queue_table.height()

    assert short <= tall
    assert window.queue_table.minimumHeight() <= 120


def test_the_queue_uses_available_workspace_height(window, application):
    window._select_section("queue")
    application.processEvents()
    assert window.queue_table.height() > 250


def test_the_queue_never_collapses_to_nothing(window, application):
    window.resize(900, 420)
    application.processEvents()
    assert window.queue_table.minimumHeight() > 0


# ------------------------------------------------------- the folders section


def test_the_folders_section_shows_one_destination_not_three(window):
    assert window.scriptsync_preview.isVisibleTo(window.folders_group)
    assert window.timecoded_preview.isHidden()
    assert window.subtitles_preview.isHidden()


def test_drop_target_uses_the_available_folders_workspace(window, application):
    window._select_section("folders")
    application.processEvents()
    assert window.drop_target.height() > 140
    assert window.drop_target.height() > window.source_field.height() * 4


def test_the_detailed_paths_live_in_the_tooltip(window, tmp_path):
    window.output_field.setText(str(tmp_path / "Projects"))
    window._refresh_output_preview()

    tooltip = window.scriptsync_preview.toolTip()
    assert "ScriptSync:" in tooltip
    assert "Timecoded:" in tooltip
    assert "Subtitles:" in tooltip
    # The concise line itself carries only the one destination folder.
    assert window.scriptsync_preview.text().endswith("Transcription")


def test_output_chooser_is_available_when_no_batch_is_running(window):
    assert window.output_button.isEnabled()


def test_output_chooser_updates_the_destination(window, monkeypatch, tmp_path):
    destination = tmp_path / "Deliverables"
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    window.output_button.click()
    assert window.output_field.text() == str(destination)


def test_single_file_chooser_updates_the_source(window, monkeypatch, tmp_path):
    source = tmp_path / "interview.mov"
    source.touch()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(source)], "Media files"),
    )
    monkeypatch.setattr(window, "_queue_sources", lambda _paths: None)
    window.source_file_button.click()
    assert window.source_field.text() == str(source)


# ------------------------------------------------------- the dropdown arrow


def test_the_arrow_rule_is_dropped_when_the_asset_is_missing(monkeypatch):
    """Pointing Qt at an image it cannot load renders no arrow at all."""
    import app.theme as theme

    monkeypatch.setattr(theme, "chevron_path", lambda: None)
    sheet = theme.build_style_sheet()

    assert "QComboBox::down-arrow" not in sheet
    assert "__CHEVRON_RULE__" not in sheet
    # The drop-down box itself is still styled.
    assert "QComboBox::drop-down" in sheet


def test_the_arrow_rule_is_used_when_the_asset_is_present():
    sheet = build_style_sheet()
    if chevron_path() is None:
        pytest.skip("the chevron asset is not present in this checkout")
    assert "QComboBox::down-arrow" in sheet
    assert "chevron.svg" in sheet
    assert "__CHEVRON_RULE__" not in sheet


def test_the_asset_is_found_in_this_checkout():
    assert chevron_path() is not None, "chevron.svg is missing from app/assets"


def test_comboboxes_are_still_styled_with_a_visible_control(window):
    boxes = window.findChildren(QComboBox)
    assert boxes, "the window has no dropdowns"
    for box in boxes:
        assert box.width() > 0


@pytest.mark.parametrize("size", [(1180, 720), (1000, 560)])
def test_expanded_speaker_fields_do_not_overlap(window, application, size):
    window.resize(*size)
    window._select_section("speakers")
    window.detect_speakers.setChecked(True)
    window.speaker_advanced_toggle.setChecked(True)
    application.processEvents()
    fields = window.speaker_setting_spins
    for above, below in zip(fields, fields[1:]):
        assert above.geometry().bottom() < below.geometry().top()
    assert window.footer.y() + window.footer.height() <= window.height()
