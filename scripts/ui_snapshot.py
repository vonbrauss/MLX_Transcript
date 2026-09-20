#!/usr/bin/env python3
"""Render the main window, save PNGs, and print measurable layout facts.

An agent without eyes cannot answer "does this look right", but it can run
this and report numbers. The PNGs are for a human to open afterwards.

Run from the project root:

    .venv/bin/python scripts/ui_snapshot.py

Exits 0 when every hard check passes, 1 otherwise. Images land in build/ui/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLabel  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from app.settings import AppSettings  # noqa: E402
from app.theme import apply_theme, build_style_sheet, chevron_path  # noqa: E402

DEFAULT_SIZE = (1180, 720)
SHORT_SIZE = (1000, 560)
OUTPUT_FOLDER = PROJECT_ROOT / "build" / "ui"

EXPECTED_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


class Report:
    """Collects pass/fail lines so the whole run is visible at once."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = 0

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        mark = "ok  " if passed else "FAIL"
        if not passed:
            self.failed += 1
        self.lines.append(f"  [{mark}] {name}{': ' + detail if detail else ''}")

    def note(self, name: str, detail: str) -> None:
        self.lines.append(f"  [info] {name}: {detail}")

    def render(self) -> str:
        return "\n".join(self.lines)


def snapshot(window: MainWindow, application: QApplication, name: str) -> Path:
    application.processEvents()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_FOLDER / f"{name}.png"
    window.grab().save(str(target))
    return target


def measure(window: MainWindow, application: QApplication, report: Report, label: str) -> None:
    """Record whether the footer survives at the current window size."""
    application.processEvents()
    footer_bottom = window.footer.y() + window.footer.height()
    report.check(
        f"{label}: footer fully inside the window",
        window.footer.isVisible() and footer_bottom <= window.height() + 1,
        f"footer bottom {footer_bottom}px, window {window.height()}px",
    )
    start = window.start_button
    position = start.mapTo(window, start.rect().topLeft())
    report.check(
        f"{label}: Start Transcription on screen",
        0 <= position.y() < window.height(),
        f"button top at y={position.y()}",
    )
    report.note(f"{label}: queue table height", f"{window.queue_table.height()}px")


def main() -> int:
    application = QApplication(sys.argv)
    apply_theme(application)
    report = Report()

    # --- theme assets -----------------------------------------------------
    chevron = chevron_path()
    report.check(
        "dropdown chevron asset found",
        chevron is not None,
        str(chevron) if chevron else "not found, Qt's own arrow will be used",
    )
    sheet = build_style_sheet()
    report.check(
        "no unresolved placeholder left in the stylesheet",
        "__CHEVRON" not in sheet,
    )
    report.check(
        "dropdown arrow rule present",
        ("QComboBox::down-arrow" in sheet) == (chevron is not None),
        "rule matches whether the asset exists",
    )

    # --- build the window -------------------------------------------------
    settings = AppSettings(output_parent=str(PROJECT_ROOT / "build" / "ui-sample"))
    window = MainWindow(settings)
    window.resize(*DEFAULT_SIZE)
    window.show()
    application.processEvents()

    # --- alignment --------------------------------------------------------
    misaligned: list[str] = []
    for section in window.sections:
        layout = section._content.layout()
        if layout is None:
            continue
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if isinstance(item.widget(), QLabel) and hasattr(layout, "setAlignment"):
                if item.alignment() and item.alignment() != EXPECTED_ALIGNMENT:
                    misaligned.append(item.widget().text()[:30])
    report.check(
        "section labels left aligned and vertically centred",
        not misaligned,
        f"misaligned: {misaligned}" if misaligned else f"{len(window.sections)} sections",
    )

    # --- the folders section ---------------------------------------------
    window._refresh_output_preview()
    application.processEvents()
    report.check(
        "one destination line, not three",
        window.timecoded_preview.isHidden() and window.subtitles_preview.isHidden(),
        f"showing: {window.scriptsync_preview.text()}",
    )
    tooltip = window.scriptsync_preview.toolTip()
    report.check(
        "detailed paths kept in the tooltip",
        all(word in tooltip for word in ("ScriptSync:", "Timecoded:", "Subtitles:")),
    )

    # --- dropdowns --------------------------------------------------------
    boxes = window.findChildren(QComboBox)
    report.check(
        "dropdowns present and sized",
        bool(boxes) and all(box.width() > 0 for box in boxes),
        f"{len(boxes)} dropdowns",
    )

    # --- horizontal section navigation -----------------------------------
    report.check(
        "seven workspace tabs present",
        window.section_tabs.count() == 7,
        ", ".join(
            window.section_tabs.tabText(index)
            for index in range(window.section_tabs.count())
        ),
    )
    for index in range(window.section_tabs.count()):
        window.section_tabs.setCurrentIndex(index)
        application.processEvents()
        report.check(
            f"tab {index + 1} opens its workspace",
            window.section_stack.currentIndex() == index,
        )
    window.section_tabs.setCurrentIndex(0)

    # --- geometry at two sizes -------------------------------------------
    measure(window, application, report, "default size")
    default_image = snapshot(window, application, "window-default")

    window.resize(*SHORT_SIZE)
    measure(window, application, report, "short window")
    short_image = snapshot(window, application, "window-short")

    window.resize(*DEFAULT_SIZE)
    window.section_tabs.setCurrentIndex(3)
    window.detect_speakers.setChecked(True)
    window.speaker_advanced_toggle.setChecked(True)
    expanded_image = snapshot(window, application, "window-speaker-settings")

    print("MLX Transcript UI snapshot")
    print()
    print(report.render())
    print()
    print("Images written for a human to open:")
    for image in (default_image, short_image, expanded_image):
        print(f"  {image}")
    print()
    if report.failed:
        print(f"{report.failed} check(s) failed.")
    else:
        print("All layout checks passed.")

    window.close()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
