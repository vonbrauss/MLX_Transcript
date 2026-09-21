"""Application-wide visual theme with accessible dark-mode contrast."""

from __future__ import annotations
from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["apply_theme", "build_style_sheet", "chevron_path"]


STYLE_SHEET = r"""
QWidget {
    color: #F2F5F9;
    font-size: 14px;
}

QMainWindow, QDialog, QMessageBox, QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #101318;
}

QFrame#appHeader {
    background-color: #141920;
    border-bottom: 1px solid #29313D;
}
QLabel[kind="brandMark"] {
    color: #FFFFFF;
    background-color: #2878D0;
    border: 1px solid #5AA7F3;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 750;
}
QLabel[kind="pageTitle"] {
    color: #F8FAFD;
    font-size: 20px;
    font-weight: 650;
}
QTextBrowser#helpBrowser {
    background: transparent;
    border: none;
    color: #DCE4EE;
}
QLabel[kind="appTitle"] {
    color: #F8FAFD;
    font-size: 22px;
    font-weight: 700;
}
QFrame#sectionNavigation, QFrame#workspace {
    background-color: #101318;
}
QTabBar#sectionTabs {
    background: transparent;
}
QTabBar#sectionTabs::tab {
    color: #AEB9C8;
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    min-height: 38px;
    padding: 3px 10px;
    font-weight: 600;
}
QTabBar#sectionTabs::tab:hover {
    color: #F4F7FB;
    background-color: #171D25;
}
QTabBar#sectionTabs::tab:selected {
    color: #FFFFFF;
    border-bottom-color: #4C9AFF;
    background-color: #171D25;
}
QTabBar#sectionTabs::tab:focus {
    color: #FFFFFF;
}
QStackedWidget#sectionStack {
    background-color: transparent;
    border: none;
}

QGroupBox {
    background-color: #181C23;
    border: 1px solid #303846;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #F7F9FC;
}

QToolButton[kind="sectionHeader"] {
    color: #F7F9FC;
    background-color: #1B2028;
    border: 1px solid #343D4B;
    border-radius: 8px;
    min-height: 28px;
    padding: 2px 11px;
    font-size: 15px;
    font-weight: 650;
    text-align: left;
}
QToolButton[kind="sectionHeader"]:hover {
    background-color: #232A34;
    border-color: #4A586C;
}
QToolButton[kind="sectionHeader"]:focus { border: 2px solid #4C9AFF; }
QFrame#sectionContent {
    background-color: #181C23;
    border: 1px solid #303846;
    border-radius: 10px;
}
QFrame#dropTarget {
    background-color: #141C27;
    border: 2px dashed #405D7A;
    border-radius: 7px;
}
QLabel[kind="dropTitle"] {
    color: #DCEBFC;
    font-size: 17px;
    font-weight: 650;
}
QFrame#dropTarget[dropActive="true"], QTableWidget[dropActive="true"] {
    background-color: #18324D;
    border: 2px solid #4C9AFF;
}
/* Once media is queued the workspace shows the inline queue, so the dashed
   invitation gives way to the same surface the Queue page uses. */
QFrame#dropTarget[mode="queue"] {
    background-color: transparent;
    border: none;
}
QStackedWidget#mediaWorkspaceStack {
    background-color: transparent;
    border: none;
}
/* The Finder-drag sheet. The tint is translucent so a loaded queue stays
   faintly readable underneath it. */
QFrame#dropOverlay {
    background-color: rgba(40, 120, 208, 58);
    border: 2px solid #4C9AFF;
    border-radius: 7px;
}
QLabel[kind="dropOverlayText"] {
    color: #EAF3FF;
    background-color: rgba(12, 24, 40, 225);
    border: 1px solid #4C9AFF;
    border-radius: 11px;
    padding: 11px 20px;
    font-size: 17px;
    font-weight: 650;
}
/* The line that shows where a dragged row will land. */
QFrame#queueInsertIndicator {
    background-color: #4C9AFF;
    border: none;
    border-radius: 1px;
}

QFrame#appFooter {
    background-color: #141920;
    border-top: 1px solid #303846;
}

QLabel[tone="secondary"] { color: #B8C1CE; }
QLabel[tone="muted"] { color: #98A4B3; }
QLabel[tone="success"] { color: #67D7A3; }
QLabel[tone="warning"] { color: #F4C66A; }
QLabel[tone="danger"] { color: #FF8B8B; }
QLabel[kind="privacy"] {
    color: #8ADDB4;
    background-color: #173328;
    border: 1px solid #285744;
    border-radius: 11px;
    padding: 5px 10px;
    font-weight: 600;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    color: #F6F8FB;
    background-color: #11151B;
    border: 1px solid #3A4453;
    border-radius: 6px;
    min-height: 26px;
    padding: 3px 8px;
    selection-background-color: #2F80ED;
}

QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #56657A;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #4C9AFF;
    padding: 2px 7px;
}

QComboBox::drop-down {
    width: 28px;
    border: none;
    border-left: 1px solid #3A4453;
}
__CHEVRON_RULE__

QComboBox QAbstractItemView {
    color: #F2F5F9;
    background-color: #202630;
    border: 1px solid #465267;
    selection-background-color: #2F6FB9;
    selection-color: #FFFFFF;
    padding: 4px;
    outline: 0;
}

QPushButton {
    color: #F2F5F9;
    background-color: #252C36;
    border: 1px solid #414C5D;
    border-radius: 7px;
    min-height: 31px;
    padding: 3px 14px;
    font-weight: 500;
}

QPushButton:hover { background-color: #303947; border-color: #5A6980; }
QPushButton:pressed { background-color: #1E242C; }
QPushButton:focus { border: 2px solid #64A8FF; padding: 2px 13px; }
QPushButton[kind="primary"] {
    color: #FFFFFF;
    background-color: #2878D0;
    border-color: #4396EE;
    font-weight: 650;
}
QPushButton[kind="primary"]:hover { background-color: #3488E2; }
QPushButton[kind="danger"] { color: #FFB1B1; border-color: #7A3E46; }
QPushButton[kind="danger"]:hover { background-color: #4A262C; }

QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QPushButton:disabled {
    color: #6F7A89;
    background-color: #171B21;
    border-color: #2B313B;
}
QLabel:disabled, QCheckBox:disabled { color: #778393; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; }

QTableWidget {
    color: #EAF0F7;
    background-color: #11151B;
    alternate-background-color: #171C24;
    border: 1px solid #303846;
    border-radius: 7px;
    gridline-color: #29313D;
    selection-background-color: #244F7D;
    selection-color: #FFFFFF;
}

QHeaderView::section {
    color: #DCE4EE;
    background-color: #222934;
    border: none;
    border-right: 1px solid #354050;
    border-bottom: 1px solid #354050;
    padding: 8px 9px;
    font-weight: 600;
}

QProgressBar {
    color: #F5F8FC;
    background-color: #202630;
    border: 1px solid #303A48;
    border-radius: 5px;
    min-height: 10px;
    text-align: center;
}
QProgressBar::chunk { background-color: #3B91ED; border-radius: 4px; }

QScrollBar:vertical {
    background: #11151B;
    width: 12px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #485568;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #62728A; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QSplitter::handle { background-color: #303846; height: 2px; }
QStatusBar { color: #AEB8C6; background-color: #151920; }
QStatusBar::item { border: none; }
QToolTip {
    color: #F7F9FC;
    background-color: #252C36;
    border: 1px solid #56657A;
    padding: 5px;
}
"""


CHEVRON_RULE = """QComboBox::down-arrow {
    image: url("%s");
    width: 12px;
    height: 8px;
}"""


def chevron_path() -> Path | None:
    """Find the dropdown arrow asset in development and inside the bundle.

    A frozen macOS application splits its payload: binaries land in
    Contents/Frameworks and data files in Contents/Resources, so the asset is
    not always where ``__file__`` suggests. Every plausible location is tried,
    and None means the caller should leave Qt's own arrow alone.
    """
    from app.runtime import resource_root

    root = resource_root()
    candidates = [
        root / "app" / "assets" / "chevron.svg",
        root.parent / "Resources" / "app" / "assets" / "chevron.svg",
        Path(__file__).resolve().parent / "assets" / "chevron.svg",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def build_style_sheet() -> str:
    """Return the stylesheet with the arrow rule resolved for this install.

    When the asset cannot be found the rule is dropped entirely rather than
    pointed at a missing file, because a stylesheet that sets an image Qt
    cannot load renders no arrow at all instead of falling back.
    """
    chevron = chevron_path()
    rule = CHEVRON_RULE % chevron.as_posix() if chevron is not None else ""
    return STYLE_SHEET.replace("__CHEVRON_RULE__", rule)


def apply_theme(application: QApplication) -> None:
    """Apply one consistent palette and stylesheet to every window."""
    application.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101318"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#F2F5F9"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#11151B"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#171C24"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#F2F5F9"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#252C36"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#F2F5F9"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2F80ED"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6F7A89"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6F7A89"))
    application.setPalette(palette)
    application.setStyleSheet(build_style_sheet())
