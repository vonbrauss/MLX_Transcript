"""Compact disclosure sections used by the main window."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["CollapsibleSection"]


class CollapsibleSection(QWidget):
    """A keyboard-accessible section with an arrow disclosure header."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._header = QToolButton(self)
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setProperty("kind", "sectionHeader")
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._header.toggled.connect(self._set_expanded)

        self._content = QFrame(self)
        self._content.setObjectName("sectionContent")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header)
        outer.addWidget(self._content)

        self._set_expanded(expanded, emit=False)

    def setContentLayout(self, layout: QLayout) -> None:  # noqa: N802
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        self._content.setLayout(layout)
        self._align_forms(layout)

    def apply_alignment(self) -> None:
        """Re-apply the shared alignment once the section has been populated.

        Sections hand their layout over before filling it, so the first pass
        in :meth:`setContentLayout` sees an empty layout and cannot align
        labels that do not exist yet.
        """
        layout = self._content.layout()
        if layout is not None:
            self._align_forms(layout)

    def setHeaderVisible(self, visible: bool) -> None:  # noqa: N802
        """Show the disclosure header, or use this section as a tab page."""
        self._header.setVisible(visible)

    #: One spacing scale for every section, so padding reads the same
    #: whichever layout a section happens to use.
    LABEL_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    HORIZONTAL_SPACING = 16
    VERTICAL_SPACING = 10

    @staticmethod
    def _align_forms(layout: QLayout) -> None:
        """Give every nested layout the same label alignment and spacing."""
        if isinstance(layout, QFormLayout):
            layout.setLabelAlignment(CollapsibleSection.LABEL_ALIGNMENT)
            layout.setHorizontalSpacing(CollapsibleSection.HORIZONTAL_SPACING)
            layout.setVerticalSpacing(CollapsibleSection.VERTICAL_SPACING)
            layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        elif isinstance(layout, QGridLayout):
            # A grid stretches a label over the full row height, so the text
            # floats away from its control unless the alignment says otherwise.
            layout.setHorizontalSpacing(CollapsibleSection.HORIZONTAL_SPACING)
            layout.setVerticalSpacing(CollapsibleSection.VERTICAL_SPACING)
            for index in range(layout.count()):
                item = layout.itemAt(index)
                widget = item.widget()
                if not isinstance(widget, QLabel):
                    continue
                # Aligning a layout item stops it from filling its cell, which
                # is what a caption wants and what a value label that has to
                # show a long path does not.
                if widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding:
                    continue
                layout.setAlignment(widget, CollapsibleSection.LABEL_ALIGNMENT)

        for index in range(layout.count()):
            child = layout.itemAt(index).layout()
            if child is not None:
                CollapsibleSection._align_forms(child)

    def isChecked(self) -> bool:  # noqa: N802 - QGroupBox compatibility
        return self._header.isChecked()

    def setChecked(self, expanded: bool) -> None:  # noqa: N802
        self._header.setChecked(expanded)

    def isExpanded(self) -> bool:  # noqa: N802
        return self.isChecked()

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        self.setChecked(expanded)

    def _set_expanded(self, expanded: bool, emit: bool = True) -> None:
        self._content.setVisible(expanded)
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._header.setToolTip(
            "Collapse this section" if expanded else "Expand this section"
        )
        if emit:
            self.toggled.emit(expanded)
