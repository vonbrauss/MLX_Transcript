"""Small shared widgets."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

__all__ = ["ElidedLabel"]


class ElidedLabel(QLabel):
    """A single-line label that shortens a long path instead of clipping it.

    A plain QLabel given a path too wide for its cell simply cuts the text off,
    which reads as a wrong path rather than a shortened one. This elides in the
    middle, so the start and the meaningful tail both stay visible, and keeps
    the full text available as the tooltip.
    """

    def __init__(
        self,
        text: str = "",
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._mode = mode
        self._explicit_tooltip = False
        # Expanding fills the cell it is given; the small minimumSizeHint below
        # is what lets it shrink instead of forcing the window wider.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setText(text)

    # ------------------------------------------------------------------ text

    def fullText(self) -> str:  # noqa: N802 - Qt naming
        """The complete text, whatever is currently displayed."""
        return self._full_text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full_text = text or ""
        if not self._explicit_tooltip:
            super().setToolTip(self._full_text)
        self._apply_elision()

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt naming
        """A caller-supplied tooltip wins over the automatic full-text one."""
        self._explicit_tooltip = bool(text)
        super().setToolTip(text)

    # --------------------------------------------------------------- sizing

    def _apply_elision(self) -> None:
        metrics = QFontMetrics(self.font())
        available = max(0, self.width() - 2)
        if available <= 0:
            super().setText(self._full_text)
            return
        super().setText(metrics.elidedText(self._full_text, self._mode, available))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_elision()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Stay one line tall and never force the window wider."""
        metrics = QFontMetrics(self.font())
        return QSize(80, metrics.height())
