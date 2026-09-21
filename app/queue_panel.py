"""One reusable view onto the media queue.

Two of these exist: the inline panel inside the Media workspace and the
dedicated Queue page. Neither owns any queue state. :class:`MainWindow`
keeps the authoritative ``items`` list and renders both panels from it, so
the two views can never drift apart.

The panel also draws its own insertion indicator. Qt only paints its
built-in drop indicator from the view's own drag handlers, and the window
intercepts those so it can tell a Finder drop from an internal row move.
Drawing the line here keeps the feedback under our control and testable.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QPoint,
    Qt,
    Signal,
)
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["QUEUE_COLUMNS", "QueuePanel", "QueueTable"]

#: The columns both queue views show, in order.
QUEUE_COLUMNS = ("File", "Folder", "Duration", "Status")

#: Height of the insertion line drawn while a row is being moved.
INDICATOR_HEIGHT = 3


class QueueTable(QTableWidget):
    """A queue table that never edits its own rows.

    ``QAbstractItemView.startDrag`` ends with::

        if (drag->exec(supportedActions, defaultDropAction) == Qt::MoveAction)
            d->clearOrRemove();

    and ``clearOrRemove`` calls ``removeRows`` on whatever was selected. That
    runs *after* the drop has been handled, so a row dragged to a new position
    was reordered correctly and then deleted from the table it came from a
    moment later: the clip vanished from the view while still sitting in the
    window's queue. It only ever happened with a real mouse, because
    ``drag->exec`` cannot return ``MoveAction`` without one.

    So this table runs the drag itself and never calls up to that branch. The
    window owns the queue and applies the move to its own list; the rows here
    are only ever redrawn from it. ``drag_finished`` fires once the drag is
    over, which is the moment Qt would have edited the rows, so the window can
    confirm the views still match the queue.
    """

    #: Emitted when a drag that started here has ended, whatever its outcome.
    drag_finished = Signal()

    def startDrag(self, supported_actions) -> None:  # noqa: N802 - Qt naming
        """Run the drag without Qt's remove-the-source-rows epilogue."""
        indexes = [
            index
            for index in self.selectedIndexes()
            if index.flags() & Qt.ItemFlag.ItemIsDragEnabled
        ]
        if not indexes:
            return
        payload = self.model().mimeData(indexes)
        if payload is None:
            return

        drag = QDrag(self)
        drag.setMimeData(payload)
        try:
            # Copy, not Move, on both counts. Nothing here is being given
            # away, and a Move result is the only thing Qt acts on.
            drag.exec(Qt.DropAction.CopyAction, Qt.DropAction.CopyAction)
        finally:
            self.drag_finished.emit()


class QueuePanel(QWidget):
    """A queue table, its selection-sensitive actions, and its summary."""

    def __init__(
        self,
        summary_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._reorder_enabled = True
        #: Row the insertion line currently points at, or ``None``.
        self.insertion_row: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.table = QueueTable(0, len(QUEUE_COLUMNS), self)
        self.table.setMinimumHeight(90)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table.setHorizontalHeaderLabels(list(QUEUE_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setAcceptDrops(True)
        self.table.viewport().setAcceptDrops(True)
        self.table.setDragEnabled(True)
        self.table.setDragDropOverwriteMode(False)
        self.table.setDropIndicatorShown(False)
        # Move is what makes Qt delete the source rows once a drag ends.
        # The window applies the reorder itself, so nothing is ever moved out
        # of this table and the action stays Copy end to end.
        self.table.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.insertion_indicator = QFrame(self.table.viewport())
        self.insertion_indicator.setObjectName("queueInsertIndicator")
        self.insertion_indicator.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        self.insertion_indicator.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.insertion_indicator.setFixedHeight(INDICATOR_HEIGHT)
        self.insertion_indicator.hide()

        self.summary = QLabel(summary_text)
        self.summary.setWordWrap(True)
        self.summary.setProperty("tone", "secondary")

        actions = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setToolTip(
            "Take the selected files out of the Media queue."
        )
        self.clear_button = QPushButton("Clear Queue")
        self.clear_button.setProperty("kind", "danger")
        self.clear_button.setToolTip("Empty the Media queue.")
        self.reveal_button = QPushButton("Reveal Source")
        self.reveal_button.setToolTip(
            "Show the selected file where it lives on disk."
        )
        actions.addWidget(self.remove_button)
        actions.addWidget(self.clear_button)
        actions.addWidget(self.reveal_button)
        actions.addStretch(1)

        layout.addLayout(actions)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.summary)

    # ------------------------------------------------------------ rendering

    def row_count(self) -> int:
        return self.table.rowCount()

    def set_row_count(self, count: int) -> None:
        """Grow or shrink the table to hold exactly ``count`` rows."""
        self.table.setRowCount(count)
        for row in range(count):
            for column in range(len(QUEUE_COLUMNS)):
                if self.table.item(row, column) is None:
                    self.table.setItem(row, column, QTableWidgetItem(""))

    def append_row(self) -> int:
        """Add one row and its cells.

        Deliberately not ``set_row_count``: that walks every row to fill in
        missing cells, which turns queueing a folder of dailies into
        quadratic work. Appending touches only the row being added.
        """
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column in range(len(QUEUE_COLUMNS)):
            self.table.setItem(row, column, QTableWidgetItem(""))
        return row

    def set_row(self, row: int, values, source_tooltip: str = "") -> None:
        """Fill one row, creating its cells if the table was just grown."""
        if not 0 <= row < self.table.rowCount():
            return
        for column, value in enumerate(values):
            cell = self.table.item(row, column)
            if cell is None:
                cell = QTableWidgetItem("")
                self.table.setItem(row, column, cell)
            cell.setText(value)
            if column == 2:
                cell.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            cell.setToolTip(source_tooltip if column == 0 else value)

    def set_summary(self, text: str, tooltip: str = "") -> None:
        self.summary.setText(text)
        self.summary.setToolTip(tooltip)

    # ------------------------------------------------------------ selection

    def selected_rows(self) -> list[int]:
        model = self.table.selectionModel()
        if model is None:
            return []
        return sorted(index.row() for index in model.selectedRows())

    def select_rows(self, rows) -> None:
        """Mirror a selection made in the other view."""
        model = self.table.selectionModel()
        if model is None:
            return
        self.table.clearSelection()
        wanted = [row for row in rows if 0 <= row < self.table.rowCount()]
        if not wanted:
            return
        selection = QItemSelection()
        columns = self.table.columnCount() - 1
        source = self.table.model()
        for row in wanted:
            selection.select(
                source.index(row, 0), source.index(row, max(columns, 0))
            )
        model.select(
            selection, QItemSelectionModel.SelectionFlag.ClearAndSelect
        )

    # ------------------------------------------------------- reorder support

    @property
    def reorder_enabled(self) -> bool:
        return self._reorder_enabled

    def set_reorder_enabled(self, enabled: bool) -> None:
        """Turn row dragging on or off without touching Finder drops."""
        enabled = bool(enabled)
        self._reorder_enabled = enabled
        self.table.setDragEnabled(enabled)
        self.table.setDragDropMode(
            QAbstractItemView.DragDropMode.DragDrop
            if enabled
            else QAbstractItemView.DragDropMode.DropOnly
        )
        if not enabled:
            self.hide_insertion()

    def insertion_row_at(self, position: QPoint) -> int:
        """Return the row a drop at ``position`` should insert before."""
        count = self.table.rowCount()
        if count == 0:
            return 0
        row = self.table.rowAt(position.y())
        if row < 0:
            # Past the last row, or above the first one.
            first = self.table.visualRect(self.table.model().index(0, 0))
            return 0 if position.y() < first.top() else count
        rect = self.table.visualRect(self.table.model().index(row, 0))
        return row + 1 if position.y() >= rect.center().y() else row

    def show_insertion_at(self, row: int) -> None:
        """Draw the insertion line above ``row``."""
        viewport = self.table.viewport()
        count = self.table.rowCount()
        row = max(0, min(int(row), count))
        if count == 0:
            top = 0
        elif row >= count:
            top = self.table.visualRect(
                self.table.model().index(count - 1, 0)
            ).bottom()
        else:
            top = self.table.visualRect(self.table.model().index(row, 0)).top()
        top = max(0, min(top - 1, max(viewport.height() - INDICATOR_HEIGHT, 0)))
        self.insertion_indicator.setGeometry(
            0, top, viewport.width(), INDICATOR_HEIGHT
        )
        self.insertion_indicator.show()
        self.insertion_indicator.raise_()
        self.insertion_row = row

    def hide_insertion(self) -> None:
        self.insertion_indicator.hide()
        self.insertion_row = None

    @property
    def insertion_visible(self) -> bool:
        return not self.insertion_indicator.isHidden()
