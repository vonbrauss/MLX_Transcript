"""The speaker review dialog shown before transcripts are written.

All of the editing logic lives in :mod:`transcription.speakers`. This module
is only the Qt surface over it, so the rules about names, merging, and
reassignment stay testable without a display.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models import ReviewDecision
from transcription.media_probe import format_duration
from transcription.speakers import SpeakerNameError, SpeakerTranscript
from transcription.timecode import TimecodeConverter

__all__ = ["SpeakerReviewDialog"]

_SPEAKER_COLUMNS = ("Speaker", "Name", "Speaking time", "Segments", "Samples")
_PREVIEW_COLUMNS = ("Timecode", "Speaker", "Text")

#: Rendering every segment of a long interview would stall the dialog.
_PREVIEW_LIMIT = 400


class SpeakerReviewDialog(QDialog):
    """Rename, merge, and correct the speakers detected in one clip."""

    def __init__(
        self,
        source: Path,
        transcript: SpeakerTranscript,
        converter: TimecodeConverter | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = Path(source)
        self.transcript = transcript
        self.converter = converter
        self.decision = ReviewDecision.CONTINUE

        self.setWindowTitle(f"Speakers in {self.source.name}")
        self.setMinimumSize(900, 620)
        self.setSizeGripEnabled(True)
        self._build_ui()
        self._reload()

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        heading = QLabel(
            f"{len(self.transcript.speakers)} speaker(s) detected in "
            f"{self.source.name}. Rename them before the transcripts are written."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_speaker_table())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        layout.addLayout(self._build_tools())
        layout.addWidget(self._build_buttons())

    def _build_speaker_table(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)

        self.speaker_table = QTableWidget(0, len(_SPEAKER_COLUMNS))
        self.speaker_table.setHorizontalHeaderLabels(list(_SPEAKER_COLUMNS))
        self.speaker_table.verticalHeader().setVisible(False)
        self.speaker_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.speaker_table.setAlternatingRowColors(True)
        header = self.speaker_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.speaker_table.itemChanged.connect(self._on_name_edited)

        box.addWidget(QLabel("Detected speakers (edit the Name column)"))
        box.addWidget(self.speaker_table)
        return panel

    def _build_preview(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)

        self.preview_table = QTableWidget(0, len(_PREVIEW_COLUMNS))
        self.preview_table.setHorizontalHeaderLabels(list(_PREVIEW_COLUMNS))
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.preview_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.preview_table.setAlternatingRowColors(True)
        preview_header = self.preview_table.horizontalHeader()
        preview_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        preview_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        preview_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.preview_note = QLabel()
        self.preview_note.setProperty("tone", "secondary")

        box.addWidget(QLabel("Transcript preview"))
        box.addWidget(self.preview_table)
        box.addWidget(self.preview_note)
        return panel

    def _build_tools(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.reset_button = QPushButton("Reset names")
        self.reset_button.clicked.connect(self._reset_names)

        self.merge_button = QPushButton("Merge selected into…")
        self.merge_button.clicked.connect(self._merge_selected)
        self.merge_target = QComboBox()

        self.reassign_button = QPushButton("Reassign segment to…")
        self.reassign_button.clicked.connect(self._reassign_selected)
        self.reassign_target = QComboBox()

        row.addWidget(self.reset_button)
        row.addSpacing(12)
        row.addWidget(self.merge_button)
        row.addWidget(self.merge_target)
        row.addSpacing(12)
        row.addWidget(self.reassign_button)
        row.addWidget(self.reassign_target)
        row.addStretch(1)
        return row

    def _build_buttons(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox()
        self.continue_button = buttons.addButton(
            "Continue and export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.continue_button.setProperty("kind", "primary")
        self.plain_button = buttons.addButton(
            "Export without speaker labels", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.cancel_button = buttons.addButton(
            "Cancel batch", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.cancel_button.setProperty("kind", "danger")
        self.continue_button.clicked.connect(self._continue)
        self.plain_button.clicked.connect(self._export_plain)
        self.cancel_button.clicked.connect(self._cancel_batch)
        return buttons

    # ---------------------------------------------------------------- reload

    def _reload(self) -> None:
        self._reload_speakers()
        self._reload_preview()
        self._reload_targets()

    def _reload_speakers(self) -> None:
        table = self.speaker_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, speaker in enumerate(self.transcript.speakers):
            table.insertRow(row)
            values = (
                speaker.identifier,
                speaker.display_name,
                format_duration(self.transcript.speaking_duration(speaker.identifier)),
                str(self.transcript.segment_count(speaker.identifier)),
                "  •  ".join(self.transcript.samples(speaker.identifier)),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column != 1:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, speaker.identifier)
                table.setItem(row, column, cell)
        table.blockSignals(False)

    def _reload_preview(self) -> None:
        table = self.preview_table
        table.setRowCount(0)
        segments = self.transcript.segments
        shown = segments[:_PREVIEW_LIMIT]

        previous_speaker: str | None = "\u0000"
        for row, segment in enumerate(shown):
            table.insertRow(row)
            name = self.transcript.display_name(segment.speaker_id)
            changed = segment.speaker_id != previous_speaker
            previous_speaker = segment.speaker_id

            timecode = (
                self.converter.at_offset(segment.start)
                if self.converter is not None
                else format_duration(segment.start)
            )
            values = (timecode, name if changed else "", segment.text)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, row)
                if column == 1 and segment.has_overlap:
                    cell.setToolTip(
                        "Overlapped speech. Primary speaker chosen by longest "
                        "overlap: " + ", ".join(segment.overlap_identifiers)
                    )
                table.setItem(row, column, cell)

        hidden = len(segments) - len(shown)
        self.preview_note.setText(
            f"{len(segments)} segment(s)."
            + (f" Showing the first {_PREVIEW_LIMIT}." if hidden > 0 else "")
        )

    def _reload_targets(self) -> None:
        for picker in (self.merge_target, self.reassign_target):
            picker.blockSignals(True)
            picker.clear()
            for speaker in self.transcript.speakers:
                picker.addItem(speaker.display_name, speaker.identifier)
            picker.blockSignals(False)
        has_speakers = bool(self.transcript.speakers)
        self.merge_button.setEnabled(len(self.transcript.speakers) > 1)
        self.reassign_button.setEnabled(has_speakers)

    # --------------------------------------------------------------- editing

    def _selected_identifier(self) -> str | None:
        row = self.speaker_table.currentRow()
        if row < 0:
            return None
        cell = self.speaker_table.item(row, 0)
        return None if cell is None else cell.data(Qt.ItemDataRole.UserRole)

    def _on_name_edited(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        identifier_cell = self.speaker_table.item(item.row(), 0)
        if identifier_cell is None:
            return
        identifier = identifier_cell.data(Qt.ItemDataRole.UserRole)
        proposed = item.text()

        try:
            self.transcript.rename(identifier, proposed)
        except SpeakerNameError as error:
            if "already named" in str(error):
                confirmed = QMessageBox.question(
                    self,
                    "Duplicate speaker name",
                    f"{error}\n\nUse the same name for both speakers anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirmed == QMessageBox.StandardButton.Yes:
                    self.transcript.rename(identifier, proposed, allow_duplicate=True)
                    self._reload()
                    return
            else:
                QMessageBox.warning(self, "That name cannot be used", str(error))
            self._reload()
            return

        self._reload()

    def _reset_names(self) -> None:
        self.transcript.reset_names()
        self._reload()

    def _merge_selected(self) -> None:
        source = self._selected_identifier()
        target = self.merge_target.currentData()
        if source is None or target is None:
            QMessageBox.information(
                self, "Choose a speaker", "Select the speaker row to merge first."
            )
            return
        if source == target:
            QMessageBox.information(
                self, "Choose a different speaker", "Pick two different speakers."
            )
            return
        self.transcript.merge(source, target)
        self._reload()

    def _reassign_selected(self) -> None:
        row = self.preview_table.currentRow()
        target = self.reassign_target.currentData()
        if row < 0 or target is None:
            QMessageBox.information(
                self,
                "Choose a segment",
                "Select a line in the transcript preview first.",
            )
            return
        self.transcript.reassign(row, target)
        self._reload()

    # --------------------------------------------------------------- finish

    def _continue(self) -> None:
        self.decision = ReviewDecision.CONTINUE
        self.accept()

    def _export_plain(self) -> None:
        self.decision = ReviewDecision.EXPORT_WITHOUT_SPEAKERS
        self.accept()

    def _cancel_batch(self) -> None:
        self.decision = ReviewDecision.CANCEL_BATCH
        self.reject()

    def review(self) -> ReviewDecision:
        """Show the dialog and return what the user decided."""
        self.exec()
        return self.decision
