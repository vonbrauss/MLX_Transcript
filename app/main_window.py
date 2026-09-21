"""The MLX Transcript main window."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QElapsedTimer,
    QEvent,
    QObject,
    QThread,
    QTimer,
    QUrl,
    Qt,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.models import (
    BatchSummary,
    ConflictChoice,
    ExistingFilePolicy,
    QueueItem,
    QueueStatus,
    ReviewDecision,
    ReviewMode,
    source_identity,
)
from app.collapsible import CollapsibleSection
from app.power import SleepBlocker
from app.settings import (
    LANGUAGE_CHOICES,
    AppSettings,
    load_settings,
    save_settings,
    list_user_presets,
    load_user_preset,
    save_user_preset,
    delete_user_preset,
)
from app.speaker_review import SpeakerReviewDialog
from app.runtime import is_frozen, missing_component_message
from app.widgets import ElidedLabel
from transcription.discovery import (
    MEDIA_EXTENSIONS,
    MEDIA_NAME_FILTER,
    is_supported_media,
)
from transcription.diarization import (
    SherpaOnnxDiarizer,
    SpeakerCountMode,
    backend_available,
)
from transcription.engine import (
    MODEL_CHOICES,
    TranscriptionEngine,
    model_cache_root,
    model_is_cached,
)
from transcription.media_probe import ffprobe_available, format_duration
from transcription.model_cache import DEFAULT_BUNDLE, ModelCache, ModelState
from transcription.outputs import (
    SOURCE_PACKAGE_CONFLICT_MESSAGE,
    FolderLayout,
    NameStyle,
    output_parent_conflicts_with_source,
    output_roots,
)
from transcription.pipeline import BatchJob
from transcription.presets import CleanupPreset, preset_values
from transcription.speaker_presets import SpeakerPreset, preset_values as speaker_preset_values
from transcription.timecode import TimecodeConverter
from app.workers import ModelDownloadWorker, ScanWorker, TranscriptionWorker

__all__ = ["MainWindow", "ConflictDialog"]

logger = logging.getLogger(__name__)

PRIVACY_TEXT = "Processing locally on this Mac"
PRIVACY_DETAIL = (
    "Your media never leaves this Mac. Models download once from Hugging Face "
    "and are reused locally."
)
SPEAKER_TOOLTIP = (
    "Detect who is speaking, locally on this Mac. Transcripts gain Speaker 1, "
    "Speaker 2, and so on, which you can rename before they are written."
)
LOADING_MODEL_TEXT = "Downloading or loading model… the first run can take a while."

#: Shown once, before the first Whisper model is fetched. The weights are a
#: multi-gigabyte download and the progress bar cannot report on it, so the
#: size and the destination are disclosed rather than discovered.
WHISPER_DOWNLOAD_TITLE = "Download the transcription model?"
WHISPER_DOWNLOAD_TEXT = (
    "{label} has not been downloaded yet.\n\n"
    "MLX Transcript will fetch it once from Hugging Face. These models are "
    "several gigabytes, so the first run can take a while on a slow "
    "connection, and the progress bar cannot show how far along it is.\n\n"
    "It is cached in {folder} and reused offline from then on.\n\n"
    + PRIVACY_DETAIL
)
FINISHING_TEXT = (
    "Finishing the current file, then closing. Transcripts already written are safe."
)


def _set_tone(widget: QWidget, tone: str) -> None:
    """Apply a semantic text color without disabling readable content."""
    widget.setProperty("tone", tone)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class ConflictDialog(QMessageBox):
    """Ask what to do about one clip whose transcripts already exist."""

    def __init__(
        self,
        source: Path,
        existing: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transcript already exists")
        self.setIcon(QMessageBox.Icon.Question)
        self.setText(f"A transcript already exists for {Path(source).name}.")
        self.setInformativeText("\n".join(str(path) for path in existing))

        self._overwrite_this = self.addButton(
            "Overwrite this file", QMessageBox.ButtonRole.AcceptRole
        )
        self._overwrite_all = self.addButton(
            "Overwrite all remaining conflicts", QMessageBox.ButtonRole.AcceptRole
        )
        self._skip = self.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        self._cancel = self.addButton(
            "Cancel batch", QMessageBox.ButtonRole.DestructiveRole
        )
        self.setDefaultButton(self._skip)

    def choice(self) -> ConflictChoice:
        """Show the dialog and return the chosen action."""
        self.exec()
        clicked = self.clickedButton()
        if clicked is self._overwrite_this:
            return ConflictChoice.OVERWRITE_THIS
        if clicked is self._overwrite_all:
            return ConflictChoice.OVERWRITE_ALL
        if clicked is self._cancel:
            return ConflictChoice.CANCEL_BATCH
        return ConflictChoice.SKIP_THIS


class MainWindow(QMainWindow):
    """Folder pickers, transcription options, and the media queue."""

    QUEUE_COLUMNS = ("File", "Folder", "Duration", "Status")

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()
        self.items: list[QueueItem] = []
        self.last_summary: BatchSummary | None = None
        # Seam for the headless harness: replace to run a batch without MLX.
        self.engine_factory: Callable[[], TranscriptionEngine] = self._build_engine
        self.diarizer_factory: Callable[[], object] = self._build_diarizer
        self.model_cache = ModelCache()
        # Set while a preset fills the numeric controls, so those programmatic
        # changes are not mistaken for the user editing a value.
        self._applying_preset = False
        self._applying_speaker_preset = False

        self._scan_thread: QThread | None = None
        self._scan_worker: ScanWorker | None = None
        self._batch_thread: QThread | None = None
        self._batch_worker: TranscriptionWorker | None = None

        self._sleep_blocker = SleepBlocker()
        self._elapsed = QElapsedTimer()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(400)
        self._settings_save_timer.timeout.connect(self._save_current_settings)
        self._auto_save_connected = False
        self._scan_sources: list[Path] = []
        # Duplicate detection reads from these rather than resolving the whole
        # queue again for every file the scan finds.
        self._queued_identities: set[Path] = set()
        self._queued_roots: list[Path] = []
        # Threads that have been asked to stop but may still be winding down.
        # The window waits on these before it lets itself be destroyed.
        self._live_threads: list[QThread] = []
        self._closing = False
        # Remembered so the progress bar can return from its indeterminate
        # model-loading state to the real count.
        self._batch_total = 0
        self._batch_done = 0
        #: ``(path, reason)`` for files the last scan turned away.
        self._skipped_files: list[tuple[Path, str]] = []

        self.setWindowTitle("MLX Transcript")
        self.setMinimumSize(940, 560)
        self.resize(1180, 720)
        self._build_ui()
        self._restore_settings()
        self._refresh_output_preview()
        self._update_actions()
        self._check_ffprobe()

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame(self)
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 15, 22, 12)
        header_layout.setSpacing(12)

        brand_mark = QLabel("MLX")
        brand_mark.setProperty("kind", "brandMark")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(48, 48)
        self.brand_mark = brand_mark

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("MLX Transcript")
        title.setProperty("kind", "appTitle")
        subtitle = QLabel("Local transcription for Apple silicon")
        subtitle.setProperty("tone", "secondary")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.app_title_label = title

        header_layout.addWidget(brand_mark)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)
        root_layout.addWidget(header)

        self.folders_group = self._build_folders_group()
        self.transcription_group = self._build_options_group()
        self.output_group = self._build_output_group()
        self.speaker_group = self._build_speaker_group()
        self.queue_group = self._build_queue_group()
        self.help_group = self._build_help_group()

        self._build_advanced_group()

        self.section_tabs = QTabBar(self)
        self.section_tabs.setObjectName("sectionTabs")
        self.section_tabs.setDocumentMode(True)
        self.section_tabs.setExpanding(True)
        self.section_tabs.setDrawBase(False)
        self.section_tabs.setUsesScrollButtons(False)

        self.section_stack = QStackedWidget(self)
        self.section_stack.setObjectName("sectionStack")
        self.section_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.section_keys = (
            "folders",
            "transcription",
            "output",
            "speakers",
            "advanced",
            "queue",
            "help",
        )
        section_pages = (
            ("Folders", self.folders_group),
            ("Transcription", self.transcription_group),
            ("Output", self.output_group),
            ("Speaker Detection", self.speaker_group),
            ("Advanced Settings", self.advanced_group),
            ("Queue", self.queue_group),
            ("Help", self.help_group),
        )
        page_guidance = (
            ("Choose your media", "Select one media file or a folder. Folders and their subfolders are scanned automatically."),
            ("Transcription preferences", "Choose the spoken language and a cleanup preset. Large v3 favors accuracy; Turbo favors speed."),
            ("Save your transcripts", "Choose formats, file names, and how to organize the results."),
            ("Identify speakers", "Add speaker labels, then review and rename them before saving."),
            ("Fine-tune transcription", "Start with a cleanup preset in Transcription. Adjust these values only when needed."),
            ("Your processing queue", "Files run in order. Select a row and hover over its status for details."),
            ("Help & getting started", "A quick guide to preparing media and saving useful transcripts."),
        )
        for index, (label, page) in enumerate(section_pages):
            heading, description = page_guidance[index]
            intro = QWidget()
            intro_box = QVBoxLayout(intro)
            intro_box.setContentsMargins(4, 0, 4, 12)
            intro_box.setSpacing(4)
            title = QLabel(heading)
            title.setProperty("kind", "pageTitle")
            note = QLabel(description)
            note.setWordWrap(True)
            note.setProperty("tone", "secondary")
            intro_box.addWidget(title)
            intro_box.addWidget(note)
            page.layout().insertWidget(1, intro)
            if index in (1, 2, 3):
                page._content.layout().addStretch(1)
            if index in (0, 1, 2, 3, 4):
                # Preserve each form's natural height on compact displays.
                # Only settings scroll; navigation and processing controls stay put.
                content = page._content
                content.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
                page.layout().removeWidget(content)
                scroll = QScrollArea()
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                scroll.setWidget(content)
                page.layout().addWidget(scroll, 1)
            page.setProperty("tabPage", True)
            page.setHeaderVisible(False)
            page.setChecked(True)
            self.section_tabs.addTab(label)
            self.section_stack.addWidget(page)
        self.section_tabs.currentChanged.connect(self.section_stack.setCurrentIndex)

        nav = QFrame(self)
        nav.setObjectName("sectionNavigation")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(18, 0, 18, 0)
        nav_layout.addWidget(self.section_tabs)
        root_layout.addWidget(nav)

        workspace = QFrame(self)
        workspace.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(18, 12, 18, 12)
        workspace_layout.addWidget(self.section_stack)
        root_layout.addWidget(workspace, stretch=1)

        # Keep progress and the primary actions visible even when the settings
        # and queue need to scroll on a shorter display.
        footer = QFrame(self)
        footer.setObjectName("appFooter")
        self.footer = footer
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(18, 9, 18, 12)
        footer_layout.setSpacing(8)
        footer_layout.addWidget(self._build_progress_block())
        footer_layout.addLayout(self._build_action_row())
        root_layout.addWidget(footer)

        self.setCentralWidget(central)
        # Each section hands its layout over before filling it, so alignment is
        # applied again now that every label and control exists.
        self.sections = (
            self.folders_group,
            self.transcription_group,
            self.output_group,
            self.speaker_group,
            self.advanced_group,
            self.queue_group,
            self.help_group,
        )
        for section in self.sections:
            section.apply_alignment()
        self.statusBar().showMessage("Choose a media file or folder to build the queue.")

    def _build_folders_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Folders")
        grid = QGridLayout()
        group.setContentLayout(grid)
        grid.setColumnStretch(1, 1)

        self.source_field = QLineEdit()
        self.source_field.setPlaceholderText("Media file or folder to transcribe")
        self.source_field.editingFinished.connect(self._on_source_text_changed)
        self.source_button = QPushButton("Choose Folder…")
        self.source_button.clicked.connect(self._choose_source_folder)
        self.source_file_button = QPushButton("Choose File…")
        self.source_file_button.clicked.connect(self._choose_source_file)

        self.output_field = QLineEdit()
        self.output_field.setPlaceholderText(
            "Where the Transcription folder is created"
        )
        self.output_field.editingFinished.connect(self._on_output_text_changed)
        self.output_button = QPushButton("Browse…")
        self.output_button.clicked.connect(self._choose_output_parent)

        grid.addWidget(QLabel("Media source"), 0, 0)
        grid.addWidget(self.source_field, 0, 1)
        grid.addWidget(self.source_button, 0, 2)
        grid.addWidget(self.source_file_button, 0, 3)
        grid.addWidget(QLabel("Save transcripts to"), 1, 0)
        grid.addWidget(self.output_field, 1, 1)
        grid.addWidget(self.output_button, 1, 3)

        # A path too wide for its cell is elided in the middle rather than
        # clipped, so a shortened path never reads as the wrong path.
        self.scriptsync_preview = ElidedLabel()
        self.timecoded_preview = QLabel()
        self.subtitles_preview = QLabel()
        for preview in (self.timecoded_preview, self.subtitles_preview):
            preview.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            preview.setWordWrap(True)
        for preview in (self.scriptsync_preview, self.timecoded_preview, self.subtitles_preview):
            _set_tone(preview, "secondary")

        grid.addWidget(QLabel("Transcripts saved in"), 2, 0)
        grid.addWidget(self.scriptsync_preview, 2, 1, 1, 3)
        self.drop_target = QFrame()
        self.drop_target.setObjectName("dropTarget")
        self.drop_target.setAcceptDrops(True)
        self.drop_target.installEventFilter(self)
        self.drop_target.setMinimumHeight(150)
        self.drop_target.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        drop_layout = QVBoxLayout(self.drop_target)
        drop_layout.setContentsMargins(20, 20, 20, 20)
        drop_layout.setSpacing(6)
        drop_layout.addStretch(1)
        drop_title = QLabel("Drop media files or folders here")
        drop_title.setProperty("kind", "dropTitle")
        drop_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        drop_hint = QLabel("They will be added to the current queue")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _set_tone(drop_hint, "secondary")
        drop_layout.addWidget(drop_title)
        drop_layout.addWidget(drop_hint)
        drop_layout.addStretch(1)
        grid.addWidget(self.drop_target, 3, 0, 1, 4)
        grid.setRowStretch(3, 1)
        self.timecoded_preview.hide()
        self.subtitles_preview.hide()
        return group

    def _build_options_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Transcription")
        row = QHBoxLayout()

        self.model_picker = QComboBox()
        self.model_picker.addItems(list(MODEL_CHOICES))
        self.model_picker.setToolTip(
            "Large v3 is the accuracy-first default. Turbo is the faster "
            "option for quicker turnaround."
        )

        self.language_picker = QComboBox()
        self.language_picker.addItems(list(LANGUAGE_CHOICES))

        self.policy_picker = QComboBox()
        for policy in ExistingFilePolicy:
            self.policy_picker.addItem(policy.label, policy)

        self.cleanup_picker = QComboBox()
        for preset in CleanupPreset:
            self.cleanup_picker.addItem(preset.label, preset)
        self.cleanup_picker.setToolTip(
            "Sets the three Whisper decoding thresholds under Advanced Settings."
        )
        self.cleanup_picker.currentIndexChanged.connect(self._on_cleanup_changed)

        self.cleanup_description = QLabel()
        self.cleanup_description.setWordWrap(True)
        _set_tone(self.cleanup_description, "secondary")

        left = QFormLayout()
        left.addRow("Model", self.model_picker)
        left.addRow("Language", self.language_picker)
        right = QFormLayout()
        right.addRow("Existing files", self.policy_picker)
        right.addRow("Transcription cleanup", self.cleanup_picker)
        right.addRow("", self.cleanup_description)

        row.addLayout(left, stretch=1)
        row.addLayout(right, stretch=1)

        outer = QVBoxLayout()
        outer.setContentsMargins(14, 12, 14, 14)
        preset_row = QHBoxLayout()
        self.user_preset_picker = QComboBox()
        self.user_preset_picker.currentIndexChanged.connect(
            self._on_user_preset_selected
        )
        self.save_preset_button = QPushButton("Save as New…")
        self.save_preset_button.clicked.connect(self._save_user_preset_as)
        self.update_preset_button = QPushButton("Update")
        self.update_preset_button.clicked.connect(self._update_user_preset)
        self.delete_preset_button = QPushButton("Delete")
        self.delete_preset_button.setProperty("kind", "danger")
        self.delete_preset_button.clicked.connect(self._delete_user_preset)
        preset_row.addWidget(QLabel("My preset"))
        preset_row.addWidget(self.user_preset_picker, 1)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.update_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        outer.addLayout(preset_row)
        outer.addLayout(row)
        group.setContentLayout(outer)
        return group

    def _build_output_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Output")
        outer = QVBoxLayout()
        group.setContentLayout(outer)

        row = QHBoxLayout()
        left = QFormLayout()
        right = QFormLayout()

        self.name_style_picker = QComboBox()
        for style in NameStyle:
            self.name_style_picker.addItem(style.label, style)
        self.name_style_picker.currentIndexChanged.connect(
            self._on_output_options_changed
        )

        self.custom_suffix_field = QLineEdit("_transcript")
        self.custom_suffix_field.setPlaceholderText("_transcript")
        self.custom_suffix_field.textChanged.connect(self._refresh_output_description)

        self.folder_layout_picker = QComboBox()
        for choice in FolderLayout:
            self.folder_layout_picker.addItem(choice.label, choice)
        self.folder_layout_picker.currentIndexChanged.connect(
            self._refresh_output_description
        )

        left.addRow("File names", self.name_style_picker)
        left.addRow("Custom suffix", self.custom_suffix_field)
        right.addRow("Folders", self.folder_layout_picker)

        formats = QWidget()
        format_row = QHBoxLayout(formats)
        format_row.setContentsMargins(0, 0, 0, 0)
        self.output_scriptsync = QCheckBox("ScriptSync")
        self.output_timecoded = QCheckBox("Timecoded")
        self.output_srt = QCheckBox("SRT")
        self.output_vtt = QCheckBox("WebVTT")
        for checkbox in (
            self.output_scriptsync,
            self.output_timecoded,
            self.output_srt,
            self.output_vtt,
        ):
            checkbox.toggled.connect(self._refresh_output_description)
            format_row.addWidget(checkbox)
        format_row.addStretch(1)
        right.addRow("Create", formats)

        row.addLayout(left, stretch=1)
        row.addLayout(right, stretch=1)
        outer.addLayout(row)

        self.output_description = QLabel()
        self.output_description.setWordWrap(True)
        _set_tone(self.output_description, "secondary")
        outer.addWidget(self.output_description)
        return group

    @Slot()
    def _on_output_options_changed(self) -> None:
        style = self._picker_enum(
            self.name_style_picker, NameStyle, NameStyle.ORIGINAL
        )
        self.custom_suffix_field.setEnabled(style is NameStyle.CUSTOM_SUFFIX)
        self._refresh_output_description()

    @Slot()
    def _refresh_output_description(self) -> None:
        if not hasattr(self, "output_description"):
            return
        layout = self._picker_enum(
            self.folder_layout_picker, FolderLayout, FolderLayout.TREE
        )
        location = (
            "Source folders will be mirrored under each output type."
            if layout is FolderLayout.TREE
            else "Files will be collected by output type; duplicate names receive (2), (3), and so on."
        )
        selected = [
            label
            for checkbox, label in (
                (self.output_scriptsync, "ScriptSync"),
                (self.output_timecoded, "Timecoded"),
                (self.output_srt, "SRT"),
                (self.output_vtt, "WebVTT"),
            )
            if checkbox.isChecked()
        ]
        formats = ", ".join(selected) if selected else "No formats selected"
        self.output_description.setText(f"{location} Creating: {formats}.")

    def _build_speaker_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Speaker Detection", expanded=False)
        outer = QVBoxLayout()
        group.setContentLayout(outer)

        self.detect_speakers = QCheckBox("Detect speakers")
        self.detect_speakers.setToolTip(SPEAKER_TOOLTIP)
        self.detect_speakers.toggled.connect(self._on_detect_speakers_toggled)
        outer.addWidget(self.detect_speakers)

        self.speaker_preset_picker = QComboBox()
        for preset in SpeakerPreset:
            self.speaker_preset_picker.addItem(preset.label, preset)
        self.speaker_preset_picker.currentIndexChanged.connect(
            self._on_speaker_preset_changed
        )

        self.speaker_preset_description = QLabel()
        self.speaker_preset_description.setWordWrap(True)
        _set_tone(self.speaker_preset_description, "secondary")

        preset_form = QFormLayout()
        preset_form.setContentsMargins(0, 0, 0, 0)
        preset_form.setVerticalSpacing(2)
        preset_form.addRow("Detection style", self.speaker_preset_picker)
        outer.addLayout(preset_form)
        preset_form.addRow("", self.speaker_preset_description)

        row = QHBoxLayout()

        self.speaker_count_picker = QComboBox()
        for mode in SpeakerCountMode:
            self.speaker_count_picker.addItem(mode.label, mode)
        self.speaker_count_picker.currentIndexChanged.connect(
            self._on_speaker_count_mode_changed
        )

        self.exact_speakers_spin = QSpinBox()
        self.exact_speakers_spin.setRange(1, 24)
        self.exact_speakers_spin.setToolTip(
            "How many people are speaking. Used only with Exact number."
        )

        self.review_mode_picker = QComboBox()
        for mode in ReviewMode:
            self.review_mode_picker.addItem(mode.label, mode)

        self.speakers_in_scriptsync = QCheckBox("Include speaker names in ScriptSync")
        self.speakers_in_scriptsync.setToolTip(
            "Timecoded transcripts always carry speakers while detection is on."
        )
        self.speakers_in_subtitles = QCheckBox("Include speaker names in subtitles")

        left = QFormLayout()
        left.addRow("Speaker count", self.speaker_count_picker)
        left.addRow("Exact number", self.exact_speakers_spin)
        right = QFormLayout()
        right.addRow("Review", self.review_mode_picker)
        right.addRow("ScriptSync", self.speakers_in_scriptsync)
        right.addRow("SRT / WebVTT", self.speakers_in_subtitles)

        row.addLayout(left, stretch=1)
        row.addLayout(right, stretch=1)
        outer.addLayout(row)

        self.speaker_advanced_toggle = QCheckBox("Advanced speaker settings")
        self.speaker_advanced_toggle.toggled.connect(
            self._on_speaker_advanced_toggled
        )
        outer.addWidget(self.speaker_advanced_toggle)

        self.speaker_advanced_panel = QWidget()
        self.speaker_advanced_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        advanced = QFormLayout(self.speaker_advanced_panel)
        advanced.setContentsMargins(18, 0, 0, 0)
        advanced.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        advanced.setVerticalSpacing(8)

        self.speaker_threshold_spin = QDoubleSpinBox()
        self.speaker_threshold_spin.setRange(0.05, 1.50)
        self.speaker_threshold_spin.setSingleStep(0.05)
        self.speaker_threshold_spin.setDecimals(2)
        self.speaker_threshold_spin.setToolTip(
            "Higher values combine similar voices; lower values create more speakers. "
            "Ignored when Exact number is selected."
        )
        advanced.addRow("Speaker separation", self.speaker_threshold_spin)

        self.min_speech_spin = QDoubleSpinBox()
        self.min_speech_spin.setRange(0.05, 5.0)
        self.min_speech_spin.setSingleStep(0.05)
        self.min_speech_spin.setDecimals(2)
        self.min_speech_spin.setSuffix(" s")
        self.min_speech_spin.setToolTip(
            "The shortest speech turn to keep. Raise it to ignore false, very short changes."
        )
        advanced.addRow("Minimum speaker turn", self.min_speech_spin)

        self.min_pause_spin = QDoubleSpinBox()
        self.min_pause_spin.setRange(0.0, 5.0)
        self.min_pause_spin.setSingleStep(0.05)
        self.min_pause_spin.setDecimals(2)
        self.min_pause_spin.setSuffix(" s")
        self.min_pause_spin.setToolTip(
            "The silence needed to separate speech regions. Lower values suit fast exchanges."
        )
        advanced.addRow("Minimum pause between turns", self.min_pause_spin)

        self.speaker_tolerance_spin = QDoubleSpinBox()
        self.speaker_tolerance_spin.setRange(0.0, 3.0)
        self.speaker_tolerance_spin.setSingleStep(0.05)
        self.speaker_tolerance_spin.setDecimals(2)
        self.speaker_tolerance_spin.setSuffix(" s")
        self.speaker_tolerance_spin.setToolTip(
            "How far a transcript word may sit from detected speech and still receive a speaker."
        )
        advanced.addRow("Word assignment tolerance", self.speaker_tolerance_spin)

        self.speaker_merge_spin = QDoubleSpinBox()
        self.speaker_merge_spin.setRange(0.05, 5.0)
        self.speaker_merge_spin.setSingleStep(0.05)
        self.speaker_merge_spin.setDecimals(2)
        self.speaker_merge_spin.setSuffix(" s")
        self.speaker_merge_spin.setToolTip(
            "Nearby words from the same person are joined when the gap is below this value."
        )
        advanced.addRow("Merge nearby speech", self.speaker_merge_spin)
        outer.addWidget(self.speaker_advanced_panel)

        self.speaker_setting_spins = (
            self.speaker_threshold_spin,
            self.min_speech_spin,
            self.min_pause_spin,
            self.speaker_tolerance_spin,
            self.speaker_merge_spin,
        )
        for spin in self.speaker_setting_spins:
            spin.valueChanged.connect(self._on_speaker_setting_edited)

        self.model_status_label = QLabel()
        self.model_status_label.setWordWrap(True)
        _set_tone(self.model_status_label, "secondary")
        outer.addWidget(self.model_status_label)

        model_actions = QHBoxLayout()
        self.check_models_button = QPushButton("Check or Repair Models")
        self.check_models_button.setToolTip(
            "Verify the cached speaker models and download them again if a "
            "file is missing or damaged."
        )
        self.check_models_button.clicked.connect(self._check_or_repair_models)
        model_actions.addWidget(self.check_models_button)
        model_actions.addStretch(1)
        outer.addLayout(model_actions)

        self.speaker_controls = (
            self.speaker_preset_picker,
            self.speaker_count_picker,
            self.exact_speakers_spin,
            self.review_mode_picker,
            self.speakers_in_scriptsync,
            self.speakers_in_subtitles,
            self.speaker_advanced_toggle,
            self.check_models_button,
            *self.speaker_setting_spins,
        )
        return group

    def _build_advanced_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Advanced Settings", expanded=False)
        self.advanced_group = group

        form = QFormLayout()
        group.setContentLayout(form)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self.preset_note = QLabel()
        self.preset_note.setWordWrap(True)
        _set_tone(self.preset_note, "secondary")
        form.addRow(self.preset_note)

        self.hallucination_spin = QDoubleSpinBox()
        self.hallucination_spin.setRange(0.0, 10.0)
        self.hallucination_spin.setSingleStep(0.1)
        self.hallucination_spin.setDecimals(2)
        self.hallucination_spin.setSuffix(" s")
        self.hallucination_spin.setToolTip(
            "hallucination_silence_threshold — seconds of silence used to "
            "reject likely hallucinations. Requires word timestamps, which "
            "stay on."
        )
        self.hallucination_field = self._advanced_field(
            self.hallucination_spin,
            "Lower values are more aggressive about removing text generated "
            "around silence.",
        )

        self.no_speech_spin = QDoubleSpinBox()
        self.no_speech_spin.setRange(0.0, 1.0)
        self.no_speech_spin.setSingleStep(0.05)
        self.no_speech_spin.setDecimals(2)
        self.no_speech_spin.setToolTip(
            "no_speech_threshold — probability above which a "
            "low-confidence segment counts as silence."
        )
        self.no_speech_field = self._advanced_field(
            self.no_speech_spin,
            "Lower values reject noise and silence more readily. Very low "
            "values may remove quiet speech.",
        )

        self.logprob_spin = QDoubleSpinBox()
        self.logprob_spin.setRange(-10.0, 0.0)
        self.logprob_spin.setSingleStep(0.1)
        self.logprob_spin.setDecimals(2)
        self.logprob_spin.setToolTip(
            "logprob_threshold — average log probability below which a "
            "segment is treated as unreliable."
        )
        self.logprob_field = self._advanced_field(
            self.logprob_spin,
            "Higher values reject more uncertain words. Very high values may "
            "remove difficult dialogue.",
        )

        form.addRow("Silence around suspicious text", self.hallucination_field)
        form.addRow("Speech detection sensitivity", self.no_speech_field)
        form.addRow("Minimum transcription confidence", self.logprob_field)

        self.advanced_fields = (
            self.preset_note,
            self.hallucination_field,
            self.no_speech_field,
            self.logprob_field,
        )
        self.threshold_spins = (
            self.hallucination_spin,
            self.no_speech_spin,
            self.logprob_spin,
        )
        for spin in self.threshold_spins:
            spin.valueChanged.connect(self._on_threshold_edited)

        group.toggled.connect(self._on_advanced_toggled)
        self._set_advanced_visible(False)
        return group

    @staticmethod
    def _advanced_field(spin: QDoubleSpinBox, description: str) -> QWidget:
        """Pair a numeric control with its plain-language explanation."""
        field = QWidget()
        box = QVBoxLayout(field)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        note = QLabel(description)
        note.setWordWrap(True)
        _set_tone(note, "secondary")
        note.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        box.addWidget(spin)
        box.addWidget(note)
        field.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        return field

    def _build_queue_group(self) -> CollapsibleSection:
        group = CollapsibleSection("Queue")
        layout = QVBoxLayout()
        group.setContentLayout(layout)

        self.queue_table = QTableWidget(0, len(self.QUEUE_COLUMNS))
        self.queue_table.setMinimumHeight(90)
        self.queue_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.queue_table.setHorizontalHeaderLabels(list(self.QUEUE_COLUMNS))
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.queue_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setAcceptDrops(True)
        self.queue_table.installEventFilter(self)
        self.queue_drop_viewport = self.queue_table.viewport()
        self.queue_drop_viewport.setAcceptDrops(True)
        self.queue_drop_viewport.installEventFilter(self)
        self.queue_table.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.queue_table.setShowGrid(False)
        self.queue_table.verticalHeader().setDefaultSectionSize(34)
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.queue_summary = QLabel("Choose a media file or folder in Folders to add it to this queue.")
        _set_tone(self.queue_summary, "secondary")

        actions = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self._remove_selected_items)
        self.clear_queue_button = QPushButton("Clear Queue")
        self.clear_queue_button.setProperty("kind", "danger")
        self.clear_queue_button.clicked.connect(self._clear_queue)
        self.reveal_source_button = QPushButton("Reveal Source")
        self.reveal_source_button.clicked.connect(self._reveal_selected_source)
        actions.addWidget(self.remove_selected_button)
        actions.addWidget(self.clear_queue_button)
        actions.addWidget(self.reveal_source_button)
        actions.addStretch(1)
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection_changed)

        layout.addLayout(actions)
        layout.addWidget(self.queue_table, stretch=1)
        layout.addWidget(self.queue_summary)
        return group

    def _build_help_group(self) -> CollapsibleSection:
        """Build the self-contained quick-start instructions page."""
        group = CollapsibleSection("Help")
        layout = QVBoxLayout()
        group.setContentLayout(layout)

        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        help_text.setReadOnly(True)
        help_text.setObjectName("helpBrowser")
        help_text.document().setDocumentMargin(18)
        help_text.document().setDefaultStyleSheet(
            "h2 {font-size:18px; color:#F2F5F9;} p, li {line-height:145%;} "
            "li {margin-bottom:7px;}"
        )
        help_text.setHtml(
            "<h2>Getting started</h2>"
            "<ol>"
            "<li>Open <b>Folders</b> and add media with the buttons, or drag files and folders into the drop area. "
            "MLX Transcript accepts any local media file FFmpeg can read that contains audio, whatever its "
            "filename, and reports anything it had to skip.</li>"
            "<li>Choose a destination under <b>Save transcripts to</b>. The app creates a Transcription folder there.</li>"
            "<li>Review, remove, or clear files in <b>Queue</b>. Multiple folders stay separate in the output tree.</li>"
            "<li>Choose your Whisper model and cleanup preset in <b>Transcription</b>.</li>"
            "<li>Use <b>Save as New</b> to keep a working setup as a named preset.</li>"
            "<li>Choose filename, folder, and format options in <b>Output</b>.</li>"
            "<li>Press <b>Start Transcription</b>. Queue opens automatically while files process.</li>"
            "</ol>"
            "<h2>Transcript formats</h2>"
            "<p><b>ScriptSync</b> creates plain text for Avid. <b>Timecoded</b> adds source "
            "timecode. <b>SRT</b> and <b>WebVTT</b> create captions.</p>"
            "<h2>Speaker detection</h2>"
            "<p>Turn on <b>Detect speakers</b> only when you need speaker labels. "
            "If you know how many people are speaking, choose <b>Exact number</b> "
            "for more reliable results. Speaker models download once and then run locally.</p>"
            "<h2>Privacy</h2>"
            f"<p><b>{PRIVACY_DETAIL}</b> There is no cloud service, no analytics, and "
            "no API key. The transcription model is fetched once on first use and "
            "the speaker models are fetched once when you turn speaker detection "
            "on. After that the application works offline.</p>"
            "<h2>Helpful notes</h2>"
            "<p>Existing output files follow the choice in Transcription. Your "
            "latest settings are saved automatically. You can cancel safely after "
            "the current file finishes, and closing the window waits for it too.</p>"
        )
        layout.addWidget(help_text, stretch=1)
        self.help_text = help_text
        return group

    def _build_progress_block(self) -> QWidget:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        row = QHBoxLayout()
        self.current_file_label = ElidedLabel("Choose a media file or folder to begin")
        _set_tone(self.current_file_label, "secondary")
        self.elapsed_label = QLabel("Elapsed 00:00:00")
        _set_tone(self.elapsed_label, "secondary")
        row.addWidget(self.current_file_label, stretch=1)
        row.addWidget(self.elapsed_label)

        layout.addWidget(self.progress_bar)
        layout.addLayout(row)
        return block

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.privacy_label = QLabel(PRIVACY_TEXT)
        self.privacy_label.setToolTip(PRIVACY_DETAIL)
        privacy_font = QFont(self.privacy_label.font())
        privacy_font.setBold(True)
        self.privacy_label.setFont(privacy_font)
        self.privacy_label.setFrameShape(QFrame.Shape.NoFrame)
        self.privacy_label.setProperty("kind", "privacy")

        self.start_button = QPushButton("Start Transcription")
        self.start_button.setDefault(True)
        self.start_button.setProperty("kind", "primary")
        self.start_button.clicked.connect(self._start_transcription)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("kind", "danger")
        self.cancel_button.clicked.connect(self._cancel)

        self.reveal_button = QPushButton("Reveal Output")
        self.reveal_button.clicked.connect(self._reveal_output)

        row.addWidget(self.privacy_label)
        row.addStretch(1)
        row.addWidget(self.reveal_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.start_button)
        return row

    # -------------------------------------------------------------- settings

    def _restore_settings(self) -> None:
        self.source_field.setText(self.settings.source_folder)
        self.output_field.setText(self.settings.output_parent)
        self.model_picker.setCurrentText(self.settings.model_label)
        self.language_picker.setCurrentText(self.settings.language_label)
        index = self.policy_picker.findData(self.settings.existing_policy)
        if index >= 0:
            self.policy_picker.setCurrentIndex(index)
        name_index = self.name_style_picker.findData(self.settings.name_style)
        if name_index >= 0:
            self.name_style_picker.setCurrentIndex(name_index)
        self.custom_suffix_field.setText(self.settings.custom_suffix)
        layout_index = self.folder_layout_picker.findData(self.settings.folder_layout)
        if layout_index >= 0:
            self.folder_layout_picker.setCurrentIndex(layout_index)
        self.output_scriptsync.setChecked(self.settings.output_scriptsync)
        self.output_timecoded.setChecked(self.settings.output_timecoded)
        self.output_srt.setChecked(self.settings.output_srt)
        self.output_vtt.setChecked(self.settings.output_vtt)
        self._on_output_options_changed()
        # The stored numbers are restored first and the picker is set without
        # firing, so a Custom combination is never overwritten at launch.
        self._applying_preset = True
        try:
            self.hallucination_spin.setValue(
                self.settings.hallucination_silence_threshold
            )
            self.no_speech_spin.setValue(self.settings.no_speech_threshold)
            self.logprob_spin.setValue(self.settings.logprob_threshold)
        finally:
            self._applying_preset = False

        cleanup_index = self.cleanup_picker.findData(self.settings.cleanup_preset)
        if cleanup_index >= 0:
            self.cleanup_picker.blockSignals(True)
            self.cleanup_picker.setCurrentIndex(cleanup_index)
            self.cleanup_picker.blockSignals(False)
        self._refresh_cleanup_text()

        for section in self.sections:
            section.setChecked(True)
        self._set_advanced_visible(True)
        self.section_tabs.setCurrentIndex(0)
        self.section_stack.setCurrentIndex(0)

        self.detect_speakers.setChecked(self.settings.detect_speakers)
        self._applying_speaker_preset = True
        try:
            self.speaker_threshold_spin.setValue(self.settings.clustering_threshold)
            self.min_speech_spin.setValue(self.settings.min_duration_on)
            self.min_pause_spin.setValue(self.settings.min_duration_off)
            self.speaker_tolerance_spin.setValue(self.settings.nearest_tolerance)
            self.speaker_merge_spin.setValue(self.settings.merge_gap)
        finally:
            self._applying_speaker_preset = False
        speaker_preset_index = self.speaker_preset_picker.findData(
            self.settings.speaker_preset
        )
        if speaker_preset_index >= 0:
            self.speaker_preset_picker.blockSignals(True)
            self.speaker_preset_picker.setCurrentIndex(speaker_preset_index)
            self.speaker_preset_picker.blockSignals(False)
        self.speaker_advanced_toggle.setChecked(
            self.settings.speaker_advanced_expanded
        )
        self._on_speaker_advanced_toggled(
            self.settings.speaker_advanced_expanded
        )
        self._refresh_speaker_preset_text()
        count_index = self.speaker_count_picker.findData(
            self.settings.speaker_count_mode
        )
        if count_index >= 0:
            self.speaker_count_picker.setCurrentIndex(count_index)
        self.exact_speakers_spin.setValue(self.settings.exact_speakers)
        review_index = self.review_mode_picker.findData(self.settings.review_mode)
        if review_index >= 0:
            self.review_mode_picker.setCurrentIndex(review_index)
        self.speakers_in_scriptsync.setChecked(self.settings.speakers_in_scriptsync)
        self.speakers_in_subtitles.setChecked(self.settings.speakers_in_subtitles)
        self._on_detect_speakers_toggled(self.settings.detect_speakers)
        self._refresh_user_presets()
        if not self._auto_save_connected:
            self._connect_auto_save()
            self._auto_save_connected = True

    def _refresh_user_presets(self, selected: str = "") -> None:
        self.user_preset_picker.blockSignals(True)
        self.user_preset_picker.clear()
        self.user_preset_picker.addItem("Choose a saved preset", "")
        for name in list_user_presets():
            self.user_preset_picker.addItem(name, name)
        index = self.user_preset_picker.findData(selected)
        self.user_preset_picker.setCurrentIndex(max(0, index))
        self.user_preset_picker.blockSignals(False)
        has_selection = bool(self.user_preset_picker.currentData())
        self.update_preset_button.setEnabled(has_selection)
        self.delete_preset_button.setEnabled(has_selection)

    def _connect_auto_save(self) -> None:
        """Save settled edits so an unexpected quit does not lose a setup."""
        controls = (
            self.source_field, self.output_field, self.custom_suffix_field,
            self.model_picker, self.language_picker, self.policy_picker,
            self.cleanup_picker, self.name_style_picker, self.folder_layout_picker,
            self.speaker_preset_picker, self.speaker_count_picker,
            self.review_mode_picker, self.output_scriptsync, self.output_timecoded,
            self.output_srt, self.output_vtt, self.detect_speakers,
            self.speaker_advanced_toggle, self.speakers_in_scriptsync,
            self.speakers_in_subtitles, self.exact_speakers_spin,
            *self.threshold_spins, *self.speaker_setting_spins,
        )
        for control in controls:
            if isinstance(control, QLineEdit):
                control.textChanged.connect(self._schedule_settings_save)
            elif isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._schedule_settings_save)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._schedule_settings_save)
            else:
                control.valueChanged.connect(self._schedule_settings_save)

    @Slot()
    def _schedule_settings_save(self, *_args) -> None:
        if not self.is_transcribing:
            self._settings_save_timer.start()

    @Slot()
    def _save_current_settings(self) -> None:
        self.settings = self._collect_settings()
        save_settings(self.settings)

    @Slot(int)
    def _on_user_preset_selected(self, _index: int) -> None:
        name = str(self.user_preset_picker.currentData() or "")
        self.update_preset_button.setEnabled(bool(name))
        self.delete_preset_button.setEnabled(bool(name))
        if not name:
            return
        current_section = self.section_keys[self.section_tabs.currentIndex()]
        loaded = load_user_preset(name, self._collect_settings())
        if loaded is None:
            self._refresh_user_presets()
            return
        self.settings = loaded
        self._restore_settings()
        self._select_section(current_section)
        self.user_preset_picker.blockSignals(True)
        self.user_preset_picker.setCurrentIndex(
            self.user_preset_picker.findData(name)
        )
        self.user_preset_picker.blockSignals(False)
        self.update_preset_button.setEnabled(True)
        self.delete_preset_button.setEnabled(True)
        self.statusBar().showMessage(f'Preset “{name}” applied.')
        self._save_current_settings()

    @Slot()
    def _save_user_preset_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = name.strip()
        if not accepted or not name:
            return
        if name in list_user_presets():
            answer = QMessageBox.question(
                self, "Replace preset?", f'A preset named “{name}” already exists.',
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Save:
                return
        save_user_preset(name, self._collect_settings())
        self._refresh_user_presets(name)
        self.statusBar().showMessage(f'Preset “{name}” saved.')

    @Slot()
    def _update_user_preset(self) -> None:
        name = str(self.user_preset_picker.currentData() or "")
        if name:
            save_user_preset(name, self._collect_settings())
            self.statusBar().showMessage(f'Preset “{name}” updated.')

    @Slot()
    def _delete_user_preset(self) -> None:
        name = str(self.user_preset_picker.currentData() or "")
        if not name:
            return
        answer = QMessageBox.question(
            self, "Delete preset?", f'Delete the preset “{name}”?',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            delete_user_preset(name)
            self._refresh_user_presets()
            self.statusBar().showMessage(f'Preset “{name}” deleted.')

    @staticmethod
    def _picker_enum(picker: QComboBox, enum_type, fallback):
        """Read a combo box's data back as a real enum member.

        Qt stores our string enums as plain strings, so ``currentData()`` hands
        back ``"skip"`` rather than ``ExistingFilePolicy.SKIP``. Every identity
        check downstream depends on getting the member itself.
        """
        try:
            return enum_type(picker.currentData())
        except (TypeError, ValueError):
            return fallback

    def _collect_settings(self) -> AppSettings:
        return AppSettings(
            source_folder=self.source_field.text().strip(),
            output_parent=self.output_field.text().strip(),
            model=MODEL_CHOICES[self.model_picker.currentText()],
            language=LANGUAGE_CHOICES[self.language_picker.currentText()],
            existing_policy=self._picker_enum(
                self.policy_picker, ExistingFilePolicy, ExistingFilePolicy.ASK
            ),
            cleanup_preset=self._current_preset(),
            hallucination_silence_threshold=self.hallucination_spin.value(),
            no_speech_threshold=self.no_speech_spin.value(),
            logprob_threshold=self.logprob_spin.value(),
            folders_expanded=self.folders_group.isChecked(),
            transcription_expanded=self.transcription_group.isChecked(),
            speaker_expanded=self.speaker_group.isChecked(),
            advanced_expanded=self.advanced_group.isChecked(),
            queue_expanded=self.queue_group.isChecked(),
            output_expanded=self.output_group.isChecked(),
            selected_section=self.section_keys[self.section_tabs.currentIndex()],
            name_style=self._picker_enum(
                self.name_style_picker, NameStyle, NameStyle.ORIGINAL
            ),
            custom_suffix=self.custom_suffix_field.text(),
            folder_layout=self._picker_enum(
                self.folder_layout_picker, FolderLayout, FolderLayout.TREE
            ),
            output_scriptsync=self.output_scriptsync.isChecked(),
            output_timecoded=self.output_timecoded.isChecked(),
            output_srt=self.output_srt.isChecked(),
            output_vtt=self.output_vtt.isChecked(),
            detect_speakers=self.detect_speakers.isChecked(),
            speaker_preset=self._current_speaker_preset(),
            speaker_advanced_expanded=self.speaker_advanced_toggle.isChecked(),
            speaker_count_mode=self._picker_enum(
                self.speaker_count_picker,
                SpeakerCountMode,
                SpeakerCountMode.AUTOMATIC,
            ),
            exact_speakers=self.exact_speakers_spin.value(),
            clustering_threshold=self.speaker_threshold_spin.value(),
            min_duration_on=self.min_speech_spin.value(),
            min_duration_off=self.min_pause_spin.value(),
            nearest_tolerance=self.speaker_tolerance_spin.value(),
            merge_gap=self.speaker_merge_spin.value(),
            speakers_in_scriptsync=self.speakers_in_scriptsync.isChecked(),
            speakers_in_subtitles=self.speakers_in_subtitles.isChecked(),
            review_mode=self._picker_enum(
                self.review_mode_picker, ReviewMode, ReviewMode.EVERY_FILE
            ),
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Persist settings, stop background work, and release the assertion.

        Closing while a batch runs never destroys the worker thread. The
        window asks the batch to stop, shows that it is finishing the current
        file, and closes itself once the worker reports back.
        """
        self.settings = self._collect_settings()
        save_settings(self.settings)

        if self.is_transcribing:
            if not self._closing and not self._confirm_close_during_batch():
                event.ignore()
                return
            self._closing = True
            self._stop_batch()
            self._enter_finishing_state()
            event.ignore()
            return

        self._stop_scan(wait=True)
        self._wait_for_retiring_threads()
        self._elapsed_timer.stop()
        self._sleep_blocker.release()
        super().closeEvent(event)

    def _confirm_close_during_batch(self) -> bool:
        """Ask whether to stop a running batch, since it cannot be closed instantly."""
        answer = QMessageBox.question(
            self,
            "A batch is still running",
            "Stop after the current file finishes and then quit?\n\n"
            "Transcripts already written stay where they are. The file being "
            "transcribed right now has to finish first, so this can take a "
            "few minutes.",
            QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Close

    # ----------------------------------------------------------------- paths

    @property
    def source_root(self) -> Path | None:
        text = self.source_field.text().strip()
        return Path(text).expanduser() if text else None

    @property
    def output_parent(self) -> Path | None:
        text = self.output_field.text().strip()
        return Path(text).expanduser() if text else None

    def _select_section(self, key: str) -> None:
        """Open one workspace tab by its stable settings key."""
        try:
            index = self.section_keys.index(key)
        except ValueError:
            return
        self.section_tabs.setCurrentIndex(index)

    def _choose_source_folder(self) -> None:
        start = self.source_field.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose source folder", start)
        if not chosen:
            return
        self.source_field.setText(chosen)
        if not self.output_field.text().strip() and (
            output_parent_conflicts_with_source(Path(chosen).expanduser()) is None
        ):
            self.output_field.setText(chosen)
        self._refresh_output_preview()
        self._queue_sources([Path(chosen)])

    def _choose_source_file(self) -> None:
        start_path = Path(self.source_field.text().strip() or Path.home())
        start = str(start_path.parent if start_path.is_file() else start_path)
        chosen, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Choose media file",
            start,
            MEDIA_NAME_FILTER,
        )
        if not chosen:
            return
        paths = [Path(path) for path in chosen]
        self.source_field.setText(str(paths[-1]))
        if not self.output_field.text().strip() and (
            output_parent_conflicts_with_source(paths[0].parent) is None
        ):
            self.output_field.setText(str(paths[0].parent))
        self._refresh_output_preview()
        self._queue_sources(paths)

    def _output_parent_is_usable(self, parent: Path | None) -> bool:
        """Refuse a destination whose output tree would land in our own source.

        On a case-insensitive filesystem ``<parent>/Transcription`` resolves
        onto this project's ``transcription/`` package, so creating the tree
        would write transcripts into the application's source folder.
        """
        conflict = output_parent_conflicts_with_source(parent)
        if conflict is None:
            return True
        QMessageBox.warning(
            self,
            "That folder cannot be used",
            f"{SOURCE_PACKAGE_CONFLICT_MESSAGE}\n\n{conflict}",
        )
        return False

    def _choose_output_parent(self) -> None:
        start = (
            self.output_field.text().strip()
            or self.source_field.text().strip()
            or str(Path.home())
        )
        chosen = QFileDialog.getExistingDirectory(self, "Choose output parent", start)
        if not chosen:
            return
        if not self._output_parent_is_usable(Path(chosen).expanduser()):
            return
        self.output_field.setText(chosen)
        self._refresh_output_preview()
        self._start_scan()

    @Slot()
    def _on_source_text_changed(self) -> None:
        self._refresh_output_preview()
        source = self.source_root
        if source is not None:
            self._queue_sources([source])

    @Slot()
    def _on_output_text_changed(self) -> None:
        # A typed destination is refused the same way a browsed one is, and the
        # field is cleared so an unusable path is never left armed for Start.
        if not self._output_parent_is_usable(self.output_parent):
            self.output_field.clear()
        self._refresh_output_preview()

    def _refresh_output_preview(self) -> None:
        parent = self.output_parent
        if parent is None:
            self.scriptsync_preview.setText("Choose a destination folder")
            self.timecoded_preview.setText("Choose a destination folder")
            self.subtitles_preview.setText("Choose a destination folder")
            self._update_actions()
            return
        roots = output_roots(parent)
        # One concise destination in the Folders section. The exact per-format
        # folders are in the tooltip and repeated in the Output section, so
        # three long paths no longer take three rows of vertical space.
        self.scriptsync_preview.setText(str(roots.transcription))
        self.scriptsync_preview.setToolTip(
            "Selected formats are written to:\n"
            f"  ScriptSync:  {roots.scriptsync}\n"
            f"  Timecoded:   {roots.timecoded}\n"
            f"  Subtitles:   {roots.subtitles}"
        )
        self.timecoded_preview.setText(str(roots.timecoded))
        self.subtitles_preview.setText(str(roots.subtitles))
        self._update_actions()

    # ------------------------------------------------------------------ scan

    def _is_covered(self, identity: Path) -> bool:
        """True when a queued folder has already been scanned for this path."""
        return any(
            identity == root or root in identity.parents
            for root in self._queued_roots
        )

    def _queue_sources(self, sources: list[Path]) -> None:
        """Append dropped or selected files and folders to the current queue.

        Three outcomes are kept apart, because they mean different things to
        the person who just dropped something: a source that is not there at
        all, a source already covered by the queue, and a source worth
        scanning.
        """
        if not sources or self.is_transcribing:
            return

        missing: list[Path] = []
        unsupported: list[Path] = []
        duplicates: list[Path] = []
        unique: list[Path] = []
        pending: set[Path] = set()
        for source in sources:
            source = Path(source).expanduser()
            if not source.exists():
                missing.append(source)
                continue
            if source.is_file() and not is_supported_media(source):
                unsupported.append(source)
                continue
            identity = source_identity(source)
            if (
                identity in self._queued_identities
                or identity in pending
                or self._is_covered(identity)
            ):
                duplicates.append(source)
                continue
            pending.add(identity)
            unique.append(source)

        if missing:
            self._report_missing_sources(missing)
        if unsupported:
            self._report_unsupported_sources(unsupported)
        if not unique:
            if duplicates and not missing and not unsupported:
                self.statusBar().showMessage(
                    "Those media files are already in the queue."
                )
            return

        if self._scan_thread is not None and self._scan_thread.isRunning():
            self._stop_scan(wait=True)

        self.progress_bar.setRange(0, 0)
        self.current_file_label.setText("Adding media to the queue…")
        self.statusBar().showMessage("Scanning selected media…")

        worker = ScanWorker(unique, self.output_parent)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.found_files.connect(self._on_found_files)
        worker.skipped_files.connect(self._on_scan_skipped)
        worker.item_ready.connect(self._on_item_ready)
        worker.progress.connect(self._on_scan_progress)
        worker.failed.connect(self._on_scan_failed)
        worker.finished.connect(self._on_scan_finished)
        worker.finished.connect(thread.quit)
        self._track_thread(thread, worker)

        self._scan_worker = worker
        self._scan_thread = thread
        self._scan_sources = unique
        # A root only counts as covered once its scan has been started, so a
        # cancelled scan does not silently block a later drop of the same
        # folder.
        for source in unique:
            self._queued_roots.append(source_identity(source))
        thread.start()
        self._update_actions()

    def _track_thread(self, thread: QThread, worker: QObject) -> None:
        """Retire a worker thread cleanly once its event loop has stopped.

        Both the worker and the thread are deleted by Qt, and the thread is
        held in ``_live_threads`` until then so closing the window never
        destroys a thread that is still running.
        """
        self._live_threads.append(thread)
        thread.finished.connect(lambda: self._forget_thread(thread))
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

    def _forget_thread(self, thread: QThread) -> None:
        if thread in self._live_threads:
            self._live_threads.remove(thread)

    def _report_unsupported_sources(self, unsupported: list[Path]) -> None:
        """Say which dropped files were not media, rather than dropping them silently."""
        kinds = sorted({path.suffix.lower() or "(no extension)" for path in unsupported})
        count = len(unsupported)
        summary = (
            f"{count} file(s) were skipped: {', '.join(kinds)} is not supported "
            "media."
            if count > 1
            else f"{unsupported[0].name} was skipped: not a supported media file."
        )
        self.statusBar().showMessage(summary)
        listed = "\n".join(path.name for path in unsupported[:12])
        if count > 12:
            listed += f"\n… and {count - 12} more"
        QMessageBox.information(
            self,
            "Not supported media",
            f"{summary}\n\n{listed}\n\n"
            "MLX Transcript reads these audio and video containers:\n"
            f"{', '.join(sorted(MEDIA_EXTENSIONS))}",
        )

    def _report_missing_sources(self, missing: list[Path]) -> None:
        """Say which dropped items are not there, which is not the same as a duplicate."""
        names = "\n".join(str(path) for path in missing)
        summary = (
            f"{len(missing)} item(s) could not be found and were not added."
            if len(missing) > 1
            else f"{missing[0].name} could not be found and was not added."
        )
        self.statusBar().showMessage(summary)
        QMessageBox.warning(
            self,
            "Media not available",
            "These items are not available right now, so nothing was added "
            "for them:\n\n"
            f"{names}\n\n"
            "If they live on an external drive or a network volume, check "
            "that it is still connected.",
        )

    # Compatibility entry point for existing callers and manual path entry.
    def _start_scan(self) -> None:
        source = self.source_root
        if source is not None:
            self._queue_sources([source])

    def _stop_scan(self, wait: bool = False) -> None:
        if self._scan_worker is not None:
            self._scan_worker.cancel()
        thread = self._scan_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if wait:
                thread.wait(5000)

    @Slot(list)
    def _on_scan_skipped(self, skipped: list) -> None:
        """Remember what the scan turned away so the summary can report it."""
        self._skipped_files = [(Path(path), str(reason)) for path, reason in skipped]

    def _skipped_summary(self) -> str:
        """One short clause naming how many files were skipped."""
        count = len(self._skipped_files)
        if not count:
            return ""
        return f" {count} file{'s' if count != 1 else ''} skipped."

    def _skipped_detail(self) -> str:
        """The per-file reasons, for the tooltip on the queue summary."""
        if not self._skipped_files:
            return ""
        shown = self._skipped_files[:15]
        lines = [f"{path.name} — {reason}" for path, reason in shown]
        if len(self._skipped_files) > len(shown):
            lines.append(f"… and {len(self._skipped_files) - len(shown)} more")
        return "Skipped:\n" + "\n".join(lines)

    @Slot(int)
    def _on_found_files(self, total: int) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(0)
        self.queue_summary.setText(f"Adding {total} media file(s)…")

    def _add_item(self, item: QueueItem) -> bool:
        """Append one scanned item unless its source is already queued.

        Duplicate detection reads the identity the scan worker already
        resolved and the set this window maintains, so adding a file costs the
        same whether the queue holds ten items or ten thousand.
        """
        identity = item.resolved_identity
        if identity in self._queued_identities:
            return False
        self._queued_identities.add(identity)
        self.items.append(item)
        self._append_row(item)
        return True

    @Slot(int, object)
    def _on_item_ready(self, index: int, item: QueueItem) -> None:
        self._add_item(item)
        self.current_file_label.setText(item.name)

    @Slot(int, int)
    def _on_scan_progress(self, done: int, total: int) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)

    @Slot(str)
    def _on_scan_failed(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "Scan problem", message)

    @Slot(list)
    def _on_scan_finished(self, items: list) -> None:
        for item in items:
            self._add_item(item)
        # Mixed source roots only gain their distinguishing labels once every
        # item is in, so the Folder column is refreshed here rather than
        # waiting for the next queue edit.
        self._refresh_queue_source_labels()
        total = len(self.items)
        failures = sum(1 for item in self.items if item.status is QueueStatus.FAILED)
        seconds = sum(item.duration_seconds or 0.0 for item in self.items)

        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(0)
        self.current_file_label.setText("Idle")
        summary = f"{total} media file(s), {format_duration(seconds)} of runtime."
        if failures:
            summary += f" {failures} could not be read."
        summary += self._skipped_summary()
        self.queue_summary.setText(summary)
        self.queue_summary.setToolTip(self._skipped_detail())
        if not items and self._scan_sources:
            # The folder was readable, it simply held nothing this application
            # can transcribe. Saying which folder is what makes that useful.
            names = ", ".join(path.name for path in self._scan_sources[:3])
            if len(self._scan_sources) > 3:
                names += ", …"
            message = f"No audio was found in {names}."
            if self._skipped_files:
                message += self._skipped_summary()
            self.statusBar().showMessage(message)
        else:
            self.statusBar().showMessage(
                summary if total else "No media with audio was found."
            )
        self._scan_thread = None
        self._scan_worker = None
        self._scan_sources = []
        self._update_actions()

    @staticmethod
    def _source_identity(path: Path) -> Path:
        """Kept as the window's own entry point onto the shared helper."""
        return source_identity(path)

    # ----------------------------------------------------------------- queue

    def _append_row(self, item: QueueItem) -> None:
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        for column in range(len(self.QUEUE_COLUMNS)):
            self.queue_table.setItem(row, column, QTableWidgetItem(""))
        self._refresh_row(row, item)

    def _refresh_row(self, row: int, item: QueueItem) -> None:
        root_label = str(item.extras.get("queue_root_label", ""))
        folder = item.relative_folder_label
        if root_label:
            folder = root_label if folder == "/" else f"{root_label} / {folder}"
        values = (
            item.name,
            folder,
            item.duration_label,
            item.status_label,
        )
        for column, value in enumerate(values):
            cell = self.queue_table.item(row, column)
            if cell is None:
                cell = QTableWidgetItem("")
                self.queue_table.setItem(row, column, cell)
            cell.setText(value)
            if column == 2:
                cell.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            cell.setToolTip(str(item.source) if column == 0 else value)

    @Slot()
    def _on_queue_selection_changed(self) -> None:
        selected = bool(self.queue_table.selectionModel().selectedRows())
        enabled = selected and not self.is_transcribing
        self.remove_selected_button.setEnabled(enabled)
        self.reveal_source_button.setEnabled(enabled)

    @Slot()
    def _remove_selected_items(self) -> None:
        rows = sorted(
            (index.row() for index in self.queue_table.selectionModel().selectedRows()),
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(self.items):
                del self.items[row]
                self.queue_table.removeRow(row)
        self._rebuild_queue_index()
        self._refresh_queue_summary()
        self._update_actions()

    @Slot()
    def _clear_queue(self) -> None:
        if self.is_transcribing:
            return
        self.items.clear()
        self.queue_table.setRowCount(0)
        self._rebuild_queue_index()
        self._refresh_queue_summary()
        self._update_actions()

    def _rebuild_queue_index(self) -> None:
        """Recompute the duplicate index after the user edits the queue.

        Removing entries has to give their sources back, so dropping the same
        folder again scans it rather than reporting a duplicate. This runs
        only on an explicit queue edit and reads identities that are already
        resolved, so it stays proportional to the queue.
        """
        self._queued_identities = {item.resolved_identity for item in self.items}
        roots: list[Path] = []
        for item in self.items:
            root = item.resolved_root_identity
            if root not in roots:
                roots.append(root)
        self._queued_roots = roots

    @Slot()
    def _reveal_selected_source(self) -> None:
        rows = self.queue_table.selectionModel().selectedRows()
        if not rows:
            return
        source = self.items[rows[0].row()].source
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(source)], check=False)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(source.parent)))

    def _refresh_queue_summary(self) -> None:
        self._refresh_queue_source_labels()
        total = len(self.items)
        seconds = sum(item.duration_seconds or 0.0 for item in self.items)
        if not total:
            self.queue_summary.setText(
                "Drop media files or folders here, or add them in Folders."
            )
            return
        failures = sum(1 for item in self.items if item.status is QueueStatus.FAILED)
        summary = f"{total} media file(s), {format_duration(seconds)} of runtime."
        if failures:
            summary += f" {failures} could not be read."
        self.queue_summary.setText(summary)

    def _refresh_queue_source_labels(self) -> None:
        roots: list[Path] = []
        for item in self.items:
            root = item.resolved_root_identity
            if root not in roots:
                roots.append(root)
        labels: dict[Path, str] = {}
        if len(roots) > 1:
            used: dict[str, int] = {}
            for root in roots:
                base = root.name or "Files"
                used[base] = used.get(base, 0) + 1
                labels[root] = base if used[base] == 1 else f"{base} ({used[base]})"
        for row, item in enumerate(self.items):
            item.extras["queue_root_label"] = labels.get(
                item.resolved_root_identity, ""
            )
            if row < self.queue_table.rowCount():
                self._refresh_row(row, item)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        """Accept Finder drops on the add-media target and the queue itself."""
        drop_target = getattr(self, "drop_target", None)
        queue_table = getattr(self, "queue_table", None)
        queue_viewport = getattr(self, "queue_drop_viewport", None)
        if watched in (drop_target, queue_table, queue_viewport) and watched is not None:
            if event.type() == QEvent.Type.DragEnter:
                mime = event.mimeData()
                if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
                    event.acceptProposedAction()
                    watched.setProperty("dropActive", True)
                    watched.style().unpolish(watched)
                    watched.style().polish(watched)
                    return True
            elif event.type() == QEvent.Type.DragLeave:
                watched.setProperty("dropActive", False)
                watched.style().unpolish(watched)
                watched.style().polish(watched)
            elif event.type() == QEvent.Type.Drop:
                paths = [
                    Path(url.toLocalFile()) for url in event.mimeData().urls()
                    if url.isLocalFile()
                ]
                watched.setProperty("dropActive", False)
                watched.style().unpolish(watched)
                watched.style().polish(watched)
                self._queue_sources(paths)
                event.acceptProposedAction()
                return True
        return super().eventFilter(watched, event)

    def _set_item_status(self, row: int, status: QueueStatus, message: str = "") -> None:
        if not 0 <= row < len(self.items):
            return
        item = self.items[row]
        item.status = status
        item.message = message
        self._refresh_row(row, item)

    # ---------------------------------------------------------- transcription

    @property
    def is_transcribing(self) -> bool:
        return self._batch_thread is not None and self._batch_thread.isRunning()

    def _build_engine(self) -> TranscriptionEngine:
        """Build the real MLX engine from the current settings."""
        return TranscriptionEngine(self.settings.transcription_options())

    def _build_diarizer(self) -> SherpaOnnxDiarizer:
        """Build the real sherpa-onnx diarizer sharing the window's cache."""
        return SherpaOnnxDiarizer(cache=self.model_cache)

    def _batch_root_labels(self, rows: list[int]) -> dict[Path, str | None]:
        """Keep mixed source trees separate without changing file names."""
        roots = []
        for row in rows:
            root = self.items[row].resolved_root_identity
            if root not in roots:
                roots.append(root)
        if len(roots) <= 1:
            return {root: None for root in roots}
        used: dict[str, int] = {}
        labels: dict[Path, str | None] = {}
        for root in roots:
            base = root.name or "Files"
            used[base] = used.get(base, 0) + 1
            labels[root] = base if used[base] == 1 else f"{base} ({used[base]})"
        return labels

    # --------------------------------------------------------- speaker setup

    def _on_detect_speakers_toggled(self, enabled: bool) -> None:
        for widget in self.speaker_controls:
            widget.setEnabled(enabled)
        if enabled:
            self._on_speaker_count_mode_changed()
        self._on_speaker_advanced_toggled(
            enabled and self.speaker_advanced_toggle.isChecked()
        )
        self._refresh_model_status()

    def _on_speaker_count_mode_changed(self, *_args: object) -> None:
        exact = (
            self._picker_enum(
                self.speaker_count_picker,
                SpeakerCountMode,
                SpeakerCountMode.AUTOMATIC,
            )
            is SpeakerCountMode.EXACT
        )
        self.exact_speakers_spin.setEnabled(
            exact and self.detect_speakers.isChecked()
        )
        self.speaker_threshold_spin.setEnabled(
            not exact and self.detect_speakers.isChecked()
        )

    def _current_speaker_preset(self) -> SpeakerPreset:
        return self._picker_enum(
            self.speaker_preset_picker, SpeakerPreset, SpeakerPreset.BALANCED
        )

    def _on_speaker_preset_changed(self, *_args: object) -> None:
        preset = self._current_speaker_preset()
        values = speaker_preset_values(preset)
        if values is not None:
            self._applying_speaker_preset = True
            try:
                self.speaker_threshold_spin.setValue(values["clustering_threshold"])
                self.min_speech_spin.setValue(values["min_duration_on"])
                self.min_pause_spin.setValue(values["min_duration_off"])
                self.speaker_tolerance_spin.setValue(values["nearest_tolerance"])
                self.speaker_merge_spin.setValue(values["merge_gap"])
            finally:
                self._applying_speaker_preset = False
        self._refresh_speaker_preset_text()

    def _on_speaker_setting_edited(self, *_args: object) -> None:
        if self._applying_speaker_preset:
            return
        if self._current_speaker_preset() is not SpeakerPreset.CUSTOM:
            index = self.speaker_preset_picker.findData(SpeakerPreset.CUSTOM)
            if index >= 0:
                self.speaker_preset_picker.blockSignals(True)
                self.speaker_preset_picker.setCurrentIndex(index)
                self.speaker_preset_picker.blockSignals(False)
        self._refresh_speaker_preset_text()

    def _refresh_speaker_preset_text(self) -> None:
        self.speaker_preset_description.setText(
            self._current_speaker_preset().description
        )

    def _on_speaker_advanced_toggled(self, expanded: bool) -> None:
        self.speaker_advanced_panel.setVisible(bool(expanded))

    def model_state(self) -> ModelState:
        """Report the diarization model state for the interface."""
        if not backend_available():
            return ModelState.UNAVAILABLE
        return self.model_cache.state(DEFAULT_BUNDLE)

    def _refresh_model_status(self) -> None:
        state = self.model_state()
        if state is ModelState.UNAVAILABLE:
            detail = (
                "A component this application ships with is missing. "
                "Download the application again."
                if is_frozen()
                else "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            )
            tone = "warning"
        elif state is ModelState.READY:
            detail = f"Cached in {self.model_cache.describe()}"
            tone = "success"
        elif state is ModelState.FAILED:
            detail = self.model_cache.last_error
            tone = "danger"
        else:
            detail = DEFAULT_BUNDLE.disclosure()
            tone = "secondary"
        _set_tone(self.model_status_label, tone)
        self.model_status_label.setText(
            f"{DEFAULT_BUNDLE.display_name} — {state.label}. {detail}"
        )

    def _start_transcription(self) -> None:
        if self.is_transcribing:
            return
        output_parent = self.output_parent
        if output_parent is None:
            QMessageBox.information(
                self,
                "Choose folders first",
                "Choose a media source and a destination folder.",
            )
            return
        if not self._output_parent_is_usable(output_parent):
            self._select_section("folders")
            return

        rows = [
            row for row, item in enumerate(self.items) if item.media is not None
        ]
        if not rows:
            QMessageBox.information(
                self, "Nothing to transcribe", "No readable media is queued."
            )
            return

        self.settings = self._collect_settings()
        if not self.settings.output_options().formats:
            QMessageBox.information(
                self,
                "Choose an output format",
                "Select at least one format in the Output section.",
            )
            self._select_section("output")
            return
        save_settings(self.settings)
        self._select_section("queue")

        root_labels = self._batch_root_labels(rows)
        jobs = [
            BatchJob(
                source=self.items[row].source,
                media=self.items[row].media,
                source_root=self.items[row].source_root,
                output_root_label=root_labels.get(
                    self.items[row].resolved_root_identity
                ),
            )
            for row in rows
        ]
        for row in rows:
            self._set_item_status(row, QueueStatus.WAITING)

        if not self._confirm_whisper_download():
            for row in rows:
                self._set_item_status(row, QueueStatus.READY)
            return

        if self.settings.detect_speakers and not self._prepare_speaker_models():
            for row in rows:
                self._set_item_status(row, QueueStatus.READY)
            return

        engine = self.engine_factory()
        worker = TranscriptionWorker(
            engine=engine,
            jobs=jobs,
            rows=rows,
            source_root=None,
            output_parent=output_parent,
            policy=self.settings.existing_policy,
            diarizer=self.diarizer_factory() if self.settings.detect_speakers else None,
            diarization=self.settings.diarization_options(),
            include_speakers_in_scriptsync=self.settings.speakers_in_scriptsync,
            include_speakers_in_subtitles=self.settings.speakers_in_subtitles,
            review_mode=self.settings.review_mode,
            output_options=self.settings.output_options(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stage_changed.connect(self._on_stage_changed)
        worker.item_finished.connect(self._on_item_finished)
        worker.progress.connect(self._on_batch_progress)
        worker.conflict.connect(self._on_conflict)
        worker.review.connect(self._on_review)
        worker.finished.connect(self._on_batch_finished)
        worker.finished.connect(thread.quit)
        self._track_thread(thread, worker)

        self._batch_worker = worker
        self._batch_thread = thread

        self._batch_total = len(jobs)
        self._batch_done = 0
        self.progress_bar.setRange(0, len(jobs))
        self.progress_bar.setValue(0)
        self.current_file_label.setText(LOADING_MODEL_TEXT)
        self.statusBar().showMessage(LOADING_MODEL_TEXT)
        self._elapsed.restart()
        self.elapsed_label.setText("Elapsed 00:00:00")
        self._elapsed_timer.start()
        if not self._sleep_blocker.acquire() and self._sleep_blocker.supported:
            self.statusBar().showMessage(
                "Could not prevent sleep; the batch will still run."
            )

        thread.start()
        self._update_actions()

    def _stop_batch(self, wait: bool = False) -> None:
        """Ask the batch to stop after the clip in flight finishes.

        ``wait`` is deliberately not used while the window is closing. A clip
        can take minutes, and blocking the main thread on it is what used to
        end with a running thread being destroyed underneath Qt. The close
        path now waits for the worker's own completion signal instead.
        """
        if self._batch_worker is not None:
            self._batch_worker.cancel()
        thread = self._batch_thread
        if wait and thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(30000)

    def _wait_for_retiring_threads(self, milliseconds: int = 5000) -> None:
        """Let already-stopped threads finish before the window is destroyed."""
        for thread in list(self._live_threads):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(milliseconds)
            except RuntimeError:
                # Qt already deleted it, which is the outcome we wanted.
                continue

    def _enter_finishing_state(self) -> None:
        """Show that the window is closing once the current file is done."""
        self.statusBar().showMessage(FINISHING_TEXT)
        self.current_file_label.setText(FINISHING_TEXT)
        self.progress_bar.setRange(0, 0)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

    @Slot(int, object)
    def _on_stage_changed(self, row: int, status: QueueStatus) -> None:
        self._set_item_status(row, status)
        if self._closing:
            # The window is waiting for this clip so it can close. Per-file
            # activity would overwrite the message that explains the wait.
            return
        if 0 <= row < len(self.items):
            name = self.items[row].name
            position = row + 1
            total = len(self.items)
            if status is QueueStatus.LOADING_MODEL:
                activity = f"{LOADING_MODEL_TEXT} File {position} of {total}: {name}"
                # There is no way to see inside the model download, so show a
                # bar that is moving rather than one that looks stuck at zero.
                self.progress_bar.setRange(0, 0)
            else:
                activity = f"{status.label} — file {position} of {total}: {name}"
                if self.progress_bar.maximum() == 0:
                    self.progress_bar.setRange(0, max(self._batch_total, 1))
                    self.progress_bar.setValue(self._batch_done)
            self.current_file_label.setText(activity)
            self.statusBar().showMessage(activity)

    @Slot(int, object, str)
    def _on_item_finished(self, row: int, status: QueueStatus, message: str) -> None:
        self._set_item_status(row, status, message)

    @Slot(int, int)
    def _on_batch_progress(self, done: int, total: int) -> None:
        self._batch_done = done
        self._batch_total = total
        if self._closing:
            # Keep the indeterminate "finishing" bar rather than snapping back
            # to a count the user is no longer waiting on.
            return
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)

    @Slot(int, object, object)
    def _on_conflict(self, row: int, source: Path, existing: list) -> None:
        """Ask on the UI thread, then release the waiting worker."""
        worker = self._batch_worker
        if worker is None:
            return
        try:
            choice = ConflictDialog(source, list(existing), self).choice()
        except Exception:
            # The worker is blocked waiting for an answer, so it always gets
            # one even if the dialog could not be shown. Raising here would
            # push the exception out through the Qt event loop instead.
            logger.exception("The conflict dialog could not be shown")
            worker.provide_conflict_choice(ConflictChoice.CANCEL_BATCH)
            return
        worker.provide_conflict_choice(choice)

    @Slot(int, object, object)
    def _on_review(self, row: int, job: object, transcript: object) -> None:
        """Show the speaker review dialog, then release the waiting worker."""
        worker = self._batch_worker
        if worker is None:
            return
        source = getattr(job, "source", Path("clip"))
        try:
            converter = self._converter_for(row)
            decision = SpeakerReviewDialog(
                source, transcript, converter, self
            ).review()
        except Exception:
            logger.exception("The speaker review dialog could not be shown")
            worker.provide_review_decision(ReviewDecision.CANCEL_BATCH)
            return
        worker.provide_review_decision(decision)

    def _converter_for(self, row: int) -> TimecodeConverter | None:
        """Build the source-timecode converter for one queued clip."""
        if not 0 <= row < len(self.items):
            return None
        media = self.items[row].media
        if media is None:
            return None
        try:
            return TimecodeConverter(media.frame_rate, media.start_timecode)
        except Exception:
            return None

    def _confirm_whisper_download(self) -> bool:
        """Disclose the first Whisper download before a batch triggers it.

        Nothing is fetched here: MLX Whisper pulls the weights itself on the
        first clip. What this adds is the one thing the progress bar cannot,
        which is telling the user in advance that several gigabytes are about
        to arrive and where they will live.
        """
        if model_is_cached(self.settings.model):
            return True
        answer = QMessageBox.question(
            self,
            WHISPER_DOWNLOAD_TITLE,
            WHISPER_DOWNLOAD_TEXT.format(
                label=self.settings.model_label,
                folder=model_cache_root(),
            ),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        return answer == QMessageBox.StandardButton.Ok

    def _confirm_model_download(self) -> bool:
        """Disclose the download, then fetch the models before the batch starts.

        Fetching here rather than inside the batch is deliberate: the download
        gets a progress dialog the user can cancel, instead of appearing as a
        transcription that has stopped responding.
        """
        state = self.model_state()
        if state is ModelState.READY:
            return True
        if state is ModelState.UNAVAILABLE:
            QMessageBox.warning(
                self,
                "Speaker detection is unavailable",
                missing_component_message("sherpa-onnx"),
            )
            return False

        answer = QMessageBox.question(
            self,
            "Download the speaker model?",
            DEFAULT_BUNDLE.disclosure()
            + f"\n\nThe files are stored in:\n{self.model_cache.describe()}",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        # PySide may return an equivalent enum wrapper rather than the same
        # Python object, so identity comparison can treat an OK click as Cancel.
        return answer == QMessageBox.StandardButton.Ok

    def _prepare_speaker_models(self) -> bool:
        """Get consent, then make sure the models are actually on disk."""
        if not self._confirm_model_download():
            return False
        if self.model_state() is ModelState.READY:
            return True
        return self._download_models(repair=False)

    @Slot()
    def _check_or_repair_models(self) -> None:
        """Verify the cached speaker models and offer to fetch them again."""
        if self.model_state() is ModelState.UNAVAILABLE:
            QMessageBox.warning(
                self,
                "Speaker detection is unavailable",
                missing_component_message("sherpa-onnx"),
            )
            return

        problems = self.model_cache.verify(DEFAULT_BUNDLE)
        if not problems:
            QMessageBox.information(
                self,
                "Speaker models are in order",
                "Every cached model file is present and matches its expected "
                f"contents.\n\nStored in:\n{self.model_cache.describe()}",
            )
            self._refresh_model_status()
            return

        listed = "\n".join(f"  {problem}" for problem in problems)
        answer = QMessageBox.question(
            self,
            "Repair the speaker models?",
            f"{len(problems)} model file(s) need attention:\n\n{listed}\n\n"
            "Download them again? "
            f"{DEFAULT_BUNDLE.disclosure()}",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        if self._download_models(repair=True):
            QMessageBox.information(
                self,
                "Speaker models repaired",
                "The model files were downloaded again and verified.",
            )

    def _download_models(self, repair: bool = False) -> bool:
        """Run the download on a thread behind a cancellable progress dialog."""
        dialog = QProgressDialog(
            "Preparing the speaker models…",
            "Cancel",
            0,
            100,
            self,
        )
        dialog.setWindowTitle(
            "Repairing speaker models" if repair else "Downloading speaker models"
        )
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)

        worker = ModelDownloadWorker(self.model_cache, DEFAULT_BUNDLE, repair=repair)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        outcome: dict[str, object] = {"ok": False, "message": ""}

        def on_progress(name: str, done: int, total: int) -> None:
            if total > 0:
                dialog.setMaximum(100)
                dialog.setValue(min(100, int(100 * done / total)))
            else:
                # Length unknown, so show movement rather than a false figure.
                dialog.setRange(0, 0)
            dialog.setLabelText(
                f"{name}: {done / (1024 * 1024):.1f} MB"
                + (f" of {total / (1024 * 1024):.1f} MB" if total else "")
            )

        def on_finished(succeeded: bool, message: str) -> None:
            outcome["ok"] = succeeded
            outcome["message"] = message
            thread.quit()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        # Direct, because the worker's thread is inside run() and will never
        # reach its event loop to deliver a queued call. Cancelling only sets
        # a flag the transfer reads, which is safe to do from here.
        dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)

        thread.start()
        while thread.isRunning():
            QApplication.processEvents()
            thread.wait(20)
        dialog.close()
        worker.deleteLater()
        thread.deleteLater()

        self._refresh_model_status()
        if not outcome["ok"] and outcome["message"]:
            QMessageBox.warning(
                self,
                "The speaker models are not ready",
                f"{outcome['message']}\n\n"
                "Transcription without speaker detection still works.",
            )
        return bool(outcome["ok"])

    @Slot(object)
    def _on_batch_finished(self, summary: BatchSummary) -> None:
        self.last_summary = summary
        self._elapsed_timer.stop()
        self._tick_elapsed()
        self._sleep_blocker.release()
        self._batch_thread = None
        self._batch_worker = None
        finished = summary.as_sentence()
        self.statusBar().showMessage(finished)
        if self._closing:
            # The user asked to quit while this batch was in flight. The
            # worker has now reported back, so the window can close without
            # ever destroying a running thread.
            self.close()
            return
        self._update_actions()
        # _update_actions enables controls and may calculate readiness, but a
        # completed run should continue to show its result until the user
        # starts another scan or batch.
        self.current_file_label.setText(finished)
        self.start_button.setToolTip("Run the queued files again.")
        self._show_batch_summary(summary)

    def _show_batch_summary(self, summary: BatchSummary) -> None:
        folder = summary.transcription_folder
        box = QMessageBox(self)
        box.setWindowTitle("Batch finished")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(
            f"Completed: {summary.completed}\n"
            f"Skipped: {summary.skipped}\n"
            f"Failed: {summary.failed}\n"
            f"Cancelled: {summary.cancelled}"
        )
        details = [f"Output folder: {folder}"] if folder else []
        if summary.warnings:
            details.append("")
            details.append(
                f"{len(summary.warnings)} file(s) exported without speaker labels:"
            )
            details.extend(
                f"  {path.name}: {message}" for path, message in summary.warnings
            )
        if summary.failures:
            details.append("")
            details.extend(
                f"{path.name}: {message}" for path, message in summary.failures
            )
        box.setInformativeText("\n".join(details))
        reveal = box.addButton("Reveal in Finder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is reveal:
            self._reveal_output()

    def _tick_elapsed(self) -> None:
        seconds = self._elapsed.elapsed() / 1000.0
        self.elapsed_label.setText(f"Elapsed {format_duration(seconds)}")

    # --------------------------------------------------------------- actions

    def _update_actions(self) -> None:
        scanning = self._scan_thread is not None and self._scan_thread.isRunning()
        transcribing = self.is_transcribing
        busy = scanning or transcribing
        ready = [item for item in self.items if item.media is not None]

        if not busy:
            if not ready:
                hint = "Choose a media file or folder to begin."
            elif self.output_parent is None:
                hint = "Choose where to save your transcripts in Folders."
            else:
                hint = f"Ready to transcribe {len(ready)} file{'s' if len(ready) != 1 else ''}."
            self.current_file_label.setText(hint)
            self.start_button.setToolTip(hint)

        self.start_button.setEnabled(
            bool(ready) and not busy and self.output_parent is not None
        )
        self.cancel_button.setEnabled(busy)
        self.reveal_button.setEnabled(self.output_parent is not None)
        self.clear_queue_button.setEnabled(bool(self.items) and not busy)
        self._on_queue_selection_changed()

        for widget in (
            self.source_field,
            self.source_button,
            self.source_file_button,
            self.output_field,
            self.output_button,
            self.model_picker,
            self.language_picker,
            self.policy_picker,
            self.cleanup_picker,
            self.output_group,
            self.advanced_group,
            self.detect_speakers,
        ):
            widget.setEnabled(not transcribing)
        for widget in self.speaker_controls:
            widget.setEnabled(not transcribing and self.detect_speakers.isChecked())
        if not transcribing:
            self._on_speaker_count_mode_changed()

    def _cancel(self) -> None:
        if self.is_transcribing:
            self.statusBar().showMessage(
                "Cancelling after the current file…"
            )
            self._stop_batch()
            return
        self._stop_scan()
        self.statusBar().showMessage("Cancelled.")
        self.current_file_label.setText("Idle")
        self._update_actions()

    def _reveal_output(self) -> None:
        parent = self.output_parent
        if parent is None:
            return
        roots = output_roots(parent)
        target = roots.transcription if roots.transcription.exists() else parent
        if not target.exists():
            QMessageBox.information(
                self,
                "Nothing to reveal yet",
                f"{target} has not been created yet.",
            )
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(target)], check=False)
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    # ----------------------------------------------------------------- misc

    # ------------------------------------------------------ cleanup presets

    def _current_preset(self) -> CleanupPreset:
        return self._picker_enum(
            self.cleanup_picker, CleanupPreset, CleanupPreset.BALANCED
        )

    def _on_cleanup_changed(self, *_args: object) -> None:
        """A preset was chosen, so fill in all three controls at once."""
        self._apply_preset(self._current_preset())

    def _apply_preset(self, preset: CleanupPreset) -> None:
        """Push a named preset's values into the three numeric controls.

        Custom carries no values of its own, so the controls keep whatever the
        user set.
        """
        values = preset_values(preset)
        if values is not None:
            self._applying_preset = True
            try:
                self.hallucination_spin.setValue(
                    values["hallucination_silence_threshold"]
                )
                self.no_speech_spin.setValue(values["no_speech_threshold"])
                self.logprob_spin.setValue(values["logprob_threshold"])
            finally:
                self._applying_preset = False
        self._refresh_cleanup_text()

    def _on_threshold_edited(self, *_args: object) -> None:
        """A manual edit means these are no longer a preset's values."""
        if self._applying_preset:
            return
        if self._current_preset() is CleanupPreset.CUSTOM:
            self._refresh_cleanup_text()
            return
        index = self.cleanup_picker.findData(CleanupPreset.CUSTOM)
        if index >= 0:
            self.cleanup_picker.blockSignals(True)
            self.cleanup_picker.setCurrentIndex(index)
            self.cleanup_picker.blockSignals(False)
        self._refresh_cleanup_text()

    def _refresh_cleanup_text(self) -> None:
        """Keep both explanations in step with the selected preset."""
        preset = self._current_preset()
        self.cleanup_description.setText(preset.description)
        if preset.is_named:
            self.preset_note.setText(
                f"These values come from the {preset.label} preset. Changing "
                "any of them switches Transcription cleanup to Custom."
            )
        else:
            self.preset_note.setText(
                "These are your own values. Transcription cleanup is set to "
                "Custom."
            )

    def _on_advanced_toggled(self, expanded: bool) -> None:
        self._set_advanced_visible(expanded)

    def _set_advanced_visible(self, expanded: bool) -> None:
        """Show or hide the numeric controls. Values are never touched here."""
        for widget in self.advanced_fields:
            widget.setVisible(expanded)
        layout = self.advanced_group.layout()
        if isinstance(layout, QFormLayout):
            for row in range(layout.rowCount()):
                label = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if label is not None and label.widget() is not None:
                    label.widget().setVisible(expanded)

    def _check_ffprobe(self) -> None:
        if ffprobe_available():
            return
        QMessageBox.warning(
            self,
            "ffprobe was not found",
            "Install FFmpeg so durations, frame rates, and source timecode "
            "can be read.\n\nWith Homebrew: brew install ffmpeg",
        )
