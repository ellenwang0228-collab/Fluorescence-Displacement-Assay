#!/usr/bin/env python3
"""
PySide6 applet for the PhosphoMAX multi-chromatic FAILPASS binding pipeline.
Run:  python app.py
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure the Desktop directory (parent of applet_versions/) is on sys.path
# so pipeline_fda.py and fda_launcher.py can be found from here too.
_HERE    = os.path.dirname(os.path.abspath(__file__))
_DESKTOP = os.path.dirname(_HERE)
for _p in (_DESKTOP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import (
    Qt, QThread, Signal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QComboBox, QTabWidget, QTableView, QTextEdit, QScrollArea,
    QDialog, QDialogButtonBox, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QToolBar, QStatusBar, QProgressBar
)

import pipeline_fda as pl
from pipeline_fda import PipelineState

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent


def _default_path(sub: str) -> str:
    return str(_BASE_DIR / sub)


# ─────────────────────────────────────────────────────────────────────────────
# Pandas table model
# ─────────────────────────────────────────────────────────────────────────────

class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else __import__("pandas").DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iloc[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(val, float):
                return f"{val:.4g}"
            return str(val) if val is not None else ""
        if role == Qt.BackgroundRole:
            col = self._df.columns[index.column()]
            if col == "Status":
                if str(val) == "PASS":
                    return QColor("#d4edda")
                if str(val) == "FAIL":
                    return QColor("#f8d7da")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None

    def set_dataframe(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()


def _make_table(df=None) -> tuple[QTableView, PandasModel]:
    model = PandasModel(df)
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    view  = QTableView()
    view.setModel(proxy)
    view.setSortingEnabled(True)
    view.horizontalHeader().setStretchLastSection(True)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QTableView.SelectRows)
    return view, model


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class StageWorker(QThread):
    progress = Signal(str)
    finished = Signal(object)   # result or Exception
    error    = Signal(str)

    def __init__(self, fn, args=(), kwargs=None):
        super().__init__()
        self._fn     = fn
        self._args   = args
        self._kwargs = kwargs or {}

    def run(self):
        try:
            result = self._fn(*self._args, progress_cb=self.progress.emit, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(traceback.format_exc())
            self.finished.emit(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Folder selector row
# ─────────────────────────────────────────────────────────────────────────────

class FolderRow(QWidget):
    def __init__(self, default_path: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit   = QLineEdit(default_path)
        self.button = QPushButton("…")
        self.button.setFixedWidth(28)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text())
        if folder:
            self.edit.setText(folder)

    @property
    def path(self) -> str:
        return self.edit.text()


# ─────────────────────────────────────────────────────────────────────────────
# Plot canvas helpers
# ─────────────────────────────────────────────────────────────────────────────

class GridCanvas(FigureCanvas):
    plot_clicked = Signal(int)   # index into items list

    def __init__(self, items: list, parent=None):
        self._items = items
        n    = max(len(items), 1)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig  = Figure(figsize=(cols * 4, rows * 3))
        super().__init__(fig)
        self._axes = []
        for i, entry in enumerate(items):
            ax = fig.add_subplot(rows, cols, i + 1)
            pl.render_plot(ax, entry)
            self._axes.append(ax)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        self.draw()

    def mousePressEvent(self, event):
        # Map click to subplot index
        fig  = self.figure
        w, h = self.width(), self.height()
        x    = event.position().x() / w
        y    = 1.0 - event.position().y() / h
        for i, ax in enumerate(self._axes):
            bbox = ax.get_position()
            if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1:
                self.plot_clicked.emit(i)
                return
        super().mousePressEvent(event)


class SinglePlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(6, 4))
        super().__init__(self._fig)
        self._ax = self._fig.add_subplot(111)

    def show_entry(self, entry):
        self._ax.clear()
        pl.render_plot(self._ax, entry)
        self._fig.tight_layout()
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Plots tab
# ─────────────────────────────────────────────────────────────────────────────

class PlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pass_items = []
        self._fail_items = []
        self._current_items = []
        self._current_idx   = 0

        layout = QVBoxLayout(self)

        # Top controls
        ctrl = QHBoxLayout()
        self._view_btn = QPushButton("Switch to Single View")
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._jump     = QComboBox()
        self._jump.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl.addWidget(self._view_btn)
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._jump)
        layout.addLayout(ctrl)

        # Sub-tabs: PASS / FAIL
        self._subtabs = QTabWidget()
        layout.addWidget(self._subtabs)

        self._grid_scroll_pass = QScrollArea()
        self._grid_scroll_pass.setWidgetResizable(True)
        self._grid_scroll_fail = QScrollArea()
        self._grid_scroll_fail.setWidgetResizable(True)

        self._single_pass = SinglePlotCanvas()
        self._single_fail = SinglePlotCanvas()

        self._subtabs.addTab(self._grid_scroll_pass, "PASS")
        self._subtabs.addTab(self._grid_scroll_fail, "FAIL")

        self._is_grid = True
        self._prev_btn.setVisible(False)
        self._next_btn.setVisible(False)
        self._jump.setVisible(False)

        self._view_btn.clicked.connect(self._toggle_view)
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn.clicked.connect(self._next)
        self._jump.currentIndexChanged.connect(self._jump_to)

        self._grid_pass_canvas = None
        self._grid_fail_canvas = None

    def load(self, pass_items: list, fail_items: list):
        self._pass_items = pass_items
        self._fail_items = fail_items
        self._build_grids()
        self._build_jump_list()

    def _build_grids(self):
        if self._pass_items:
            canvas = GridCanvas(self._pass_items)
            canvas.plot_clicked.connect(lambda i: self._open_single("PASS", i))
            self._grid_scroll_pass.setWidget(canvas)
            self._grid_pass_canvas = canvas
        if self._fail_items:
            canvas = GridCanvas(self._fail_items)
            canvas.plot_clicked.connect(lambda i: self._open_single("FAIL", i))
            self._grid_scroll_fail.setWidget(canvas)
            self._grid_fail_canvas = canvas

    def _build_jump_list(self):
        self._jump.clear()
        for entry in self._pass_items:
            self._jump.addItem(f"[PASS] {entry['host']}-{entry['dye']}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry['host']}-{entry['dye']}")

    def _toggle_view(self):
        self._is_grid = not self._is_grid
        self._subtabs.clear()
        if self._is_grid:
            self._view_btn.setText("Switch to Single View")
            self._prev_btn.setVisible(False)
            self._next_btn.setVisible(False)
            self._jump.setVisible(False)
            self._subtabs.addTab(self._grid_scroll_pass, "PASS")
            self._subtabs.addTab(self._grid_scroll_fail, "FAIL")
        else:
            self._view_btn.setText("Switch to Grid View")
            self._prev_btn.setVisible(True)
            self._next_btn.setVisible(True)
            self._jump.setVisible(True)
            self._subtabs.addTab(self._single_pass, "PASS")
            self._subtabs.addTab(self._single_fail, "FAIL")
            self._show_single(0)

    def _open_single(self, kind: str, idx: int):
        self._is_grid = True   # force toggle
        self._toggle_view()
        self._subtabs.setCurrentIndex(0 if kind == "PASS" else 1)
        self._current_items = self._pass_items if kind == "PASS" else self._fail_items
        self._current_idx   = idx
        canvas = self._single_pass if kind == "PASS" else self._single_fail
        canvas.show_entry(self._current_items[idx])

    def _show_single(self, idx: int):
        tab = self._subtabs.currentIndex()
        items  = self._pass_items if tab == 0 else self._fail_items
        canvas = self._single_pass if tab == 0 else self._single_fail
        if items:
            self._current_idx = idx % len(items)
            canvas.show_entry(items[self._current_idx])

    def _prev(self):
        self._show_single(self._current_idx - 1)

    def _next(self):
        self._show_single(self._current_idx + 1)

    def _jump_to(self, i: int):
        if i < 0:
            return
        n_pass = len(self._pass_items)
        if i < n_pass:
            self._subtabs.setCurrentIndex(0)
            self._show_single(i)
        else:
            self._subtabs.setCurrentIndex(1)
            self._show_single(i - n_pass)


# ─────────────────────────────────────────────────────────────────────────────
# Save preview dialog
# ─────────────────────────────────────────────────────────────────────────────

class SavePreviewDialog(QDialog):
    def __init__(self, file_list: list, n_pass: int, n_fail: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Preview")
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)

        summary = QLabel(f"<b>PASS: {n_pass}  |  FAIL: {n_fail}</b>")
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)

        layout.addWidget(QLabel("The following files will be written:"))
        text = QTextEdit()
        text.setReadOnly(True)
        for f in file_list:
            text.append(f"• {f['description']}\n  {f['path']}\n")
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

STAGE_NAMES = ["Mappings", "Raw Data", "FI-F0", "Fit Results", "Plots"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX – Dye Screen Pipeline")
        self.resize(1300, 800)

        self._state   = PipelineState()
        self._worker  = None
        self._stage   = 0       # next stage to run (0–4; 5 = all done)
        self._run_all = False

        self._build_ui()
        self._set_stage_buttons_enabled()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._run_all_btn  = QPushButton("Run All")
        self._run_step_btn = QPushButton("Run Next Step")
        self._save_btn     = QPushButton("Save…")
        self._save_btn.setEnabled(False)
        self._rerun_combo  = QComboBox()
        self._rerun_combo.addItems(STAGE_NAMES)
        self._rerun_combo.setToolTip("Select a stage to re-run")
        self._rerun_btn    = QPushButton("Re-run")
        self._switch_btn   = QPushButton("⇄ Competitive Binding")
        self._status_lbl   = QLabel("Ready")

        for w in (self._run_all_btn, self._run_step_btn,
                  QLabel("  |  Re-run:"), self._rerun_combo, self._rerun_btn,
                  QLabel("  |  "), self._save_btn,
                  QLabel("  |  "), self._switch_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        self._run_all_btn.clicked.connect(self._on_run_all)
        self._run_step_btn.clicked.connect(self._on_run_step)
        self._rerun_btn.clicked.connect(self._on_rerun_selected)
        self._save_btn.clicked.connect(self._on_save)
        self._switch_btn.clicked.connect(self._on_open_competitive)

        # Central splitter (horizontal: config | content)
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_config_panel())

        # Right: tabs + log
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_log_panel())
        right.setSizes([600, 150])
        splitter.addWidget(right)
        splitter.setSizes([280, 1020])

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # Folders
        folders_box = QGroupBox("Folders")
        fform = QFormLayout(folders_box)
        self._input_row  = FolderRow(str(_BASE_DIR))
        self._output_row = FolderRow(str(_BASE_DIR))
        fform.addRow("Experiment:", self._input_row)
        fform.addRow("Output:",     self._output_row)

        # Show the derived subfolders as read-only hint
        self._subfolder_lbl = QLabel()
        self._subfolder_lbl.setWordWrap(True)
        self._subfolder_lbl.setStyleSheet("color: grey; font-size: 10px;")
        fform.addRow(self._subfolder_lbl)
        layout.addWidget(folders_box)

        self._input_row.edit.textChanged.connect(self._update_subfolder_hint)
        self._update_subfolder_hint(str(_BASE_DIR))

        # Parameters
        params_box = QGroupBox("Parameters")
        pform = QFormLayout(params_box)

        self._r2_spin = QDoubleSpinBox()
        self._r2_spin.setRange(0.0, 1.0)
        self._r2_spin.setSingleStep(0.05)
        self._r2_spin.setValue(pl.PASS_R2_DEFAULT)
        self._r2_spin.setDecimals(2)
        self._r2_spin.valueChanged.connect(self._on_r2_changed)

        self._grubbs_spin = QDoubleSpinBox()
        self._grubbs_spin.setRange(0.001, 0.2)
        self._grubbs_spin.setSingleStep(0.005)
        self._grubbs_spin.setValue(0.05)
        self._grubbs_spin.setDecimals(3)
        self._grubbs_spin.valueChanged.connect(self._on_grubbs_changed)

        from PySide6.QtWidgets import QCheckBox
        self._cross_conc_chk = QCheckBox("Cross-conc. Grubbs")
        self._cross_conc_chk.setChecked(True)
        self._cross_conc_chk.setToolTip(
            "Apply Grubbs outlier test across concentration means\n"
            "to detect hook effects. Uncheck if the highest\n"
            "concentration point is being incorrectly removed.")
        self._cross_conc_chk.stateChanged.connect(self._on_grubbs_changed)

        self._cross_conc_alpha_spin = QDoubleSpinBox()
        self._cross_conc_alpha_spin.setRange(0.001, 0.1)
        self._cross_conc_alpha_spin.setSingleStep(0.005)
        self._cross_conc_alpha_spin.setValue(0.01)
        self._cross_conc_alpha_spin.setDecimals(3)
        self._cross_conc_alpha_spin.setEnabled(True)
        self._cross_conc_alpha_spin.valueChanged.connect(self._on_grubbs_changed)
        self._cross_conc_chk.stateChanged.connect(
            lambda s: self._cross_conc_alpha_spin.setEnabled(bool(s)))

        self._kd_lo_spin = QDoubleSpinBox()
        self._kd_lo_spin.setRange(0.0, 1.0)
        self._kd_lo_spin.setSingleStep(0.05)
        self._kd_lo_spin.setValue(pl.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_lo_spin.setDecimals(2)
        self._kd_lo_spin.setToolTip(
            "FAIL if Kd < factor × min([Host])\n"
            "(Kd is below the measured range — extrapolating left)")
        self._kd_lo_spin.valueChanged.connect(self._on_r2_changed)

        self._kd_hi_spin = QDoubleSpinBox()
        self._kd_hi_spin.setRange(1.0, 1000.0)
        self._kd_hi_spin.setSingleStep(1.0)
        self._kd_hi_spin.setValue(pl.KD_RANGE_FACTOR_HI_DEFAULT)
        self._kd_hi_spin.setDecimals(1)
        self._kd_hi_spin.setToolTip(
            "FAIL if Kd > factor × max([Host])\n"
            "(never reaches saturation — extrapolating right)")
        self._kd_hi_spin.valueChanged.connect(self._on_r2_changed)

        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "Auto (AICc)", "One-site only", "Quadratic only", "Stern-Volmer only"])

        pform.addRow("R² threshold:",    self._r2_spin)
        pform.addRow("Kd range lo:",     self._kd_lo_spin)
        pform.addRow("Kd range hi:",     self._kd_hi_spin)
        pform.addRow("Grubbs α:",        self._grubbs_spin)
        pform.addRow(self._cross_conc_chk, self._cross_conc_alpha_spin)
        pform.addRow("Model:",           self._model_combo)
        layout.addWidget(params_box)

        # Reset button
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)

        layout.addStretch()
        return panel

    def _build_tabs(self) -> QTabWidget:
        self._tabs = QTabWidget()

        # Tab 0: Mappings
        self._map_view, self._map_model = _make_table()
        self._tabs.addTab(self._map_view, "Mappings")

        # Tab 1: Raw Data
        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        self._raw_count_lbl = QLabel("")
        self._raw_view, self._raw_model = _make_table()
        raw_layout.addWidget(self._raw_count_lbl)
        raw_layout.addWidget(self._raw_view)
        self._tabs.addTab(raw_widget, "Raw Data")

        # Tab 2: FI-F0
        self._fi_view, self._fi_model = _make_table()
        self._tabs.addTab(self._fi_view, "FI-F0")

        # Tab 3: Fit Results
        fit_widget = QWidget()
        fit_layout = QVBoxLayout(fit_widget)
        self._fit_count_lbl = QLabel("")
        self._fit_count_lbl.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self._fit_count_lbl.setFont(font)
        self._fit_view, self._fit_model = _make_table()
        fit_layout.addWidget(self._fit_count_lbl)
        fit_layout.addWidget(self._fit_view)
        self._tabs.addTab(fit_widget, "Fit Results")

        # Tab 4: Plots
        self._plots_tab = PlotsTab()
        self._tabs.addTab(self._plots_tab, "Plots")

        # Disable all tabs initially except Mappings placeholder
        for i in range(self._tabs.count()):
            self._tabs.setTabEnabled(i, False)

        return self._tabs

    def _build_log_panel(self) -> QWidget:
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self._log_text)
        return box

    # ── Config helpers ────────────────────────────────────────────────────────

    def _get_config(self) -> dict:
        model_map = {
            "Auto (AICc)":       "auto",
            "One-site only":     "one_site",
            "Quadratic only":    "quadratic",
            "Stern-Volmer only": "stern_volmer",
        }
        base = self._input_row.path
        return {
            "raw_folder":            os.path.join(base, "Raw"),
            "dye_folder":            os.path.join(base, "dye_map"),
            "host_folder":           os.path.join(base, "host_map"),
            "blank_folder":          os.path.join(base, "blank_map"),
            "output_folder":         self._output_row.path,
            "r2_threshold":          self._r2_spin.value(),
            "kd_range_lo":           self._kd_lo_spin.value(),
            "kd_range_hi":           self._kd_hi_spin.value(),
            "grubbs_alpha":          self._grubbs_spin.value(),
            "use_cross_conc_grubbs": self._cross_conc_chk.isChecked(),
            "cross_conc_alpha":      self._cross_conc_alpha_spin.value(),
            "model_preference":      model_map[self._model_combo.currentText()],
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl.PASS_R2_DEFAULT)
        self._kd_lo_spin.setValue(pl.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_hi_spin.setValue(pl.KD_RANGE_FACTOR_HI_DEFAULT)
        self._grubbs_spin.setValue(0.05)
        self._cross_conc_chk.setChecked(True)
        self._cross_conc_alpha_spin.setValue(0.01)
        self._model_combo.setCurrentIndex(0)

    def _update_subfolder_hint(self, base: str):
        self._subfolder_lbl.setText(
            f"Raw/  dye_map/  host_map/  blank_map/\n← expected inside experiment folder")

    # ── Stage orchestration ───────────────────────────────────────────────────

    def _set_stage_buttons_enabled(self):
        busy = self._worker is not None and self._worker.isRunning()
        self._run_all_btn.setEnabled(not busy)
        self._run_step_btn.setEnabled(not busy and self._stage < 5)
        self._rerun_btn.setEnabled(not busy)
        self._rerun_combo.setEnabled(not busy)

    def _on_run_all(self):
        self._stage    = 0
        self._run_all  = True
        self._state    = PipelineState()
        self._run_current_stage()

    def _on_run_step(self):
        self._run_all = False
        self._run_current_stage()

    def _on_rerun_selected(self):
        selected = self._rerun_combo.currentIndex()
        # Check prerequisites
        prereqs = {
            2: ("merged_mapping", "fluorescence"),
            3: ("merged",),
            4: ("fi_df",),
        }
        missing = [p for p in prereqs.get(selected, [])
                   if getattr(self._state, p) is None]
        if missing:
            QMessageBox.warning(
                self, "Missing prerequisite",
                f"Stage '{STAGE_NAMES[selected]}' requires earlier stages to have run first.\n"
                f"Missing: {', '.join(missing)}")
            return
        self._run_all  = False
        self._stage    = selected
        self._run_current_stage()

    def _run_current_stage(self):
        cfg = self._get_config()
        if self._stage == 0:
            self._launch(pl.load_mappings,
                         args=(cfg["dye_folder"], cfg["host_folder"]))
        elif self._stage == 1:
            self._launch(pl.load_plates,
                         args=(cfg["raw_folder"],))
        elif self._stage == 2:
            self._launch(pl.merge_blanks,
                         args=(self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]))
        elif self._stage == 3:
            self._launch(pl.subtract_background,
                         args=(self._state.merged,))
        elif self._stage == 4:
            self._launch(pl.fit_curves,
                         args=(self._state.fi_df,),
                         kwargs={"grubbs_alpha":          cfg["grubbs_alpha"],
                                 "use_cross_conc_grubbs":  cfg["use_cross_conc_grubbs"],
                                 "cross_conc_alpha":       cfg["cross_conc_alpha"],
                                 "model_preference":       cfg["model_preference"]})

    def _launch(self, fn, args=(), kwargs=None):
        # Strip progress_cb from kwargs — StageWorker injects it
        if kwargs is None:
            kwargs = {}
        self._worker = StageWorker(fn, args=args, kwargs=kwargs)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_stage_finished)
        self._worker.error.connect(self._on_error)
        stage_label = STAGE_NAMES[self._stage] if self._stage < len(STAGE_NAMES) else "?"
        self._status_lbl.setText(f"Running: {stage_label}…")
        self._set_stage_buttons_enabled()
        self._worker.start()

    def _on_progress(self, msg: str):
        self._log_text.append(msg)

    def _on_error(self, tb: str):
        self._log_text.append(f"\n[ERROR]\n{tb}")
        QMessageBox.critical(self, "Pipeline Error",
                             "An error occurred. See the log panel for details.")

    def _on_stage_finished(self, result):
        if isinstance(result, Exception):
            self._status_lbl.setText("Error — see log")
            self._set_stage_buttons_enabled()
            return

        cfg = self._get_config()

        if self._stage == 0:
            self._state.merged_mapping = result
            self._map_model.set_dataframe(result)
            self._tabs.setTabEnabled(0, True)
            self._tabs.setCurrentIndex(0)

        elif self._stage == 1:
            self._state.fluorescence = result
            self._raw_model.set_dataframe(result)
            self._raw_count_lbl.setText(f"{len(result):,} rows  |  "
                                         f"{result['Plate'].nunique()} plate(s)  |  "
                                         f"{result['Chromatic'].nunique()} chromatic(s)")
            self._tabs.setTabEnabled(1, True)
            self._tabs.setCurrentIndex(1)

        elif self._stage == 2:
            self._state.merged = result
            # FI-F0 tab enabled after background subtraction (stage 3)

        elif self._stage == 3:
            self._state.fi_df = result
            self._fi_model.set_dataframe(result)
            self._tabs.setTabEnabled(2, True)
            self._tabs.setCurrentIndex(2)

        elif self._stage == 4:
            fit_results, plot_data = result
            self._state.fit_results = fit_results
            self._state.plot_data   = plot_data
            self._refresh_fit_results()
            self._refresh_plots()
            self._tabs.setTabEnabled(3, True)
            self._tabs.setTabEnabled(4, True)
            self._tabs.setCurrentIndex(3)
            self._save_btn.setEnabled(True)

        self._stage += 1
        self._status_lbl.setText(f"Stage {self._stage}/5 done")
        self._set_stage_buttons_enabled()

        if self._run_all and self._stage < 5:
            self._run_current_stage()

    # ── Live threshold updates ────────────────────────────────────────────────

    def _on_r2_changed(self, value: float):
        if not self._state.fit_results:
            return
        self._refresh_fit_results()
        self._refresh_plots()

    def _on_grubbs_changed(self, value: float):
        if not self._state.fit_results:
            return
        QMessageBox.information(
            self, "Re-run required",
            "Grubbs α affects outlier removal during curve fitting.\n"
            "Please re-run from the Fit Results stage (Run Step) to apply the change."
        )

    def _refresh_fit_results(self, r2_threshold: float = None):
        cfg = self._get_config()
        df = pl.apply_thresholds(
            self._state.fit_results, self._state.plot_data,
            r2_threshold   = cfg["r2_threshold"],
            kd_range_lo    = cfg["kd_range_lo"],
            kd_range_hi    = cfg["kd_range_hi"],
        )
        self._state.df_results = df
        self._fit_model.set_dataframe(df)
        if not df.empty:
            n_pass = (df["Status"] == "PASS").sum()
            n_fail = (df["Status"] == "FAIL").sum()
            self._fit_count_lbl.setText(
                f"PASS: {n_pass}   |   FAIL: {n_fail}   "
                f"(R² threshold = {cfg['r2_threshold']:.2f})")

    def _refresh_plots(self, _r2_threshold: float = None):
        if not self._state.plot_data:
            return
        pass_items = [e for e in self._state.plot_data if e.get("status") == "PASS"]
        fail_items = [e for e in self._state.plot_data if e.get("status") != "PASS"]
        self._plots_tab.load(pass_items, fail_items)

    # ── Save ─────────────────────────────────────────────────────────────────

    def _on_save(self):
        cfg       = self._get_config()
        file_list = pl.preview_save_files(self._state, cfg["output_folder"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        # Run save in background worker
        # save_outputs doesn't take progress_cb as positional arg, so we wrap it
        def _save_fn(*args, progress_cb=None, **kwargs):
            return pl.save_outputs(self._state, cfg["output_folder"],
                                   r2_threshold=cfg["r2_threshold"],
                                   progress_cb=progress_cb)

        self._worker = StageWorker(_save_fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_save_finished)
        self._worker.error.connect(self._on_error)
        self._status_lbl.setText("Saving…")
        self._save_btn.setEnabled(False)
        self._worker.start()

    def _on_save_finished(self, result):
        self._save_btn.setEnabled(True)
        self._status_lbl.setText("Saved.")
        if isinstance(result, list):
            self._log_text.append(f"\nSaved {len(result)} file(s).")
            QMessageBox.information(self, "Saved",
                                    f"{len(result)} file(s) written successfully.")


    def _on_open_competitive(self):
        try:
            import fda_launcher as lnch
            win = lnch.CompMainWindow()
            win.show()
            app = QApplication.instance()
            if not hasattr(app, "_extra_windows"):
                app._extra_windows = []
            app._extra_windows.append(win)
        except Exception as e:
            QMessageBox.critical(self, "Could not open Competitive Binding",
                                 f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
