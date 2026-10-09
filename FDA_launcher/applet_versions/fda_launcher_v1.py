#!/usr/bin/env python3
"""
PhosphoMAX Analysis Launcher.
Opens a mode picker, then launches either:
  - Direct Binding (FDA.MainWindow via pipeline_fda)
  - Competitive Binding (CompMainWindow via pipeline_ki)
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure both the launcher directory (Desktop) and applet_versions/ are on sys.path
# so that `import FDA`, `import pipeline_fda`, `import pipeline_ki` all resolve.
_HERE    = os.path.dirname(os.path.abspath(__file__))
_APPLETS = os.path.join(_HERE, "applet_versions")
for _p in (_HERE, _APPLETS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QTabWidget, QTableView, QTableWidget, QTableWidgetItem,
    QTextEdit, QScrollArea, QDialogButtonBox, QFileDialog,
    QMessageBox, QSizePolicy, QToolBar, QAbstractItemView
)

import pipeline_ki as pl_ki
from pipeline_ki import PipelineStateKi

_BASE_DIR = Path(__file__).parent

# ── Shared helpers (reused from app.py) ───────────────────────────────────────

class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        import pandas as pd
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):    return len(self._df)
    def columnCount(self, parent=QModelIndex()): return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iloc[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return f"{val:.4g}" if isinstance(val, float) else (str(val) if val is not None else "")
        if role == Qt.BackgroundRole and self._df.columns[index.column()] == "Status":
            return QColor("#d4edda") if str(val) == "PASS" else QColor("#f8d7da") if str(val) == "FAIL" else None
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return (str(self._df.columns[section]) if orientation == Qt.Horizontal
                    else str(section + 1))
        return None

    def set_dataframe(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()


def _make_table(df=None):
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


class FolderRow(QWidget):
    def __init__(self, default="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit   = QLineEdit(default)
        self.button = QPushButton("…")
        self.button.setFixedWidth(28)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text())
        if d:
            self.edit.setText(d)

    @property
    def path(self): return self.edit.text()


class StageWorker(QThread):
    progress = Signal(str)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, fn, args=(), kwargs=None):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs or {}

    def run(self):
        try:
            result = self._fn(*self._args, progress_cb=self.progress.emit, **self._kwargs)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())
            self.finished.emit(Exception())


# ── Mode picker dialog ────────────────────────────────────────────────────────

class ModePicker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhosphoMAX — Select Analysis Mode")
        self.setFixedSize(420, 200)
        self.choice = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Select analysis mode:</h3>"))

        btn_direct = QPushButton("Direct Binding  (Kd)")
        btn_direct.setFixedHeight(52)
        btn_direct.setStyleSheet("font-size: 14px;")
        btn_direct.clicked.connect(lambda: self._pick("direct"))

        btn_comp = QPushButton("Competitive Binding  (Ki)")
        btn_comp.setFixedHeight(52)
        btn_comp.setStyleSheet("font-size: 14px;")
        btn_comp.clicked.connect(lambda: self._pick("competitive"))

        layout.addWidget(btn_direct)
        layout.addWidget(btn_comp)

    def _pick(self, choice):
        self.choice = choice
        self.accept()


# ── Chromatic → Dye mapping widget ───────────────────────────────────────────

class ChromaticMappingWidget(QWidget):
    def __init__(self, default: dict = None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Order", "Dye"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(120)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self._table)

        btns = QHBoxLayout()
        add_btn = QPushButton("+ Row")
        del_btn = QPushButton("− Row")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._del_row)
        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        layout.addLayout(btns)

        for k, v in (default or {1: "DAPI", 2: "DASPI", 3: "H33"}).items():
            self._insert(k, v)

    def _insert(self, order, dye):
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(str(order)))
        self._table.setItem(r, 1, QTableWidgetItem(str(dye)))

    def _add_row(self):
        r = self._table.rowCount()
        self._insert(r + 1, "")

    def _del_row(self):
        rows = {i.row() for i in self._table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self._table.removeRow(r)

    @property
    def mapping(self) -> dict:
        result = {}
        for r in range(self._table.rowCount()):
            try:
                order = int(self._table.item(r, 0).text())
                dye   = self._table.item(r, 1).text().strip()
                if dye:
                    result[order] = dye
            except (ValueError, AttributeError):
                pass
        return result


# ── QC plots tab ─────────────────────────────────────────────────────────────

class QCPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout     = QVBoxLayout(self)
        ctrl       = QHBoxLayout()
        self._prev = QPushButton("◀ Prev")
        self._next = QPushButton("Next ▶")
        self._lbl  = QLabel("—")
        self._lbl.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self._prev)
        ctrl.addWidget(self._lbl)
        ctrl.addWidget(self._next)
        layout.addLayout(ctrl)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        self._figures = []
        self._idx     = 0
        self._prev.clicked.connect(self._go_prev)
        self._next.clicked.connect(self._go_next)

    def load(self, figures: list):
        self._figures = figures
        self._idx     = 0
        self._show()

    def _show(self):
        if not self._figures:
            self._lbl.setText("No QC figures")
            return
        label, fig = self._figures[self._idx]
        canvas = FigureCanvas(fig)
        canvas.draw()
        self._scroll.setWidget(canvas)
        self._lbl.setText(f"{label}  ({self._idx + 1}/{len(self._figures)})")

    def _go_prev(self):
        if self._figures:
            self._idx = (self._idx - 1) % len(self._figures)
            self._show()

    def _go_next(self):
        if self._figures:
            self._idx = (self._idx + 1) % len(self._figures)
            self._show()


# ── Competitive binding plots tab ─────────────────────────────────────────────

class CompPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self._prev  = QPushButton("◀ Prev")
        self._next  = QPushButton("Next ▶")
        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self._prev)
        ctrl.addWidget(self._label)
        ctrl.addWidget(self._next)
        layout.addLayout(ctrl)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        self._items = []
        self._idx   = 0
        self._pass_items = []
        self._fail_items = []

        self._prev.clicked.connect(self._go_prev)
        self._next.clicked.connect(self._go_next)

    def load(self, pass_items: list, fail_items: list):
        self._pass_items = pass_items
        self._fail_items = fail_items
        self._items      = pass_items + fail_items
        self._idx        = 0
        self._render()

    def _render(self):
        if not self._items:
            self._label.setText("No plots")
            return
        entry = self._items[self._idx]
        fig   = Figure(figsize=(5.5, 4))
        ax    = fig.add_subplot(111)
        pl_ki.render_plot_ki(ax, entry)
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        self._scroll.setWidget(canvas)
        status = entry.get("status", "?")
        self._label.setText(
            f"[{status}] {entry['host']} | {entry['dye']} | {entry['guest']}  "
            f"({self._idx + 1}/{len(self._items)})")

    def _go_prev(self):
        if self._items:
            self._idx = (self._idx - 1) % len(self._items)
            self._render()

    def _go_next(self):
        if self._items:
            self._idx = (self._idx + 1) % len(self._items)
            self._render()


# ── Save preview dialog ───────────────────────────────────────────────────────

class SavePreviewDialog(QDialog):
    def __init__(self, file_list, n_pass, n_fail, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Preview")
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>PASS: {n_pass}  |  FAIL: {n_fail}</b>"))
        layout.addWidget(QLabel("Files to be written:"))
        text = QTextEdit()
        text.setReadOnly(True)
        for f in file_list:
            text.append(f"• {f['description']}\n  {f['path']}\n")
        layout.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


# ── Competitive binding main window ───────────────────────────────────────────

STAGE_NAMES_KI = ["Mappings & Kd", "Raw Data", "Merge & Blanks",
                  "Background & QC", "Competitive Fitting"]


class CompMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Competitive Binding (Ki)")
        self.resize(1300, 820)

        self._state   = PipelineStateKi()
        self._worker  = None
        self._stage   = 0
        self._run_all = False

        self._build_ui()
        self._set_buttons_enabled()

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._run_all_btn  = QPushButton("Run All")
        self._run_step_btn = QPushButton("Run Next Step")
        self._save_btn     = QPushButton("Save…")
        self._save_btn.setEnabled(False)
        from PySide6.QtWidgets import QComboBox
        self._rerun_combo  = QComboBox()
        self._rerun_combo.addItems(STAGE_NAMES_KI)
        self._rerun_btn    = QPushButton("Re-run")
        self._switch_btn   = QPushButton("⇄ Direct Binding")
        self._status_lbl   = QLabel("Ready")

        for w in (self._run_all_btn, self._run_step_btn,
                  QLabel("  |  Re-run:"), self._rerun_combo, self._rerun_btn,
                  QLabel("  |  "), self._save_btn,
                  QLabel("  |  "), self._switch_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        self._run_all_btn.clicked.connect(self._on_run_all)
        self._run_step_btn.clicked.connect(self._on_run_step)
        self._rerun_btn.clicked.connect(self._on_rerun)
        self._save_btn.clicked.connect(self._on_save)
        self._switch_btn.clicked.connect(self._on_open_direct)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_config())

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_log())
        right.setSizes([620, 150])
        splitter.addWidget(right)
        splitter.setSizes([300, 1000])

    def _build_config(self):
        panel = QWidget()
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # Folders
        box_f = QGroupBox("Folders")
        ff    = QFormLayout(box_f)
        self._input_row  = FolderRow(str(_BASE_DIR))
        self._kd_row     = FolderRow(str(_BASE_DIR / "Comp_Kd"))
        self._output_row = FolderRow(str(_BASE_DIR))
        ff.addRow("Experiment:",   self._input_row)
        ff.addRow("Kd table dir:", self._kd_row)
        ff.addRow("Output:",       self._output_row)
        hint = QLabel("Experiment must contain:\nRaw/ dye_map/ host_map/\nguest_map/ blank_map/")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        ff.addRow(hint)
        layout.addWidget(box_f)

        # Chromatic mapping
        box_c = QGroupBox("Chromatic → Dye")
        cc    = QVBoxLayout(box_c)
        self._chrom_map = ChromaticMappingWidget()
        cc.addWidget(self._chrom_map)
        layout.addWidget(box_c)

        # Parameters
        box_p = QGroupBox("Parameters")
        pf    = QFormLayout(box_p)

        self._r2_spin = QDoubleSpinBox()
        self._r2_spin.setRange(0.0, 1.0)
        self._r2_spin.setSingleStep(0.05)
        self._r2_spin.setValue(pl_ki.PASS_R2_DEFAULT_KI)
        self._r2_spin.setDecimals(2)
        self._r2_spin.valueChanged.connect(self._on_r2_changed)

        self._grubbs_spin = QDoubleSpinBox()
        self._grubbs_spin.setRange(0.001, 0.2)
        self._grubbs_spin.setSingleStep(0.005)
        self._grubbs_spin.setValue(0.05)
        self._grubbs_spin.setDecimals(3)

        self._morrison_spin = QDoubleSpinBox()
        self._morrison_spin.setRange(0.1, 100.0)
        self._morrison_spin.setSingleStep(0.5)
        self._morrison_spin.setValue(pl_ki.MORRISON_GATE_DEFAULT)
        self._morrison_spin.setDecimals(1)
        self._morrison_spin.setToolTip(
            "Morrison model used when Ki_standard < gate × [Host].\n"
            "Increase to be more permissive, decrease to restrict.")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Morrison gate:", self._morrison_spin)
        layout.addWidget(box_p)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return panel

    def _build_tabs(self):
        self._tabs = QTabWidget()

        self._map_view, self._map_model = _make_table()
        self._tabs.addTab(self._map_view, "Mappings")

        self._kd_view, self._kd_model = _make_table()
        self._tabs.addTab(self._kd_view, "Kd Table")

        raw_w = QWidget()
        raw_l = QVBoxLayout(raw_w)
        self._raw_lbl = QLabel("")
        self._raw_view, self._raw_model = _make_table()
        raw_l.addWidget(self._raw_lbl)
        raw_l.addWidget(self._raw_view)
        self._tabs.addTab(raw_w, "Raw Data")

        self._fi_view, self._fi_model = _make_table()
        self._tabs.addTab(self._fi_view, "FI-F0")

        self._qc_tab = QCPlotsTab()
        self._tabs.addTab(self._qc_tab, "QC Plots")

        fit_w = QWidget()
        fit_l = QVBoxLayout(fit_w)
        self._fit_lbl = QLabel("")
        self._fit_lbl.setAlignment(Qt.AlignCenter)
        font = QFont(); font.setBold(True)
        self._fit_lbl.setFont(font)
        self._fit_view, self._fit_model = _make_table()
        fit_l.addWidget(self._fit_lbl)
        fit_l.addWidget(self._fit_view)
        self._tabs.addTab(fit_w, "Fit Results")

        self._plots_tab = CompPlotsTab()
        self._tabs.addTab(self._plots_tab, "Plots")

        for i in range(self._tabs.count()):
            self._tabs.setTabEnabled(i, False)
        return self._tabs

    def _build_log(self):
        box = QGroupBox("Log")
        l   = QVBoxLayout(box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 9))
        l.addWidget(self._log)
        return box

    def _get_config(self) -> dict:
        base = self._input_row.path
        return {
            "dye_folder":    os.path.join(base, "dye_map"),
            "host_folder":   os.path.join(base, "host_map"),
            "guest_folder":  os.path.join(base, "guest_map"),
            "blank_folder":  os.path.join(base, "blank_map"),
            "raw_folder":    os.path.join(base, "Raw"),
            "kd_folder":     self._kd_row.path,
            "output_folder": self._output_row.path,
            "chromatic_to_dye": self._chrom_map.mapping,
            "r2_threshold":     self._r2_spin.value(),
            "grubbs_alpha":     self._grubbs_spin.value(),
            "morrison_gate":    self._morrison_spin.value(),
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_ki.PASS_R2_DEFAULT_KI)
        self._grubbs_spin.setValue(0.05)
        self._morrison_spin.setValue(pl_ki.MORRISON_GATE_DEFAULT)

    def _set_buttons_enabled(self):
        busy = self._worker is not None and self._worker.isRunning()
        self._run_all_btn.setEnabled(not busy)
        self._run_step_btn.setEnabled(not busy and self._stage < 5)
        self._rerun_btn.setEnabled(not busy)
        self._rerun_combo.setEnabled(not busy)

    def _on_run_all(self):
        self._stage   = 0
        self._run_all = True
        self._state   = PipelineStateKi()
        self._run_stage()

    def _on_run_step(self):
        self._run_all = False
        self._run_stage()

    def _on_rerun(self):
        sel = self._rerun_combo.currentIndex()
        prereqs = {2: ("merged_mapping", "fluorescence"),
                   3: ("merged",),
                   4: ("fi_df",)}
        missing = [p for p in prereqs.get(sel, []) if getattr(self._state, p) is None]
        if missing:
            QMessageBox.warning(self, "Missing prerequisite",
                                f"Stage '{STAGE_NAMES_KI[sel]}' needs earlier stages first.\n"
                                f"Missing: {', '.join(missing)}")
            return
        self._run_all = False
        self._stage   = sel
        self._run_stage()

    def _run_stage(self):
        cfg = self._get_config()

        def _wrap(fn, *args, **kwargs):
            def _inner(*_a, progress_cb=None, **_kw):
                return fn(*args, progress_cb=progress_cb, **kwargs)
            return _inner

        if self._stage == 0:
            self._launch(_wrap(pl_ki.load_mappings_and_kd,
                               cfg["dye_folder"], cfg["host_folder"],
                               cfg["guest_folder"], cfg["kd_folder"]))
        elif self._stage == 1:
            self._launch(_wrap(pl_ki.load_plates_ki,
                               cfg["raw_folder"], cfg["chromatic_to_dye"]))
        elif self._stage == 2:
            self._launch(_wrap(pl_ki.merge_ki,
                               self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]))
        elif self._stage == 3:
            self._launch(_wrap(pl_ki.subtract_background_ki,
                               self._state.merged))
        elif self._stage == 4:
            self._launch(_wrap(pl_ki.fit_curves_ki,
                               self._state.fi_df, self._state.hot_df,
                               grubbs_alpha=cfg["grubbs_alpha"],
                               morrison_gate=cfg["morrison_gate"]))

    def _launch(self, fn):
        self._worker = StageWorker(fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_stage_done)
        self._worker.error.connect(lambda tb: (self._log.append(f"\n[ERROR]\n{tb}"),
                                               QMessageBox.critical(self, "Error", "See log.")))
        stage_lbl = STAGE_NAMES_KI[self._stage] if self._stage < len(STAGE_NAMES_KI) else "?"
        self._status_lbl.setText(f"Running: {stage_lbl}…")
        self._set_buttons_enabled()
        self._worker.start()

    def _on_stage_done(self, result):
        if isinstance(result, Exception):
            self._status_lbl.setText("Error — see log")
            self._set_buttons_enabled()
            return

        cfg = self._get_config()

        if self._stage == 0:
            self._state.merged_mapping = result["merged_mapping"]
            self._state.hot_df         = result["hot_df"]
            self._map_model.set_dataframe(result["merged_mapping"])
            self._kd_model.set_dataframe(result["hot_df"])
            self._tabs.setTabEnabled(0, True)
            self._tabs.setTabEnabled(1, True)
            self._tabs.setCurrentIndex(0)

        elif self._stage == 1:
            self._state.fluorescence = result
            self._raw_model.set_dataframe(result)
            self._raw_lbl.setText(f"{len(result):,} rows  |  "
                                   f"{result['Plate'].nunique()} plate(s)  |  "
                                   f"{result['Chromatic'].nunique()} chromatic(s)")
            self._tabs.setTabEnabled(2, True)
            self._tabs.setCurrentIndex(2)

        elif self._stage == 2:
            self._state.merged = result

        elif self._stage == 3:
            self._state.fi_df       = result["fi_df"]
            self._state.qc_figures  = result["qc_figures"]
            self._fi_model.set_dataframe(result["fi_df"])
            self._qc_tab.load(result["qc_figures"])
            self._tabs.setTabEnabled(3, True)
            self._tabs.setTabEnabled(4, True)
            self._tabs.setCurrentIndex(3)

        elif self._stage == 4:
            fit_results, plot_data = result
            self._state.fit_results = fit_results
            self._state.plot_data   = plot_data
            self._refresh_results()
            self._tabs.setTabEnabled(5, True)
            self._tabs.setTabEnabled(6, True)
            self._tabs.setCurrentIndex(5)
            self._save_btn.setEnabled(True)

        self._stage += 1
        self._status_lbl.setText(f"Stage {self._stage}/{len(STAGE_NAMES_KI)} done")
        self._set_buttons_enabled()

        if self._run_all and self._stage < len(STAGE_NAMES_KI):
            self._run_stage()

    def _on_r2_changed(self, value):
        if self._state.fit_results:
            self._refresh_results()

    def _refresh_results(self):
        cfg = self._get_config()
        df  = pl_ki.apply_thresholds_ki(
            self._state.fit_results, self._state.plot_data, cfg["r2_threshold"])
        self._state.df_results = df
        self._fit_model.set_dataframe(df)
        if not df.empty:
            n_pass = (df["Status"] == "PASS").sum()
            n_fail = (df["Status"] == "FAIL").sum()
            self._fit_lbl.setText(
                f"PASS: {n_pass}   |   FAIL: {n_fail}   "
                f"(R² threshold = {cfg['r2_threshold']:.2f})")
        pass_items = [e for e in self._state.plot_data if e.get("status") == "PASS"]
        fail_items = [e for e in self._state.plot_data if e.get("status") != "PASS"]
        self._plots_tab.load(pass_items, fail_items)

    def _on_save(self):
        cfg       = self._get_config()
        file_list = pl_ki.preview_save_files_ki(self._state, cfg["output_folder"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_ki.save_outputs_ki(self._state, cfg["output_folder"],
                                         cfg["r2_threshold"], progress_cb)
        self._worker = StageWorker(_save_fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_save_done)
        self._worker.error.connect(lambda tb: self._log.append(f"\n[ERROR]\n{tb}"))
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Saving…")
        self._worker.start()

    def _on_save_done(self, result):
        self._save_btn.setEnabled(True)
        self._status_lbl.setText("Saved.")
        if isinstance(result, list):
            self._log.append(f"\nSaved {len(result)} file(s).")
            QMessageBox.information(self, "Saved",
                                    f"{len(result)} file(s) written successfully.")

    def _on_open_direct(self):
        try:
            import FDA
            win = FDA.MainWindow()
            win.show()
            app = QApplication.instance()
            if not hasattr(app, "_extra_windows"):
                app._extra_windows = []
            app._extra_windows.append(win)
        except Exception as e:
            QMessageBox.critical(self, "Could not open Direct Binding",
                                 f"{type(e).__name__}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    picker = ModePicker()
    if picker.exec() != QDialog.Accepted or picker.choice is None:
        sys.exit(0)

    if picker.choice == "direct":
        import FDA
        win = FDA.MainWindow()
    else:
        win = CompMainWindow()

    app._extra_windows = [win]
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
