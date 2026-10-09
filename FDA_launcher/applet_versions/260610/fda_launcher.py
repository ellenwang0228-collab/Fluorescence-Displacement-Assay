#!/usr/bin/env python3
"""
PhosphoMAX Analysis Launcher.
Opens a mode picker, then launches either:
  - Direct Binding   (DirectMainWindow via pipeline_fda)
  - Competitive Binding (CompMainWindow via pipeline_ki)
"""

import os
import sys
import traceback
from pathlib import Path

_HERE    = os.path.dirname(os.path.abspath(__file__))
_APPLETS = os.path.join(_HERE, "applet_versions")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns

from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QCheckBox, QComboBox,
    QRadioButton, QTabWidget, QTableView, QTableWidget, QTableWidgetItem, QListWidget,
    QTextEdit, QScrollArea, QDialogButtonBox, QFileDialog, QColorDialog,
    QMessageBox, QSizePolicy, QToolBar, QAbstractItemView
)

import importlib.util as _ilu

def _load_local(name: str):
    """Load a pipeline module explicitly from _HERE, bypassing sys.path lookup."""
    spec = _ilu.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod  = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pl_fda = _load_local("pipeline_fda")
pl_ki  = _load_local("pipeline_ki")
PipelineState   = pl_fda.PipelineState
PipelineStateKi = pl_ki.PipelineStateKi

_BASE_DIR = Path(__file__).parent

# Height ratios for residual panels: [curve, residual]
_RESID_RATIO = [3.5, 1]
_RESID_HSPACE = 0.06


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


class ColorPickerRow(QWidget):
    """Colour swatch button + hex text input, kept in sync. Emits color_changed(str)."""
    color_changed = Signal(str)

    def __init__(self, label: str, default: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))

        self._color = QColor(default).name()

        self._btn = QPushButton()
        self._btn.setFixedSize(26, 22)
        self._btn.setToolTip("Click to open colour picker")
        self._btn.clicked.connect(self._pick)
        layout.addWidget(self._btn)

        self._edit = QLineEdit(self._color)
        self._edit.setMaximumWidth(72)
        self._edit.setPlaceholderText("#rrggbb")
        self._edit.editingFinished.connect(self._on_text)
        layout.addWidget(self._edit)
        layout.addStretch()
        self._sync_btn()

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str):
        self._color = QColor(color).name()
        self._edit.setText(self._color)
        self._sync_btn()

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Choose colour")
        if col.isValid():
            self._apply(col.name())

    def _on_text(self):
        text = self._edit.text().strip()
        if not text.startswith("#"):
            text = "#" + text
        col = QColor(text)
        if col.isValid():
            self._apply(col.name())
        else:
            self._edit.setText(self._color)

    def _apply(self, hex_color: str):
        self._color = hex_color
        self._edit.setText(hex_color)
        self._sync_btn()
        self.color_changed.emit(hex_color)

    def _sync_btn(self):
        self._btn.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #888;")


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


# ── Mode picker dialog ────────────────────────────────────────────────────────

# ── Heatmap / Bar-chart standalone window ────────────────────────────────────

class HeatmapWindow(QMainWindow):
    """
    Load one or more pipeline output Excel files (or a folder of them)
    and generate pKd / pKi heatmaps or bar charts for export as PDF.

    Accepts two formats automatically:
      • Pipeline output  — sheets 'PASS' / 'FAIL' with Kd or Ki_uM columns
      • Kd_Tables_Gen    — sheets 'Kd_Direct' / 'Ki_Competition' (AVERAGE_Kd/Ki)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Heatmap / Bar Chart")
        self.resize(1250, 820)
        self._file_paths   = []
        self._current_figs = []
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_config())

        right = QWidget()
        rl    = QVBoxLayout(right)
        self._status_lbl = QLabel("Add files or a folder, then click  Generate.")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet("color: grey;")
        rl.addWidget(self._status_lbl)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        rl.addWidget(self._scroll)
        splitter.addWidget(right)
        splitter.setSizes([290, 960])

    def _build_config(self):
        panel = QWidget()
        panel.setFixedWidth(290)
        cl    = QVBoxLayout(panel)
        cl.setAlignment(Qt.AlignTop)

        # ── Files ──────────────────────────────────────────────────────────
        gb_f = QGroupBox("Input Files")
        fv   = QVBoxLayout(gb_f)
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(110)
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setToolTip(
            "Accepted: pipeline output (PASS/FAIL sheets) or\n"
            "Kd_Tables_Gen format (Kd_Direct / Ki_Competition sheets)")
        fv.addWidget(self._file_list)
        btn_row = QHBoxLayout()
        add_f   = QPushButton("Add Files…")
        add_d   = QPushButton("Add Folder…")
        rem_btn = QPushButton("Remove")
        add_f.clicked.connect(self._add_files)
        add_d.clicked.connect(self._add_folder)
        rem_btn.clicked.connect(self._remove_files)
        btn_row.addWidget(add_f)
        btn_row.addWidget(add_d)
        btn_row.addWidget(rem_btn)
        fv.addLayout(btn_row)
        cl.addWidget(gb_f)

        # ── Chart settings ─────────────────────────────────────────────────
        gb_c  = QGroupBox("Chart")
        cv    = QVBoxLayout(gb_c)
        type_row = QHBoxLayout()
        self._heat_rb = QRadioButton("Heatmap")
        self._bar_rb  = QRadioButton("Bar chart")
        self._heat_rb.setChecked(True)
        type_row.addWidget(self._heat_rb)
        type_row.addWidget(self._bar_rb)
        cv.addLayout(type_row)
        cff = QFormLayout()
        self._data_combo   = QComboBox()
        self._data_combo.addItems(["Both (Kd + Ki)", "Kd only", "Ki only"])
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["PASS only", "FAIL only", "All"])
        cff.addRow("Data:",   self._data_combo)
        cff.addRow("Filter:", self._filter_combo)
        cv.addLayout(cff)
        cl.addWidget(gb_c)

        # ── Heatmap options ────────────────────────────────────────────────
        self._gb_heat = QGroupBox("Heatmap Options")
        hf = QFormLayout(self._gb_heat)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            "Blues", "viridis", "plasma", "magma",
            "inferno", "coolwarm", "RdYlGn", "YlOrRd", "Greens"])
        self._annot_combo = QComboBox()
        self._annot_combo.addItems(
            ["Raw µM (.2g)", "pK (.2f)", "log₁₀ (.2f)", "None"])
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(
            ["By median (tightest first)", "Alphabetical"])
        self._tile_spin = QDoubleSpinBox()
        self._tile_spin.setRange(0.2, 2.0)
        self._tile_spin.setValue(0.5)
        self._tile_spin.setSingleStep(0.1)
        self._tile_spin.setDecimals(1)
        self._tile_spin.setToolTip("Cell size in inches (width = height)")
        self._shared_chk = QCheckBox("Shared colour scale (Kd + Ki)")
        self._shared_chk.setChecked(True)
        self._grey_chk   = QCheckBox("Grey for tried-but-failed")
        self._grey_chk.setChecked(True)
        hf.addRow("Colormap:",   self._cmap_combo)
        hf.addRow("Annotate:",   self._annot_combo)
        hf.addRow("Sort rows:",  self._sort_combo)
        hf.addRow("Tile size:",  self._tile_spin)
        hf.addRow(self._shared_chk)
        hf.addRow(self._grey_chk)
        cl.addWidget(self._gb_heat)

        # ── Bar chart options ──────────────────────────────────────────────
        self._gb_bar = QGroupBox("Bar Options")
        bv = QVBoxLayout(self._gb_bar)
        self._bar_col = ColorPickerRow("Bar colour:", "#4878CF")
        self._err_col = ColorPickerRow("Error bars:", "#222222")
        bv.addWidget(self._bar_col)
        bv.addWidget(self._err_col)
        bfl = QFormLayout()
        self._grp_combo   = QComboBox()
        self._grp_combo.addItems(["None (single colour)", "Host", "Dye"])
        self._bsort_combo = QComboBox()
        self._bsort_combo.addItems(
            ["pK ↓ (strongest first)", "pK ↑ (weakest first)", "Alphabetical"])
        self._byscale_combo = QComboBox()
        self._byscale_combo.addItems(
            ["pK  (−log₁₀ K/M)", "Raw µM", "log₁₀(K/µM)"])
        self._blbl_rot = QComboBox()
        self._blbl_rot.addItems(["90°", "45°", "30°"])
        self._bbar_w = QDoubleSpinBox()
        self._bbar_w.setRange(0.4, 3.0)
        self._bbar_w.setValue(1.0)
        self._bbar_w.setSingleStep(0.1)
        self._bbar_w.setDecimals(1)
        self._bbar_w.setToolTip("Inches of figure width allocated per bar")
        bfl.addRow("Colour by:", self._grp_combo)
        bfl.addRow("Sort:",      self._bsort_combo)
        bfl.addRow("Y-axis:",    self._byscale_combo)
        bfl.addRow("Label rot:", self._blbl_rot)
        bfl.addRow("Width/bar:", self._bbar_w)
        bv.addLayout(bfl)
        cl.addWidget(self._gb_bar)
        self._gb_bar.setVisible(False)

        # ── Font ───────────────────────────────────────────────────────────
        gb_font = QGroupBox("Font")
        ff_lay  = QFormLayout(gb_font)
        self._fam = QComboBox()
        self._fam.addItems([
            "sans-serif", "serif", "monospace",
            "DejaVu Sans", "Arial", "Helvetica"])
        self._fsz = QDoubleSpinBox()
        self._fsz.setRange(6, 22)
        self._fsz.setValue(9)
        self._fsz.setSingleStep(0.5)
        self._fsz.setDecimals(1)
        ff_lay.addRow("Family:", self._fam)
        ff_lay.addRow("Size:",   self._fsz)
        cl.addWidget(gb_font)

        # ── Actions ────────────────────────────────────────────────────────
        gen_btn = QPushButton("Generate")
        gen_btn.setFixedHeight(34)
        gen_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        gen_btn.clicked.connect(self._generate)
        exp_btn = QPushButton("Export PDF…")
        exp_btn.setFixedHeight(34)
        exp_btn.clicked.connect(self._export)
        cl.addWidget(gen_btn)
        cl.addWidget(exp_btn)
        cl.addStretch()

        self._heat_rb.toggled.connect(self._on_type_changed)
        return panel

    def _on_type_changed(self, heat):
        self._gb_heat.setVisible(heat)
        self._gb_bar.setVisible(not heat)

    # ── File management ────────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Excel files", "",
            "Excel files (*.xlsx);;All files (*)")
        for p in paths:
            if p not in self._file_paths:
                self._file_paths.append(p)
                self._file_list.addItem(os.path.basename(p))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if not folder:
            return
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".xlsx") and not f.startswith("~$"):
                p = os.path.join(folder, f)
                if p not in self._file_paths:
                    self._file_paths.append(p)
                    self._file_list.addItem(f)

    def _remove_files(self):
        rows = sorted(
            {i.row() for i in self._file_list.selectedIndexes()}, reverse=True)
        for r in rows:
            self._file_list.takeItem(r)
            del self._file_paths[r]

    # ── Data loading ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_fmt(xl) -> str:
        sheets = set(xl.sheet_names)
        if "Kd_Direct" in sheets or "Ki_Competition" in sheets:
            return "kd_tables"
        if "PASS" in sheets or "FAIL" in sheets:
            return "pipeline"
        return "unknown"

    @staticmethod
    def _load_pipeline(xl):
        import pandas as pd
        frames = []
        for s in xl.sheet_names:
            if s in ("PASS", "FAIL"):
                df = xl.parse(s)
                df["Status"] = s
                frames.append(df)
        if not frames:
            return None, None
        combined = pd.concat(frames, ignore_index=True)
        kd = ki = None
        if "Kd" in combined.columns:
            cols = [c for c in ["Host","Dye","Dye_Concentration","Kd","Kd_SE","Status"]
                    if c in combined.columns]
            kd = combined[cols].copy()
            kd["Kd"] = pd.to_numeric(kd["Kd"], errors="coerce")
        if "Ki_uM" in combined.columns and "Guest" in combined.columns:
            cols = [c for c in ["Host","Dye","Guest","Ki_uM","Ki_err_uM","Status"]
                    if c in combined.columns]
            ki = combined[cols].copy()
            ki["Ki_uM"] = pd.to_numeric(ki["Ki_uM"], errors="coerce")
        return kd, ki

    @staticmethod
    def _load_kd_tables(xl):
        import pandas as pd
        kd = ki = None
        if "Kd_Direct" in xl.sheet_names:
            try:
                raw     = xl.parse("Kd_Direct", header=None)
                avg_col = raw.iloc[1].tolist().index("AVERAGE_Kd")
                df      = raw.iloc[2:, [0, 1, 2, avg_col]].copy()
                df.columns = ["Host", "Dye", "Dye_Concentration", "Kd"]
                df["Host"]             = df["Host"].ffill()
                df["Kd"]               = pd.to_numeric(df["Kd"], errors="coerce")
                df["Dye_Concentration"] = pd.to_numeric(
                    df["Dye_Concentration"], errors="coerce")
                df = df.dropna(subset=["Host", "Dye"])
                df["Status"] = df["Kd"].apply(
                    lambda v: "PASS" if pd.notna(v) and 0 < v < 1e6 else "FAIL")
                kd = df
            except (ValueError, IndexError):
                pass
        if "Ki_Competition" in xl.sheet_names:
            try:
                raw     = xl.parse("Ki_Competition", header=None)
                avg_col = raw.iloc[1].tolist().index("AVERAGE_Ki")
                df      = raw.iloc[2:, [0, 1, 2, avg_col]].copy()
                df.columns = ["Host", "Dye", "Guest", "Ki_uM"]
                df["Host"]  = df["Host"].ffill()
                df["Ki_uM"] = pd.to_numeric(df["Ki_uM"], errors="coerce")
                df["Guest"] = df["Guest"].astype(str).str.strip().str.title()
                df = df.dropna(subset=["Host", "Guest"])
                df["Status"] = df["Ki_uM"].apply(
                    lambda v: "PASS" if pd.notna(v) and 0 < v < 1e6 else "FAIL")
                ki = df
            except (ValueError, IndexError):
                pass
        return kd, ki

    def _load_all(self):
        import pandas as pd
        kd_parts, ki_parts, errors = [], [], []
        for path in self._file_paths:
            try:
                xl  = pd.ExcelFile(path)
                fmt = self._detect_fmt(xl)
                if fmt == "pipeline":
                    kd, ki = self._load_pipeline(xl)
                elif fmt == "kd_tables":
                    kd, ki = self._load_kd_tables(xl)
                else:
                    errors.append(f"{os.path.basename(path)}: unrecognised format")
                    continue
                if kd is not None and not kd.empty: kd_parts.append(kd)
                if ki is not None and not ki.empty: ki_parts.append(ki)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        import pandas as pd
        kd_all = pd.concat(kd_parts, ignore_index=True) if kd_parts else None
        ki_all = pd.concat(ki_parts, ignore_index=True) if ki_parts else None
        return kd_all, ki_all, errors

    # ── Generate ───────────────────────────────────────────────────────────

    def _generate(self):
        if not self._file_paths:
            QMessageBox.warning(self, "No files", "Add at least one file first.")
            return
        self._status_lbl.setText("Loading…")
        self._status_lbl.setStyleSheet("color: grey;")
        QApplication.processEvents()

        kd_all, ki_all, errors = self._load_all()
        if errors:
            QMessageBox.warning(self, "Load warnings",
                                "Some files had issues:\n" + "\n".join(errors))

        # Apply filter
        filt = self._filter_combo.currentText()
        def _filt(df):
            if df is None: return None
            if filt == "PASS only": return df[df["Status"] == "PASS"].copy()
            if filt == "FAIL only": return df[df["Status"] == "FAIL"].copy()
            return df
        kd = _filt(kd_all);  ki = _filt(ki_all)

        sel = self._data_combo.currentText()
        if "Kd" not in sel:  kd = None
        if "Ki" not in sel:  ki = None

        n_kd = 0 if kd is None else len(kd)
        n_ki = 0 if ki is None else len(ki)
        if n_kd == 0 and n_ki == 0:
            self._status_lbl.setText("No data found for this filter/selection.")
            self._status_lbl.setStyleSheet("color: red;")
            return

        fs = self._fsz.value();  ff = self._fam.currentText()
        try:
            if self._heat_rb.isChecked():
                figs = self._render_heatmaps(kd, ki, fs, ff)
            else:
                figs = self._render_barcharts(kd, ki, fs, ff)
            self._current_figs = figs

            container = QWidget()
            vl = QVBoxLayout(container)
            for fig in figs:
                canvas = FigureCanvas(fig)
                canvas.draw()
                vl.addWidget(canvas)
            self._scroll.setWidget(container)
            self._status_lbl.setText(
                f"Generated {len(figs)} chart(s).  "
                + (f"Kd: {n_kd} rows.  " if n_kd else "")
                + (f"Ki: {n_ki} rows." if n_ki else ""))
            self._status_lbl.setStyleSheet("")
        except Exception as exc:
            import traceback as _tb
            self._status_lbl.setText(f"Render error — {exc}")
            self._status_lbl.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Render error", _tb.format_exc())

    # ── Heatmap ────────────────────────────────────────────────────────────

    @staticmethod
    def _sort_pivot(pivot):
        import pandas as pd
        log     = np.log10(pivot.replace(0, np.nan))
        medians = log.median(axis=1).fillna(999)
        return pivot.loc[medians.sort_values().index]

    def _prep_pivot(self, df, val_col, row_col, col_col):
        import pandas as pd
        attempted = (df.pivot_table(index=row_col, columns=col_col,
                                    values=val_col, aggfunc="count")
                     .notna())
        good = df[df[val_col].notna() & (df[val_col] > 0) & (df[val_col] < 1e6)]
        pivot = good.pivot_table(index=row_col, columns=col_col,
                                 values=val_col, aggfunc="mean")
        return pivot, attempted

    def _render_heatmaps(self, kd_df, ki_df, fs, ff):
        import pandas as pd
        cmap       = self._cmap_combo.currentText()
        annot_mode = self._annot_combo.currentText()
        sort_alpha = "Alpha" in self._sort_combo.currentText()
        shared     = self._shared_chk.isChecked()
        grey_fail  = self._grey_chk.isChecked()
        tile       = self._tile_spin.value()
        MARG_W, MARG_H = 3.5, 2.5

        entries = []   # (label, pivot, attempted, xlabel)
        if kd_df is not None and not kd_df.empty:
            p, m = self._prep_pivot(kd_df, "Kd", "Host", "Dye")
            if not p.empty:
                if not sort_alpha: p = self._sort_pivot(p)
                m = m.reindex(index=p.index, columns=p.columns).fillna(False)
                entries.append(("Direct Binding — Kd (Host–Dye)", p, m, "Dye"))
        if ki_df is not None and not ki_df.empty:
            p, m = self._prep_pivot(ki_df, "Ki_uM", "Host", "Guest")
            if not p.empty:
                if not sort_alpha: p = self._sort_pivot(p)
                m = m.reindex(index=p.index, columns=p.columns).fillna(False)
                entries.append(("Competition Binding — Ki (Host–Guest)", p, m, "Guest"))
        if not entries:
            return []

        # Colour scale
        if shared and len(entries) > 1:
            all_lv = np.concatenate([
                np.log10(e[1].replace(0, np.nan)).values.flatten()
                for e in entries])
            all_lv = all_lv[~np.isnan(all_lv)]
            gvmin = float(np.floor(np.nanpercentile(all_lv, 2)))
            gvmax = float(np.ceil(np.nanpercentile(all_lv, 98)))
            scales = [(gvmin, gvmax)] * len(entries)
        else:
            scales = []
            for _, p, _, _ in entries:
                lv = np.log10(p.replace(0, np.nan)).values.flatten()
                lv = lv[~np.isnan(lv)]
                if len(lv):
                    scales.append((float(np.floor(np.nanpercentile(lv, 2))),
                                   float(np.ceil(np.nanpercentile(lv, 98)))))
                else:
                    scales.append((0.0, 1.0))

        figs = []
        for (title, pivot, mask, xlabel), (vmin_, vmax_) in zip(entries, scales):
            nr, nc = pivot.shape
            w = max(4.0, nc * tile + MARG_W)
            h = max(4.0, nr * tile + MARG_H)
            fig = Figure(figsize=(w, h));  ax = fig.add_subplot(111)
            ax.set_facecolor("white")
            log_df = np.log10(pivot.replace(0, np.nan))

            sns.heatmap(
                log_df, ax=ax, mask=log_df.isna(),
                cmap=cmap, vmin=vmin_, vmax=vmax_,
                linewidths=0.4, linecolor="white",
                cbar_kws={"shrink": 0.7, "pad": 0.02},
                annot=False)

            if grey_fail:
                failed = mask & log_df.isna()
                for i in range(nr):
                    for j in range(nc):
                        if failed.iloc[i, j]:
                            ax.add_patch(plt.Rectangle(
                                (j, i), 1, 1,
                                facecolor="#AAAAAA", edgecolor="white",
                                linewidth=0.4, zorder=2))

            if annot_mode != "None":
                span = max(vmax_ - vmin_, 1e-9)
                for i in range(nr):
                    for j in range(nc):
                        val = pivot.iloc[i, j]
                        if pd.notna(val) and val > 0:
                            lv   = np.log10(val)
                            tc   = "white" if (lv - vmin_) / span > 0.55 else "black"
                            txt  = (f"{val:.2g}"      if "Raw"  in annot_mode else
                                    f"{6 - lv:.2f}"   if "pK"   in annot_mode else
                                    f"{lv:.2f}")
                            ax.text(j + 0.5, i + 0.5, txt,
                                    ha="center", va="center",
                                    fontsize=fs * 0.85, fontfamily=ff, color=tc)

            cbar = ax.collections[0].colorbar
            ticks = np.arange(int(vmin_), int(vmax_) + 1)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([r"$10^{%d}$" % t for t in ticks],
                                fontsize=fs * 0.9)
            cbar.set_label("µM", fontsize=fs, labelpad=4)

            ax.set_title(title, fontsize=fs + 2, fontweight="bold",
                         fontfamily=ff, pad=8)
            ax.set_xlabel(xlabel, fontsize=fs, fontfamily=ff)
            ax.set_ylabel("Host", fontsize=fs, fontfamily=ff)
            for lbl in ax.get_yticklabels():
                lbl.set_rotation(0); lbl.set_fontsize(fs * 0.9); lbl.set_fontfamily(ff)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45); lbl.set_ha("right")
                lbl.set_fontsize(fs * 0.9); lbl.set_fontfamily(ff)

            fig.suptitle("PhosphoMAX Binding Constants",
                         fontsize=fs + 4, fontweight="bold", fontfamily=ff, y=1.01)
            fig.tight_layout()
            figs.append(fig)
        return figs

    # ── Bar chart ──────────────────────────────────────────────────────────

    def _render_barcharts(self, kd_df, ki_df, fs, ff):
        figs = []
        if kd_df is not None and not kd_df.empty:
            f = self._bar_one(kd_df, "Kd", "pKd",
                              "pKd = −log₁₀(Kd/M)", "Direct Binding — pKd",
                              "Dye", fs, ff)
            if f: figs.append(f)
        if ki_df is not None and not ki_df.empty:
            f = self._bar_one(ki_df, "Ki_uM", "pKi",
                              "pKi = −log₁₀(Ki/M)", "Competition Binding — pKi",
                              "Guest", fs, ff)
            if f: figs.append(f)
        return figs

    def _bar_one(self, df, val_col, pv, y_label, title, secondary_col, fs, ff):
        import pandas as pd

        yscale   = self._byscale_combo.currentText()
        rot_deg  = int(self._blbl_rot.currentText().replace("°", ""))
        bar_w_in = self._bbar_w.value()

        df = df.copy()
        # Compute the display column and y-axis label from the chosen scale
        raw_col = val_col
        if "pK" in yscale:
            df["_y"] = 6.0 - np.log10(df[raw_col].clip(lower=1e-12))
            se_col   = "Kd_SE" if val_col == "Kd" else "Ki_err_uM"
            if se_col in df.columns:
                df["_se"] = (df[se_col].clip(lower=0) /
                             (df[raw_col].clip(lower=1e-12) * np.log(10)))
            else:
                df["_se"] = np.nan
            used_ylabel = "pKd = −log₁₀(Kd/M)" if val_col == "Kd" else "pKi = −log₁₀(Ki/M)"
        elif "Raw" in yscale:
            df["_y"]  = df[raw_col]
            df["_se"] = df.get("Kd_SE" if val_col == "Kd" else "Ki_err_uM",
                                pd.Series(np.nan, index=df.index))
            used_ylabel = "Kd (µM)" if val_col == "Kd" else "Ki (µM)"
        else:  # log10
            df["_y"] = np.log10(df[raw_col].clip(lower=1e-12))
            df["_se"] = np.nan   # log-scale SE would need separate propagation
            used_ylabel = "log₁₀(Kd / µM)" if val_col == "Kd" else "log₁₀(Ki / µM)"

        # Labels
        if val_col == "Kd":
            df["_lbl"] = df["Host"] + " | " + df["Dye"]
            if "Dye_Concentration" in df.columns:
                df["_lbl"] += "\n[" + df["Dye_Concentration"].astype(str) + " µM]"
        else:
            df["_lbl"] = df["Guest"] + "\n" + df["Host"]

        grp  = self._grp_combo.currentText()
        stxt = self._bsort_combo.currentText()
        if "↓" in stxt:   df = df.sort_values("_y", ascending=False)
        elif "↑" in stxt: df = df.sort_values("_y", ascending=True)
        else:              df = df.sort_values("_lbl")
        df = df.reset_index(drop=True)

        n   = len(df)
        w   = max(8, n * bar_w_in)
        fig = Figure(figsize=(w, 5.5));  ax = fig.add_subplot(111)

        single = self._bar_col.color
        if grp == "None (single colour)" or grp not in df.columns:
            colors, leg = [single] * n, []
        else:
            cats   = sorted(df[grp].dropna().unique())
            pal    = dict(zip(cats, sns.color_palette("tab10", len(cats))))
            colors = [pal.get(g, "#888") for g in df[grp]]
            leg    = [mpatches.Patch(color=pal[c], label=c) for c in cats]

        yerr = df["_se"].fillna(0).values
        ax.bar(np.arange(n), df["_y"].values, color=colors, width=0.65, zorder=3,
               yerr=yerr if not np.all(yerr == 0) else None,
               ecolor=self._err_col.color, capsize=3,
               error_kw={"lw": 1.2, "zorder": 4})

        ha  = "right" if rot_deg > 0 else "center"
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(df["_lbl"].values, rotation=rot_deg, ha=ha,
                           fontsize=fs * 0.85, fontfamily=ff)
        ax.set_ylabel(used_ylabel, fontsize=fs, fontfamily=ff)
        ax.set_title(title, fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        ax.tick_params(axis="y", labelsize=fs * 0.9)
        ax.grid(axis="y", alpha=0.35, zorder=0);  ax.set_axisbelow(True)
        if leg:
            ax.legend(handles=leg, title=grp, fontsize=fs * 0.85,
                      title_fontsize=fs * 0.85,
                      bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        return fig

    # ── Export ─────────────────────────────────────────────────────────────

    def _export(self):
        if not self._current_figs:
            QMessageBox.warning(self, "Nothing to export",
                                "Generate charts first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "heatmap_export.pdf",
            "PDF (*.pdf);;All files (*)")
        if not path:
            return
        try:
            with PdfPages(path) as pdf:
                for fig in self._current_figs:
                    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            QMessageBox.information(
                self, "Exported",
                f"Saved {len(self._current_figs)} page(s) to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


# ── Mode picker dialog ────────────────────────────────────────────────────────

class ModePicker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhosphoMAX — Select Analysis Mode")
        self.setFixedSize(420, 275)
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

        btn_heat = QPushButton("Heatmap / Bar Chart  (from results)")
        btn_heat.setFixedHeight(52)
        btn_heat.setStyleSheet("font-size: 14px;")
        btn_heat.setToolTip(
            "Load pipeline output files (binding_fit_results.xlsx /\n"
            "competitive_fit_results.xlsx) or Kd_Tables_Gen format\n"
            "and export pKd / pKi heatmaps or bar charts as PDF.")
        btn_heat.clicked.connect(lambda: self._pick("heatmap"))

        layout.addWidget(btn_direct)
        layout.addWidget(btn_comp)
        layout.addWidget(btn_heat)

    def _pick(self, choice):
        self.choice = choice
        self.accept()


# ── Chromatic → Dye mapping widget (competitive binding) ──────────────────────

class ChromaticMappingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Plate (substring)", "Order", "Dye"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(140)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setToolTip(
            "Leave 'Plate (substring)' empty to apply this row to ALL plates.\n"
            "Enter a filename substring to apply only to plates whose filename\n"
            "contains that text (case-insensitive).\n\n"
            "Example — two dyes on separate plates:\n"
            "  PlateA  | 1 | DAPI\n"
            "  PlateB  | 1 | DASPI")
        layout.addWidget(self._table)

        btns = QHBoxLayout()
        add_btn = QPushButton("+ Row")
        del_btn = QPushButton("− Row")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._del_row)
        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        layout.addLayout(btns)

        for plate, order, dye in [("", 1, "DAPI"), ("", 2, "DASPI"), ("", 3, "H33")]:
            self._insert(plate, order, dye)

    def _insert(self, plate, order, dye):
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(str(plate)))
        self._table.setItem(r, 1, QTableWidgetItem(str(order)))
        self._table.setItem(r, 2, QTableWidgetItem(str(dye)))

    def _add_row(self):
        self._insert("", self._table.rowCount() + 1, "")

    def _del_row(self):
        rows = {i.row() for i in self._table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self._table.removeRow(r)

    @property
    def mapping(self) -> dict:
        """
        Returns {"": {order: dye}, "pattern": {order: dye}, ...}
        Empty string key = global fallback (applies to plates not matched by any pattern).
        """
        result = {}
        for r in range(self._table.rowCount()):
            try:
                plate = (self._table.item(r, 0).text().strip()
                         if self._table.item(r, 0) else "")
                order = int(self._table.item(r, 1).text())
                dye   = self._table.item(r, 2).text().strip()
                if dye:
                    result.setdefault(plate, {})[order] = dye
            except (ValueError, AttributeError):
                pass
        return result


class GridCanvas(FigureCanvas):
    plot_clicked = Signal(int)

    def __init__(self, items: list, color_data: str = None, color_fit: str = None,
                 color_resid: str = None, show_residuals: bool = False, parent=None):
        self._items = items
        n    = max(len(items), 1)
        cols = min(int(np.ceil(np.sqrt(n))), 4)   # max 4 columns for readability
        rows = int(np.ceil(n / cols))
        cd   = color_data  or pl_fda.PLOT_COLOR
        cf   = color_fit   or pl_fda.PLOT_COLOR
        cr   = color_resid or cd

        if show_residuals:
            fig = Figure(figsize=(cols * 4, rows * 4.2))
            gs  = GridSpec(rows * 2, cols, figure=fig,
                           height_ratios=_RESID_RATIO * rows,
                           hspace=_RESID_HSPACE, wspace=0.4)
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                row_i, col_i = divmod(i, cols)
                ax   = fig.add_subplot(gs[row_i * 2,     col_i])
                ax_r = fig.add_subplot(gs[row_i * 2 + 1, col_i], sharex=ax)
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                                   color_resid=cr, ax_resid=ax_r)
                self._axes.append(ax)
        else:
            fig = Figure(figsize=(cols * 4, rows * 3.5))
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ax = fig.add_subplot(rows, cols, i + 1)
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf, color_resid=cr)
                self._axes.append(ax)
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        self.draw()

    def mousePressEvent(self, event):
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

    def show_entry(self, entry, color_data: str = None, color_fit: str = None,
                   color_resid: str = None, show_residuals: bool = False):
        self._fig.clear()
        cd = color_data  or pl_fda.PLOT_COLOR
        cf = color_fit   or pl_fda.PLOT_COLOR
        cr = color_resid or cd
        if show_residuals:
            gs   = GridSpec(2, 1, figure=self._fig,
                            height_ratios=_RESID_RATIO, hspace=_RESID_HSPACE)
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
        else:
            ax   = self._fig.add_subplot(111)
            ax_r = None
        pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                           color_resid=cr, ax_resid=ax_r)
        self._fig.tight_layout()
        self.draw()


class DirectPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pass_items     = []
        self._fail_items     = []
        self._current_items  = []
        self._current_idx    = 0
        self._color_data     = pl_fda.PLOT_COLOR
        self._color_fit      = pl_fda.PLOT_COLOR
        self._color_resid    = pl_fda.PLOT_COLOR
        self._show_residuals = False

        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self._view_btn   = QPushButton("Switch to Single View")
        self._prev_btn   = QPushButton("◀ Prev")
        self._next_btn   = QPushButton("Next ▶")
        self._jump       = QComboBox()
        self._jump.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._plate_lbl  = QLabel("Plate:")
        self._plate_combo = QComboBox()
        self._plate_combo.setMinimumWidth(100)
        ctrl.addWidget(self._view_btn)
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._jump)
        ctrl.addWidget(self._plate_lbl)
        ctrl.addWidget(self._plate_combo)
        layout.addLayout(ctrl)

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
        self._plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    def _plates_for(self, items):
        seen, out = set(), []
        for e in items:
            p = e.get("plate", "")
            if p not in seen:
                seen.add(p); out.append(p)
        return sorted(out)

    def _items_for_plate(self, items, plate):
        if not plate:
            return items
        return [e for e in items if e.get("plate", "") == plate]

    def load(self, pass_items: list, fail_items: list,
             color_data: str = None, color_fit: str = None,
             color_resid: str = None, show_residuals: bool = None):
        self._pass_items = pass_items
        self._fail_items = fail_items
        if color_data     is not None: self._color_data     = color_data
        if color_fit      is not None: self._color_fit      = color_fit
        if color_resid    is not None: self._color_resid    = color_resid
        if show_residuals is not None: self._show_residuals = show_residuals
        # Rebuild plate list (union of all plates)
        all_plates = sorted(set(
            e.get("plate", "") for e in pass_items + fail_items if e.get("plate", "")))
        self._plate_combo.blockSignals(True)
        self._plate_combo.clear()
        self._plate_combo.addItem("All plates")
        for p in all_plates:
            self._plate_combo.addItem(p)
        self._plate_combo.blockSignals(False)
        self._build_grids()
        self._build_jump_list()

    def _current_plate(self):
        txt = self._plate_combo.currentText()
        return "" if txt == "All plates" else txt

    def _build_grids(self):
        plate = self._current_plate()
        pass_items = self._items_for_plate(self._pass_items, plate)
        fail_items = self._items_for_plate(self._fail_items, plate)
        if pass_items:
            canvas = GridCanvas(pass_items,
                                color_data=self._color_data, color_fit=self._color_fit,
                                color_resid=self._color_resid,
                                show_residuals=self._show_residuals)
            canvas.plot_clicked.connect(lambda i: self._open_single("PASS", i))
            self._grid_scroll_pass.setWidget(canvas)
        else:
            self._grid_scroll_pass.setWidget(QLabel("  No PASS results for this plate."))
        if fail_items:
            canvas = GridCanvas(fail_items,
                                color_data=self._color_data, color_fit=self._color_fit,
                                color_resid=self._color_resid,
                                show_residuals=self._show_residuals)
            canvas.plot_clicked.connect(lambda i: self._open_single("FAIL", i))
            self._grid_scroll_fail.setWidget(canvas)
        else:
            self._grid_scroll_fail.setWidget(QLabel("  No FAIL results for this plate."))

    def _build_jump_list(self):
        self._jump.clear()
        for entry in self._pass_items:
            self._jump.addItem(f"[PASS] {entry['host']}-{entry['dye']}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry['host']}-{entry['dye']}")

    def _on_plate_changed(self, _):
        if self._is_grid:
            self._build_grids()

    def _on_subtab_changed(self, _):
        if not self._is_grid:
            self._show_single(0)

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
        self._is_grid = True
        self._toggle_view()
        self._subtabs.setCurrentIndex(0 if kind == "PASS" else 1)
        self._current_items = self._pass_items if kind == "PASS" else self._fail_items
        self._current_idx   = idx
        canvas = self._single_pass if kind == "PASS" else self._single_fail
        canvas.show_entry(self._current_items[idx],
                          color_data=self._color_data, color_fit=self._color_fit,
                          color_resid=self._color_resid,
                          show_residuals=self._show_residuals)

    def _show_single(self, idx: int):
        tab    = self._subtabs.currentIndex()
        items  = self._pass_items if tab == 0 else self._fail_items
        canvas = self._single_pass if tab == 0 else self._single_fail
        if items:
            self._current_idx = idx % len(items)
            canvas.show_entry(items[self._current_idx],
                              color_data=self._color_data, color_fit=self._color_fit,
                              color_resid=self._color_resid,
                              show_residuals=self._show_residuals)

    def _prev(self): self._show_single(self._current_idx - 1)
    def _next(self): self._show_single(self._current_idx + 1)

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


# ── QC plots tab (used by both modes) ────────────────────────────────────────

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


class CompGridCanvas(FigureCanvas):
    plot_clicked = Signal(int)

    def __init__(self, items: list, color_fit: str = None, color_data: str = None,
                 color_resid: str = None, show_residuals: bool = False,
                 title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "bold", title_fontstyle: str = "normal",
                 parent=None):
        self._items = items
        n    = max(len(items), 1)
        cols = min(int(np.ceil(np.sqrt(n))), 4)   # max 4 columns for readability
        rows = int(np.ceil(n / cols))
        cf   = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd   = color_data  or pl_ki.PLOT_COLOR_DATA
        cr   = color_resid or cd
        tkw  = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                    title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                    stats_in_title=False)

        if show_residuals:
            # Nested GridSpec: constrained_layout handles inter-row spacing;
            # inner subgridspec keeps each main+residual pair tight.
            fig = Figure(figsize=(cols * 3, rows * 5.5), layout="constrained")
            outer_gs = GridSpec(rows, cols, figure=fig)
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                row_i, col_i = divmod(i, cols)
                inner_gs = outer_gs[row_i, col_i].subgridspec(
                    2, 1, height_ratios=_RESID_RATIO, hspace=0.05)
                ax   = fig.add_subplot(inner_gs[0])
                ax_r = fig.add_subplot(inner_gs[1], sharex=ax)
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, ax_resid=ax_r, **tkw)
                self._axes.append(ax)
        else:
            # constrained_layout resolves all text spacing at draw time.
            fig = Figure(figsize=(cols * 3, rows * 5.0), layout="constrained")
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ax = fig.add_subplot(rows, cols, i + 1)
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, **tkw)
                self._axes.append(ax)

        self.draw()

    def mousePressEvent(self, event):
        w, h = self.width(), self.height()
        x    = event.position().x() / w
        y    = 1.0 - event.position().y() / h
        for i, ax in enumerate(self._axes):
            bbox = ax.get_position()
            if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1:
                self.plot_clicked.emit(i)
                return
        super().mousePressEvent(event)


class CompSinglePlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(6, 4))
        super().__init__(self._fig)

    def show_entry(self, entry, color_fit: str = None, color_data: str = None,
                   color_resid: str = None, show_residuals: bool = False,
                   title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "bold", title_fontstyle: str = "normal"):
        self._fig.clear()
        cf = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd = color_data  or pl_ki.PLOT_COLOR_DATA
        cr = color_resid or cd
        if show_residuals:
            gs   = GridSpec(2, 1, figure=self._fig,
                            height_ratios=_RESID_RATIO, hspace=_RESID_HSPACE)
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
        else:
            ax   = self._fig.add_subplot(111)
            ax_r = None
        pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                             color_resid=cr, ax_resid=ax_r,
                             title_fontsize=title_fontsize,
                             title_fontfamily=title_fontfamily,
                             title_fontweight=title_fontweight,
                             title_fontstyle=title_fontstyle)
        self._fig.tight_layout()
        self.draw()


class CompPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pass_items     = []
        self._fail_items     = []
        self._current_items  = []
        self._current_idx    = 0
        self._color_fit        = pl_ki.PLOT_COLOR_FIT
        self._color_data       = pl_ki.PLOT_COLOR_DATA
        self._color_resid      = pl_ki.PLOT_COLOR_DATA
        self._show_residuals   = False
        self._title_fontsize   = 8.0
        self._title_fontfamily = "sans-serif"
        self._title_fontweight = "bold"
        self._title_fontstyle  = "normal"

        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self._view_btn    = QPushButton("Switch to Single View")
        self._prev_btn    = QPushButton("◀ Prev")
        self._next_btn    = QPushButton("Next ▶")
        self._jump        = QComboBox()
        self._jump.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._host_lbl    = QLabel("Host:")
        self._host_combo  = QComboBox()
        self._host_combo.setMinimumWidth(100)
        self._plate_lbl   = QLabel("Plate:")
        self._plate_combo = QComboBox()
        self._plate_combo.setMinimumWidth(120)
        ctrl.addWidget(self._view_btn)
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._jump)
        ctrl.addWidget(self._host_lbl)
        ctrl.addWidget(self._host_combo)
        ctrl.addWidget(self._plate_lbl)
        ctrl.addWidget(self._plate_combo)
        layout.addLayout(ctrl)

        self._subtabs = QTabWidget()
        layout.addWidget(self._subtabs)

        self._grid_scroll_pass = QScrollArea()
        self._grid_scroll_pass.setWidgetResizable(True)
        self._grid_scroll_fail = QScrollArea()
        self._grid_scroll_fail.setWidgetResizable(True)
        self._single_pass = CompSinglePlotCanvas()
        self._single_fail = CompSinglePlotCanvas()

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
        self._host_combo.currentIndexChanged.connect(self._on_host_changed)
        self._plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    def _items_for_filters(self, items, host, plate):
        result = items
        if host:
            result = [e for e in result if e.get("host", "") == host]
        if plate:
            result = [e for e in result if e.get("plate", "") == plate]
        return result

    def load(self, pass_items: list, fail_items: list,
             color_fit: str = None, color_data: str = None,
             color_resid: str = None, show_residuals: bool = None,
             title_fontsize: float = None, title_fontfamily: str = None,
             title_fontweight: str = None, title_fontstyle: str = None):
        self._pass_items = pass_items
        self._fail_items = fail_items
        if color_fit        is not None: self._color_fit        = color_fit
        if color_data       is not None: self._color_data       = color_data
        if color_resid      is not None: self._color_resid      = color_resid
        if show_residuals   is not None: self._show_residuals   = show_residuals
        if title_fontsize   is not None: self._title_fontsize   = title_fontsize
        if title_fontfamily is not None: self._title_fontfamily = title_fontfamily
        if title_fontweight is not None: self._title_fontweight = title_fontweight
        if title_fontstyle  is not None: self._title_fontstyle  = title_fontstyle

        all_items = pass_items + fail_items
        all_hosts = sorted(set(e.get("host", "") for e in all_items if e.get("host", "")))
        self._host_combo.blockSignals(True)
        self._host_combo.clear()
        self._host_combo.addItem("All hosts")
        for h in all_hosts:
            self._host_combo.addItem(h)
        self._host_combo.blockSignals(False)

        all_plates = sorted(set(e.get("plate", "") for e in all_items if e.get("plate", "")))
        self._plate_combo.blockSignals(True)
        self._plate_combo.clear()
        self._plate_combo.addItem("All plates")
        for p in all_plates:
            self._plate_combo.addItem(p)
        self._plate_combo.blockSignals(False)

        self._build_grids()
        self._build_jump_list()

    def _current_host(self):
        txt = self._host_combo.currentText()
        return "" if txt == "All hosts" else txt

    def _current_plate(self):
        txt = self._plate_combo.currentText()
        return "" if txt == "All plates" else txt

    def _build_grids(self):
        host  = self._current_host()
        plate = self._current_plate()
        pass_items = self._items_for_filters(self._pass_items, host, plate)
        fail_items = self._items_for_filters(self._fail_items, host, plate)
        tkw = dict(title_fontsize=self._title_fontsize,
                   title_fontfamily=self._title_fontfamily,
                   title_fontweight=self._title_fontweight,
                   title_fontstyle=self._title_fontstyle)
        no_pass_lbl = "  No PASS results for this selection."
        no_fail_lbl = "  No FAIL results for this selection."
        if pass_items:
            canvas = CompGridCanvas(pass_items,
                                    color_fit=self._color_fit, color_data=self._color_data,
                                    color_resid=self._color_resid,
                                    show_residuals=self._show_residuals, **tkw)
            canvas.plot_clicked.connect(lambda i: self._open_single("PASS", i))
            self._grid_scroll_pass.setWidget(canvas)
        else:
            self._grid_scroll_pass.setWidget(QLabel(no_pass_lbl))
        if fail_items:
            canvas = CompGridCanvas(fail_items,
                                    color_fit=self._color_fit, color_data=self._color_data,
                                    color_resid=self._color_resid,
                                    show_residuals=self._show_residuals, **tkw)
            canvas.plot_clicked.connect(lambda i: self._open_single("FAIL", i))
            self._grid_scroll_fail.setWidget(canvas)
        else:
            self._grid_scroll_fail.setWidget(QLabel(no_fail_lbl))

    def _build_jump_list(self):
        self._jump.clear()
        for entry in self._pass_items:
            self._jump.addItem(f"[PASS] {entry['host']} | {entry['dye']} | {entry['guest']}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry['host']} | {entry['dye']} | {entry['guest']}")

    def _on_host_changed(self, _):
        if self._is_grid:
            self._build_grids()

    def _on_plate_changed(self, _):
        if self._is_grid:
            self._build_grids()

    def _on_subtab_changed(self, _):
        if not self._is_grid:
            self._show_single(0)

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
        self._is_grid = True
        self._toggle_view()
        self._subtabs.setCurrentIndex(0 if kind == "PASS" else 1)
        self._current_items = self._pass_items if kind == "PASS" else self._fail_items
        self._current_idx   = idx
        canvas = self._single_pass if kind == "PASS" else self._single_fail
        canvas.show_entry(self._current_items[idx],
                          color_fit=self._color_fit, color_data=self._color_data,
                          color_resid=self._color_resid,
                          show_residuals=self._show_residuals,
                          title_fontsize=self._title_fontsize,
                          title_fontfamily=self._title_fontfamily,
                          title_fontweight=self._title_fontweight,
                          title_fontstyle=self._title_fontstyle)

    def _show_single(self, idx: int):
        tab    = self._subtabs.currentIndex()
        items  = self._pass_items if tab == 0 else self._fail_items
        canvas = self._single_pass if tab == 0 else self._single_fail
        if items:
            self._current_idx = idx % len(items)
            canvas.show_entry(items[self._current_idx],
                              color_fit=self._color_fit, color_data=self._color_data,
                              color_resid=self._color_resid,
                              show_residuals=self._show_residuals,
                              title_fontsize=self._title_fontsize,
                              title_fontfamily=self._title_fontfamily,
                              title_fontweight=self._title_fontweight,
                              title_fontstyle=self._title_fontstyle)

    def _prev(self): self._show_single(self._current_idx - 1)
    def _next(self): self._show_single(self._current_idx + 1)

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


# ── Summary tab  (pKd / pKi bar-chart & heatmap) ─────────────────────────────

class SummaryTab(QWidget):
    """Interactive pKd or pKi bar-chart / heatmap with configurable style."""

    def __init__(self, mode: str = "kd", parent=None):
        """mode: 'kd' (Direct Binding) or 'ki' (Competitive Binding)."""
        super().__init__(parent)
        self._mode        = mode
        self._df          = None
        self._current_fig = None   # last rendered figure — used by Export
        self._build_ui()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _val_col(self):  return "Kd"    if self._mode == "kd" else "Ki_uM"
    def _se_col(self):   return "Kd_SE" if self._mode == "kd" else "Ki_err_uM"
    def _pval(self):     return "pKd"   if self._mode == "kd" else "pKi"
    def _y_label(self):
        return ("pKd  =  −log₁₀(Kd / M)" if self._mode == "kd"
                else "pKi  =  −log₁₀(Ki / M)")

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QHBoxLayout(self)

        # ── left controls ─────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(240)
        cl   = QVBoxLayout(ctrl)
        cl.setAlignment(Qt.AlignTop)

        # Filter
        gb_f = QGroupBox("Filter")
        fl   = QFormLayout(gb_f)
        self._filt = QComboBox()
        self._filt.addItems(["PASS only", "FAIL only", "All"])
        fl.addRow("Show:", self._filt)
        cl.addWidget(gb_f)

        # Chart type
        gb_t = QGroupBox("Chart Type")
        tl   = QVBoxLayout(gb_t)
        self._bar_rb  = QRadioButton("Bar chart")
        self._heat_rb = QRadioButton("Heatmap")
        self._bar_rb.setChecked(True)
        tl.addWidget(self._bar_rb)
        tl.addWidget(self._heat_rb)
        cl.addWidget(gb_t)

        # Bar options ──────────────────────────────────────────────────────────
        self._gb_bar = QGroupBox("Bar Options")
        bv = QVBoxLayout(self._gb_bar)
        self._bar_col = ColorPickerRow("Bar colour:", "#4878CF")
        self._err_col = ColorPickerRow("Error bars:", "#222222")
        bv.addWidget(self._bar_col)
        bv.addWidget(self._err_col)
        bf = QFormLayout()
        self._grp = QComboBox()
        if self._mode == "kd":
            self._grp.addItems(["None (single colour)", "Dye", "Host"])
        else:
            self._grp.addItems(["None (single colour)", "Host", "Dye", "Guest"])
        self._sort = QComboBox()
        self._sort.addItems(["pK ↓ (strongest first)", "pK ↑ (weakest first)",
                              "Alphabetical"])
        self._yscale = QComboBox()
        self._yscale.addItems(["pK  (−log₁₀ K/M)", "Raw µM", "log₁₀(K/µM)"])
        self._lbl_rot = QComboBox()
        self._lbl_rot.addItems(["90°", "45°", "30°"])
        self._bar_w = QDoubleSpinBox()
        self._bar_w.setRange(0.4, 3.0)
        self._bar_w.setValue(1.0)
        self._bar_w.setSingleStep(0.1)
        self._bar_w.setDecimals(1)
        self._bar_w.setToolTip("Inches of figure width allocated per bar")
        bf.addRow("Colour by:", self._grp)
        bf.addRow("Sort:",      self._sort)
        bf.addRow("Y-axis:",    self._yscale)
        bf.addRow("Label rot:", self._lbl_rot)
        bf.addRow("Width/bar:", self._bar_w)
        bv.addLayout(bf)
        cl.addWidget(self._gb_bar)

        # Heatmap options ──────────────────────────────────────────────────────
        self._gb_heat = QGroupBox("Heatmap Options")
        hf = QFormLayout(self._gb_heat)
        self._cmap = QComboBox()
        self._cmap.addItems(["viridis", "plasma", "magma", "inferno",
                              "coolwarm", "RdYlGn", "Blues", "YlOrRd", "Greens"])
        self._heat_col_combo = QComboBox()   # ki only: columns variable
        if self._mode == "ki":
            self._heat_col_combo.addItems(["Host", "Dye", "Host | Dye"])
        self._annot = QCheckBox("Show values")
        self._annot.setChecked(True)
        self._annot_fmt = QComboBox()
        self._annot_fmt.addItems([".2f", ".1f", ".3f"])
        hf.addRow("Colormap:",   self._cmap)
        if self._mode == "ki":
            hf.addRow("Columns:", self._heat_col_combo)
        hf.addRow("Format:",     self._annot_fmt)
        hf.addRow(self._annot)
        cl.addWidget(self._gb_heat)
        self._gb_heat.setVisible(False)

        # Font options ─────────────────────────────────────────────────────────
        gb_font = QGroupBox("Font")
        ff_lay  = QFormLayout(gb_font)
        self._fam = QComboBox()
        self._fam.addItems(["sans-serif", "serif", "monospace",
                             "DejaVu Sans", "Arial", "Helvetica", "Times New Roman"])
        self._fsz = QDoubleSpinBox()
        self._fsz.setRange(6, 22)
        self._fsz.setValue(9)
        self._fsz.setSingleStep(0.5)
        self._fsz.setDecimals(1)
        ff_lay.addRow("Family:", self._fam)
        ff_lay.addRow("Size:",   self._fsz)
        cl.addWidget(gb_font)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._render)
        cl.addWidget(refresh_btn)

        export_btn = QPushButton("Export…")
        export_btn.setToolTip("Save the current chart as PDF, PNG or SVG")
        export_btn.clicked.connect(self._export)
        cl.addWidget(export_btn)

        cl.addStretch()
        outer.addWidget(ctrl)

        # ── right: scrollable canvas ──────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll, stretch=1)

        # signals
        self._bar_rb.toggled.connect(self._on_type)
        self._filt.currentIndexChanged.connect(self._render)

    def _on_type(self, bar):
        self._gb_bar.setVisible(bar)
        self._gb_heat.setVisible(not bar)
        self._render()

    # ── public ─────────────────────────────────────────────────────────────────
    def load(self, df):
        self._df = df
        self._render()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self):
        if self._df is None or self._df.empty:
            return
        df   = self._df.copy()
        filt = self._filt.currentText()
        if filt == "PASS only":  df = df[df["Status"] == "PASS"]
        elif filt == "FAIL only": df = df[df["Status"] == "FAIL"]
        if df.empty:
            self._scroll.setWidget(QLabel("  No data for this filter."))
            return

        vc = self._val_col();  sc = self._se_col();  pv = self._pval()
        df[pv]    = 6.0 - np.log10(df[vc].clip(lower=1e-12))
        df["p_SE"] = (df[sc] / (df[vc].clip(lower=1e-12) * np.log(10))
                      if sc in df.columns else np.nan)

        fs = self._fsz.value();  ff = self._fam.currentText()
        try:
            fig    = self._bar_chart(df, pv, fs, ff) if self._bar_rb.isChecked() \
                     else self._heatmap(df, pv, fs, ff)
            self._current_fig = fig
            canvas = FigureCanvas(fig)
            canvas.draw()
            self._scroll.setWidget(canvas)
        except Exception as exc:
            import traceback as _tb
            self._current_fig = None
            self._scroll.setWidget(QLabel(f"  Render error: {exc}\n{_tb.format_exc()}"))

    def _export(self):
        if self._current_fig is None:
            QMessageBox.warning(self, "Nothing to export",
                                "Generate a chart first by clicking Refresh.")
            return
        default_name = f"summary_{'pKd' if self._mode == 'kd' else 'pKi'}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary Chart", default_name,
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)")
        if not path:
            return
        try:
            self._current_fig.savefig(path, bbox_inches="tight", dpi=150)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _row_label(self, row):
        if self._mode == "kd":
            return f"{row['Host']} | {row['Dye']}\n[{row['Dye_Concentration']} µM]"
        else:
            return f"{row['Guest']}\n{row['Host']} | {row['Dye']}"

    def _bar_chart(self, df, pv, fs, ff):
        grp      = self._grp.currentText()
        stxt     = self._sort.currentText()
        yscale   = self._yscale.currentText()
        rot_deg  = int(self._lbl_rot.currentText().replace("°", ""))
        bar_w_in = self._bar_w.value()

        vc = self._val_col();  sc = self._se_col()
        df = df.copy()
        df["_lbl"] = df.apply(self._row_label, axis=1)

        # Build the y column from the chosen scale
        if "pK" in yscale:
            df["_y"]  = df[pv]          # already computed in _render
            df["_se"] = df["p_SE"]
            used_ylabel = self._y_label()
        elif "Raw" in yscale:
            df["_y"]  = df[vc]
            df["_se"] = df[sc] if sc in df.columns else np.nan
            used_ylabel = "Kd (µM)" if self._mode == "kd" else "Ki (µM)"
        else:  # log10
            df["_y"]  = np.log10(df[vc].clip(lower=1e-12))
            df["_se"] = np.nan
            used_ylabel = "log₁₀(Kd / µM)" if self._mode == "kd" else "log₁₀(Ki / µM)"

        if "↓" in stxt:   df = df.sort_values("_y", ascending=False)
        elif "↑" in stxt: df = df.sort_values("_y", ascending=True)
        else:              df = df.sort_values("_lbl")
        df = df.reset_index(drop=True)

        n   = len(df)
        w   = max(8, n * bar_w_in)
        fig = Figure(figsize=(w, 5.5));  ax = fig.add_subplot(111)

        single = self._bar_col.color
        if grp == "None (single colour)" or grp not in df.columns:
            colors, leg = [single] * n, []
        else:
            cats    = sorted(df[grp].dropna().unique())
            pal     = dict(zip(cats, sns.color_palette("tab10", len(cats))))
            colors  = [pal.get(g, "#888") for g in df[grp]]
            leg     = [mpatches.Patch(color=pal[c], label=c) for c in cats]

        yerr = df["_se"].fillna(0).values
        ax.bar(np.arange(n), df["_y"].values, color=colors, width=0.65, zorder=3,
               yerr=yerr if not np.all(yerr == 0) else None,
               ecolor=self._err_col.color, capsize=3,
               error_kw={"lw": 1.2, "zorder": 4})

        ha = "right" if rot_deg > 0 else "center"
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(df["_lbl"].values, rotation=rot_deg, ha=ha,
                           fontsize=fs * 0.85, fontfamily=ff)
        ax.set_ylabel(used_ylabel, fontsize=fs, fontfamily=ff)
        ax.set_title(f"{'pKd' if self._mode == 'kd' else 'pKi'} — summary",
                     fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        ax.tick_params(axis="y", labelsize=fs * 0.9)
        ax.grid(axis="y", alpha=0.35, zorder=0);  ax.set_axisbelow(True)
        if leg:
            ax.legend(handles=leg, title=grp, fontsize=fs * 0.85,
                      title_fontsize=fs * 0.85,
                      bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
        fig.tight_layout()
        return fig

    def _heatmap(self, df, pv, fs, ff):
        cmap  = self._cmap.currentText()
        annot = self._annot.isChecked()
        fmt   = self._annot_fmt.currentText()

        if self._mode == "kd":
            row_col = "Host";  col_col = "Dye"
        else:
            row_col = "Guest"
            cm = self._heat_col_combo.currentText()
            if cm == "Host | Dye":
                df = df.copy()
                df["_hd"] = df["Host"] + " | " + df["Dye"]
                col_col = "_hd"
            else:
                col_col = cm   # "Host" or "Dye"

        pivot = df.groupby([row_col, col_col])[pv].mean().unstack(col_col)
        nr, nc = pivot.shape
        fig = Figure(figsize=(max(6, nc * 0.8 + 2), max(4, nr * 0.55 + 1.5)))
        ax  = fig.add_subplot(111)

        data = pivot.values.astype(float)
        vmin = float(np.nanmin(data)) if not np.all(np.isnan(data)) else 0.0
        vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right",
                           fontsize=fs * 0.9, fontfamily=ff)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(pivot.index, fontsize=fs * 0.9, fontfamily=ff)

        if annot:
            span = max(vmax - vmin, 1e-9)
            for i in range(nr):
                for j in range(nc):
                    v = data[i, j]
                    if not np.isnan(v):
                        txt_c = "white" if (v - vmin) / span > 0.5 else "black"
                        ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                                fontsize=fs * 0.85, fontfamily=ff, color=txt_c)

        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(self._y_label(), fontsize=fs * 0.9, fontfamily=ff)
        cb.ax.tick_params(labelsize=fs * 0.85)
        ax.set_xlabel(col_col.replace("_hd", "Host | Dye"), fontsize=fs, fontfamily=ff)
        ax.set_ylabel(row_col, fontsize=fs, fontfamily=ff)
        ax.set_title(f"{'pKd' if self._mode == 'kd' else 'pKi'} — heatmap",
                     fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        fig.tight_layout()
        return fig


# ── Direct Binding main window ────────────────────────────────────────────────
# Tab layout: 0=Mappings, 1=Raw Data, 2=FI-F0, 3=QC Plots, 4=Fit Results, 5=Plots

STAGE_NAMES_FDA = ["Mappings", "Raw Data", "FI-F0", "Fit Results", "Plots"]


class DirectMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Direct Binding (Kd)")
        self.resize(1300, 800)

        self._state   = PipelineState()
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
        self._rerun_combo  = QComboBox()
        self._rerun_combo.addItems(STAGE_NAMES_FDA)
        self._rerun_btn  = QPushButton("Re-run")
        self._switch_btn = QPushButton("⇄ Competitive Binding")
        self._status_lbl = QLabel("Ready")

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
        self._switch_btn.clicked.connect(self._on_open_competitive)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_config())

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_log())
        right.setSizes([600, 150])
        splitter.addWidget(right)
        splitter.setSizes([280, 1020])

    def _build_config(self):
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        box_f = QGroupBox("Folders")
        ff    = QFormLayout(box_f)
        self._input_row  = FolderRow(str(_BASE_DIR))
        self._output_row = FolderRow(str(_BASE_DIR))
        ff.addRow("Experiment:", self._input_row)
        ff.addRow("Output:",     self._output_row)
        hint = QLabel("Raw/  dye_map/  host_map/  blank_map/\n← expected inside experiment folder")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        ff.addRow(hint)
        layout.addWidget(box_f)

        box_p = QGroupBox("Parameters")
        pf    = QFormLayout(box_p)

        self._r2_spin = QDoubleSpinBox()
        self._r2_spin.setRange(0.0, 1.0)
        self._r2_spin.setSingleStep(0.05)
        self._r2_spin.setValue(pl_fda.PASS_R2_DEFAULT)
        self._r2_spin.setDecimals(2)
        self._r2_spin.valueChanged.connect(self._on_r2_changed)

        self._kd_lo_spin = QDoubleSpinBox()
        self._kd_lo_spin.setRange(0.0, 1.0)
        self._kd_lo_spin.setSingleStep(0.05)
        self._kd_lo_spin.setValue(pl_fda.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_lo_spin.setDecimals(2)
        self._kd_lo_spin.setToolTip(
            "FAIL if Kd < factor × min([Host])\n(Kd below measured range — extrapolating left)")
        self._kd_lo_spin.valueChanged.connect(self._on_r2_changed)

        self._kd_hi_spin = QDoubleSpinBox()
        self._kd_hi_spin.setRange(1.0, 1000.0)
        self._kd_hi_spin.setSingleStep(1.0)
        self._kd_hi_spin.setValue(pl_fda.KD_RANGE_FACTOR_HI_DEFAULT)
        self._kd_hi_spin.setDecimals(1)
        self._kd_hi_spin.setToolTip(
            "FAIL if Kd > factor × max([Host])\n(never reaches saturation — extrapolating right)")
        self._kd_hi_spin.valueChanged.connect(self._on_r2_changed)

        self._grubbs_spin = QDoubleSpinBox()
        self._grubbs_spin.setRange(0.001, 0.2)
        self._grubbs_spin.setSingleStep(0.005)
        self._grubbs_spin.setValue(0.05)
        self._grubbs_spin.setDecimals(3)
        self._grubbs_spin.valueChanged.connect(self._on_grubbs_changed)

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

        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["Auto (AICc)", "One-site only", "Quadratic only", "Stern-Volmer only"])

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Kd range lo:",  self._kd_lo_spin)
        pf.addRow("Kd range hi:",  self._kd_hi_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow(self._cross_conc_chk, self._cross_conc_alpha_spin)
        pf.addRow("Model:",        self._model_combo)
        layout.addWidget(box_p)

        box_col = QGroupBox("Plot Colors")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", pl_fda.PLOT_COLOR)
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", pl_fda.PLOT_COLOR)
        self._color_resid_row = ColorPickerRow("Residuals:  ", pl_fda.PLOT_COLOR)
        self._color_data_row.color_changed.connect(self._on_color_changed)
        self._color_fit_row.color_changed.connect(self._on_color_changed)
        self._color_resid_row.color_changed.connect(self._on_color_changed)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        layout.addWidget(box_col)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(False)
        self._resid_chk.setToolTip("Show residual sub-panel beneath each binding curve")
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On Save: write one PDF per curve into\n"
            "results/plots/individual/PASS|FAIL/")
        self._summary_chk = QCheckBox("Show summary tab")
        self._summary_chk.setChecked(True)
        self._summary_chk.setToolTip("pKd bar-chart / heatmap summary tab")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return panel

    def _build_tabs(self):
        self._tabs = QTabWidget()

        self._map_view, self._map_model = _make_table()
        self._tabs.addTab(self._map_view, "Mappings")           # 0

        raw_w = QWidget()
        raw_l = QVBoxLayout(raw_w)
        self._raw_lbl = QLabel("")
        self._raw_view, self._raw_model = _make_table()
        raw_l.addWidget(self._raw_lbl)
        raw_l.addWidget(self._raw_view)
        self._tabs.addTab(raw_w, "Raw Data")                    # 1

        self._fi_view, self._fi_model = _make_table()
        self._tabs.addTab(self._fi_view, "FI-F0")               # 2

        self._qc_tab = QCPlotsTab()
        self._tabs.addTab(self._qc_tab, "QC Plots")             # 3

        fit_w = QWidget()
        fit_l = QVBoxLayout(fit_w)
        self._fit_lbl = QLabel("")
        self._fit_lbl.setAlignment(Qt.AlignCenter)
        font = QFont(); font.setBold(True)
        self._fit_lbl.setFont(font)
        self._fit_view, self._fit_model = _make_table()
        fit_l.addWidget(self._fit_lbl)
        fit_l.addWidget(self._fit_view)
        self._tabs.addTab(fit_w, "Fit Results")                 # 4

        self._plots_tab = DirectPlotsTab()
        self._tabs.addTab(self._plots_tab, "Plots")             # 5

        self._summary_tab = SummaryTab(mode="kd")
        self._tabs.addTab(self._summary_tab, "Summary")         # 6

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
            "color_data":            self._color_data_row.color,
            "color_fit":             self._color_fit_row.color,
            "color_resid":           self._color_resid_row.color,
            "show_residuals":        self._resid_chk.isChecked(),
            "export_individual":     self._ind_export_chk.isChecked(),
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_fda.PASS_R2_DEFAULT)
        self._kd_lo_spin.setValue(pl_fda.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_hi_spin.setValue(pl_fda.KD_RANGE_FACTOR_HI_DEFAULT)
        self._grubbs_spin.setValue(0.05)
        self._cross_conc_chk.setChecked(True)
        self._cross_conc_alpha_spin.setValue(0.01)
        self._model_combo.setCurrentIndex(0)
        self._color_data_row.set_color(pl_fda.PLOT_COLOR)
        self._color_fit_row.set_color(pl_fda.PLOT_COLOR)
        self._color_resid_row.set_color(pl_fda.PLOT_COLOR)
        self._resid_chk.setChecked(False)
        self._ind_export_chk.setChecked(False)
        self._summary_chk.setChecked(True)
        if self._state.plot_data:
            self._refresh_results()

    def _on_color_changed(self, _hex):
        if self._state.plot_data:
            self._refresh_results()

    def _on_display_changed(self, _state):
        if self._state.plot_data:
            self._refresh_results()

    def _on_summary_toggle(self, state):
        self._tabs.setTabVisible(6, bool(state))
        if state and self._state.df_results is not None:
            self._summary_tab.load(self._state.df_results)

    def _set_buttons_enabled(self):
        busy = self._worker is not None and self._worker.isRunning()
        self._run_all_btn.setEnabled(not busy)
        self._run_step_btn.setEnabled(not busy and self._stage < 5)
        self._rerun_btn.setEnabled(not busy)
        self._rerun_combo.setEnabled(not busy)

    def _on_run_all(self):
        self._stage   = 0
        self._run_all = True
        self._state   = PipelineState()
        self._run_stage()

    def _on_run_step(self):
        self._run_all = False
        self._run_stage()

    def _on_rerun(self):
        sel = self._rerun_combo.currentIndex()
        prereqs = {2: ("merged_mapping", "fluorescence"), 3: ("merged",), 4: ("fi_df",)}
        missing = [p for p in prereqs.get(sel, []) if getattr(self._state, p) is None]
        if missing:
            QMessageBox.warning(self, "Missing prerequisite",
                                f"Stage '{STAGE_NAMES_FDA[sel]}' requires earlier stages first.\n"
                                f"Missing: {', '.join(missing)}")
            return
        self._run_all = False
        self._stage   = sel
        self._run_stage()

    def _run_stage(self):
        cfg = self._get_config()
        if self._stage == 0:
            self._launch(pl_fda.load_mappings,
                         args=(cfg["dye_folder"], cfg["host_folder"]))
        elif self._stage == 1:
            self._launch(pl_fda.load_plates,
                         args=(cfg["raw_folder"],))
        elif self._stage == 2:
            self._launch(pl_fda.merge_blanks,
                         args=(self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]))
        elif self._stage == 3:
            self._launch(pl_fda.subtract_background,
                         args=(self._state.merged,))
        elif self._stage == 4:
            self._launch(pl_fda.fit_curves,
                         args=(self._state.fi_df,),
                         kwargs={"grubbs_alpha":          cfg["grubbs_alpha"],
                                 "use_cross_conc_grubbs": cfg["use_cross_conc_grubbs"],
                                 "cross_conc_alpha":      cfg["cross_conc_alpha"],
                                 "model_preference":      cfg["model_preference"]})

    def _launch(self, fn, args=(), kwargs=None):
        self._worker = StageWorker(fn, args=args, kwargs=kwargs or {})
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_stage_done)
        self._worker.error.connect(lambda tb: (self._log.append(f"\n[ERROR]\n{tb}"),
                                               QMessageBox.critical(self, "Error", "See log.")))
        stage_lbl = STAGE_NAMES_FDA[self._stage] if self._stage < len(STAGE_NAMES_FDA) else "?"
        self._status_lbl.setText(f"Running: {stage_lbl}…")
        self._set_buttons_enabled()
        self._worker.start()

    def _on_stage_done(self, result):
        if isinstance(result, Exception):
            self._status_lbl.setText("Error — see log")
            self._set_buttons_enabled()
            return

        if self._stage == 0:
            self._state.merged_mapping = result
            self._map_model.set_dataframe(result)
            self._tabs.setTabEnabled(0, True)
            self._tabs.setCurrentIndex(0)

        elif self._stage == 1:
            self._state.fluorescence = result
            self._raw_model.set_dataframe(result)
            self._raw_lbl.setText(f"{len(result):,} rows  |  "
                                   f"{result['Plate'].nunique()} plate(s)  |  "
                                   f"{result['Chromatic'].nunique()} chromatic(s)")
            self._tabs.setTabEnabled(1, True)
            self._tabs.setCurrentIndex(1)

        elif self._stage == 2:
            self._state.merged = result

        elif self._stage == 3:
            # subtract_background returns a plain DataFrame
            self._state.fi_df = result
            self._fi_model.set_dataframe(result)
            self._tabs.setTabEnabled(2, True)
            self._tabs.setCurrentIndex(2)
            # Generate QC figures here in the main thread (seaborn is not
            # thread-safe and must not run inside StageWorker)
            try:
                qc = pl_fda.make_qc_figures_fda(result)
            except Exception:
                import traceback as _tb
                self._log.append(f"\n[QC warning]\n{_tb.format_exc()}")
                qc = []
            self._state.qc_figures = qc
            self._qc_tab.load(qc)
            self._tabs.setTabEnabled(3, True)

        elif self._stage == 4:
            fit_results, plot_data = result
            self._state.fit_results = fit_results
            self._state.plot_data   = plot_data
            self._tabs.setTabEnabled(4, True)
            self._tabs.setTabEnabled(5, True)
            if self._summary_chk.isChecked():
                self._tabs.setTabEnabled(6, True)
            self._tabs.setCurrentIndex(4)
            self._save_btn.setEnabled(True)
            try:
                self._refresh_results()
            except Exception:
                import traceback as _tb
                self._log.append(f"\n[ERROR in refresh]\n{_tb.format_exc()}")

        self._stage += 1
        self._status_lbl.setText(f"Stage {self._stage}/{len(STAGE_NAMES_FDA)} done")
        self._set_buttons_enabled()

        if self._run_all and self._stage < len(STAGE_NAMES_FDA):
            self._run_stage()

    def _on_r2_changed(self, _value):
        if self._state.fit_results:
            self._refresh_results()

    def _on_grubbs_changed(self, _value):
        if self._state.fit_results:
            QMessageBox.information(
                self, "Re-run required",
                "Grubbs α affects outlier removal during curve fitting.\n"
                "Please re-run from the Fit Results stage to apply the change.")

    def _refresh_results(self):
        cfg = self._get_config()
        df  = pl_fda.apply_thresholds(
            self._state.fit_results, self._state.plot_data,
            r2_threshold=cfg["r2_threshold"],
            kd_range_lo=cfg["kd_range_lo"],
            kd_range_hi=cfg["kd_range_hi"])
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
        self._plots_tab.load(pass_items, fail_items,
                             color_data=cfg["color_data"], color_fit=cfg["color_fit"],
                             color_resid=cfg["color_resid"],
                             show_residuals=cfg["show_residuals"])
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)

    def _on_save(self):
        cfg       = self._get_config()
        file_list = pl_fda.preview_save_files(self._state, cfg["output_folder"],
                                              export_individual=cfg["export_individual"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        color_data = cfg["color_data"]
        color_fit  = cfg["color_fit"]
        color_resid = cfg["color_resid"]
        show_resid  = cfg["show_residuals"]
        export_ind  = cfg["export_individual"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_fda.save_outputs(self._state, cfg["output_folder"],
                                       cfg["r2_threshold"], progress_cb,
                                       plot_color_data=color_data,
                                       plot_color_fit=color_fit,
                                       plot_color_resid=color_resid,
                                       show_residuals=show_resid,
                                       export_individual=export_ind)
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

    def _on_open_competitive(self):
        try:
            win = CompMainWindow()
            win.show()
            app = QApplication.instance()
            if not hasattr(app, "_extra_windows"):
                app._extra_windows = []
            app._extra_windows.append(win)
        except Exception as e:
            QMessageBox.critical(self, "Could not open Competitive Binding",
                                 f"{type(e).__name__}: {e}")


# ── Competitive Binding main window ───────────────────────────────────────────

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
        self._rerun_combo  = QComboBox()
        self._rerun_combo.addItems(STAGE_NAMES_KI)
        self._rerun_btn  = QPushButton("Re-run")
        self._switch_btn = QPushButton("⇄ Direct Binding")
        self._status_lbl = QLabel("Ready")

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

        box_c = QGroupBox("Chromatic → Dye")
        cc    = QVBoxLayout(box_c)
        self._multi_chrom_chk = QCheckBox("Multi-chromatic plates")
        self._multi_chrom_chk.setChecked(False)
        self._multi_chrom_chk.setToolTip(
            "Tick this if any plate contains more than one chromatic block.\n"
            "When unticked, only the first chromatic block of each plate is\n"
            "loaded — correct for single-dye experiments and avoids mixing\n"
            "fluorescence channels on multi-block plates.\n\n"
            "When ticked, specify the dye for each chromatic position below.\n"
            "Use the 'Plate (substring)' column to assign different dyes to\n"
            "specific plates; leave it empty for a global mapping.")
        self._chrom_map = ChromaticMappingWidget()
        self._chrom_map.setVisible(False)
        self._multi_chrom_chk.toggled.connect(self._chrom_map.setVisible)
        cc.addWidget(self._multi_chrom_chk)
        cc.addWidget(self._chrom_map)
        layout.addWidget(box_c)

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
        self._morrison_spin.setRange(0.01, 100.0)
        self._morrison_spin.setSingleStep(0.1)
        self._morrison_spin.setValue(pl_ki.MORRISON_GATE_DEFAULT)
        self._morrison_spin.setDecimals(2)
        self._morrison_spin.setToolTip(
            "Morrison tight-binding model is tried when:\n"
            "  Ki_standard < gate × [Host]\n\n"
            "Morrison corrects for depletion of free inhibitor\n"
            "when a significant fraction is bound (Ki ≈ [Host]).\n\n"
            "Default 1.0 = only when Ki ≤ [Host] (true tight binding).\n"
            "Raise to e.g. 2–5 to be more permissive;\n"
            "lower to 0.1 to restrict to very tight binders only.")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Morrison gate:", self._morrison_spin)
        layout.addWidget(box_p)

        box_col = QGroupBox("Plot Colors")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", pl_ki.PLOT_COLOR_DATA)
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", pl_ki.PLOT_COLOR_FIT)
        self._color_resid_row = ColorPickerRow("Residuals:  ", pl_ki.PLOT_COLOR_DATA)
        self._color_data_row.color_changed.connect(self._on_color_changed)
        self._color_fit_row.color_changed.connect(self._on_color_changed)
        self._color_resid_row.color_changed.connect(self._on_color_changed)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        layout.addWidget(box_col)

        box_tf = QGroupBox("Title Font")
        tf_lay = QFormLayout(box_tf)
        self._title_fam = QComboBox()
        self._title_fam.addItems([
            "sans-serif", "serif", "monospace",
            "DejaVu Sans", "Arial", "Helvetica"])
        self._title_fsz = QDoubleSpinBox()
        self._title_fsz.setRange(5, 18)
        self._title_fsz.setValue(8)
        self._title_fsz.setSingleStep(0.5)
        self._title_fsz.setDecimals(1)
        self._title_style = QComboBox()
        self._title_style.addItems(["Bold", "Normal", "Italic", "Bold Italic"])
        self._title_fam.currentIndexChanged.connect(self._on_title_font_changed)
        self._title_fsz.valueChanged.connect(self._on_title_font_changed)
        self._title_style.currentIndexChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        tf_lay.addRow("Style:",  self._title_style)
        layout.addWidget(box_tf)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(False)
        self._resid_chk.setToolTip("Show residual sub-panel beneath each binding curve")
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On Save: write one PDF per curve into\n"
            "results/plots/individual/PASS|FAIL/")
        self._summary_chk = QCheckBox("Show summary tab")
        self._summary_chk.setChecked(True)
        self._summary_chk.setToolTip("pKi bar-chart / heatmap summary tab")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

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
        self._tabs.addTab(self._plots_tab, "Plots")             # 6

        self._summary_tab = SummaryTab(mode="ki")
        self._tabs.addTab(self._summary_tab, "Summary")         # 7

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

    @staticmethod
    def _parse_title_style(style_text: str) -> tuple[str, str]:
        """Return (fontweight, fontstyle) from a combo label."""
        return {
            "Bold":        ("bold",   "normal"),
            "Normal":      ("normal", "normal"),
            "Italic":      ("normal", "italic"),
            "Bold Italic": ("bold",   "italic"),
        }.get(style_text, ("bold", "normal"))

    def _get_config(self) -> dict:
        fw, fs = self._parse_title_style(self._title_style.currentText())
        base = self._input_row.path
        return {
            "dye_folder":         os.path.join(base, "dye_map"),
            "host_folder":        os.path.join(base, "host_map"),
            "guest_folder":       os.path.join(base, "guest_map"),
            "blank_folder":       os.path.join(base, "blank_map"),
            "raw_folder":         os.path.join(base, "Raw"),
            "kd_folder":          self._kd_row.path,
            "output_folder":      self._output_row.path,
            "chromatic_to_dye":   (self._chrom_map.mapping
                                      if self._multi_chrom_chk.isChecked() else {}),
            "r2_threshold":       self._r2_spin.value(),
            "grubbs_alpha":       self._grubbs_spin.value(),
            "morrison_gate":      self._morrison_spin.value(),
            "color_data":         self._color_data_row.color,
            "color_fit":          self._color_fit_row.color,
            "color_resid":        self._color_resid_row.color,
            "show_residuals":     self._resid_chk.isChecked(),
            "export_individual":  self._ind_export_chk.isChecked(),
            "title_fontsize":     self._title_fsz.value(),
            "title_fontfamily":   self._title_fam.currentText(),
            "title_fontweight":   fw,
            "title_fontstyle":    fs,
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_ki.PASS_R2_DEFAULT_KI)
        self._grubbs_spin.setValue(0.05)
        self._morrison_spin.setValue(pl_ki.MORRISON_GATE_DEFAULT)
        self._color_data_row.set_color(pl_ki.PLOT_COLOR_DATA)
        self._color_fit_row.set_color(pl_ki.PLOT_COLOR_FIT)
        self._color_resid_row.set_color(pl_ki.PLOT_COLOR_DATA)
        self._multi_chrom_chk.setChecked(False)
        self._resid_chk.setChecked(False)
        self._ind_export_chk.setChecked(False)
        self._summary_chk.setChecked(True)
        self._title_fsz.setValue(8)
        self._title_fam.setCurrentText("sans-serif")
        self._title_style.setCurrentText("Bold")
        if self._state.fit_results:
            self._refresh_results()

    def _on_color_changed(self, _hex):
        if self._state.fit_results:
            self._refresh_results()

    def _on_display_changed(self, _state):
        if self._state.fit_results:
            self._refresh_results()

    def _on_title_font_changed(self, _=None):
        if self._state.fit_results:
            self._refresh_results()

    def _on_summary_toggle(self, state):
        self._tabs.setTabVisible(7, bool(state))
        if state and self._state.df_results is not None:
            self._summary_tab.load(self._state.df_results)

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
        prereqs = {2: ("merged_mapping", "fluorescence"), 3: ("merged",), 4: ("fi_df",)}
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
            # subtract_background_ki returns {"fi_df": df}
            self._state.fi_df = result["fi_df"]
            self._fi_model.set_dataframe(result["fi_df"])
            self._tabs.setTabEnabled(3, True)
            self._tabs.setTabEnabled(4, True)
            self._tabs.setCurrentIndex(3)
            # Generate QC figures in the main thread (seaborn is not thread-safe)
            try:
                qc = pl_ki.make_qc_figures_ki(result["fi_df"])
            except Exception:
                import traceback as _tb
                self._log.append(f"\n[QC warning]\n{_tb.format_exc()}")
                qc = []
            self._state.qc_figures = qc
            self._qc_tab.load(qc)

        elif self._stage == 4:
            fit_results, plot_data = result
            self._state.fit_results = fit_results
            self._state.plot_data   = plot_data
            self._tabs.setTabEnabled(5, True)
            self._tabs.setTabEnabled(6, True)
            if self._summary_chk.isChecked():
                self._tabs.setTabEnabled(7, True)
            self._tabs.setCurrentIndex(5)
            self._save_btn.setEnabled(True)
            try:
                self._refresh_results()
            except Exception:
                import traceback as _tb
                self._log.append(f"\n[ERROR in refresh]\n{_tb.format_exc()}")

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
        self._plots_tab.load(pass_items, fail_items,
                             color_fit=cfg["color_fit"], color_data=cfg["color_data"],
                             color_resid=cfg["color_resid"],
                             show_residuals=cfg["show_residuals"],
                             title_fontsize=cfg["title_fontsize"],
                             title_fontfamily=cfg["title_fontfamily"],
                             title_fontweight=cfg["title_fontweight"],
                             title_fontstyle=cfg["title_fontstyle"])
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)

    def _on_save(self):
        cfg       = self._get_config()
        file_list = pl_ki.preview_save_files_ki(self._state, cfg["output_folder"],
                                                export_individual=cfg["export_individual"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        color_data  = cfg["color_data"]
        color_fit   = cfg["color_fit"]
        color_resid = cfg["color_resid"]
        show_resid  = cfg["show_residuals"]
        export_ind  = cfg["export_individual"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_ki.save_outputs_ki(self._state, cfg["output_folder"],
                                         cfg["r2_threshold"], progress_cb,
                                         plot_color_fit=color_fit,
                                         plot_color_data=color_data,
                                         plot_color_resid=color_resid,
                                         show_residuals=show_resid,
                                         export_individual=export_ind,
                                         title_fontsize=cfg["title_fontsize"],
                                         title_fontfamily=cfg["title_fontfamily"],
                                         title_fontweight=cfg["title_fontweight"],
                                         title_fontstyle=cfg["title_fontstyle"])
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
            win = DirectMainWindow()
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
        win = DirectMainWindow()
    elif picker.choice == "competitive":
        win = CompMainWindow()
    else:
        win = HeatmapWindow()
    app._extra_windows = [win]
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
