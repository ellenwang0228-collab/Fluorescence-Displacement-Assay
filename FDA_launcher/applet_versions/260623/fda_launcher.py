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
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
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

pl_fda  = _load_local("pipeline_fda")
pl_ki   = _load_local("pipeline_ki")
pl_spec = _load_local("pipeline_spectral")
PipelineState         = pl_fda.PipelineState
PipelineStateKi       = pl_ki.PipelineStateKi
PipelineStateSpectral = pl_spec.PipelineStateSpectral

_BASE_DIR = Path(__file__).parent

# Height ratios for residual panels: [curve, residual]
_RESID_RATIO = [3.5, 1]
_RESID_HSPACE = 0.06

# Colours for Compare Runs (up to 10 runs)
_RUN_COLORS = ["#6495ED", "#8B0000", "#2ca02c", "#ff7f0e",
               "#9467bd", "#8c564b", "#e377c2", "#17becf",
               "#bcbd22", "#7f7f7f"]


# ── Shared helpers ─────────────────────────────────────────────────────────────

# ── Compare Runs — parsing & curve-reconstruction helpers ─────────────────────

def _detect_compare_mode(path: str):
    """Return 'ki', 'kd', or None (unrecognised) by inspecting file columns."""
    import pandas as _pd
    try:
        xl = _pd.ExcelFile(path)
    except Exception:
        return None
    for sheet in ("PASS", "FAIL"):
        if sheet not in xl.sheet_names:
            continue
        try:
            cols = set(xl.parse(sheet, nrows=0).columns)
        except Exception:
            continue
        if "Guest" in cols and "Ki_uM" in cols:
            return "ki"
        if "Kd" in cols:
            return "kd"
    return None


def _find_companion_data(fit_path: str):
    """Return path to the matching *_data.xlsx beside fit_results, or None."""
    import re as _re
    bn   = os.path.basename(fit_path)
    dire = os.path.dirname(fit_path)
    candidates = []
    if "competitive_fit_results" in bn:
        candidates.append(bn.replace("competitive_fit_results.xlsx",
                                     "competitive_data.xlsx"))
    if "binding_fit_results" in bn:
        candidates.append(bn.replace("binding_fit_results.xlsx",
                                     "comp_binding_data.xlsx"))
        candidates.append(bn.replace("binding_fit_results.xlsx", "data.xlsx"))
    stem = os.path.splitext(bn)[0]
    candidates += [stem + "_competitive_data.xlsx",
                   stem + "_comp_binding_data.xlsx",
                   stem + "_data.xlsx"]
    m = _re.match(r'^(\d{6}_\d{6})_', bn)
    if m:
        pfx = m.group(1)
        candidates += [pfx + "_competitive_data.xlsx",
                       pfx + "_comp_binding_data.xlsx",
                       pfx + "_data.xlsx"]
    for comp in candidates:
        full = os.path.join(dire, comp)
        if os.path.exists(full):
            return full
    return None


def _parse_fit_results(path: str, mode: str) -> dict:
    """Return {key_tuple: row_dict} from a fit_results xlsx (PASS + FAIL combined).

    Ki key: (host, dye, guest)
    Kd key: (host, dye, dye_conc_str)   e.g. ('PMP5-Na', 'DAPI', '1')
    """
    import pandas as _pd
    result: dict = {}
    try:
        xl = _pd.ExcelFile(path)
    except Exception:
        return result
    for sheet in ("PASS", "FAIL"):
        if sheet not in xl.sheet_names:
            continue
        try:
            df = xl.parse(sheet)
        except Exception:
            continue
        if df.empty:
            continue
        df["_status"] = sheet
        for _, row in df.iterrows():
            r = row.to_dict()
            if mode == "ki":
                key = (str(r.get("Host", "")).strip(),
                       str(r.get("Dye", "")).strip(),
                       str(r.get("Guest", "")).strip().title())
            else:
                dc = r.get("Dye_Concentration", float("nan"))
                try:
                    dc_str = f"{float(dc):.4g}"
                except Exception:
                    dc_str = str(dc)
                key = (str(r.get("Host", "")).strip(),
                       str(r.get("Dye", "")).strip(), dc_str)
            if key in result and result[key].get("_status") == "PASS" and sheet == "FAIL":
                continue
            result[key] = r
    return result


def _parse_run_data(path: str) -> dict:
    """Parse a competitive_data.xlsx file produced by the Ki pipeline.

    Returns {"PASS": {(host,dye,guest): {"concs": arr, "reps": [arr,...]}},
             "FAIL":  {...}}.
    Each group of 4 columns encodes one (host|dye|guest) curve: the first
    column holds guest concentrations, the next three hold FI-F0 triplicates.
    """
    import re as _re
    import pandas as _pd
    result: dict = {}
    try:
        xl = _pd.ExcelFile(path)
    except Exception:
        return result
    for sheet in ("PASS", "FAIL"):
        if sheet not in xl.sheet_names:
            continue
        df = xl.parse(sheet)
        entries: dict = {}
        cols = list(df.columns)
        i = 0
        while i < len(cols):
            raw = str(cols[i])
            clean = _re.sub(r'\.\d+$', '', raw).strip()
            if "|" in clean:
                parts = [p.strip() for p in clean.split("|")]
                if len(parts) == 3:
                    host, dye, guest = parts[0], parts[1], parts[2].title()
                    conc_series = df.iloc[:, i].dropna()
                    n = len(conc_series)
                    concs = conc_series.values.astype(float)
                    reps = []
                    for j in range(1, 4):
                        if i + j < len(cols):
                            reps.append(df.iloc[:n, i + j].values.astype(float))
                    entries[(host, dye, guest)] = {"concs": concs, "reps": reps}
                    i += 4
                    continue
            i += 1
        result[sheet] = entries
    return result


def _parse_raw_ki(path: str) -> dict:
    """competitive_data.xlsx → {(host,dye,guest): {'x_log': arr, 'reps': [arr,…]}}"""
    raw = _parse_run_data(path)
    result: dict = {}
    for sheet_data in raw.values():
        for (host, dye, guest), entry in sheet_data.items():
            concs = entry["concs"]
            pos   = concs > 0
            result[(host, dye, guest)] = {
                "x_log": np.log10(concs[pos]),
                "reps":  [r[pos] for r in entry["reps"]],
            }
    return result


def _parse_raw_kd(path: str) -> dict:
    """binding _data.xlsx → {(host,dye,dye_conc_str): {'x': arr, 'reps': [arr,…]}}"""
    import re as _re
    import pandas as _pd
    col_re = _re.compile(r'^(.+?)\s*\|\s*(.+?)\s*\[(.+?)\]\s*µM')
    result: dict = {}
    try:
        xl = _pd.ExcelFile(path)
    except Exception:
        return result
    for sheet in ("PASS", "FAIL"):
        if sheet not in xl.sheet_names:
            continue
        df   = xl.parse(sheet)
        cols = list(df.columns)
        i    = 0
        while i < len(cols):
            clean = _re.sub(r'\.\d+$', '', str(cols[i])).strip()
            m = col_re.match(clean)
            if m:
                host, dye = m.group(1).strip(), m.group(2).strip()
                dc = m.group(3).strip()
                try:
                    dc_str = f"{float(dc):.4g}"
                except Exception:
                    dc_str = dc
                cs = df.iloc[:, i].dropna()
                n  = len(cs)
                reps = []
                for j in range(1, 4):
                    if i + j < len(cols):
                        reps.append(df.iloc[:n, i + j].values.astype(float))
                result[(host, dye, dc_str)] = {"x": cs.values.astype(float), "reps": reps}
                i += 4
                continue
            i += 1
    return result


def _reconstruct_ki_curve(fit_row: dict, x_lo: float, x_hi: float):
    """Return (x_dense, y_norm, warn_str) for a Ki fit row, or (None, None, msg)."""
    def _f(k):
        v = fit_row.get(k)
        try:
            return float(v)
        except Exception:
            return np.nan

    DyeConc  = _f("DyeConc_uM")
    DyeKd    = _f("DyeKd_uM")
    HostConc = _f("HostConc_uM")
    Top      = _f("Top_fit")
    Bottom   = _f("Bottom_fit")
    logKi    = _f("logKi")
    model    = str(fit_row.get("Best_Model", "Standard"))

    if any(np.isnan(v) for v in [DyeConc, DyeKd, Top, Bottom, logKi]):
        return None, None, "missing parameters"
    span = Top - Bottom
    if abs(span) < 1e-10:
        return None, None, "Top ≈ Bottom"

    x    = np.linspace(x_lo - 0.5, x_hi + 0.5, 300)
    warn = ""
    try:
        if model == "Standard":
            fn, popt = pl_ki._comp_standard(DyeConc, DyeKd), [Top, Bottom, logKi]
        elif model == "HillSlope":
            hs = _f("HillSlope")
            if np.isnan(hs):
                warn = "HillSlope not saved — Standard curve shown"
                fn, popt = pl_ki._comp_standard(DyeConc, DyeKd), [Top, Bottom, logKi]
            else:
                fn, popt = pl_ki._comp_hill(DyeConc, DyeKd), [Top, Bottom, logKi, hs]
        elif model == "Morrison":
            if np.isnan(HostConc):
                return None, None, "HostConc missing"
            fn, popt = pl_ki._morrison(DyeConc, DyeKd, HostConc), [Top, Bottom, logKi]
        elif model == "Biphasic":
            lK2  = _f("logKi2")
            frac = _f("Frac_site1")
            if np.isnan(lK2) or np.isnan(frac):
                return None, None, "Biphasic params missing"
            fn   = pl_ki._biphasic(DyeConc, DyeKd)
            popt = [Top, Bottom, logKi, lK2, frac]
        else:
            fn, popt = pl_ki._comp_standard(DyeConc, DyeKd), [Top, Bottom, logKi]

        y_norm = (fn(x, *popt) - Bottom) / span
        return x, y_norm, warn
    except Exception as exc:
        return None, None, str(exc)


def _reconstruct_kd_curve(fit_row: dict, x_max: float):
    """Return (x_dense, y_norm, '') for a Kd fit row, or (None, None, msg)."""
    def _f(k):
        v = fit_row.get(k)
        try:
            return float(v)
        except Exception:
            return np.nan

    Bmax    = _f("Bmax")
    Kd      = _f("Kd")
    D_fixed = _f("Dye_Concentration")
    model   = str(fit_row.get("Model", "one_site"))

    if np.isnan(Bmax) or np.isnan(Kd) or abs(Bmax) < 1e-10:
        return None, None, "missing/zero Bmax"

    x = np.linspace(0.0, x_max * 1.5, 300)
    try:
        if model == "quadratic" and not np.isnan(D_fixed):
            y = pl_fda._quadratic_binding(x, Bmax, Kd, D_fixed)
        else:
            y = pl_fda._one_site(x, Bmax, Kd)
        return x, y / Bmax, ""
    except Exception as exc:
        return None, None, str(exc)


def _render_compare_combo(ax_curve, ax_dots, combo: tuple, runs: list, mode: str,
                          show_legend: bool = False, flip_norm: bool = False,
                          pooled_mode: bool = False, ax_resid=None,
                          title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                          title_fontweight: str = "bold", title_fontstyle: str = "normal",
                          color_data: str = "#1e4572", color_fit: str = "#6495ED",
                          color_resid: str = None):
    """Render one combo's normalised curve panel and optional Ki/Kd dot panel.

    ax_dots may be None when the Ki/Kd panel is hidden.
    ax_resid may be None when residuals are hidden.
    flip_norm inverts the y-axis (0↔1) for curves whose natural direction is reversed.
    pooled_mode pools normalised raw data from all runs and fits a single curve.
    """
    from scipy.optimize import curve_fit as _curve_fit

    ki_vals, ki_errs, dot_labels, dot_colors = [], [], [], []
    legend_lines, legend_labels_l = [], []
    _resid_col = color_resid or color_data

    # Collect x range from raw data across all runs
    all_x: list = []
    for run in runs:
        rd = run["raw"].get(combo)
        if rd:
            xv = rd.get("x_log") if mode == "ki" else rd.get("x")
            if xv is not None and len(xv):
                all_x.extend(xv.tolist())
    if all_x:
        x_lo = min(all_x)
        x_hi = max(all_x)
    elif mode == "ki":
        logki_vals = []
        for run in runs:
            fr = run["fit"].get(combo)
            if fr:
                try: logki_vals.append(float(fr["logKi"]))
                except (TypeError, ValueError, KeyError): pass
        logki_vals = [v for v in logki_vals if not np.isnan(v)]
        mid = np.mean(logki_vals) if logki_vals else 0.0
        x_lo, x_hi = mid - 2.0, mid + 2.0
    else:
        kd_vals_fb = []
        for run in runs:
            fr = run["fit"].get(combo)
            if fr:
                try: kd_vals_fb.append(float(fr["Kd"]))
                except (TypeError, ValueError, KeyError): pass
        kd_vals_fb = [v for v in kd_vals_fb if not np.isnan(v) and v > 0]
        x_lo = 0.0
        x_hi = max(kd_vals_fb) * 5.0 if kd_vals_fb else 2.0

    # ── collect normalised data for all runs ──
    pooled_x, pooled_y = [], []
    for run in runs:
        fit_row = run["fit"].get(combo)
        if fit_row is None:
            continue
        color = run["color"]
        label = run["label"]

        rd = run["raw"].get(combo)
        if rd is not None:
            try:
                if mode == "ki":
                    top    = float(fit_row.get("Top_fit",    np.nan))
                    bottom = float(fit_row.get("Bottom_fit", np.nan))
                    span   = top - bottom
                    if not np.isnan(span) and abs(span) > 1e-10:
                        xv = rd["x_log"]
                        norm_reps = []
                        for rep in rd["reps"]:
                            if len(rep) == len(xv):
                                yn = (rep.astype(float) - bottom) / span
                                if flip_norm: yn = 1.0 - yn
                                norm_reps.append(yn)
                        if norm_reps:
                            if pooled_mode:
                                run_mean = np.mean(norm_reps, axis=0)
                                run_sem  = (np.std(norm_reps, axis=0, ddof=1)
                                            / np.sqrt(len(norm_reps))
                                            if len(norm_reps) > 1
                                            else np.zeros_like(run_mean))
                                ax_curve.errorbar(xv, run_mean, yerr=run_sem,
                                                  fmt="o", color=color,
                                                  ecolor=color, elinewidth=1,
                                                  markersize=4, capsize=2,
                                                  alpha=0.7, zorder=2)
                                pooled_x.extend(xv.tolist())
                                pooled_y.extend(run_mean.tolist())
                            else:
                                for yn in norm_reps:
                                    ax_curve.scatter(xv, yn, color=color,
                                                     alpha=0.22, s=7,
                                                     linewidths=0, zorder=2)
                else:
                    bmax = float(fit_row.get("Bmax", np.nan))
                    if not np.isnan(bmax) and abs(bmax) > 1e-10:
                        xv = rd["x"]
                        norm_reps = []
                        for rep in rd["reps"]:
                            if len(rep) == len(xv):
                                yn = rep.astype(float) / bmax
                                if flip_norm: yn = 1.0 - yn
                                norm_reps.append(yn)
                        if norm_reps:
                            if pooled_mode:
                                run_mean = np.mean(norm_reps, axis=0)
                                run_sem  = (np.std(norm_reps, axis=0, ddof=1)
                                            / np.sqrt(len(norm_reps))
                                            if len(norm_reps) > 1
                                            else np.zeros_like(run_mean))
                                ax_curve.errorbar(xv, run_mean, yerr=run_sem,
                                                  fmt="o", color=color,
                                                  ecolor=color, elinewidth=1,
                                                  markersize=4, capsize=2,
                                                  alpha=0.7, zorder=2)
                                pooled_x.extend(xv.tolist())
                                pooled_y.extend(run_mean.tolist())
                            else:
                                for yn in norm_reps:
                                    ax_curve.scatter(xv, yn, color=color,
                                                     alpha=0.22, s=7,
                                                     linewidths=0, zorder=2)
            except Exception:
                pass

        # ── per-run reconstructed fit curve ──
        if not pooled_mode:
            try:
                if mode == "ki":
                    xc, yc, warn = _reconstruct_ki_curve(fit_row, x_lo, x_hi)
                else:
                    xc, yc, warn = _reconstruct_kd_curve(fit_row, x_hi)
            except Exception as exc:
                xc, yc, warn = None, None, str(exc)

            if xc is not None:
                if flip_norm: yc = 1.0 - yc
                lbl  = label + (" †" if warn else "")
                line, = ax_curve.plot(xc, yc, color=color, lw=2, zorder=3)
                if lbl not in legend_labels_l:
                    legend_lines.append(line)
                    legend_labels_l.append(lbl)
                if warn:
                    existing = ax_curve.texts
                    y_off = 0.04 + 0.09 * len([t for t in existing if "†" in t.get_text()])
                    ax_curve.text(0.02, y_off, f"† {warn}",
                                  transform=ax_curve.transAxes,
                                  fontsize=4, color=color, va="bottom", clip_on=True)

                # per-run residuals
                if ax_resid is not None and rd is not None:
                    try:
                        xk = "x_log" if mode == "ki" else "x"
                        xv = rd[xk]
                        for rep in rd["reps"]:
                            if len(rep) == len(xv):
                                if mode == "ki":
                                    top_    = float(fit_row.get("Top_fit", np.nan))
                                    bot_    = float(fit_row.get("Bottom_fit", np.nan))
                                    sp_     = top_ - bot_
                                    yn_rep  = (rep.astype(float) - bot_) / sp_
                                else:
                                    yn_rep = rep.astype(float) / float(fit_row.get("Bmax", 1.0))
                                if flip_norm: yn_rep = 1.0 - yn_rep
                                y_pred = np.interp(xv, xc, yc)
                                resid  = yn_rep - y_pred
                                ax_resid.scatter(xv, resid, color=color,
                                                 alpha=0.3, s=5, linewidths=0, zorder=2)
                    except Exception:
                        pass

        # ── Ki/Kd value ──
        try:
            if mode == "ki":
                ki  = float(fit_row.get("Ki_uM",    np.nan))
                err = float(fit_row.get("Ki_err_uM", np.nan))
            else:
                ki  = float(fit_row.get("Kd",    np.nan))
                err = float(fit_row.get("Kd_SE", np.nan))
        except Exception:
            ki, err = np.nan, np.nan

        if not np.isnan(ki):
            ki_vals.append(ki)
            ki_errs.append(err if not np.isnan(err) else None)
            dot_labels.append(label)
            dot_colors.append(color)

    # ── pooled-mode: fit a single curve to run-level means ──
    if pooled_mode and len(pooled_x) >= 4:
        px = np.asarray(pooled_x)
        py = np.asarray(pooled_y)
        mask = np.isfinite(px) & np.isfinite(py)
        px, py = px[mask], py[mask]
        try:
            if mode == "ki":
                def _sig(x, top, bot, logIC50):
                    return bot + (top - bot) / (1.0 + 10.0 ** (x - logIC50))
                p0 = [max(py), min(py), np.median(px)]
                popt, pcov = _curve_fit(_sig, px, py, p0=p0, maxfev=5000)
                xd = np.linspace(x_lo - 0.5, x_hi + 0.5, 300)
                yd = _sig(xd, *popt)
            else:
                def _bind(x, top, kd):
                    return top * x / (kd + x)
                p0 = [max(py), np.median(px)]
                popt, pcov = _curve_fit(_bind, px, py, p0=p0, maxfev=5000)
                xd = np.linspace(0, x_hi * 1.5, 300)
                yd = _bind(xd, *popt)
            ax_curve.plot(xd, yd, color=color_fit, lw=2.0, zorder=3)
            if ax_resid is not None:
                y_pred = _sig(px, *popt) if mode == "ki" else _bind(px, *popt)
                resid = py - y_pred
                ax_resid.scatter(px, resid, color=_resid_col, alpha=0.5, s=8,
                                 linewidths=0, zorder=2)
        except Exception:
            ax_curve.text(0.5, 0.5, "pooled fit failed",
                          transform=ax_curve.transAxes, ha="center", va="center",
                          fontsize=6, color="red")
    elif pooled_mode and len(pooled_x) < 4:
        ax_curve.text(0.5, 0.5, "not enough raw data\nfor pooled fit",
                      transform=ax_curve.transAxes, ha="center", va="center",
                      fontsize=6, color="gray")

    # ── residuals cosmetics ──
    if ax_resid is not None:
        ax_resid.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5, zorder=1)
        ax_resid.tick_params(labelsize=6)
        ax_resid.set_ylabel("Resid.", fontsize=7)
        ax_resid.set_xlabel(
            "log[Guest] (µM)" if mode == "ki" else "[Host] (µM)", fontsize=7)

    # ── curve panel cosmetics ──
    ax_curve.set_ylim(-0.18, 1.28)
    ax_curve.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)
    ax_curve.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)
    ax_curve.tick_params(labelsize=6)
    ax_curve.set_ylabel("Norm. response", fontsize=7)
    if ax_resid is None:
        ax_curve.set_xlabel(
            "log[Guest] (µM)" if mode == "ki" else "[Host] (µM)", fontsize=7)
    else:
        ax_curve.tick_params(labelbottom=False)

    # ── compute title stats ──
    _mkey = "Best_Model" if mode == "ki" else "Model"
    _models_seen = [str(run["fit"][combo].get(_mkey, "?"))
                    for run in runs if combo in run["fit"]]
    _unique_models = sorted(set(_models_seen))
    _model_mismatch = len(_unique_models) > 1
    model_str = ", ".join(_unique_models) if _unique_models else "?"

    # Title with stats (matching render_plot_ki format)
    tkw = dict(fontsize=title_fontsize, fontfamily=title_fontfamily,
               fontweight=title_fontweight, fontstyle=title_fontstyle)

    if mode == "ki":
        h, d, g = combo
        id_line = f"{h} | {d} | {g}"
        if ki_vals:
            mean_v = float(np.mean(ki_vals))
            sd_v = float(np.std(ki_vals, ddof=1)) if len(ki_vals) >= 2 else np.nan
            sd_str = f" ± {sd_v:.2f}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs if combo in run["fit"]]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Ki = {mean_v:.2f}{sd_str} µM (n={len(ki_vals)})  |  {r2_str}"
        else:
            stats_line = "no Ki data"
        ax_curve.set_title(f"{id_line}\n{stats_line}\n[{model_str}]", **tkw)
    else:
        h, d, dc = combo
        id_line = f"{h} – {d}  [{dc} µM]"
        if ki_vals:  # ki_vals holds Kd values in kd mode
            mean_v = float(np.mean(ki_vals))
            sd_v = float(np.std(ki_vals, ddof=1)) if len(ki_vals) >= 2 else np.nan
            sd_str = f" ± {sd_v:.2f}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs if combo in run["fit"]]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Kd = {mean_v:.2f}{sd_str} µM (n={len(ki_vals)})  |  {r2_str}"
        else:
            stats_line = "no Kd data"
        ax_curve.set_title(f"{id_line}\n{stats_line}  |  [{model_str}]", **tkw)

    if _model_mismatch:
        ax_curve.set_facecolor("#fff8e1")
        if ax_dots is not None:
            ax_dots.set_facecolor("#fff8e1")
        ax_curve.text(0.5, 0.98,
                      f"⚠ Mixed models: {', '.join(_unique_models)}",
                      transform=ax_curve.transAxes, fontsize=3.8,
                      color="#b85c00", ha="center", va="top",
                      bbox=dict(boxstyle="round,pad=0.15", fc="#fff8e1",
                                ec="#f0a030", lw=0.6))

    if show_legend and legend_lines:
        ax_curve.legend(legend_lines, legend_labels_l, fontsize=4.5,
                        loc="upper right", framealpha=0.7,
                        handlelength=1.2, borderpad=0.3)

    # ── Ki/Kd dot panel ──
    if ax_dots is None:
        return
    if ki_vals:
        xs = list(range(len(ki_vals)))
        for xi, (ki, err, col) in enumerate(zip(ki_vals, ki_errs, dot_colors)):
            if err is not None:
                ax_dots.errorbar(xi, ki, yerr=err, fmt="o", color=col,
                                 markersize=4, elinewidth=1.0, capsize=2.5, zorder=3)
            else:
                ax_dots.scatter([xi], [ki], color=col, s=20, zorder=3)

        if len(ki_vals) >= 2:
            mean_ki = float(np.nanmean(ki_vals))
            sd_ki   = float(np.nanstd(ki_vals, ddof=1))
            ax_dots.axhline(mean_ki, color="#555", lw=0.8, zorder=1)
            ax_dots.axhspan(mean_ki - sd_ki, mean_ki + sd_ki,
                            color="gray", alpha=0.12, zorder=0)
            valid = [v for v in ki_vals if v > 0]
            if valid and max(valid) / min(valid) > 5:
                ax_dots.set_yscale("log")

        ax_dots.set_xticks(xs)
        ax_dots.set_xticklabels(
            dot_labels if show_legend else [""] * len(dot_labels),
            fontsize=4.5, rotation=30, ha="right")
        ax_dots.set_xlim(-0.6, len(ki_vals) - 0.4)
        ax_dots.set_ylabel("Ki (µM)" if mode == "ki" else "Kd (µM)", fontsize=5.5)
        ax_dots.tick_params(axis="y", labelsize=5)
        ax_dots.tick_params(axis="x", length=2)
    else:
        ax_dots.text(0.5, 0.5, "no data", transform=ax_dots.transAxes,
                     ha="center", va="center", fontsize=6, color="gray")
        ax_dots.set_xticks([]); ax_dots.set_yticks([])


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
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        home_btn = QPushButton("⌂ Home")
        home_btn.clicked.connect(self._on_home)
        toolbar.addWidget(home_btn)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(290)
        panel = QWidget()
        scroll.setWidget(panel)
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
        self._grp_combo.addItems(["None (single colour)", "Host", "Dye", "Guest"])
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
        return scroll

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

        entries = []   # (label, pivot, attempted, xlabel, se_pivot)
        if kd_df is not None and not kd_df.empty:
            p, m = self._prep_pivot(kd_df, "Kd", "Host", "Dye")
            se_p = None
            if "Kd_SE" in kd_df.columns:
                good = kd_df[kd_df["Kd"].notna() & (kd_df["Kd"] > 0) & (kd_df["Kd"] < 1e6)]
                se_p = good.pivot_table(index="Host", columns="Dye",
                                        values="Kd_SE", aggfunc="mean")
            if not p.empty:
                if not sort_alpha: p = self._sort_pivot(p)
                m = m.reindex(index=p.index, columns=p.columns).fillna(False)
                if se_p is not None:
                    se_p = se_p.reindex(index=p.index, columns=p.columns)
                entries.append(("Direct Binding — Kd (Host–Dye)", p, m, "Dye", se_p))
        if ki_df is not None and not ki_df.empty:
            ki_work = ki_df.copy()
            if "Dye" in ki_work.columns:
                ki_work["Guest | Dye"] = ki_work["Guest"] + " | " + ki_work["Dye"]
                col_name = "Guest | Dye"
            else:
                col_name = "Guest"
            p, m = self._prep_pivot(ki_work, "Ki_uM", "Host", col_name)
            se_p = None
            if "Ki_err_uM" in ki_work.columns:
                good = ki_work[ki_work["Ki_uM"].notna() & (ki_work["Ki_uM"] > 0) & (ki_work["Ki_uM"] < 1e6)]
                se_p = good.pivot_table(index="Host", columns=col_name,
                                        values="Ki_err_uM", aggfunc="mean")
            if not p.empty:
                if not sort_alpha: p = self._sort_pivot(p)
                m = m.reindex(index=p.index, columns=p.columns).fillna(False)
                if se_p is not None:
                    se_p = se_p.reindex(index=p.index, columns=p.columns)
                entries.append(("Competition Binding — Ki (Host–Guest)", p, m, col_name, se_p))
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
            for _, p, _, _, _ in entries:
                lv = np.log10(p.replace(0, np.nan)).values.flatten()
                lv = lv[~np.isnan(lv)]
                if len(lv):
                    scales.append((float(np.floor(np.nanpercentile(lv, 2))),
                                   float(np.ceil(np.nanpercentile(lv, 98)))))
                else:
                    scales.append((0.0, 1.0))

        figs = []
        for (title, pivot, mask, xlabel, se_pivot), (vmin_, vmax_) in zip(entries, scales):
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
                            se_val = (se_pivot.iloc[i, j]
                                      if se_pivot is not None
                                      and i < se_pivot.shape[0]
                                      and j < se_pivot.shape[1]
                                      else np.nan)
                            if "Raw" in annot_mode:
                                txt = f"{val:.2g}"
                                if pd.notna(se_val) and se_val > 0:
                                    txt += f"\n±{se_val:.2g}"
                            elif "pK" in annot_mode:
                                pk = 6 - lv
                                txt = f"{pk:.2f}"
                                if pd.notna(se_val) and se_val > 0:
                                    pk_se = se_val / (val * np.log(10))
                                    txt += f"\n±{pk_se:.2f}"
                            else:
                                txt = f"{lv:.2f}"
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
            if "Dye" in df.columns:
                df["_lbl"] += " | " + df["Dye"]

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

    def _on_home(self):
        _go_home(self)


# ── Mode picker dialog ────────────────────────────────────────────────────────

class ModePicker(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhosphoMAX — Select Analysis Mode")
        self.setFixedSize(420, 400)
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

        btn_spectral = QPushButton("Spectral Scan  (Ex/Em + Binding)")
        btn_spectral.setFixedHeight(52)
        btn_spectral.setStyleSheet("font-size: 14px;")
        btn_spectral.setToolTip(
            "Load excitation/emission spectral-scan exports (.csv), view\n"
            "Ex/Em spectra per host concentration, and fit a Kd from FI-F0\n"
            "at a single chosen wavelength — same mapping/blank convention\n"
            "as Direct Binding.")
        btn_spectral.clicked.connect(lambda: self._pick("spectral"))

        btn_heat = QPushButton("Heatmap / Bar Chart  (from results)")
        btn_heat.setFixedHeight(52)
        btn_heat.setStyleSheet("font-size: 14px;")
        btn_heat.setToolTip(
            "Load pipeline output files (binding_fit_results.xlsx /\n"
            "competitive_fit_results.xlsx) or Kd_Tables_Gen format\n"
            "and export pKd / pKi heatmaps or bar charts as PDF.")
        btn_heat.clicked.connect(lambda: self._pick("heatmap"))

        btn_compare = QPushButton("Compare Runs  (reproducibility)")
        btn_compare.setFixedHeight(52)
        btn_compare.setStyleSheet("font-size: 14px;")
        btn_compare.setToolTip(
            "Load multiple competitive_data.xlsx output files and overlay\n"
            "FI-F0 curves for matching host / dye / guest combinations\n"
            "to assess reproducibility across independent runs.")
        btn_compare.clicked.connect(lambda: self._pick("compare"))

        layout.addWidget(btn_direct)
        layout.addWidget(btn_comp)
        layout.addWidget(btn_spectral)
        layout.addWidget(btn_heat)
        layout.addWidget(btn_compare)

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
                 color_resid: str = None, show_residuals: bool = False,
                 title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "bold", title_fontstyle: str = "normal",
                 layout_cfg: dict = None, dpi: int = 100, parent=None):
        self._items = items
        lc   = layout_cfg or {}
        n    = max(len(items), 1)
        cols = lc.get("pdf_cols", min(int(np.ceil(np.sqrt(n))), 4))
        rows = int(np.ceil(n / cols))
        cd   = color_data  or pl_fda.PLOT_COLOR
        cf   = color_fit   or pl_fda.PLOT_COLOR
        cr   = color_resid or cd
        tkw  = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                    title_fontweight=title_fontweight, title_fontstyle=title_fontstyle)
        fig_w   = lc.get("fig_w", cols * 4.0)
        col_w   = fig_w / cols

        if show_residuals:
            fig = Figure(figsize=(fig_w, rows * max(2.5, col_w * 1.35)), dpi=dpi)
            outer = GridSpec(rows, cols, figure=fig,
                             hspace=lc.get("hspace", 0.45),
                             wspace=lc.get("wspace", 0.40))
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ri, ci = divmod(i, cols)
                igs = outer[ri, ci].subgridspec(
                    2, 1, height_ratios=_RESID_RATIO,
                    hspace=lc.get("resid_gap", _RESID_HSPACE))
                ax   = fig.add_subplot(igs[0])
                ax_r = fig.add_subplot(igs[1], sharex=ax)
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                                   color_resid=cr, ax_resid=ax_r, **tkw)
                self._axes.append(ax)
        else:
            fig = Figure(figsize=(fig_w, rows * max(2.0, col_w * 0.95)), dpi=dpi)
            outer = GridSpec(rows, cols, figure=fig,
                             hspace=lc.get("hspace", 0.45),
                             wspace=lc.get("wspace", 0.40))
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ri, ci = divmod(i, cols)
                ax = fig.add_subplot(outer[ri, ci])
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                                   color_resid=cr, **tkw)
                self._axes.append(ax)

        fig.subplots_adjust(left=lc.get("left", 0.08), right=lc.get("right", 0.97),
                            top=lc.get("top", 0.96), bottom=lc.get("bottom", 0.08))
        dpi_v = fig.dpi
        self.setMinimumSize(int(fig.get_figwidth() * dpi_v),
                            int(fig.get_figheight() * dpi_v))
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
        self._fig = Figure(figsize=(5.5, 4.0))
        super().__init__(self._fig)

    def show_entry(self, entry, color_data: str = None, color_fit: str = None,
                   color_resid: str = None, show_residuals: bool = False,
                   title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "bold", title_fontstyle: str = "normal",
                   layout_cfg: dict = None):
        lc  = layout_cfg or {}
        fw  = lc.get("fig_w", 5.5)
        if show_residuals:
            fh = fw * (sum(_RESID_RATIO) / _RESID_RATIO[0]) * 0.75
        else:
            fh = fw * 0.75
        self._fig.set_size_inches(fw, fh)
        self._fig.clear()
        cd = color_data  or pl_fda.PLOT_COLOR
        cf = color_fit   or pl_fda.PLOT_COLOR
        cr = color_resid or cd
        tkw = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                   title_fontweight=title_fontweight, title_fontstyle=title_fontstyle)
        if show_residuals:
            gs   = GridSpec(2, 1, figure=self._fig,
                            height_ratios=_RESID_RATIO,
                            hspace=lc.get("resid_gap", _RESID_HSPACE))
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
            if ax_r: ax.tick_params(labelbottom=False)
        else:
            ax   = self._fig.add_subplot(111)
            ax_r = None
        self._fig.subplots_adjust(
            left=lc.get("left", 0.12), right=lc.get("right", 0.97),
            top=lc.get("top", 0.90), bottom=lc.get("bottom", 0.12))
        pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                           color_resid=cr, ax_resid=ax_r, **tkw)
        dpi_v = self._fig.dpi
        self.setMinimumSize(int(fw * dpi_v), int(fh * dpi_v))
        self.draw()


class DirectPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pass_items       = []
        self._fail_items       = []
        self._current_items    = []
        self._current_idx      = 0
        self._color_data       = pl_fda.PLOT_COLOR
        self._color_fit        = pl_fda.PLOT_COLOR
        self._color_resid      = pl_fda.PLOT_COLOR
        self._show_residuals   = False
        self._layout_cfg       = {}
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
        self._plate_combo.setMinimumWidth(100)
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

        # setWidgetResizable(False) prevents the scroll area from squeezing the
        # canvas smaller than its natural size — which would cause axes to overlap.
        self._grid_scroll_pass = QScrollArea()
        self._grid_scroll_pass.setWidgetResizable(False)
        self._grid_scroll_fail = QScrollArea()
        self._grid_scroll_fail.setWidgetResizable(False)
        self._single_pass = SinglePlotCanvas()
        self._single_fail = SinglePlotCanvas()
        self._single_scroll_pass = QScrollArea()
        self._single_scroll_pass.setWidgetResizable(False)
        self._single_scroll_pass.setWidget(self._single_pass)
        self._single_scroll_fail = QScrollArea()
        self._single_scroll_fail.setWidgetResizable(False)
        self._single_scroll_fail.setWidget(self._single_fail)

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
             color_data: str = None, color_fit: str = None,
             color_resid: str = None, show_residuals: bool = None,
             title_fontsize: float = None, title_fontfamily: str = None,
             title_fontweight: str = None, title_fontstyle: str = None,
             layout_cfg: dict = None):
        self._pass_items = pass_items
        self._fail_items = fail_items
        if color_data       is not None: self._color_data       = color_data
        if color_fit        is not None: self._color_fit        = color_fit
        if color_resid      is not None: self._color_resid      = color_resid
        if show_residuals   is not None: self._show_residuals   = show_residuals
        if title_fontsize   is not None: self._title_fontsize   = title_fontsize
        if title_fontfamily is not None: self._title_fontfamily = title_fontfamily
        if title_fontweight is not None: self._title_fontweight = title_fontweight
        if title_fontstyle  is not None: self._title_fontstyle  = title_fontstyle
        if layout_cfg       is not None: self._layout_cfg       = layout_cfg

        all_items  = pass_items + fail_items
        all_hosts  = sorted(set(e.get("host", "") for e in all_items if e.get("host", "")))
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

    def _tkw(self):
        return dict(title_fontsize=self._title_fontsize,
                    title_fontfamily=self._title_fontfamily,
                    title_fontweight=self._title_fontweight,
                    title_fontstyle=self._title_fontstyle)

    def _build_grids(self):
        host  = self._current_host()
        plate = self._current_plate()
        pass_items = self._items_for_filters(self._pass_items, host, plate)
        fail_items = self._items_for_filters(self._fail_items, host, plate)
        tkw = self._tkw()
        no_pass = "  No PASS results for this selection."
        no_fail = "  No FAIL results for this selection."
        lc = self._layout_cfg or None
        if pass_items:
            canvas = GridCanvas(pass_items,
                                color_data=self._color_data, color_fit=self._color_fit,
                                color_resid=self._color_resid,
                                show_residuals=self._show_residuals,
                                layout_cfg=lc, **tkw)
            canvas.plot_clicked.connect(
                lambda i, fi=pass_items: self._open_single_filtered("PASS", fi, i))
            self._grid_scroll_pass.setWidget(canvas)
        else:
            self._grid_scroll_pass.setWidget(QLabel(no_pass))
        if fail_items:
            canvas = GridCanvas(fail_items,
                                color_data=self._color_data, color_fit=self._color_fit,
                                color_resid=self._color_resid,
                                show_residuals=self._show_residuals,
                                layout_cfg=lc, **tkw)
            canvas.plot_clicked.connect(
                lambda i, fi=fail_items: self._open_single_filtered("FAIL", fi, i))
            self._grid_scroll_fail.setWidget(canvas)
        else:
            self._grid_scroll_fail.setWidget(QLabel(no_fail))

    def _build_jump_list(self):
        self._jump.clear()
        for entry in self._pass_items:
            self._jump.addItem(f"[PASS] {entry['host']}-{entry['dye']}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry['host']}-{entry['dye']}")

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
            self._subtabs.addTab(self._single_scroll_pass, "PASS")
            self._subtabs.addTab(self._single_scroll_fail, "FAIL")
            self._show_single(0)

    def _sync_jump(self, tab: int, idx: int):
        """Keep the jump combo's selection in lockstep with what's plotted."""
        jump_idx = idx if tab == 0 else len(self._pass_items) + idx
        self._jump.blockSignals(True)
        self._jump.setCurrentIndex(jump_idx)
        self._jump.blockSignals(False)

    def _open_single_filtered(self, kind: str, filtered_items: list, filtered_idx: int):
        """Translate a filtered-grid index to the full-list index before opening."""
        full_items = self._pass_items if kind == "PASS" else self._fail_items
        item       = filtered_items[filtered_idx]
        # Use identity comparison so dicts with identical content don't collide.
        full_idx   = next((j for j, e in enumerate(full_items) if e is item),
                          filtered_idx)
        self._open_single(kind, full_idx)

    def _open_single(self, kind: str, idx: int):
        self._is_grid = True
        self._toggle_view()
        tab = 0 if kind == "PASS" else 1
        self._subtabs.setCurrentIndex(tab)
        self._current_items = self._pass_items if kind == "PASS" else self._fail_items
        self._current_idx   = idx
        canvas = self._single_pass if kind == "PASS" else self._single_fail
        canvas.show_entry(self._current_items[idx],
                          color_data=self._color_data, color_fit=self._color_fit,
                          color_resid=self._color_resid,
                          show_residuals=self._show_residuals,
                          layout_cfg=self._layout_cfg, **self._tkw())
        self._sync_jump(tab, idx)

    def _show_single(self, idx: int):
        tab    = self._subtabs.currentIndex()
        items  = self._pass_items if tab == 0 else self._fail_items
        canvas = self._single_pass if tab == 0 else self._single_fail
        if items:
            self._current_idx = idx % len(items)
            canvas.show_entry(items[self._current_idx],
                              color_data=self._color_data, color_fit=self._color_fit,
                              color_resid=self._color_resid,
                              show_residuals=self._show_residuals,
                              layout_cfg=self._layout_cfg, **self._tkw())
            self._sync_jump(tab, self._current_idx)

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

        # setWidgetResizable(False) prevents the scroll area from resizing the
        # canvas on each navigation, which would cause the figure to grow
        # progressively larger as the scroll area's viewport expands.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
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
        # Pin the canvas to the figure's natural pixel size so its size is
        # stable across repeated next/prev calls regardless of window size.
        dpi = fig.dpi
        canvas.resize(int(fig.get_figwidth() * dpi), int(fig.get_figheight() * dpi))
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
                 layout_cfg: dict = None, dpi: int = 100, parent=None):
        self._items = items
        lc   = layout_cfg or {}
        n    = max(len(items), 1)
        cols = lc.get("pdf_cols", min(int(np.ceil(np.sqrt(n))), 4))
        rows = int(np.ceil(n / cols))
        cf   = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd   = color_data  or pl_ki.PLOT_COLOR_DATA
        cr   = color_resid or cd
        tkw  = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                    title_fontweight=title_fontweight, title_fontstyle=title_fontstyle)
        fig_w = lc.get("fig_w", cols * 3.0)
        col_w = fig_w / cols

        if show_residuals:
            fig = Figure(figsize=(fig_w, rows * max(2.5, col_w * 1.70)), dpi=dpi)
            outer = GridSpec(rows, cols, figure=fig,
                             hspace=lc.get("hspace", 0.20),
                             wspace=lc.get("wspace", 0.40))
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ri, ci = divmod(i, cols)
                igs = outer[ri, ci].subgridspec(
                    2, 1, height_ratios=_RESID_RATIO,
                    hspace=lc.get("resid_gap", 0.06))
                ax   = fig.add_subplot(igs[0])
                ax_r = fig.add_subplot(igs[1], sharex=ax)
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, ax_resid=ax_r, **tkw)
                self._axes.append(ax)
        else:
            fig = Figure(figsize=(fig_w, rows * max(2.0, col_w * 1.55)), dpi=dpi)
            outer = GridSpec(rows, cols, figure=fig,
                             hspace=lc.get("hspace", 0.40),
                             wspace=lc.get("wspace", 0.40))
            super().__init__(fig)
            self._axes = []
            for i, entry in enumerate(items):
                ri, ci = divmod(i, cols)
                ax = fig.add_subplot(outer[ri, ci])
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, **tkw)
                self._axes.append(ax)

        fig.subplots_adjust(left=lc.get("left", 0.08), right=lc.get("right", 0.97),
                            top=lc.get("top", 0.96), bottom=lc.get("bottom", 0.08))
        dpi_v = fig.dpi
        self.setMinimumSize(int(fig.get_figwidth() * dpi_v),
                            int(fig.get_figheight() * dpi_v))
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
        self._fig = Figure(figsize=(5.5, 5.0))
        super().__init__(self._fig)

    def show_entry(self, entry, color_fit: str = None, color_data: str = None,
                   color_resid: str = None, show_residuals: bool = False,
                   title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "bold", title_fontstyle: str = "normal",
                   layout_cfg: dict = None):
        lc  = layout_cfg or {}
        fw  = lc.get("fig_w", 5.5)
        if show_residuals:
            fh = fw * (sum(_RESID_RATIO) / _RESID_RATIO[0]) * 0.95
        else:
            fh = fw * 0.95
        self._fig.set_size_inches(fw, fh)
        self._fig.clear()
        cf = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd = color_data  or pl_ki.PLOT_COLOR_DATA
        cr = color_resid or cd
        if show_residuals:
            gs   = GridSpec(2, 1, figure=self._fig,
                            height_ratios=_RESID_RATIO,
                            hspace=lc.get("resid_gap", 0.06))
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
            ax.tick_params(labelbottom=False)
        else:
            ax   = self._fig.add_subplot(111)
            ax_r = None
        self._fig.subplots_adjust(
            left=lc.get("left", 0.12), right=lc.get("right", 0.97),
            top=lc.get("top", 0.88), bottom=lc.get("bottom", 0.12))
        pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                             color_resid=cr, ax_resid=ax_r,
                             title_fontsize=title_fontsize,
                             title_fontfamily=title_fontfamily,
                             title_fontweight=title_fontweight,
                             title_fontstyle=title_fontstyle)
        dpi_v = self._fig.dpi
        self.setMinimumSize(int(fw * dpi_v), int(fh * dpi_v))
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
        self._layout_cfg       = {}
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
        self._grid_scroll_pass.setWidgetResizable(False)
        self._grid_scroll_fail = QScrollArea()
        self._grid_scroll_fail.setWidgetResizable(False)
        self._single_pass = CompSinglePlotCanvas()
        self._single_fail = CompSinglePlotCanvas()
        self._single_scroll_pass = QScrollArea()
        self._single_scroll_pass.setWidgetResizable(False)
        self._single_scroll_pass.setWidget(self._single_pass)
        self._single_scroll_fail = QScrollArea()
        self._single_scroll_fail.setWidgetResizable(False)
        self._single_scroll_fail.setWidget(self._single_fail)

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
             title_fontweight: str = None, title_fontstyle: str = None,
             layout_cfg: dict = None):
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
        if layout_cfg       is not None: self._layout_cfg       = layout_cfg

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
        lc = self._layout_cfg or None
        no_pass_lbl = "  No PASS results for this selection."
        no_fail_lbl = "  No FAIL results for this selection."
        if pass_items:
            canvas = CompGridCanvas(pass_items,
                                    color_fit=self._color_fit, color_data=self._color_data,
                                    color_resid=self._color_resid,
                                    show_residuals=self._show_residuals,
                                    layout_cfg=lc, **tkw)
            canvas.plot_clicked.connect(
                lambda i, fi=pass_items: self._open_single_filtered("PASS", fi, i))
            self._grid_scroll_pass.setWidget(canvas)
        else:
            self._grid_scroll_pass.setWidget(QLabel(no_pass_lbl))
        if fail_items:
            canvas = CompGridCanvas(fail_items,
                                    color_fit=self._color_fit, color_data=self._color_data,
                                    color_resid=self._color_resid,
                                    show_residuals=self._show_residuals,
                                    layout_cfg=lc, **tkw)
            canvas.plot_clicked.connect(
                lambda i, fi=fail_items: self._open_single_filtered("FAIL", fi, i))
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
            self._subtabs.addTab(self._single_scroll_pass, "PASS")
            self._subtabs.addTab(self._single_scroll_fail, "FAIL")
            self._show_single(0)

    def _sync_jump(self, tab: int, idx: int):
        """Keep the jump combo's selection in lockstep with what's plotted —
        without this, Prev/Next/grid-click leaves the dropdown showing a stale
        host-dye-guest label that no longer matches the displayed curve."""
        jump_idx = idx if tab == 0 else len(self._pass_items) + idx
        self._jump.blockSignals(True)
        self._jump.setCurrentIndex(jump_idx)
        self._jump.blockSignals(False)

    def _open_single_filtered(self, kind: str, filtered_items: list, filtered_idx: int):
        """Translate a filtered-grid index to the full-list index before opening."""
        full_items = self._pass_items if kind == "PASS" else self._fail_items
        item       = filtered_items[filtered_idx]
        full_idx   = next((j for j, e in enumerate(full_items) if e is item),
                          filtered_idx)
        self._open_single(kind, full_idx)

    def _open_single(self, kind: str, idx: int):
        self._is_grid = True
        self._toggle_view()
        tab = 0 if kind == "PASS" else 1
        self._subtabs.setCurrentIndex(tab)
        self._current_items = self._pass_items if kind == "PASS" else self._fail_items
        self._current_idx   = idx
        canvas = self._single_pass if kind == "PASS" else self._single_fail
        canvas.show_entry(self._current_items[idx],
                          color_fit=self._color_fit, color_data=self._color_data,
                          color_resid=self._color_resid,
                          show_residuals=self._show_residuals,
                          layout_cfg=self._layout_cfg,
                          title_fontsize=self._title_fontsize,
                          title_fontfamily=self._title_fontfamily,
                          title_fontweight=self._title_fontweight,
                          title_fontstyle=self._title_fontstyle)
        self._sync_jump(tab, idx)

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
                              layout_cfg=self._layout_cfg,
                              title_fontsize=self._title_fontsize,
                              title_fontfamily=self._title_fontfamily,
                              title_fontweight=self._title_fontweight,
                              title_fontstyle=self._title_fontstyle)
            self._sync_jump(tab, self._current_idx)

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
            # "Host | Dye" colours each assay condition as a unit — important
            # when multiple dyes are used so bars for the same guest are not
            # mis-grouped across dye experiments.
            self._grp.addItems(["None (single colour)", "Host | Dye",
                                 "Host", "Dye", "Guest"])
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
            # "Host | Dye" must be the default: using "Host" or "Dye" alone
            # triggers a groupby mean that averages across the other dimension,
            # silently collapsing data when more than one dye is used.
            self._heat_col_combo.addItems(["Host | Dye", "Host", "Dye"])
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
        # Build the composite grouping column before label/colour logic so
        # "Host | Dye" is available as a real column for colour-by grouping.
        if self._mode == "ki" and "Host" in df.columns and "Dye" in df.columns:
            df["Host | Dye"] = df["Host"] + " | " + df["Dye"]
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

STAGE_NAMES_FDA = ["Mappings", "Raw Data", "Merge & Blanks", "FI-F0", "Fit Results"]


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
        self._home_btn   = QPushButton("⌂ Home")
        self._status_lbl = QLabel("Ready")

        for w in (self._run_all_btn, self._run_step_btn,
                  QLabel("  |  Re-run:"), self._rerun_combo, self._rerun_btn,
                  QLabel("  |  "), self._save_btn,
                  QLabel("  |  "), self._switch_btn,
                  QLabel("  |  "), self._home_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        self._run_all_btn.clicked.connect(self._on_run_all)
        self._run_step_btn.clicked.connect(self._on_run_step)
        self._rerun_btn.clicked.connect(self._on_rerun)
        self._save_btn.clicked.connect(self._on_save)
        self._switch_btn.clicked.connect(self._on_open_competitive)
        self._home_btn.clicked.connect(self._on_home)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(300)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        box_f = QGroupBox("Folders")
        ff    = QFormLayout(box_f)
        self._input_row     = FolderRow(str(_BASE_DIR))
        self._output_row    = FolderRow(str(_BASE_DIR))
        self._chrom_db_row  = FolderRow(str(_BASE_DIR / "Chromatic_DB"))
        ff.addRow("Experiment:",   self._input_row)
        ff.addRow("Output:",       self._output_row)
        ff.addRow("Chromatic DB:", self._chrom_db_row)
        hint = QLabel("Raw/  dye_map/  host_map/  blank_map/\n← expected inside experiment folder\n\n"
                      "Chromatic DB: optional Dye/Filter lookup table\n"
                      "(folder of .xlsx, columns Dye/Filter) used to\n"
                      "auto-resolve multichromatic raw files by their\n"
                      "own filter-settings header.")
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
        self._cross_conc_chk.setChecked(False)   # pipeline default is OFF (v4 docstring)
        self._cross_conc_chk.setToolTip(
            "Apply Grubbs outlier test across concentration means.\n"
            "Off by default — can remove real hook-effect points at the\n"
            "highest host concentrations. Enable only if you are confident\n"
            "that outlier concentrations are artefacts, not real biology.")
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
            ["Auto (AICc)", "One-site only", "Quadratic only",
             "One-site (quenching only)"])

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
        self._summary_chk.setToolTip("pKd bar-chart / heatmap summary tab")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

        self._layout_panel = _LayoutPanel()
        self._layout_panel.changed.connect(self._on_layout_changed)
        layout.addWidget(self._layout_panel)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return scroll

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

        prev_w = QWidget()
        prev_l = QVBoxLayout(prev_w)
        ctrl_row = QHBoxLayout()
        self._prev_refresh_btn  = QPushButton("Refresh Preview")
        self._prev_pass_chk     = QCheckBox("PASS")
        self._prev_pass_chk.setChecked(True)
        self._prev_fail_chk     = QCheckBox("FAIL")
        self._prev_fail_chk.setChecked(False)
        ctrl_row.addWidget(self._prev_refresh_btn)
        ctrl_row.addWidget(self._prev_pass_chk)
        ctrl_row.addWidget(self._prev_fail_chk)
        ctrl_row.addStretch()
        note = QLabel("Shows plots at configured PDF dimensions.  "
                       "Adjust Figure Layout controls, then refresh.")
        note.setStyleSheet("color: gray; font-size: 10px;")
        ctrl_row.addWidget(note)
        prev_l.addLayout(ctrl_row)
        self._prev_scroll = QScrollArea()
        self._prev_scroll.setWidgetResizable(False)
        prev_l.addWidget(self._prev_scroll)
        self._prev_refresh_btn.clicked.connect(self._refresh_preview)
        self._prev_pass_chk.stateChanged.connect(self._refresh_preview)
        self._prev_fail_chk.stateChanged.connect(self._refresh_preview)
        self._tabs.addTab(prev_w, "PDF Preview")                # 7

        self._tabs.currentChanged.connect(self._on_tab_changed)
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
        model_map = {
            "Auto (AICc)":             "auto",
            "One-site only":           "one_site",
            "Quadratic only":          "quadratic",
            "One-site (quenching only)": "stern_volmer",  # internal key kept for compatibility
        }
        fw, fs = self._parse_title_style(self._title_style.currentText())
        base = self._input_row.path
        return {
            "raw_folder":            os.path.join(base, "Raw"),
            "dye_folder":            os.path.join(base, "dye_map"),
            "host_folder":           os.path.join(base, "host_map"),
            "blank_folder":          os.path.join(base, "blank_map"),
            "chromatic_folder":      self._chrom_db_row.path,
            "output_folder":         self._output_row.path,
            "input_folder":          base,
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
            "title_fontsize":        self._title_fsz.value(),
            "title_fontfamily":      self._title_fam.currentText(),
            "title_fontweight":      fw,
            "title_fontstyle":       fs,
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_fda.PASS_R2_DEFAULT)
        self._kd_lo_spin.setValue(pl_fda.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_hi_spin.setValue(pl_fda.KD_RANGE_FACTOR_HI_DEFAULT)
        self._grubbs_spin.setValue(0.05)
        self._cross_conc_chk.setChecked(False)   # default OFF per pipeline v4
        self._cross_conc_alpha_spin.setValue(0.01)
        self._model_combo.setCurrentIndex(0)
        self._color_data_row.set_color(pl_fda.PLOT_COLOR)
        self._color_fit_row.set_color(pl_fda.PLOT_COLOR)
        self._color_resid_row.set_color(pl_fda.PLOT_COLOR)
        self._title_fsz.setValue(8)
        self._title_fam.setCurrentText("sans-serif")
        self._title_style.setCurrentText("Bold")
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

    def _on_title_font_changed(self, _=None):
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
                         args=(cfg["raw_folder"],),
                         kwargs={"chromatic_folder": cfg["chromatic_folder"]})
        elif self._stage == 2:
            self._launch(pl_fda.merge_blanks,
                         args=(self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]),
                         kwargs={"chromatic_folder": cfg["chromatic_folder"]})
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
                qc = pl_fda.make_qc_figures_fda(result, progress_cb=self._log.append)
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
            self._tabs.setTabEnabled(7, True)
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
                             show_residuals=cfg["show_residuals"],
                             title_fontsize=cfg["title_fontsize"],
                             title_fontfamily=cfg["title_fontfamily"],
                             title_fontweight=cfg["title_fontweight"],
                             title_fontstyle=cfg["title_fontstyle"],
                             layout_cfg=self._layout_panel.get_cfg())
        self._tabs.setTabEnabled(7, True)
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)

    def _on_layout_changed(self):
        if self._state.plot_data:
            self._refresh_results()

    def _on_tab_changed(self, idx):
        if self._tabs.tabText(idx) == "PDF Preview" and self._state.plot_data:
            self._refresh_preview()

    def _refresh_preview(self, *_):
        if not self._state.plot_data:
            return
        lc  = self._layout_panel.get_cfg()
        cfg = self._get_config()
        show_pass = self._prev_pass_chk.isChecked()
        show_fail = self._prev_fail_chk.isChecked()
        if not show_pass and not show_fail:
            self._prev_scroll.setWidget(QLabel("  No items selected (tick PASS and/or FAIL)."))
            return

        # Build plate-ordered list matching PDF layout (1 plate per page, PASS+FAIL together).
        seen_plates: list = []
        for e in self._state.plot_data:
            p = e.get("plate", "")
            if p not in seen_plates:
                seen_plates.append(p)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(12)
        vbox.setContentsMargins(6, 6, 6, 6)

        any_shown = False
        for plate in seen_plates:
            plate_items = [e for e in self._state.plot_data if e.get("plate", "") == plate]
            visible = []
            if show_pass:
                visible += [e for e in plate_items if e.get("status") == "PASS"]
            if show_fail:
                visible += [e for e in plate_items if e.get("status") != "PASS"]
            if not visible:
                continue
            any_shown = True
            hdr = QLabel(f"  — Plate: {plate} —")
            hdr.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 8px;")
            vbox.addWidget(hdr)
            canvas = GridCanvas(visible,
                                color_data=cfg["color_data"], color_fit=cfg["color_fit"],
                                color_resid=cfg["color_resid"],
                                show_residuals=cfg["show_residuals"],
                                title_fontsize=cfg["title_fontsize"],
                                title_fontfamily=cfg["title_fontfamily"],
                                title_fontweight=cfg["title_fontweight"],
                                title_fontstyle=cfg["title_fontstyle"],
                                layout_cfg=lc, dpi=100)
            vbox.addWidget(canvas)

        if not any_shown:
            self._prev_scroll.setWidget(QLabel("  No items to show for selected filters."))
            return
        vbox.addStretch(1)
        container.adjustSize()
        self._prev_scroll.setWidget(container)

    def _on_save(self):
        cfg       = self._get_config()
        lc        = self._layout_panel.get_cfg()
        file_list = pl_fda.preview_save_files(self._state, cfg["output_folder"],
                                              export_individual=cfg["export_individual"],
                                              input_folder=cfg["input_folder"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None and not df.empty else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None and not df.empty else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        color_data  = cfg["color_data"]
        color_fit   = cfg["color_fit"]
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
                                       export_individual=export_ind,
                                       layout_cfg=lc,
                                       input_folder=cfg["input_folder"])
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

    def _on_home(self):
        _go_home(self)


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
        self._home_btn   = QPushButton("⌂ Home")
        self._status_lbl = QLabel("Ready")

        for w in (self._run_all_btn, self._run_step_btn,
                  QLabel("  |  Re-run:"), self._rerun_combo, self._rerun_btn,
                  QLabel("  |  "), self._save_btn,
                  QLabel("  |  "), self._switch_btn,
                  QLabel("  |  "), self._home_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        self._run_all_btn.clicked.connect(self._on_run_all)
        self._run_step_btn.clicked.connect(self._on_run_step)
        self._rerun_btn.clicked.connect(self._on_rerun)
        self._save_btn.clicked.connect(self._on_save)
        self._switch_btn.clicked.connect(self._on_open_direct)
        self._home_btn.clicked.connect(self._on_home)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(300)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        box_f = QGroupBox("Folders")
        ff    = QFormLayout(box_f)
        self._input_row    = FolderRow(str(_BASE_DIR))
        self._kd_row       = FolderRow(str(_BASE_DIR / "Comp_Kd"))
        self._chrom_db_row = FolderRow(str(_BASE_DIR / "Chromatic_DB"))
        self._output_row   = FolderRow(str(_BASE_DIR))
        ff.addRow("Experiment:",    self._input_row)
        ff.addRow("Kd table dir:",  self._kd_row)
        ff.addRow("Chromatic DB:",  self._chrom_db_row)
        ff.addRow("Output:",        self._output_row)
        hint = QLabel("Experiment must contain:\nRaw/ dye_map/ host_map/\nguest_map/ blank_map/\n\n"
                      "Chromatic DB: optional Dye/Filter lookup table\n"
                      "used to auto-resolve chromatic blocks from each\n"
                      "raw file's own filter-settings header.")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        ff.addRow(hint)
        layout.addWidget(box_f)

        box_c = QGroupBox("Chromatic → Dye (manual override)")
        cc    = QVBoxLayout(box_c)
        self._multi_chrom_chk = QCheckBox("Multi-chromatic plates")
        self._multi_chrom_chk.setChecked(False)
        self._multi_chrom_chk.setToolTip(
            "Plates are auto-resolved from the Chromatic DB above using each\n"
            "raw file's own filter-settings header — this works for both\n"
            "single-dye-per-plate and multi-dye-per-plate runs without any\n"
            "manual entry below.\n\n"
            "Tick this only to override the auto-detected mapping (e.g. a\n"
            "dye not yet in the Chromatic DB, or a raw file with no\n"
            "filter-settings header). Entries below take precedence over\n"
            "auto-detection. Use the 'Plate (substring)' column to target\n"
            "specific plates; leave it empty for a global override.")
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
        self._grubbs_spin.valueChanged.connect(self._on_grubbs_changed)

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

        self._layout_panel = _LayoutPanel()
        self._layout_panel.changed.connect(self._on_layout_changed)
        layout.addWidget(self._layout_panel)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        layout.addStretch()
        return scroll

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

        prev_w = QWidget()
        prev_l = QVBoxLayout(prev_w)
        ctrl_row = QHBoxLayout()
        self._prev_refresh_btn = QPushButton("Refresh Preview")
        self._prev_pass_chk    = QCheckBox("PASS")
        self._prev_pass_chk.setChecked(True)
        self._prev_fail_chk    = QCheckBox("FAIL")
        self._prev_fail_chk.setChecked(False)
        ctrl_row.addWidget(self._prev_refresh_btn)
        ctrl_row.addWidget(self._prev_pass_chk)
        ctrl_row.addWidget(self._prev_fail_chk)
        ctrl_row.addStretch()
        note = QLabel("Shows plots at configured PDF dimensions.  "
                       "Adjust Figure Layout controls, then refresh.")
        note.setStyleSheet("color: gray; font-size: 10px;")
        ctrl_row.addWidget(note)
        prev_l.addLayout(ctrl_row)
        self._prev_scroll = QScrollArea()
        self._prev_scroll.setWidgetResizable(False)
        prev_l.addWidget(self._prev_scroll)
        self._prev_refresh_btn.clicked.connect(self._refresh_preview)
        self._prev_pass_chk.stateChanged.connect(self._refresh_preview)
        self._prev_fail_chk.stateChanged.connect(self._refresh_preview)
        self._tabs.addTab(prev_w, "PDF Preview")                # 8

        self._tabs.currentChanged.connect(self._on_tab_changed)
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
            "chromatic_folder":   self._chrom_db_row.path,
            "output_folder":      self._output_row.path,
            "input_folder":       base,
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
                               cfg["raw_folder"], cfg["chromatic_to_dye"],
                               chromatic_folder=cfg["chromatic_folder"]))
        elif self._stage == 2:
            self._launch(_wrap(pl_ki.merge_ki,
                               self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"],
                               chromatic_folder=cfg["chromatic_folder"]))
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
            self._tabs.setTabEnabled(8, True)
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

    def _on_grubbs_changed(self, _value):
        if self._state.fit_results:
            QMessageBox.information(
                self, "Re-run required",
                "Grubbs α affects outlier removal during curve fitting.\n"
                "Please re-run from the Competitive Fitting stage to apply the change.")

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
                             title_fontstyle=cfg["title_fontstyle"],
                             layout_cfg=self._layout_panel.get_cfg())
        self._tabs.setTabEnabled(8, True)
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)

    def _on_layout_changed(self):
        if self._state.plot_data:
            self._refresh_results()

    def _on_tab_changed(self, idx):
        if self._tabs.tabText(idx) == "PDF Preview" and self._state.plot_data:
            self._refresh_preview()

    def _refresh_preview(self, *_):
        if not self._state.plot_data:
            return
        lc  = self._layout_panel.get_cfg()
        cfg = self._get_config()
        show_pass = self._prev_pass_chk.isChecked()
        show_fail = self._prev_fail_chk.isChecked()
        if not show_pass and not show_fail:
            self._prev_scroll.setWidget(QLabel("  No items selected (tick PASS and/or FAIL)."))
            return

        # Build plate-ordered list matching PDF layout (1 plate per page, PASS+FAIL together).
        seen_plates: list = []
        for e in self._state.plot_data:
            p = e.get("plate", "")
            if p not in seen_plates:
                seen_plates.append(p)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(12)
        vbox.setContentsMargins(6, 6, 6, 6)

        any_shown = False
        for plate in seen_plates:
            plate_items = [e for e in self._state.plot_data if e.get("plate", "") == plate]
            visible = []
            if show_pass:
                visible += [e for e in plate_items if e.get("status") == "PASS"]
            if show_fail:
                visible += [e for e in plate_items if e.get("status") != "PASS"]
            if not visible:
                continue
            any_shown = True
            hdr = QLabel(f"  — Plate: {plate} —")
            hdr.setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 8px;")
            vbox.addWidget(hdr)
            canvas = CompGridCanvas(visible,
                                    color_fit=cfg["color_fit"], color_data=cfg["color_data"],
                                    color_resid=cfg["color_resid"],
                                    show_residuals=cfg["show_residuals"],
                                    title_fontsize=cfg["title_fontsize"],
                                    title_fontfamily=cfg["title_fontfamily"],
                                    title_fontweight=cfg["title_fontweight"],
                                    title_fontstyle=cfg["title_fontstyle"],
                                    layout_cfg=lc, dpi=100)
            vbox.addWidget(canvas)

        if not any_shown:
            self._prev_scroll.setWidget(QLabel("  No items to show for selected filters."))
            return
        vbox.addStretch(1)
        container.adjustSize()
        self._prev_scroll.setWidget(container)

    def _on_save(self):
        cfg       = self._get_config()
        lc        = self._layout_panel.get_cfg()
        file_list = pl_ki.preview_save_files_ki(self._state, cfg["output_folder"],
                                                export_individual=cfg["export_individual"],
                                                input_folder=cfg["input_folder"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None and not df.empty else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None and not df.empty else 0

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
                                         title_fontstyle=cfg["title_fontstyle"],
                                         layout_cfg=lc,
                                         input_folder=cfg["input_folder"])
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

    def _on_home(self):
        _go_home(self)


# ── Spectral Scan main window ─────────────────────────────────────────────────
# Tab layout: 0=Mappings, 1=Raw Data, 2=FI-F0, 3=Spectra, 4=Fit Results, 5=Plots, 6=Summary

STAGE_NAMES_SPECTRAL = ["Mappings", "Raw Data", "Merge & Blanks", "FI-F0"]


class SpectralMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Spectral Scan")
        self.resize(1300, 820)

        self._state   = PipelineStateSpectral()
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
        self._rerun_combo.addItems(STAGE_NAMES_SPECTRAL)
        self._rerun_btn  = QPushButton("Re-run")
        self._home_btn   = QPushButton("⌂ Home")
        self._status_lbl = QLabel("Ready")

        for w in (self._run_all_btn, self._run_step_btn,
                  QLabel("  |  Re-run:"), self._rerun_combo, self._rerun_btn,
                  QLabel("  |  "), self._save_btn,
                  QLabel("  |  "), self._home_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        self._run_all_btn.clicked.connect(self._on_run_all)
        self._run_step_btn.clicked.connect(self._on_run_step)
        self._rerun_btn.clicked.connect(self._on_rerun)
        self._save_btn.clicked.connect(self._on_save)
        self._home_btn.clicked.connect(self._on_home)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._build_config())

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self._build_log())
        right.setSizes([620, 150])
        splitter.addWidget(right)
        splitter.setSizes([300, 1020])

    def _build_config(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(300)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        box_f = QGroupBox("Folders")
        ff    = QFormLayout(box_f)
        self._input_row  = FolderRow(str(_BASE_DIR))
        self._output_row = FolderRow(str(_BASE_DIR))
        ff.addRow("Experiment:", self._input_row)
        ff.addRow("Output:",     self._output_row)
        hint = QLabel("Raw/  dye_map/  host_map/  blank_map/\n← expected inside experiment folder\n\n"
                      "Raw/ holds spectral-scan exports (.csv).\n"
                      "dye_map / host_map / blank_map are the same\n"
                      "Echo-export mapping files used by Direct Binding.")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        ff.addRow(hint)
        layout.addWidget(box_f)

        box_b = QGroupBox("Binding Fit")
        bf    = QFormLayout(box_b)
        self._scan_combo = QComboBox()
        self._scan_combo.addItems(["Emission", "Excitation"])
        self._wavelength_spin = QDoubleSpinBox()
        self._wavelength_spin.setRange(100.0, 1200.0)
        self._wavelength_spin.setValue(460.0)
        self._wavelength_spin.setSingleStep(1.0)
        self._wavelength_spin.setDecimals(1)
        self._wavelength_spin.setSuffix(" nm")
        self._wavelength_spin.setToolTip(
            "Target wavelength for the binding fit. The scanned wavelength\n"
            "step nearest this value is used (snapped, logged on fit).")
        self._per_dye_wl_chk = QCheckBox("Per-dye wavelength")
        self._per_dye_wl_chk.setChecked(False)
        self._per_dye_wl_chk.setToolTip(
            "When checked, choose a different target wavelength for each dye.")
        self._per_dye_wl_container = QWidget()
        self._per_dye_wl_layout = QFormLayout(self._per_dye_wl_container)
        self._per_dye_wl_layout.setContentsMargins(0, 0, 0, 0)
        self._per_dye_wl_container.setVisible(False)
        self._per_dye_wl_spins: dict[str, QDoubleSpinBox] = {}
        self._per_dye_wl_chk.toggled.connect(self._on_per_dye_wl_toggled)

        self._fit_binding_btn = QPushButton("Fit Binding")
        self._fit_binding_btn.setEnabled(False)
        self._fit_binding_btn.clicked.connect(self._on_fit_binding)
        bf.addRow("Scan:",       self._scan_combo)
        bf.addRow("Wavelength:", self._wavelength_spin)
        bf.addRow(self._per_dye_wl_chk)
        bf.addRow(self._per_dye_wl_container)
        bf.addRow(self._fit_binding_btn)
        layout.addWidget(box_b)

        box_p = QGroupBox("Fit Parameters")
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
        self._kd_lo_spin.valueChanged.connect(self._on_r2_changed)

        self._kd_hi_spin = QDoubleSpinBox()
        self._kd_hi_spin.setRange(1.0, 1000.0)
        self._kd_hi_spin.setSingleStep(1.0)
        self._kd_hi_spin.setValue(pl_fda.KD_RANGE_FACTOR_HI_DEFAULT)
        self._kd_hi_spin.setDecimals(1)
        self._kd_hi_spin.valueChanged.connect(self._on_r2_changed)

        self._grubbs_spin = QDoubleSpinBox()
        self._grubbs_spin.setRange(0.001, 0.2)
        self._grubbs_spin.setSingleStep(0.005)
        self._grubbs_spin.setValue(0.05)
        self._grubbs_spin.setDecimals(3)

        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["Auto (AICc)", "One-site only", "Quadratic only",
             "One-site (quenching only)"])

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Kd range lo:",  self._kd_lo_spin)
        pf.addRow("Kd range hi:",  self._kd_hi_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
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

        box_spec = QGroupBox("Spectra Display")
        sl       = QVBoxLayout(box_spec)

        spec_form = QFormLayout()
        self._spectra_palette_combo = QComboBox()
        self._spectra_palette_combo.addItems([
            "viridis", "plasma", "magma", "inferno", "coolwarm",
            "Blues", "tab10", "Set2"])
        self._spectra_palette_combo.currentIndexChanged.connect(self._on_spectra_display_changed)
        spec_form.addRow("Palette:", self._spectra_palette_combo)

        self._spectra_display_combo = QComboBox()
        self._spectra_display_combo.addItems(["Side by side", "Combined", "Separate"])
        self._spectra_display_combo.currentIndexChanged.connect(self._on_spectra_display_changed)
        spec_form.addRow("Layout:", self._spectra_display_combo)
        sl.addLayout(spec_form)

        self._conc_show_all_chk = QCheckBox("Show all concentrations")
        self._conc_show_all_chk.setChecked(True)
        self._conc_show_all_chk.toggled.connect(self._on_conc_show_all_toggled)
        sl.addWidget(self._conc_show_all_chk)

        self._conc_scroll = QScrollArea()
        self._conc_scroll.setWidgetResizable(True)
        self._conc_scroll.setMaximumHeight(120)
        self._conc_inner = QWidget()
        self._conc_layout = QVBoxLayout(self._conc_inner)
        self._conc_layout.setContentsMargins(2, 2, 2, 2)
        self._conc_layout.setSpacing(2)
        self._conc_scroll.setWidget(self._conc_inner)
        self._conc_scroll.setVisible(False)
        self._conc_checks: list[tuple[float, QCheckBox]] = []
        sl.addWidget(self._conc_scroll)

        self._override_scroll = QScrollArea()
        self._override_scroll.setWidgetResizable(True)
        self._override_scroll.setMaximumHeight(120)
        self._override_inner = QWidget()
        self._override_layout = QVBoxLayout(self._override_inner)
        self._override_layout.setContentsMargins(2, 2, 2, 2)
        self._override_layout.setSpacing(2)
        self._override_scroll.setWidget(self._override_inner)
        self._override_scroll.setVisible(False)
        self._override_rows: list[tuple[float, ColorPickerRow]] = []
        override_lbl = QLabel("Per-concentration colour overrides:")
        override_lbl.setStyleSheet("color: grey; font-size: 10px;")
        sl.addWidget(override_lbl)
        sl.addWidget(self._override_scroll)

        layout.addWidget(box_spec)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(False)
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._summary_chk = QCheckBox("Show summary tab")
        self._summary_chk.setChecked(True)
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

        layout.addStretch()
        return scroll

    def _build_tabs(self):
        self._tabs = QTabWidget()

        self._map_view, self._map_model = _make_table()
        self._tabs.addTab(self._map_view, "Mappings")            # 0

        raw_w = QWidget()
        raw_l = QVBoxLayout(raw_w)
        self._raw_lbl = QLabel("")
        self._raw_view, self._raw_model = _make_table()
        raw_l.addWidget(self._raw_lbl)
        raw_l.addWidget(self._raw_view)
        self._tabs.addTab(raw_w, "Raw Data")                      # 1

        self._fi_view, self._fi_model = _make_table()
        self._tabs.addTab(self._fi_view, "FI-F0")                 # 2

        self._spectra_tab = QCPlotsTab()
        self._tabs.addTab(self._spectra_tab, "Spectra")           # 3

        fit_w = QWidget()
        fit_l = QVBoxLayout(fit_w)
        self._fit_lbl = QLabel("")
        self._fit_lbl.setAlignment(Qt.AlignCenter)
        font = QFont(); font.setBold(True)
        self._fit_lbl.setFont(font)
        self._fit_view, self._fit_model = _make_table()
        fit_l.addWidget(self._fit_lbl)
        fit_l.addWidget(self._fit_view)
        self._tabs.addTab(fit_w, "Fit Results")                   # 4

        self._plots_tab = DirectPlotsTab()
        self._tabs.addTab(self._plots_tab, "Plots")                # 5

        self._summary_tab = SummaryTab(mode="kd")
        self._tabs.addTab(self._summary_tab, "Summary")            # 6

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
            "Auto (AICc)":               "auto",
            "One-site only":             "one_site",
            "Quadratic only":            "quadratic",
            "One-site (quenching only)": "stern_volmer",
        }
        display_map = {"Side by side": "side_by_side", "Combined": "combined",
                       "Separate": "separate"}
        color_overrides = {}
        for conc, row in self._override_rows:
            default_hex = "#000000"
            if row.color != default_hex:
                color_overrides[conc] = row.color
        selected_concs = None
        if not self._conc_show_all_chk.isChecked():
            selected_concs = [c for c, chk in self._conc_checks if chk.isChecked()]
        base = self._input_row.path
        return {
            "raw_folder":     os.path.join(base, "Raw"),
            "dye_folder":     os.path.join(base, "dye_map"),
            "host_folder":    os.path.join(base, "host_map"),
            "blank_folder":   os.path.join(base, "blank_map"),
            "output_folder":  self._output_row.path,
            "input_folder":   base,
            "scan_type":      "ex" if self._scan_combo.currentText() == "Excitation" else "em",
            "wavelength":     self._wavelength_spin.value(),
            "r2_threshold":   self._r2_spin.value(),
            "kd_range_lo":    self._kd_lo_spin.value(),
            "kd_range_hi":    self._kd_hi_spin.value(),
            "grubbs_alpha":   self._grubbs_spin.value(),
            "model_preference": model_map[self._model_combo.currentText()],
            "color_data":     self._color_data_row.color,
            "color_fit":      self._color_fit_row.color,
            "color_resid":    self._color_resid_row.color,
            "show_residuals": self._resid_chk.isChecked(),
            "export_individual": self._ind_export_chk.isChecked(),
            "spectra_palette": self._spectra_palette_combo.currentText(),
            "spectra_display": display_map.get(self._spectra_display_combo.currentText(),
                                               "side_by_side"),
            "spectra_color_overrides": color_overrides if color_overrides else None,
            "selected_concs": selected_concs,
            "per_dye_wl":     self._per_dye_wl_chk.isChecked(),
            "wavelength_map":  {dye: spin.value()
                                for dye, spin in self._per_dye_wl_spins.items()},
        }

    def _on_color_changed(self, _hex):
        if self._state.plot_data:
            self._refresh_results()

    def _on_display_changed(self, _state):
        if self._state.plot_data:
            self._refresh_results()

    def _on_spectra_display_changed(self, _index=None):
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_show_all_toggled(self, checked):
        self._conc_scroll.setVisible(not checked)
        for _c, chk in self._conc_checks:
            chk.setEnabled(not checked)
            if checked:
                chk.setChecked(True)
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_check_changed(self, _state):
        if self._state.fi_df is not None and not self._conc_show_all_chk.isChecked():
            self._regenerate_spectra()

    def _on_override_color_changed(self, _hex):
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_per_dye_wl_toggled(self, checked):
        self._wavelength_spin.setVisible(not checked)
        self._per_dye_wl_container.setVisible(checked)

    def _populate_spectra_controls(self, fi_df):
        """Populate concentration checkboxes, colour overrides, and per-dye
        wavelength spinboxes once fi_df is available."""
        concs = sorted(fi_df["Host_Concentration"].dropna().unique())
        # Concentration checkboxes
        for _c, chk in self._conc_checks:
            self._conc_layout.removeWidget(chk)
            chk.deleteLater()
        self._conc_checks.clear()
        for c in concs:
            chk = QCheckBox(f"{c:g} µM")
            chk.setChecked(True)
            chk.stateChanged.connect(self._on_conc_check_changed)
            self._conc_layout.addWidget(chk)
            self._conc_checks.append((c, chk))

        # Per-concentration colour override rows
        for _c, row in self._override_rows:
            self._override_layout.removeWidget(row)
            row.deleteLater()
        self._override_rows.clear()
        for c in concs:
            row = ColorPickerRow(f"{c:g} µM:", "#000000")
            row.color_changed.connect(self._on_override_color_changed)
            self._override_layout.addWidget(row)
            self._override_rows.append((c, row))
        self._override_scroll.setVisible(True)

        # Per-dye wavelength spinboxes
        dyes = sorted(fi_df["Dye"].dropna().unique())
        for spin in self._per_dye_wl_spins.values():
            self._per_dye_wl_layout.removeWidget(spin)
            spin.deleteLater()
        # Also remove labels from the form layout
        while self._per_dye_wl_layout.rowCount() > 0:
            self._per_dye_wl_layout.removeRow(0)
        self._per_dye_wl_spins.clear()
        for dye in dyes:
            spin = QDoubleSpinBox()
            spin.setRange(100.0, 1200.0)
            spin.setValue(self._wavelength_spin.value())
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setSuffix(" nm")
            self._per_dye_wl_layout.addRow(f"{dye}:", spin)
            self._per_dye_wl_spins[dye] = spin
        self._per_dye_wl_chk.setVisible(len(dyes) > 1)
        if len(dyes) <= 1:
            self._per_dye_wl_chk.setChecked(False)

    def _regenerate_spectra(self):
        """Re-render spectra figures from fi_df with current display settings."""
        cfg = self._get_config()
        try:
            figures = pl_spec.make_spectra_figures(
                self._state.fi_df,
                progress_cb=self._log.append,
                palette=cfg["spectra_palette"],
                color_overrides=cfg["spectra_color_overrides"],
                display_mode=cfg["spectra_display"],
                selected_concs=cfg["selected_concs"])
        except Exception:
            import traceback as _tb
            self._log.append(f"\n[Spectra warning]\n{_tb.format_exc()}")
            figures = []
        self._state.spectra_figures = figures
        self._spectra_tab.load(figures)

    def _on_summary_toggle(self, state):
        self._tabs.setTabVisible(6, bool(state))
        if state and self._state.df_results is not None:
            self._summary_tab.load(self._state.df_results)

    def _set_buttons_enabled(self):
        busy = self._worker is not None and self._worker.isRunning()
        self._run_all_btn.setEnabled(not busy)
        self._run_step_btn.setEnabled(not busy and self._stage < len(STAGE_NAMES_SPECTRAL))
        self._rerun_btn.setEnabled(not busy)
        self._rerun_combo.setEnabled(not busy)
        self._fit_binding_btn.setEnabled(not busy and self._state.fi_df is not None)

    def _on_run_all(self):
        self._stage   = 0
        self._run_all = True
        self._state   = PipelineStateSpectral()
        self._run_stage()

    def _on_run_step(self):
        self._run_all = False
        self._run_stage()

    def _on_rerun(self):
        sel = self._rerun_combo.currentIndex()
        prereqs = {1: (), 2: ("merged_mapping", "fluorescence"), 3: ("merged",)}
        missing = [p for p in prereqs.get(sel, []) if getattr(self._state, p) is None]
        if missing:
            QMessageBox.warning(self, "Missing prerequisite",
                                f"Stage '{STAGE_NAMES_SPECTRAL[sel]}' requires earlier stages first.\n"
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
            self._launch(pl_spec.load_plates_spectral,
                         args=(cfg["raw_folder"],))
        elif self._stage == 2:
            self._launch(pl_spec.merge_blanks_spectral,
                         args=(self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]))
        elif self._stage == 3:
            self._launch(pl_fda.subtract_background,
                         args=(self._state.merged,))

    def _launch(self, fn, args=(), kwargs=None):
        self._worker = StageWorker(fn, args=args, kwargs=kwargs or {})
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_stage_done)
        self._worker.error.connect(lambda tb: (self._log.append(f"\n[ERROR]\n{tb}"),
                                               QMessageBox.critical(self, "Error", "See log.")))
        stage_lbl = STAGE_NAMES_SPECTRAL[self._stage] if self._stage < len(STAGE_NAMES_SPECTRAL) else "?"
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
            n_ex = result.loc[result["ScanType"] == "ex", "Chromatic"].nunique()
            n_em = result.loc[result["ScanType"] == "em", "Chromatic"].nunique()
            self._raw_lbl.setText(f"{len(result):,} rows  |  "
                                   f"{result['Plate'].nunique()} plate(s)  |  "
                                   f"{n_ex} excitation step(s), {n_em} emission step(s)")
            self._tabs.setTabEnabled(1, True)
            self._tabs.setCurrentIndex(1)

        elif self._stage == 2:
            self._state.merged = result

        elif self._stage == 3:
            self._state.fi_df = result
            self._fi_model.set_dataframe(result)
            self._tabs.setTabEnabled(2, True)
            self._tabs.setCurrentIndex(2)
            self._populate_spectra_controls(result)
            cfg = self._get_config()
            try:
                figures = pl_spec.make_spectra_figures(
                    result, progress_cb=self._log.append,
                    palette=cfg["spectra_palette"],
                    color_overrides=cfg["spectra_color_overrides"],
                    display_mode=cfg["spectra_display"],
                    selected_concs=cfg["selected_concs"])
            except Exception:
                import traceback as _tb
                self._log.append(f"\n[Spectra warning]\n{_tb.format_exc()}")
                figures = []
            self._state.spectra_figures = figures
            self._spectra_tab.load(figures)
            self._tabs.setTabEnabled(3, True)

        self._stage += 1
        self._status_lbl.setText(f"Stage {self._stage}/{len(STAGE_NAMES_SPECTRAL)} done")
        self._set_buttons_enabled()

        if self._run_all and self._stage < len(STAGE_NAMES_SPECTRAL):
            self._run_stage()

    def _on_r2_changed(self, _value):
        if self._state.fit_results:
            self._refresh_results()

    def _on_fit_binding(self):
        cfg = self._get_config()
        if cfg["per_dye_wl"] and cfg["wavelength_map"]:
            self._worker = StageWorker(
                pl_spec.fit_binding_per_dye,
                args=(self._state.fi_df, cfg["scan_type"], cfg["wavelength_map"]),
                kwargs={"grubbs_alpha": cfg["grubbs_alpha"],
                        "model_preference": cfg["model_preference"]})
        else:
            self._worker = StageWorker(
                pl_spec.fit_binding_at_wavelength,
                args=(self._state.fi_df, cfg["scan_type"], cfg["wavelength"]),
                kwargs={"grubbs_alpha": cfg["grubbs_alpha"],
                        "model_preference": cfg["model_preference"]})
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_fit_done)
        self._worker.error.connect(lambda tb: (self._log.append(f"\n[ERROR]\n{tb}"),
                                               QMessageBox.critical(self, "Error", "See log.")))
        self._status_lbl.setText("Fitting binding curves…")
        self._set_buttons_enabled()
        self._worker.start()

    def _on_fit_done(self, result):
        self._set_buttons_enabled()
        if isinstance(result, Exception) or not isinstance(result, dict):
            self._status_lbl.setText("Error — see log")
            return

        self._state.fit_results    = result["fit_results"]
        self._state.plot_data      = result["plot_data"]
        self._state.fit_wavelength = result["nearest_wavelength"]
        self._state.fit_scan_type  = self._get_config()["scan_type"]

        self._tabs.setTabEnabled(4, True)
        self._tabs.setTabEnabled(5, True)
        if self._summary_chk.isChecked():
            self._tabs.setTabEnabled(6, True)
        self._tabs.setCurrentIndex(4)
        self._save_btn.setEnabled(True)
        nw = result["nearest_wavelength"]
        if isinstance(nw, dict):
            wl_str = ", ".join(f"{d}: {w:.1f}" for d, w in nw.items())
            self._status_lbl.setText(f"Binding fit done — λ per dye: {wl_str}")
        else:
            self._status_lbl.setText(
                f"Binding fit done — nearest wavelength {nw:.1f} nm")
        try:
            self._refresh_results()
        except Exception:
            import traceback as _tb
            self._log.append(f"\n[ERROR in refresh]\n{_tb.format_exc()}")

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
            nw = self._state.fit_wavelength
            if isinstance(nw, dict):
                wl_lbl = ", ".join(f"{d}: {w:.1f}" for d, w in nw.items())
            elif nw is not None:
                wl_lbl = f"{nw:.1f} nm"
            else:
                wl_lbl = "?"
            self._fit_lbl.setText(
                f"PASS: {n_pass}   |   FAIL: {n_fail}   "
                f"(R² threshold = {cfg['r2_threshold']:.2f}, λ = {wl_lbl})")
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
        file_list = pl_spec.preview_save_files_spectral(
            self._state, cfg["output_folder"], export_individual=cfg["export_individual"],
            input_folder=cfg["input_folder"])
        df        = self._state.df_results
        n_pass    = int((df["Status"] == "PASS").sum()) if df is not None and not df.empty else 0
        n_fail    = int((df["Status"] == "FAIL").sum()) if df is not None and not df.empty else 0

        dlg = SavePreviewDialog(file_list, n_pass, n_fail, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        color_data  = cfg["color_data"]
        color_fit   = cfg["color_fit"]
        color_resid = cfg["color_resid"]
        show_resid  = cfg["show_residuals"]
        export_ind  = cfg["export_individual"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_spec.save_outputs_spectral(
                self._state, cfg["output_folder"], cfg["r2_threshold"], progress_cb,
                plot_color_data=color_data, plot_color_fit=color_fit,
                plot_color_resid=color_resid, show_residuals=show_resid,
                export_individual=export_ind,
                input_folder=cfg["input_folder"])
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

    def _on_home(self):
        _go_home(self)


# ── Layout panel (shared by DirectMainWindow and CompMainWindow) ───────────────

class _LayoutPanel(QGroupBox):
    """Group box for controlling figure dimensions and spacing."""
    changed = Signal()
    _DEFS = dict(fig_w=7.0, pdf_cols=4,
                 hspace=0.40, wspace=0.40, resid_gap=0.06,
                 left=0.08, right=0.97, top=0.96, bottom=0.08)

    def __init__(self, parent=None):
        super().__init__("Figure Layout", parent)
        fl = QFormLayout(self)
        fl.setSpacing(3)

        def _d(lo, hi, v, step=0.05, dec=2):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setValue(v)
            sb.setSingleStep(step); sb.setDecimals(dec)
            sb.valueChanged.connect(self.changed)
            return sb

        self._fig_w    = _d(2.0, 20.0, 7.0, 0.5, 1)
        self._pdf_cols = QSpinBox()
        self._pdf_cols.setRange(1, 8); self._pdf_cols.setValue(4)
        self._pdf_cols.valueChanged.connect(self.changed)
        self._hspace    = _d(0.0, 2.0,  0.40)
        self._wspace    = _d(0.0, 2.0,  0.40)
        self._resid_gap = _d(0.0, 0.5,  0.06)
        self._left      = _d(0.0, 0.40, 0.08)
        self._right     = _d(0.6, 1.0,  0.97)
        self._top       = _d(0.6, 1.0,  0.96)
        self._bottom    = _d(0.0, 0.40, 0.08)

        fl.addRow("Fig. width (in):", self._fig_w)
        fl.addRow("PDF columns:",     self._pdf_cols)
        fl.addRow("H-space:",         self._hspace)
        fl.addRow("W-space:",         self._wspace)
        fl.addRow("Resid. gap:",      self._resid_gap)
        fl.addRow("Left margin:",     self._left)
        fl.addRow("Right margin:",    self._right)
        fl.addRow("Top margin:",      self._top)
        fl.addRow("Bottom margin:",   self._bottom)

        rst = QPushButton("Reset layout")
        rst.clicked.connect(self._reset)
        fl.addRow(rst)

    def get_cfg(self) -> dict:
        return {
            "fig_w":    self._fig_w.value(),
            "pdf_cols": self._pdf_cols.value(),
            "hspace":   self._hspace.value(),
            "wspace":   self._wspace.value(),
            "resid_gap": self._resid_gap.value(),
            "left":     self._left.value(),
            "right":    self._right.value(),
            "top":      self._top.value(),
            "bottom":   self._bottom.value(),
        }

    def _reset(self):
        for key, val in self._DEFS.items():
            w = getattr(self, f"_{key}")
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.changed.emit()


# ── Compare Runs window ────────────────────────────────────────────────────────

class _RunFileItem(QWidget):
    """A row in the run-file list: colour swatch + label + data indicator + remove."""
    remove_requested = Signal(object)
    color_changed    = Signal(object)
    data_attached    = Signal(object)

    def __init__(self, path: str, color: str, has_data: bool = False, parent=None):
        super().__init__(parent)
        self.path   = path
        self._color = color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._swatch = QPushButton()
        self._swatch.setFixedSize(18, 18)
        self._swatch.setToolTip("Click to change run colour")
        self._swatch.clicked.connect(self._pick_color)
        self._sync_swatch()
        lay.addWidget(self._swatch)
        self._lbl = QLineEdit(os.path.splitext(os.path.basename(path))[0])
        self._lbl.setPlaceholderText("Run label")
        lay.addWidget(self._lbl, 1)
        self._data_lbl = QLabel()
        self._data_lbl.setFixedWidth(14)
        lay.addWidget(self._data_lbl)
        self._attach_btn = QPushButton("📎")
        self._attach_btn.setFixedSize(22, 20)
        self._attach_btn.setToolTip("Attach companion data file")
        self._attach_btn.clicked.connect(self._on_attach)
        lay.addWidget(self._attach_btn)
        rm_btn = QPushButton("×")
        rm_btn.setFixedSize(20, 20)
        rm_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        lay.addWidget(rm_btn)
        self.set_has_data(has_data)

    def set_has_data(self, ok: bool):
        self._has_data = ok
        if ok:
            self._data_lbl.setText("✓")
            self._data_lbl.setStyleSheet("color: green; font-weight: bold;")
            self._data_lbl.setToolTip("Raw data loaded")
        else:
            self._data_lbl.setText("✗")
            self._data_lbl.setStyleSheet("color: #999;")
            self._data_lbl.setToolTip("No raw data — attach a companion _data file")

    def _on_attach(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select companion data file", os.path.dirname(self.path),
            "Excel files (*.xlsx *.xls);;All files (*)")
        if path:
            self._attached_data_path = path
            self.data_attached.emit(self)

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Choose run colour")
        if col.isValid():
            self._color = col.name()
            self._sync_swatch()
            self.color_changed.emit(self)

    def _sync_swatch(self):
        self._swatch.setStyleSheet(
            f"background-color:{self._color}; border:1px solid #666; border-radius:2px;")

    @property
    def label(self) -> str:
        return self._lbl.text().strip() or os.path.basename(self.path)

    @property
    def color(self) -> str:
        return self._color


class RunCompareCanvas(FigureCanvas):
    """Grid of curve panels with optional residual / Ki-Kd dot panels."""
    plot_clicked = Signal(int)

    def __init__(self, combos: list, runs: list, mode: str,
                 ncols: int = 4, fig_w: float = None, dpi_val: int = 100,
                 show_dots_panel: bool = True, show_residuals: bool = False,
                 flip_norm: bool = False,
                 pooled_mode: bool = False, show_legend: bool = False,
                 title_fontsize: float = 8, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "bold", title_fontstyle: str = "normal",
                 color_data: str = "#1e4572", color_fit: str = "#6495ED",
                 color_resid: str = None,
                 parent=None, **_extra):
        n     = len(combos)
        ncols = min(ncols, n) if n else 1
        nrows = max(1, -(-n // ncols))
        pw    = (fig_w / ncols) if fig_w else 3.4
        kw    = dict(show_legend=show_legend, flip_norm=flip_norm,
                     pooled_mode=pooled_mode,
                     title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                     title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                     color_data=color_data, color_fit=color_fit, color_resid=color_resid)

        sub_rows = 1
        ratios   = [3]
        if show_residuals:
            sub_rows += 1
            ratios.append(1)
        if show_dots_panel:
            sub_rows += 1
            ratios.append(1)

        h_mult = sum(ratios) / 3.0
        fig = Figure(figsize=(ncols * pw, nrows * pw * h_mult * 0.9), dpi=dpi_val)
        super().__init__(fig)
        self._axes = []
        outer = GridSpec(nrows, ncols, figure=fig,
                         hspace=0.50, wspace=0.48,
                         left=0.09, right=0.97, top=0.95, bottom=0.10)

        for k, combo in enumerate(combos):
            ri, ci = divmod(k, ncols)
            inner  = outer[ri, ci].subgridspec(
                sub_rows, 1, height_ratios=ratios, hspace=0.12)
            idx     = 0
            ax_curve = fig.add_subplot(inner[idx]); idx += 1
            ax_resid = fig.add_subplot(inner[idx], sharex=ax_curve) if show_residuals else None
            if show_residuals: idx += 1
            ax_dots  = fig.add_subplot(inner[idx]) if show_dots_panel else None

            _render_compare_combo(ax_curve, ax_dots, combo, runs, mode,
                                  ax_resid=ax_resid, **kw)
            self._axes.append(ax_curve)

        fig.text(0.01, 0.003,
                 "Curves normalised per run using fitted Top & Bottom.  "
                 "Error bars = fit SE (nonlinear regression).  "
                 "Grey band = mean ± SD across runs.",
                 fontsize=4.5, color="gray", va="bottom")

        dpi = fig.dpi
        self.setMinimumSize(int(fig.get_figwidth() * dpi),
                            int(fig.get_figheight() * dpi))
        self.draw()

    def mousePressEvent(self, event):
        w, h = self.width(), self.height()
        x = event.position().x() / w
        y = 1.0 - event.position().y() / h
        for i, ax in enumerate(self._axes):
            bbox = ax.get_position()
            if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1:
                self.plot_clicked.emit(i)
                return
        super().mousePressEvent(event)


class CompareSingleCanvas(FigureCanvas):
    """Single-combo view for Compare Runs, matching grid cell proportions."""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(5.5, 5.0))
        super().__init__(self._fig)

    def show_combo(self, combo, runs, mode,
                   show_dots_panel=True, show_residuals=False,
                   flip_norm=False, pooled_mode=False, show_legend=True,
                   title_fontsize=8, title_fontfamily="sans-serif",
                   title_fontweight="bold", title_fontstyle="normal",
                   color_data="#1e4572", color_fit="#6495ED", color_resid=None,
                   layout_cfg=None, **_extra):
        lc = layout_cfg or {}
        fw = lc.get("fig_w", 5.5)

        sub_rows = 1
        ratios = [3]
        if show_residuals:
            sub_rows += 1; ratios.append(1)
        if show_dots_panel:
            sub_rows += 1; ratios.append(1)

        fh = fw * sum(ratios) / ratios[0] * 0.75
        self._fig.set_size_inches(fw, fh)
        self._fig.clear()

        gs = GridSpec(sub_rows, 1, figure=self._fig, height_ratios=ratios,
                      hspace=lc.get("resid_gap", 0.12))
        idx = 0
        ax_curve = self._fig.add_subplot(gs[idx]); idx += 1
        ax_resid = self._fig.add_subplot(gs[idx], sharex=ax_curve) if show_residuals else None
        if show_residuals: idx += 1
        ax_dots = self._fig.add_subplot(gs[idx]) if show_dots_panel else None

        _render_compare_combo(
            ax_curve, ax_dots, combo, runs, mode,
            ax_resid=ax_resid, show_legend=show_legend,
            flip_norm=flip_norm, pooled_mode=pooled_mode,
            title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
            title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
            color_data=color_data, color_fit=color_fit, color_resid=color_resid)

        self._fig.subplots_adjust(
            left=lc.get("left", 0.12), right=lc.get("right", 0.97),
            top=lc.get("top", 0.90), bottom=lc.get("bottom", 0.12))
        dpi_v = self._fig.dpi
        self.setMinimumSize(int(fw * dpi_v), int(fh * dpi_v))
        self.draw()


class ComparePlotsTab(QWidget):
    """Grid/single view tab for Compare Runs, matching CompPlotsTab pattern."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._combos = []
        self._runs = []
        self._mode = "ki"
        self._current_idx = 0
        self._plot_kw = {}

        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self._view_btn = QPushButton("Switch to Single View")
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._jump = QComboBox()
        self._jump.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl.addWidget(self._view_btn)
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._jump)
        layout.addLayout(ctrl)

        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(False)
        self._grid_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._single_canvas = CompareSingleCanvas()
        self._single_scroll = QScrollArea()
        self._single_scroll.setWidgetResizable(False)
        self._single_scroll.setWidget(self._single_canvas)

        self._stack = QWidget()
        self._stack_lay = QVBoxLayout(self._stack)
        self._stack_lay.setContentsMargins(0, 0, 0, 0)
        self._stack_lay.addWidget(self._grid_scroll)
        layout.addWidget(self._stack)

        self._is_grid = True
        self._prev_btn.setVisible(False)
        self._next_btn.setVisible(False)
        self._jump.setVisible(False)

        self._view_btn.clicked.connect(self._toggle_view)
        self._prev_btn.clicked.connect(lambda: self._navigate(-1))
        self._next_btn.clicked.connect(lambda: self._navigate(1))
        self._jump.currentIndexChanged.connect(self._jump_to)

    def load(self, combos, runs, mode, **plot_kw):
        self._combos = combos
        self._runs = runs
        self._mode = mode
        self._plot_kw = plot_kw
        self._build_grid()
        self._build_jump_list()
        if not self._is_grid and self._combos:
            self._show_single(min(self._current_idx, len(self._combos) - 1))

    def _build_grid(self):
        if not self._combos:
            self._grid_scroll.setWidget(QLabel("No matching combinations."))
            return
        canvas = RunCompareCanvas(self._combos, self._runs, self._mode, **self._plot_kw)
        canvas.plot_clicked.connect(self._on_grid_click)
        self._grid_scroll.setWidget(canvas)

    def _build_jump_list(self):
        self._jump.blockSignals(True)
        self._jump.clear()
        for i, combo in enumerate(self._combos):
            if self._mode == "ki":
                lbl = f"[{i+1}] {combo[0]} | {combo[1]} | {combo[2]}"
            else:
                lbl = f"[{i+1}] {combo[0]} | {combo[1]} [{combo[2]} µM]"
            self._jump.addItem(lbl)
        self._jump.blockSignals(False)

    def _toggle_view(self):
        self._is_grid = not self._is_grid
        # Remove current widget from stack
        while self._stack_lay.count():
            self._stack_lay.takeAt(0).widget().setParent(None)
        if self._is_grid:
            self._view_btn.setText("Switch to Single View")
            self._prev_btn.setVisible(False)
            self._next_btn.setVisible(False)
            self._jump.setVisible(False)
            self._stack_lay.addWidget(self._grid_scroll)
            self._grid_scroll.setVisible(True)
        else:
            self._view_btn.setText("Switch to Grid View")
            self._prev_btn.setVisible(True)
            self._next_btn.setVisible(True)
            self._jump.setVisible(True)
            self._stack_lay.addWidget(self._single_scroll)
            self._single_scroll.setVisible(True)
            self._show_single(0)

    def _on_grid_click(self, idx):
        if idx < len(self._combos):
            self._is_grid = True  # force toggle
            self._toggle_view()   # switches to single
            self._show_single(idx)

    def _show_single(self, idx):
        if not self._combos:
            return
        idx = idx % len(self._combos)
        self._current_idx = idx
        combo = self._combos[idx]
        self._single_canvas.show_combo(
            combo, self._runs, self._mode, **self._plot_kw)
        self._jump.blockSignals(True)
        self._jump.setCurrentIndex(idx)
        self._jump.blockSignals(False)

    def _navigate(self, delta):
        if self._combos:
            self._show_single((self._current_idx + delta) % len(self._combos))

    def _jump_to(self, idx):
        if 0 <= idx < len(self._combos):
            self._show_single(idx)


class CompareRunsWindow(QMainWindow):
    """Compare fitted Ki/Kd curves and values across independent runs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Compare Runs")
        self.resize(1340, 860)
        self._run_items: list = []
        self._runs:      list = []   # {"item", "fit", "raw", "path"}
        self._mode:      str  = "ki"
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        home_btn       = QPushButton("⌂ Home")
        export_btn     = QPushButton("Export Grid PDF…")
        export_ind_btn = QPushButton("Export Individual PDFs…")
        home_btn.clicked.connect(self._on_home)
        export_btn.clicked.connect(self._on_export)
        export_ind_btn.clicked.connect(self._on_export_individual)
        self._status_lbl = QLabel("Add fit_results files to begin.")
        for w in (home_btn, QLabel("  |  "), export_btn, export_ind_btn,
                  QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ── left panel ──────────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(340)
        outer_lay = QVBoxLayout(panel)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QScrollArea.NoFrame)
        outer_lay.addWidget(panel_scroll)
        panel_inner = QWidget()
        panel_scroll.setWidget(panel_inner)
        lay = QVBoxLayout(panel_inner)
        lay.setAlignment(Qt.AlignTop)

        # 1. Run Files
        box_f = QGroupBox("Run Files  (fit_results.xlsx)")
        fl = QVBoxLayout(box_f)
        self._mode_lbl = QLabel("Mode: —")
        self._mode_lbl.setStyleSheet("color: gray; font-size: 10px;")
        fl.addWidget(self._mode_lbl)
        self._file_list_widget = QWidget()
        self._file_list_lay    = QVBoxLayout(self._file_list_widget)
        self._file_list_lay.setAlignment(Qt.AlignTop)
        self._file_list_lay.setSpacing(3)
        sa = QScrollArea()
        sa.setWidget(self._file_list_widget)
        sa.setWidgetResizable(True)
        sa.setFixedHeight(160)
        fl.addWidget(sa)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Files…")
        add_btn.clicked.connect(self._on_add_files)
        add_folder_btn = QPushButton("+ Add Folder…")
        add_folder_btn.clicked.connect(self._on_add_folder)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_folder_btn)
        fl.addLayout(btn_row)
        lay.addWidget(box_f)

        # 2. Filters
        box_flt = QGroupBox("Filters")
        flt_lay = QFormLayout(box_flt)
        self._host_flt  = QComboBox(); self._host_flt.addItem("All")
        self._dye_flt   = QComboBox(); self._dye_flt.addItem("All")
        self._guest_flt = QComboBox(); self._guest_flt.addItem("All")
        self._guest_lbl = QLabel("Guest:")
        self._shared_chk = QCheckBox("Shared combos only (≥ 2 runs)")
        self._shared_chk.setChecked(True)
        self._include_fail_chk = QCheckBox("Include failed fits")
        self._include_fail_chk.setChecked(False)
        self._include_fail_chk.setToolTip(
            "By default only PASS entries are shown.\n"
            "Tick to also include FAIL entries.")
        flt_lay.addRow("Host:",        self._host_flt)
        flt_lay.addRow("Dye:",         self._dye_flt)
        flt_lay.addRow(self._guest_lbl, self._guest_flt)
        flt_lay.addRow(self._shared_chk)
        flt_lay.addRow(self._include_fail_chk)
        lay.addWidget(box_flt)

        # 3. Plot Colors
        box_col = QGroupBox("Plot Colors")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", "#1e4572")
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", "#6495ED")
        self._color_resid_row = ColorPickerRow("Residuals:  ", "#1e4572")
        self._color_data_row.color_changed.connect(self._rebuild)
        self._color_fit_row.color_changed.connect(self._rebuild)
        self._color_resid_row.color_changed.connect(self._rebuild)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        lay.addWidget(box_col)

        # 4. Title Font
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
        self._title_fam.currentIndexChanged.connect(self._rebuild)
        self._title_fsz.valueChanged.connect(self._rebuild)
        self._title_style.currentIndexChanged.connect(self._rebuild)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        tf_lay.addRow("Style:",  self._title_style)
        lay.addWidget(box_tf)

        # 5. Display
        box_p = QGroupBox("Display")
        p_lay = QVBoxLayout(box_p)
        self._dots_chk   = QCheckBox("Show Ki/Kd panel")
        self._dots_chk.setChecked(True)
        self._resid_chk  = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(False)
        self._flip_chk   = QCheckBox("Flip normalisation (invert Y)")
        self._flip_chk.setChecked(True)
        self._legend_chk = QCheckBox("Show per-plot legend")
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On 'Export Individual PDFs', write one PDF per combo\n"
            "into <output>/individual/")

        avg_row = QHBoxLayout()
        self._all_rb    = QRadioButton("Per-run curves")
        self._pooled_rb = QRadioButton("Pooled fit")
        self._all_rb.setChecked(True)
        self._pooled_rb.setToolTip(
            "Pool normalised raw data from all runs and\n"
            "fit a single curve to the combined dataset.")
        avg_row.addWidget(QLabel("Mode:"))
        avg_row.addWidget(self._all_rb)
        avg_row.addWidget(self._pooled_rb)
        avg_row.addStretch()

        p_lay.addWidget(self._dots_chk)
        p_lay.addWidget(self._resid_chk)
        p_lay.addWidget(self._flip_chk)
        p_lay.addWidget(self._legend_chk)
        p_lay.addWidget(self._ind_export_chk)
        p_lay.addLayout(avg_row)
        lay.addWidget(box_p)

        # 6. Figure Layout
        self._layout_panel = _LayoutPanel()
        self._layout_panel.changed.connect(self._rebuild)
        lay.addWidget(self._layout_panel)

        # 7. Output
        box_out = QGroupBox("Output")
        out_lay = QFormLayout(box_out)
        self._output_row = FolderRow("")
        out_lay.addRow("Folder:", self._output_row)
        lay.addWidget(box_out)

        for sig in (self._host_flt.currentIndexChanged,
                    self._dye_flt.currentIndexChanged,
                    self._guest_flt.currentIndexChanged,
                    self._shared_chk.stateChanged,
                    self._include_fail_chk.stateChanged,
                    self._dots_chk.stateChanged,
                    self._resid_chk.stateChanged,
                    self._flip_chk.stateChanged,
                    self._legend_chk.stateChanged,
                    self._all_rb.toggled,
                    self._pooled_rb.toggled):
            sig.connect(self._rebuild)

        splitter.addWidget(panel)

        # ── right: Curves tab + Summary tab ─────────────────────────────────
        self._tabs = QTabWidget()
        self._plots_tab = ComparePlotsTab()
        self._tabs.addTab(self._plots_tab, "Curves")

        self._summary_tbl = QTableWidget()
        self._summary_tbl.setSortingEnabled(True)
        self._summary_tbl.horizontalHeader().setStretchLastSection(True)
        self._summary_tbl.setAlternatingRowColors(True)
        self._tabs.addTab(self._summary_tbl, "Summary")

        splitter.addWidget(self._tabs)
        splitter.setSizes([340, 1000])

    @staticmethod
    def _parse_title_style(style_text: str):
        """Return (fontweight, fontstyle) from a combo label."""
        return {
            "Bold":        ("bold",   "normal"),
            "Normal":      ("normal", "normal"),
            "Italic":      ("normal", "italic"),
            "Bold Italic": ("bold",   "italic"),
        }.get(style_text, ("bold", "normal"))

    def _plot_kw(self) -> dict:
        cfg = self._layout_panel.get_cfg()
        fw, fs = self._parse_title_style(self._title_style.currentText())
        return dict(
            ncols=cfg["pdf_cols"],
            fig_w=cfg["fig_w"],
            show_dots_panel=self._dots_chk.isChecked(),
            show_residuals=self._resid_chk.isChecked(),
            flip_norm=self._flip_chk.isChecked(),
            pooled_mode=self._pooled_rb.isChecked(),
            show_legend=self._legend_chk.isChecked(),
            title_fontsize=self._title_fsz.value(),
            title_fontfamily=self._title_fam.currentText(),
            title_fontweight=fw,
            title_fontstyle=fs,
            color_data=self._color_data_row.color,
            color_fit=self._color_fit_row.color,
            color_resid=self._color_resid_row.color,
            layout_cfg=cfg,
        )

    # ── file management ──────────────────────────────────────────────────────

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select fit_results.xlsx file(s)", "",
            "Excel files (*.xlsx *.xls);;All files (*)")
        if paths:
            self._load_paths(paths)

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing fit_results files")
        if not folder:
            return
        paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".xlsx") and not f.startswith("~$")
        )
        if not paths:
            QMessageBox.information(self, "No files found",
                                    f"No .xlsx files found in:\n{folder}")
            return
        self._load_paths(paths)

    def _load_paths(self, paths: list):
        for path in paths:
            if any(ri.path == path for ri in self._run_items):
                continue
            mode = _detect_compare_mode(path)
            if mode is None:
                QMessageBox.warning(
                    self, "Unrecognised file",
                    f"'{os.path.basename(path)}' does not appear to be a "
                    f"fit_results file.\n\n"
                    f"Expected: pipeline output with PASS/FAIL sheets "
                    f"containing Ki_uM or Kd columns.")
                continue
            if self._runs and mode != self._mode:
                QMessageBox.warning(
                    self, "Mode mismatch",
                    f"'{os.path.basename(path)}' appears to be {mode.upper()} but "
                    f"loaded files are {self._mode.upper()}.\n"
                    "Mix Ki and Kd runs is not supported.")
                continue
            self._mode = mode
            color     = _RUN_COLORS[len(self._run_items) % len(_RUN_COLORS)]
            fit       = _parse_fit_results(path, mode)
            data_path = _find_companion_data(path)
            raw       = (_parse_raw_ki(data_path) if mode == "ki"
                         else _parse_raw_kd(data_path)) if data_path else {}
            item = _RunFileItem(path, color, has_data=bool(raw))
            item.remove_requested.connect(self._on_remove_file)
            item._lbl.editingFinished.connect(self._rebuild)
            item.color_changed.connect(self._rebuild)
            item.data_attached.connect(self._on_data_attached)
            self._file_list_lay.addWidget(item)
            self._run_items.append(item)
            self._runs.append({"item": item, "fit": fit, "raw": raw, "path": path})
        self._update_mode_label()
        self._rebuild_filters()
        self._rebuild()

    def _on_remove_file(self, item):
        idx = next((i for i, ri in enumerate(self._run_items) if ri is item), None)
        if idx is None:
            return
        self._run_items.pop(idx)
        self._runs.pop(idx)
        item.deleteLater()
        if not self._run_items:
            self._mode = "ki"
        self._update_mode_label()
        self._rebuild_filters()
        self._rebuild()

    def _on_data_attached(self, item):
        idx = next((i for i, ri in enumerate(self._run_items) if ri is item), None)
        if idx is None:
            return
        data_path = getattr(item, "_attached_data_path", None)
        if not data_path:
            return
        try:
            raw = (_parse_raw_ki(data_path) if self._mode == "ki"
                   else _parse_raw_kd(data_path))
        except Exception as exc:
            QMessageBox.warning(self, "Data load failed",
                                f"Could not parse data file:\n{exc}")
            return
        if not raw:
            QMessageBox.warning(self, "No data found",
                                "No matching concentration/replicate data "
                                "found in this file.")
            return
        self._runs[idx]["raw"] = raw
        item.set_has_data(True)
        self._rebuild()

    def _update_mode_label(self):
        if not self._runs:
            self._mode_lbl.setText("Mode: —")
            self._guest_lbl.setText("Guest:")
        elif self._mode == "ki":
            self._mode_lbl.setText("Mode: Competitive (Ki)")
            self._guest_lbl.setText("Guest:")
        else:
            self._mode_lbl.setText("Mode: Direct binding (Kd)")
            self._guest_lbl.setText("Dye conc:")

    # ── filter / rebuild helpers ──────────────────────────────────────────────

    def _all_keys_with_counts(self) -> dict:
        include_fail = self._include_fail_chk.isChecked()
        result: dict = {}
        for run in self._runs:
            for key, row in run["fit"].items():
                if not include_fail and row.get("_status") == "FAIL":
                    continue
                result[key] = result.get(key, 0) + 1
        return result

    def _rebuild_filters(self):
        kc    = self._all_keys_with_counts()
        keys  = list(kc.keys())
        hosts  = sorted({k[0] for k in keys})
        dyes   = sorted({k[1] for k in keys})
        guests = sorted({k[2] for k in keys})
        for combo_w, items in ((self._host_flt,  hosts),
                               (self._dye_flt,   dyes),
                               (self._guest_flt, guests)):
            cur = combo_w.currentText()
            combo_w.blockSignals(True)
            combo_w.clear()
            combo_w.addItem("All")
            combo_w.addItems(items)
            idx = combo_w.findText(cur)
            combo_w.setCurrentIndex(max(0, idx))
            combo_w.blockSignals(False)

    def _filtered_combos(self) -> list:
        kc      = self._all_keys_with_counts()
        host_f  = self._host_flt.currentText()
        dye_f   = self._dye_flt.currentText()
        guest_f = self._guest_flt.currentText()
        min_r   = 2 if self._shared_chk.isChecked() else 1
        return sorted([
            k for k, n in kc.items()
            if n >= min_r
            and (host_f  == "All" or k[0] == host_f)
            and (dye_f   == "All" or k[1] == dye_f)
            and (guest_f == "All" or k[2] == guest_f)
        ])

    def _active_runs(self) -> list:
        include_fail = self._include_fail_chk.isChecked()
        result = []
        for ri, run in zip(self._run_items, self._runs):
            fit = run["fit"]
            if not include_fail:
                fit = {k: v for k, v in fit.items()
                       if v.get("_status") != "FAIL"}
            result.append({"label": ri.label, "color": ri.color,
                           "fit": fit, "raw": run["raw"]})
        return result

    def _rebuild(self, *_):
        if not self._runs:
            self._plots_tab.load([], [], self._mode, **self._plot_kw())
            self._status_lbl.setText("Add fit_results files to begin.")
            return
        self._rebuild_filters()
        combos = self._filtered_combos()
        if not combos:
            msg = "No matching combinations."
            if self._shared_chk.isChecked():
                msg += "  Try unchecking 'Shared combos only'."
            self._plots_tab.load([], [], self._mode, **self._plot_kw())
            self._status_lbl.setText(msg)
            return

        runs = self._active_runs()
        self._plots_tab.load(combos, runs, self._mode, **self._plot_kw())
        self._rebuild_summary(combos, runs)
        mode_str = "Ki" if self._mode == "ki" else "Kd"
        self._status_lbl.setText(
            f"{len(combos)} combo(s)  |  {len(runs)} run(s)  |  mode: {mode_str}")

    # ── summary table ─────────────────────────────────────────────────────────

    def _rebuild_summary(self, combos: list, runs: list):
        from PySide6.QtGui import QColor
        run_labels = [r["label"] for r in runs]
        val_lbl    = "Ki (µM)" if self._mode == "ki" else "Kd (µM)"
        _mkey      = "Best_Model" if self._mode == "ki" else "Model"
        run_hdrs   = []
        for lbl in run_labels:
            run_hdrs += [f"{lbl}\n{val_lbl}", f"{lbl}\nfit SE", f"{lbl}\nR²adj"]
        headers = ["Combination"] + run_hdrs + ["Mean", "SD", "CV%", "n", "Models"]
        self._summary_tbl.setRowCount(len(combos))
        self._summary_tbl.setColumnCount(len(headers))
        self._summary_tbl.setHorizontalHeaderLabels(headers)
        self._summary_tbl.setSortingEnabled(False)
        warn_bg = QColor("#fff3cd")
        warn_fg = QColor("#856404")

        for ri, combo in enumerate(combos):
            if self._mode == "ki":
                h, d, g = combo
                combo_str = f"{h} | {d} | {g}"
            else:
                h, d, dc = combo
                combo_str = f"{h} | {d} [{dc} µM]"
            self._summary_tbl.setItem(ri, 0, QTableWidgetItem(combo_str))

            vals: list = []
            models_seen: list = []
            for ci, run in enumerate(runs):
                fr   = run["fit"].get(combo)
                base = 1 + ci * 3
                if fr is None:
                    for j in range(3):
                        self._summary_tbl.setItem(ri, base + j, QTableWidgetItem("—"))
                    continue
                models_seen.append(str(fr.get(_mkey, "?")))
                try:
                    v  = float(fr.get("Ki_uM",    np.nan) if self._mode == "ki"
                               else fr.get("Kd",    np.nan))
                    se = float(fr.get("Ki_err_uM", np.nan) if self._mode == "ki"
                               else fr.get("Kd_SE", np.nan))
                    r2 = float(fr.get("R2_adj", np.nan))
                except Exception:
                    v = se = r2 = np.nan
                def _g(x, dp=3): return f"{x:.{dp}g}" if not np.isnan(x) else "—"
                self._summary_tbl.setItem(ri, base,     QTableWidgetItem(_g(v)))
                self._summary_tbl.setItem(ri, base + 1, QTableWidgetItem(_g(se)))
                self._summary_tbl.setItem(ri, base + 2, QTableWidgetItem(_g(r2, 3)))
                if not np.isnan(v):
                    vals.append(v)

            unique_models   = sorted(set(models_seen))
            model_mismatch  = len(unique_models) > 1
            model_str       = ", ".join(unique_models) if unique_models else "—"

            bs = 1 + len(runs) * 3
            if vals:
                mean = float(np.mean(vals))
                sd   = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
                cv   = sd / mean * 100 if (not np.isnan(sd) and mean) else np.nan
                self._summary_tbl.setItem(ri, bs,     QTableWidgetItem(f"{mean:.3g}"))
                self._summary_tbl.setItem(ri, bs + 1, QTableWidgetItem(f"{sd:.3g}" if not np.isnan(sd) else "—"))
                self._summary_tbl.setItem(ri, bs + 2, QTableWidgetItem(f"{cv:.1f}%" if not np.isnan(cv) else "—"))
                self._summary_tbl.setItem(ri, bs + 3, QTableWidgetItem(str(len(vals))))
            else:
                for j in range(4):
                    self._summary_tbl.setItem(ri, bs + j, QTableWidgetItem("—"))

            mdl_item = QTableWidgetItem(
                ("⚠ " if model_mismatch else "") + model_str)
            if model_mismatch:
                mdl_item.setForeground(warn_fg)
            self._summary_tbl.setItem(ri, bs + 4, mdl_item)

            if model_mismatch:
                for ci in range(len(headers)):
                    it = self._summary_tbl.item(ri, ci)
                    if it:
                        it.setBackground(warn_bg)

        self._summary_tbl.setSortingEnabled(True)
        self._summary_tbl.resizeColumnsToContents()

    # ── PDF export ────────────────────────────────────────────────────────────

    def _on_export(self):
        combos = self._filtered_combos()
        if not combos:
            QMessageBox.warning(self, "Nothing to export",
                                "Build the grid first.")
            return

        out_dir = self._output_row.path
        default = (os.path.join(out_dir, "compare_runs.pdf")
                   if out_dir else "compare_runs.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Grid PDF", default,
            "PDF (*.pdf);;All files (*)")
        if not path:
            return

        try:
            runs = self._active_runs()
            kw = self._plot_kw()
            kw.pop("layout_cfg", None)
            canvas = RunCompareCanvas(
                combos, runs, self._mode, dpi_val=300, **kw)
            with PdfPages(path) as pdf:
                pdf.savefig(canvas.figure, bbox_inches="tight", facecolor="white")
            del canvas
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_export_individual(self):
        import re as _re
        combos = self._filtered_combos()
        if not combos:
            QMessageBox.warning(self, "Nothing to export",
                                "Build the grid first.")
            return

        out_dir = self._output_row.path
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(
                self, "Select output folder for individual PDFs")
            if not out_dir:
                return
            self._output_row.edit.setText(out_dir)

        ind_dir = os.path.join(out_dir, "individual")
        os.makedirs(ind_dir, exist_ok=True)

        runs = self._active_runs()
        kw = self._plot_kw()
        cfg = kw.get("layout_cfg") or {}
        fw  = cfg.get("fig_w", 5.5)
        show_dots = kw["show_dots_panel"]
        show_resid = kw["show_residuals"]

        sub_rows = 1
        ratios = [3]
        if show_resid:
            sub_rows += 1; ratios.append(1)
        if show_dots:
            sub_rows += 1; ratios.append(1)
        fh = fw * sum(ratios) / ratios[0] * 0.75

        n_written = 0
        try:
            for combo in combos:
                fig = Figure(figsize=(fw, fh), dpi=300)
                gs = GridSpec(sub_rows, 1, figure=fig, height_ratios=ratios,
                              hspace=cfg.get("resid_gap", 0.12))
                idx = 0
                ax_curve = fig.add_subplot(gs[idx]); idx += 1
                ax_resid = (fig.add_subplot(gs[idx], sharex=ax_curve)
                            if show_resid else None)
                if show_resid: idx += 1
                ax_dots = fig.add_subplot(gs[idx]) if show_dots else None

                _render_compare_combo(
                    ax_curve, ax_dots, combo, runs, self._mode,
                    ax_resid=ax_resid, show_legend=kw["show_legend"],
                    flip_norm=kw["flip_norm"], pooled_mode=kw["pooled_mode"],
                    title_fontsize=kw["title_fontsize"],
                    title_fontfamily=kw["title_fontfamily"],
                    title_fontweight=kw["title_fontweight"],
                    title_fontstyle=kw["title_fontstyle"],
                    color_data=kw["color_data"], color_fit=kw["color_fit"],
                    color_resid=kw["color_resid"])
                fig.subplots_adjust(
                    left=cfg.get("left", 0.12), right=cfg.get("right", 0.97),
                    top=cfg.get("top", 0.90), bottom=cfg.get("bottom", 0.12))

                if self._mode == "ki":
                    h, d, g = combo
                    stem = f"{h}_{d}_{g}"
                else:
                    h, d, dc = combo
                    stem = f"{h}_{d}_{dc}uM"
                safe_nm = _re.sub(r"[^\w\-]", "_", stem)
                out_path = os.path.join(ind_dir, f"{safe_nm}.pdf")
                with PdfPages(out_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                n_written += 1
            QMessageBox.information(
                self, "Exported",
                f"{n_written} individual PDF(s) written to:\n{ind_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_home(self):
        _go_home(self)


# ── Shared home navigation ─────────────────────────────────────────────────────

def _make_window(choice: str) -> QMainWindow:
    if choice == "direct":
        return DirectMainWindow()
    if choice == "competitive":
        return CompMainWindow()
    if choice == "spectral":
        return SpectralMainWindow()
    if choice == "compare":
        return CompareRunsWindow()
    return HeatmapWindow()


def _go_home(current_window: QMainWindow):
    """Show ModePicker; if user picks a mode, open it and close current window."""
    picker = ModePicker(parent=current_window)
    if picker.exec() != QDialog.Accepted or not picker.choice:
        return
    try:
        win = _make_window(picker.choice)
        win.show()
        app = QApplication.instance()
        if not hasattr(app, "_extra_windows"):
            app._extra_windows = []
        app._extra_windows.append(win)
    except Exception as exc:
        QMessageBox.critical(current_window, "Error opening window",
                             f"{type(exc).__name__}: {exc}")
    current_window.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    picker = ModePicker()
    if picker.exec() != QDialog.Accepted or picker.choice is None:
        sys.exit(0)

    win = _make_window(picker.choice)
    app._extra_windows = [win]
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
