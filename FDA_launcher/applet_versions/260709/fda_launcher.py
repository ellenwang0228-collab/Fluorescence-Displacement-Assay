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
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import seaborn as sns

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
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

import layout_utils

_BASE_DIR = Path(__file__).parent

# Colours for Compare Runs (up to 10 runs)
_RUN_COLORS = ["#6495ED", "#8B0000", "#2ca02c", "#ff7f0e",
               "#9467bd", "#8c564b", "#e377c2", "#17becf",
               "#bcbd22", "#7f7f7f"]


def _fmt_uM(v: float) -> str:
    """Format a Kd/Ki value in µM: fixed-point when compact, scientific when >= 5 digits."""
    if np.isnan(v):
        return "NaN"
    if abs(v) >= 10000:
        return f"{v:.4E}"
    return f"{v:.2f}"


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

        lo = min(Top, Bottom)
        y_norm = (fn(x, *popt) - lo) / abs(span)
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


def _pooled_ki_fit(combo: tuple, runs: list):
    """Run AICc model selection on pooled normalised Ki data for one combo.

    Returns dict with keys: ki, ki_err, r2_adj, model, cp_shift, popt, fn
    or None if the fit fails or insufficient data.
    """
    from scipy.optimize import curve_fit as _curve_fit

    all_x, pooled_x, pooled_y = [], [], []
    for run in runs:
        fit_row = run["fit"].get(combo)
        if fit_row is None:
            continue
        rd = run["raw"].get(combo)
        if rd is None:
            continue
        try:
            top    = float(fit_row.get("Top_fit",    np.nan))
            bottom = float(fit_row.get("Bottom_fit", np.nan))
            span   = top - bottom
            if np.isnan(span) or abs(span) < 1e-10:
                continue
            lo = min(top, bottom)
            xv = rd["x_log"]
            all_x.extend(xv.tolist())
            norm_reps = []
            for rep in rd["reps"]:
                if len(rep) == len(xv):
                    norm_reps.append((rep.astype(float) - lo) / abs(span))
            if norm_reps:
                pooled_x.extend(xv.tolist())
                pooled_y.extend(np.mean(norm_reps, axis=0).tolist())
        except Exception:
            continue

    if len(pooled_x) < 4:
        return None

    px = np.asarray(pooled_x)
    py = np.asarray(pooled_y)
    mask = np.isfinite(px) & np.isfinite(py)
    px, py = px[mask], py[mask]
    n_pts = len(py)
    if n_pts < 4:
        return None

    top0, bot0 = float(max(py)), float(min(py))
    med_x = float(np.median(px))

    hs_vals, hc_val = [], np.nan
    lk1_vals, lk2_vals, frac_vals, cp_shift = [], [], [], 0.0
    for run in runs:
        fr = run["fit"].get(combo)
        if fr:
            try: hs_vals.append(abs(float(fr["HillSlope"])))
            except (TypeError, ValueError, KeyError): pass
            try:
                hc = float(fr["HostConc_uM"])
                if not np.isnan(hc): hc_val = hc
            except (TypeError, ValueError, KeyError): pass
            try: lk1_vals.append(float(fr["logKi"]))
            except (TypeError, ValueError, KeyError): pass
            try: lk2_vals.append(float(fr["logKi2"]))
            except (TypeError, ValueError, KeyError): pass
            try: frac_vals.append(float(fr["Frac_site1"]))
            except (TypeError, ValueError, KeyError): pass
            try:
                dc = float(fr["DyeConc_uM"])
                dk = float(fr["DyeKd_uM"])
                if dk > 0:
                    cp_shift = np.log10(1.0 + dc / dk)
            except (TypeError, ValueError, KeyError): pass

    hs0    = float(np.mean(hs_vals)) if hs_vals else 1.0
    lEC1_0 = (float(np.mean(lk1_vals)) + cp_shift) if lk1_vals else med_x - 1
    lEC2_0 = (float(np.mean(lk2_vals)) + cp_shift) if lk2_vals else med_x + 1
    frac_0 = float(np.mean(frac_vals)) if frac_vals else 0.5

    def _aicc(n, rss, k):
        if n <= k + 1: return np.inf
        if rss <= 0:   return -np.inf
        return n * np.log(rss / n) + 2.0*k + (2.0*k*(k+1)) / (n - k - 1)

    def _std(x, top, bot, logEC50):
        return bot + (top - bot) / (1.0 + 10.0 ** (x - logEC50))
    def _hill(x, top, bot, logEC50, hs):
        return bot + (top - bot) / (1.0 + 10.0 ** (hs * (x - logEC50)))
    def _biph(x, top, bot, logEC1, logEC2, frac_s):
        frac_s = np.clip(frac_s, 0.0, 1.0)
        return bot + (top - bot) * (
            frac_s / (1.0 + 10.0 ** (x - logEC1)) +
            (1.0 - frac_s) / (1.0 + 10.0 ** (x - logEC2)))

    # Preliminary Standard fit — gates Morrison exactly as pipeline_ki does:
    # Morrison only enters the candidate set when Ki_std < MORRISON_GATE * [Host].
    _std_Ki   = np.nan
    _std_pfit = None
    try:
        _po_s, _pc_s = _curve_fit(_std, px, py, p0=[top0, bot0, med_x], maxfev=5000)
        _std_Ki   = 10.0 ** (float(_po_s[2]) - cp_shift)
        _std_pfit = (_po_s, _pc_s)
    except (RuntimeError, ValueError):
        pass

    _use_morrison = (not np.isnan(hc_val) and not np.isnan(_std_Ki)
                     and _std_Ki < pl_ki.MORRISON_GATE_DEFAULT * hc_val)

    candidates = [
        {"name": "Standard",  "fn": _std,  "k": 3, "p0": [top0, bot0, med_x],
         "prefit": _std_pfit},
        {"name": "HillSlope", "fn": _hill, "k": 4, "p0": [top0, bot0, med_x, hs0]},
    ]
    if pl_ki.BIPHASIC_ENABLED:
        candidates.append({"name": "Biphasic", "fn": _biph, "k": 5,
                            "p0": [top0, bot0, lEC1_0, lEC2_0, frac_0]})
    if _use_morrison:
        _hc = hc_val
        def _morr(x, top, bot, logKi_app):
            Ki_app = 10.0 ** logKi_app
            I_T    = 10.0 ** x
            A      = _hc + I_T + Ki_app
            disc   = np.maximum(A**2 - 4.0 * _hc * I_T, 0.0)
            frac_b = np.clip((A - np.sqrt(disc)) / (2.0 * _hc), 0.0, 1.0)
            return top - (top - bot) * frac_b
        candidates.append({"name": "Morrison", "fn": _morr, "k": 3,
                           "p0": [top0, bot0, med_x]})

    x_lo_pool = float(np.min(px))
    x_hi_pool = float(np.max(px))

    best = None
    for cand in candidates:
        if n_pts <= cand["k"]:
            continue
        prefit = cand.get("prefit")
        if prefit is not None:
            po, pc = prefit
        else:
            try:
                po, pc = _curve_fit(cand["fn"], px, py, p0=cand["p0"], maxfev=5000)
            except (RuntimeError, ValueError):
                continue
        if cand["name"] == "Morrison":
            Ki_fit = 10.0 ** (float(po[2]) - cp_shift)
            if Ki_fit < pl_ki.MORRISON_KI_FRAC_DEFAULT * hc_val:
                continue
            if pl_ki._transition_span(cand["fn"], po, x_lo_pool, x_hi_pool) < 1.0:
                continue
        yp    = cand["fn"](px, *po)
        rss   = float(np.sum((py - yp) ** 2))
        ss_t  = float(np.sum((py - np.mean(py)) ** 2))
        r2    = 1.0 - rss / ss_t if ss_t > 0 else 0.0
        r2a   = (1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - cand["k"] - 1)
                 if n_pts > cand["k"] + 1 else r2)
        aic_v = _aicc(n_pts, rss, cand["k"])
        if best is None or aic_v < best["aic"]:
            best = {"fn": cand["fn"], "popt": po, "pcov": pc, "aic": aic_v,
                    "name": cand["name"], "k": cand["k"], "r2_adj": r2a}

    if best is None:
        return None

    logEC  = float(best["popt"][2])
    ki_val = 10.0 ** (logEC - cp_shift)
    try:
        se_logEC = float(np.sqrt(best["pcov"][2, 2]))
        ki_err   = ki_val * np.log(10) * se_logEC
    except Exception:
        ki_err = np.nan

    return {"ki": ki_val, "ki_err": ki_err, "r2_adj": best["r2_adj"],
            "model": best["name"], "cp_shift": cp_shift,
            "popt": best["popt"], "fn": best["fn"]}


def _render_compare_combo(ax_curve, ax_dots, combo: tuple, runs: list, mode: str,
                          show_legend: bool = False, flip_norm: bool = False,
                          pooled_mode: bool = False, ax_resid=None,
                          title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                          title_fontweight: str = "normal", title_fontstyle: str = "normal",
                          axis_fontsize: float = 14, tick_fontsize: float = 12,
                          color_data: str = "#1e4572", color_fit: str = "#6495ED",
                          color_resid: str = None,
                          normalise_y: bool = False, sci_notation_y: bool = False,
                          lw_fit: float = 2.0, ms_data: float = 4.0,
                          lw_errorbar: float = 1.5, ms_resid: float = 2.5):
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
                        lo = min(top, bottom)
                        xv = rd["x_log"]
                        norm_reps = []
                        for rep in rd["reps"]:
                            if len(rep) == len(xv):
                                yn = (rep.astype(float) - lo) / abs(span)
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
                                                  ecolor=color, elinewidth=lw_errorbar,
                                                  markersize=ms_data, capsize=2,
                                                  alpha=0.7, zorder=2)
                                pooled_x.extend(xv.tolist())
                                pooled_y.extend(run_mean.tolist())
                            else:
                                for yn in norm_reps:
                                    ax_curve.scatter(xv, yn, color=color,
                                                     alpha=0.22, s=ms_data**2 * 0.44,
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
                                                  ecolor=color, elinewidth=lw_errorbar,
                                                  markersize=ms_data, capsize=2,
                                                  alpha=0.7, zorder=2)
                                pooled_x.extend(xv.tolist())
                                pooled_y.extend(run_mean.tolist())
                            else:
                                for yn in norm_reps:
                                    ax_curve.scatter(xv, yn, color=color,
                                                     alpha=0.22, s=ms_data**2 * 0.44,
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
                line, = ax_curve.plot(xc, yc, color=color, lw=lw_fit, zorder=3)
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
                                    lo_     = min(top_, bot_)
                                    yn_rep  = (rep.astype(float) - lo_) / abs(sp_)
                                else:
                                    yn_rep = rep.astype(float) / float(fit_row.get("Bmax", 1.0))
                                if flip_norm: yn_rep = 1.0 - yn_rep
                                y_pred = np.interp(xv, xc, yc)
                                resid  = yn_rep - y_pred
                                ax_resid.scatter(xv, resid, color=color,
                                                 alpha=0.3, s=ms_resid**2 * 0.8, linewidths=0, zorder=2)
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

    _pool_ki = _pool_ki_err = _pool_r2a = np.nan
    _pool_model = None

    # ── pooled-mode: fit a single curve to run-level means (AICc selection) ──
    if pooled_mode and len(pooled_x) >= 4:
        px = np.asarray(pooled_x)
        py = np.asarray(pooled_y)
        mask = np.isfinite(px) & np.isfinite(py)
        px, py = px[mask], py[mask]
        try:
            if mode == "ki":
                n_pts = len(py)
                top0, bot0 = float(max(py)), float(min(py))
                med_x = float(np.median(px))

                hs_vals, hc_val = [], np.nan
                lk1_vals, lk2_vals, frac_vals, cp_shift = [], [], [], 0.0
                for run in runs:
                    fr = run["fit"].get(combo)
                    if fr:
                        try: hs_vals.append(abs(float(fr["HillSlope"])))
                        except (TypeError, ValueError, KeyError): pass
                        try:
                            hc = float(fr["HostConc_uM"])
                            if not np.isnan(hc): hc_val = hc
                        except (TypeError, ValueError, KeyError): pass
                        try: lk1_vals.append(float(fr["logKi"]))
                        except (TypeError, ValueError, KeyError): pass
                        try: lk2_vals.append(float(fr["logKi2"]))
                        except (TypeError, ValueError, KeyError): pass
                        try: frac_vals.append(float(fr["Frac_site1"]))
                        except (TypeError, ValueError, KeyError): pass
                        try:
                            dc = float(fr["DyeConc_uM"])
                            dk = float(fr["DyeKd_uM"])
                            if dk > 0:
                                cp_shift = np.log10(1.0 + dc / dk)
                        except (TypeError, ValueError, KeyError): pass

                hs0    = float(np.mean(hs_vals)) if hs_vals else 1.0
                lEC1_0 = (float(np.mean(lk1_vals)) + cp_shift) if lk1_vals else med_x - 1
                lEC2_0 = (float(np.mean(lk2_vals)) + cp_shift) if lk2_vals else med_x + 1
                frac_0 = float(np.mean(frac_vals)) if frac_vals else 0.5

                def _pool_aicc(n, rss, k):
                    if n <= k + 1: return np.inf
                    if rss <= 0:   return -np.inf
                    return n * np.log(rss / n) + 2.0*k + (2.0*k*(k+1)) / (n - k - 1)

                def _p_std(x, top, bot, logEC50):
                    return bot + (top - bot) / (1.0 + 10.0 ** (x - logEC50))

                def _p_hill(x, top, bot, logEC50, hs):
                    return bot + (top - bot) / (1.0 + 10.0 ** (hs * (x - logEC50)))

                def _p_biph(x, top, bot, logEC1, logEC2, frac_s):
                    frac_s = np.clip(frac_s, 0.0, 1.0)
                    return bot + (top - bot) * (
                        frac_s       / (1.0 + 10.0 ** (x - logEC1)) +
                        (1.0 - frac_s) / (1.0 + 10.0 ** (x - logEC2)))

                # Preliminary Standard fit — gates Morrison exactly as pipeline_ki does.
                _std_Ki_pool   = np.nan
                _std_pfit_pool = None
                try:
                    _po_s, _pc_s = _curve_fit(_p_std, px, py,
                                              p0=[top0, bot0, med_x], maxfev=5000)
                    _std_Ki_pool   = 10.0 ** (float(_po_s[2]) - cp_shift)
                    _std_pfit_pool = (_po_s, _pc_s)
                except (RuntimeError, ValueError):
                    pass

                _use_morrison = (not np.isnan(hc_val) and not np.isnan(_std_Ki_pool)
                                 and _std_Ki_pool < pl_ki.MORRISON_GATE_DEFAULT * hc_val)

                _candidates = [
                    {"name": "Standard",  "fn": _p_std,  "k": 3,
                     "p0": [top0, bot0, med_x], "prefit": _std_pfit_pool},
                    {"name": "HillSlope", "fn": _p_hill, "k": 4,
                     "p0": [top0, bot0, med_x, hs0]},
                ]
                if pl_ki.BIPHASIC_ENABLED:
                    _candidates.append(
                        {"name": "Biphasic", "fn": _p_biph, "k": 5,
                         "p0": [top0, bot0, lEC1_0, lEC2_0, frac_0]})
                if _use_morrison:
                    _hc_closed = hc_val
                    def _p_morr(x, top, bot, logKi_app):
                        Ki_app = 10.0 ** logKi_app
                        I_T    = 10.0 ** x
                        A      = _hc_closed + I_T + Ki_app
                        disc   = np.maximum(A**2 - 4.0 * _hc_closed * I_T, 0.0)
                        frac_b = np.clip((A - np.sqrt(disc)) / (2.0 * _hc_closed),
                                         0.0, 1.0)
                        return top - (top - bot) * frac_b
                    _candidates.append(
                        {"name": "Morrison", "fn": _p_morr, "k": 3,
                         "p0": [top0, bot0, med_x]})

                _best_pool = None
                for _cand in _candidates:
                    if n_pts <= _cand["k"]:
                        continue
                    _prefit = _cand.get("prefit")
                    if _prefit is not None:
                        _po, _pc = _prefit
                    else:
                        try:
                            _po, _pc = _curve_fit(_cand["fn"], px, py,
                                                  p0=_cand["p0"], maxfev=5000)
                        except (RuntimeError, ValueError):
                            continue
                    if _cand["name"] == "Morrison":
                        _Ki_fit = 10.0 ** (float(_po[2]) - cp_shift)
                        if _Ki_fit < pl_ki.MORRISON_KI_FRAC_DEFAULT * hc_val:
                            continue
                        if pl_ki._transition_span(_cand["fn"], _po, x_lo, x_hi) < 1.0:
                            continue
                    _yp  = _cand["fn"](px, *_po)
                    _rss = float(np.sum((py - _yp) ** 2))
                    _aic = _pool_aicc(n_pts, _rss, _cand["k"])
                    _ss_tot = float(np.sum((py - np.mean(py)) ** 2))
                    _r2 = 1.0 - _rss / _ss_tot if _ss_tot > 0 else 0.0
                    _r2a = (1.0 - (1.0 - _r2) * (n_pts - 1) / (n_pts - _cand["k"] - 1)
                            if n_pts > _cand["k"] + 1 else _r2)
                    if _best_pool is None or _aic < _best_pool["aic"]:
                        _best_pool = {"fn": _cand["fn"], "popt": _po, "pcov": _pc,
                                      "aic": _aic, "name": _cand["name"],
                                      "k": _cand["k"], "r2_adj": _r2a,
                                      "rss": _rss, "cp_shift": cp_shift}

                if _best_pool is None:
                    raise RuntimeError("all pooled models failed")

                _pool_fn = _best_pool["fn"]
                popt     = _best_pool["popt"]
                xd = np.linspace(x_lo - 0.5, x_hi + 0.5, 300)
                yd = _pool_fn(xd, *popt)

                _pool_logEC = float(popt[2])
                _pool_ki    = 10.0 ** (_pool_logEC - cp_shift)
                try:
                    _se_logEC = float(np.sqrt(_best_pool["pcov"][2, 2]))
                    _pool_ki_err = _pool_ki * np.log(10) * _se_logEC
                except Exception:
                    _pool_ki_err = np.nan
                _pool_r2a   = _best_pool["r2_adj"]
                _pool_model = _best_pool["name"]
            else:
                def _pool_fn(x, top, kd):
                    return top * x / (kd + x)
                p0 = [max(py), np.median(px)]
                popt, pcov = _curve_fit(_pool_fn, px, py, p0=p0, maxfev=5000)
                xd = np.linspace(0, x_hi * 1.5, 300)
                yd = _pool_fn(xd, *popt)
            ax_curve.plot(xd, yd, color=color_fit, lw=lw_fit, zorder=3)
            if ax_resid is not None:
                y_pred = _pool_fn(px, *popt)
                resid = py - y_pred
                ax_resid.scatter(px, resid, color=_resid_col, alpha=0.5, s=ms_resid**2 * 1.28,
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
        ax_resid.tick_params(labelsize=tick_fontsize)
        ax_resid.yaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True))
        ax_resid.set_ylabel("Resid.", fontsize=axis_fontsize)
        ax_resid.set_xlabel(
            "log[Guest] (µM)" if mode == "ki" else "[Host] (µM)",
            fontsize=axis_fontsize)

    # ── curve panel cosmetics ──
    ax_curve.set_ylim(-0.18, 1.28)
    ax_curve.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)
    ax_curve.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)
    ax_curve.tick_params(labelsize=tick_fontsize)
    ax_curve.xaxis.set_major_locator(MultipleLocator(1))
    ax_curve.set_ylabel("Norm. response", fontsize=axis_fontsize)
    if ax_resid is None:
        ax_curve.set_xlabel(
            "log[Guest] (µM)" if mode == "ki" else "[Host] (µM)",
            fontsize=axis_fontsize)
    else:
        ax_curve.tick_params(labelbottom=False)

    # ── compute title stats ──
    _mkey = "Best_Model" if mode == "ki" else "Model"
    _models_seen = [str(run["fit"][combo].get(_mkey, "?"))
                    for run in runs if combo in run["fit"]]
    _unique_models = sorted(set(_models_seen))
    model_str = ", ".join(_unique_models) if _unique_models else "?"

    # Title with stats (matching render_plot_ki format)
    tkw = dict(fontsize=title_fontsize, fontfamily=title_fontfamily,
               fontweight=title_fontweight, fontstyle=title_fontstyle)

    _has_pooled = _pool_model is not None

    if mode == "ki":
        h, d, g = combo
        id_line = f"{h} | {d} | {g}"
        if ki_vals:
            mean_v = float(np.mean(ki_vals))
            sd_v = float(np.std(ki_vals, ddof=1)) if len(ki_vals) >= 2 else np.nan
            sd_str = f" ± {_fmt_uM(sd_v)}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs if combo in run["fit"]]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Ki_mean = {_fmt_uM(mean_v)}{sd_str} µM (n={len(ki_vals)})  |  {r2_str}"
            if _has_pooled:
                _pe = f" ± {_fmt_uM(_pool_ki_err)}" if not np.isnan(_pool_ki_err) else ""
                stats_line += f"\nKi_pooled = {_fmt_uM(_pool_ki)}{_pe} µM  |  R²adj = {_pool_r2a:.3f}"
        else:
            stats_line = "no Ki data"
        model_line = f"[{model_str}]"
        if _has_pooled:
            model_line += f"  pooled: [{_pool_model}]"
        ax_curve.set_title(f"{id_line}\n{stats_line}\n{model_line}", **tkw)
    else:
        h, d, dc = combo
        id_line = f"{h} – {d}  [{dc} µM]"
        if ki_vals:  # ki_vals holds Kd values in kd mode
            mean_v = float(np.mean(ki_vals))
            sd_v = float(np.std(ki_vals, ddof=1)) if len(ki_vals) >= 2 else np.nan
            sd_str = f" ± {_fmt_uM(sd_v)}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs if combo in run["fit"]]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Kd = {_fmt_uM(mean_v)}{sd_str} µM (n={len(ki_vals)})  |  {r2_str}"
        else:
            stats_line = "no Kd data"
        ax_curve.set_title(f"{id_line}\n{stats_line}  |  [{model_str}]", **tkw)

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
            fontsize=tick_fontsize * 0.75, rotation=30, ha="right")
        ax_dots.set_xlim(-0.6, len(ki_vals) - 0.4)
        ax_dots.set_ylabel("Ki (µM)" if mode == "ki" else "Kd (µM)",
                           fontsize=axis_fontsize)
        ax_dots.tick_params(axis="y", labelsize=tick_fontsize)
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


def _parse_title_style(style_text: str) -> tuple[str, str]:
    return {
        "Bold":        ("bold",   "normal"),
        "Normal":      ("normal", "normal"),
        "Italic":      ("normal", "italic"),
        "Bold Italic": ("bold",   "italic"),
    }.get(style_text, ("bold", "normal"))


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

# ── Summary Plots standalone window ──────────────────────────────────────────

class HeatmapWindow(QMainWindow):
    """
    Load one or more pipeline output Excel files (or a folder of them)
    and generate pKd / pKi summary plots for export as PDF.

    Accepts two formats automatically:
      • Pipeline output  — sheets 'PASS' / 'FAIL' with Kd or Ki_uM columns
      • Kd_Tables_Gen    — sheets 'Kd_Direct' / 'Ki_Competition' (AVERAGE_Kd/Ki)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Summary Plots")
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
        self._scroll.setWidgetResizable(False)
        rl.addWidget(self._scroll)
        splitter.addWidget(right)
        splitter.setSizes([290, 960])

    def _build_config(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(290)
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
        self._heat_rb   = QRadioButton("Heatmap")
        self._bar_rb    = QRadioButton("Bar chart")
        self._spider_rb = QRadioButton("Spider")
        self._heat_rb.setChecked(True)
        type_row.addWidget(self._heat_rb)
        type_row.addWidget(self._bar_rb)
        type_row.addWidget(self._spider_rb)
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
            ["By median (tightest first)", "Alphabetical", "By Dye", "By Guest"])
        self._tile_w_spin = QDoubleSpinBox()
        self._tile_w_spin.setRange(0.3, 5.0)
        self._tile_w_spin.setValue(1.2)
        self._tile_w_spin.setSingleStep(0.1)
        self._tile_w_spin.setDecimals(1)
        self._tile_h_spin = QDoubleSpinBox()
        self._tile_h_spin.setRange(0.3, 5.0)
        self._tile_h_spin.setValue(0.8)
        self._tile_h_spin.setSingleStep(0.1)
        self._tile_h_spin.setDecimals(1)
        self._heat_fig_w = QDoubleSpinBox()
        self._heat_fig_w.setRange(0.0, 40.0)
        self._heat_fig_w.setValue(0.0)
        self._heat_fig_w.setSingleStep(1.0)
        self._heat_fig_w.setDecimals(1)
        self._heat_fig_w.setToolTip("0 = auto (from tile size)")
        self._heat_fig_h = QDoubleSpinBox()
        self._heat_fig_h.setRange(0.0, 40.0)
        self._heat_fig_h.setValue(0.0)
        self._heat_fig_h.setSingleStep(1.0)
        self._heat_fig_h.setDecimals(1)
        self._heat_fig_h.setToolTip("0 = auto (from tile size)")
        self._shared_chk = QCheckBox("Shared colour scale (Kd + Ki)")
        self._shared_chk.setChecked(True)
        self._grey_combo = QComboBox()
        self._grey_combo.addItems([
            "Hide failed combos",
            "Grey within filtered",
            "Grey all attempted"])
        self._grey_combo.setCurrentIndex(1)
        self._grey_combo.setToolTip(
            "Hide failed combos — only show combos with valid results\n"
            "Grey within filtered — grey cells within the filtered rows/cols\n"
            "Grey all attempted — expand to show every attempted combo")
        self._cmap_reverse_chk = QCheckBox("Reverse colormap")
        self._cmap_reverse_chk.setChecked(False)
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItems(["Landscape", "Portrait"])
        self._heat_layout_combo = QComboBox()
        self._heat_layout_combo.addItems([
            "Single figure",
            "Split by Host",
            "Split by Dye",
            "Split by Guest",
            "Paginated (max cols)"])
        self._heat_layout_combo.setToolTip(
            "Single figure — one heatmap per data type\n"
            "Split by Host/Dye/Guest — one page per unique value\n"
            "Paginated — break wide heatmaps into pages of N columns")
        self._heat_page_cols = QSpinBox()
        self._heat_page_cols.setRange(2, 30)
        self._heat_page_cols.setValue(8)
        self._heat_page_cols.setToolTip("Max columns per page (paginated mode)")
        hf.addRow("Colormap:",    self._cmap_combo)
        hf.addRow("Annotate:",    self._annot_combo)
        hf.addRow("Sort rows:",   self._sort_combo)
        hf.addRow("Layout:",      self._heat_layout_combo)
        hf.addRow("Page cols:",   self._heat_page_cols)
        hf.addRow("Tile width:",  self._tile_w_spin)
        hf.addRow("Tile height:", self._tile_h_spin)
        hf.addRow("Fig width:",   self._heat_fig_w)
        hf.addRow("Fig height:",  self._heat_fig_h)
        hf.addRow(self._shared_chk)
        hf.addRow("Failed cells:", self._grey_combo)
        hf.addRow(self._cmap_reverse_chk)
        hf.addRow("Orientation:", self._orientation_combo)
        self._heat_ki_pivot = QComboBox()
        self._heat_ki_pivot.addItems(["Host vs Guest|Dye", "Dye vs Guest (per Host)"])
        self._heat_ki_host = QComboBox()
        self._heat_ki_host_lbl = QLabel("Host:")
        self._heat_ki_host.setVisible(False)
        self._heat_ki_host_lbl.setVisible(False)
        self._heat_ki_pivot.currentTextChanged.connect(self._on_ki_pivot_changed)
        hf.addRow("Ki pivot:", self._heat_ki_pivot)
        hf.addRow(self._heat_ki_host_lbl, self._heat_ki_host)
        self._traffic_chk = QCheckBox("Traffic light (discrete bands)")
        self._traffic_chk.setChecked(False)
        self._traffic_steps_lbl = QLabel("Steps:")
        self._traffic_steps_spin = QSpinBox()
        self._traffic_steps_spin.setRange(2, 20)
        self._traffic_steps_spin.setValue(7)
        self._traffic_steps_spin.setToolTip("Number of discrete colour bands")
        self._traffic_steps_lbl.setVisible(False)
        self._traffic_steps_spin.setVisible(False)
        self._traffic_chk.stateChanged.connect(
            lambda s: (self._traffic_steps_lbl.setVisible(bool(s)),
                       self._traffic_steps_spin.setVisible(bool(s)),
                       self._traffic_range_chk.setVisible(bool(s)),
                       self._traffic_colors_chk.setVisible(bool(s))))
        hf.addRow(self._traffic_chk)
        hf.addRow(self._traffic_steps_lbl, self._traffic_steps_spin)
        self._traffic_range_chk = QCheckBox("Manual range")
        self._traffic_range_chk.setChecked(False)
        self._traffic_range_chk.setVisible(False)
        self._traffic_vmin_lbl = QLabel("Min (log₁₀ µM):")
        self._traffic_vmin_spin = QDoubleSpinBox()
        self._traffic_vmin_spin.setRange(-6.0, 6.0)
        self._traffic_vmin_spin.setValue(-2.0)
        self._traffic_vmin_spin.setDecimals(1)
        self._traffic_vmin_spin.setSingleStep(0.5)
        self._traffic_vmax_lbl = QLabel("Max (log₁₀ µM):")
        self._traffic_vmax_spin = QDoubleSpinBox()
        self._traffic_vmax_spin.setRange(-6.0, 6.0)
        self._traffic_vmax_spin.setValue(2.0)
        self._traffic_vmax_spin.setDecimals(1)
        self._traffic_vmax_spin.setSingleStep(0.5)
        for _w in (self._traffic_vmin_lbl, self._traffic_vmin_spin,
                   self._traffic_vmax_lbl, self._traffic_vmax_spin):
            _w.setVisible(False)
        self._traffic_range_chk.stateChanged.connect(
            lambda s: [_w.setVisible(bool(s)) for _w in (
                self._traffic_vmin_lbl, self._traffic_vmin_spin,
                self._traffic_vmax_lbl, self._traffic_vmax_spin)])
        hf.addRow(self._traffic_range_chk)
        hf.addRow(self._traffic_vmin_lbl, self._traffic_vmin_spin)
        hf.addRow(self._traffic_vmax_lbl, self._traffic_vmax_spin)
        self._traffic_colors_chk = QCheckBox("Custom band colours")
        self._traffic_colors_chk.setChecked(False)
        self._traffic_colors_chk.setVisible(False)
        self._traffic_colors_widget = QWidget()
        _tcvbox = QVBoxLayout(self._traffic_colors_widget)
        _tcvbox.setContentsMargins(0, 0, 0, 0)
        _tcvbox.setSpacing(2)
        self._traffic_color_rows = []
        self._traffic_colors_widget.setVisible(False)
        self._traffic_colors_chk.stateChanged.connect(
            lambda s: (self._traffic_colors_widget.setVisible(bool(s)),
                       self._rebuild_combined_traffic_colors() if bool(s) else None))
        self._traffic_steps_spin.valueChanged.connect(
            lambda _: self._rebuild_combined_traffic_colors()
                      if self._traffic_colors_chk.isChecked() else None)
        hf.addRow(self._traffic_colors_chk)
        hf.addRow(self._traffic_colors_widget)
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

        # ── Spider options ─────────────────────────────────────────────────
        self._gb_spider = QGroupBox("Spider Options")
        sv = QVBoxLayout(self._gb_spider)
        sfl = QFormLayout()
        self._spider_yscale = QComboBox()
        self._spider_yscale.addItems(["pK  (−log₁₀ K/M)", "Raw µM", "log₁₀(K/µM)"])
        self._spider_cmap = QComboBox()
        self._spider_cmap.addItems(["Blues", "viridis", "plasma", "magma",
                                     "inferno", "coolwarm", "RdYlGn", "YlOrRd"])
        self._spider_reverse_chk = QCheckBox("Reverse colormap")
        self._spider_reverse_chk.setChecked(False)
        self._spider_fill_alpha = QDoubleSpinBox()
        self._spider_fill_alpha.setRange(0.0, 1.0)
        self._spider_fill_alpha.setValue(0.9)
        self._spider_fill_alpha.setSingleStep(0.05)
        self._spider_fill_alpha.setDecimals(2)
        self._spider_line_w = QDoubleSpinBox()
        self._spider_line_w.setRange(0.1, 5.0)
        self._spider_line_w.setValue(0.5)
        self._spider_line_w.setSingleStep(0.5)
        self._spider_line_w.setDecimals(1)
        self._spider_line_style = QComboBox()
        self._spider_line_style.addItems(["solid", "dotted", "dashed", "dashdot"])
        self._spider_line_col = ColorPickerRow("Outline:", "#000000")
        self._spider_fig_sz = QDoubleSpinBox()
        self._spider_fig_sz.setRange(4.0, 16.0)
        self._spider_fig_sz.setValue(7.0)
        self._spider_fig_sz.setSingleStep(0.5)
        self._spider_fig_sz.setDecimals(1)
        self._spider_margin = QDoubleSpinBox()
        self._spider_margin.setRange(0.5, 6.0)
        self._spider_margin.setValue(2.0)
        self._spider_margin.setSingleStep(0.25)
        self._spider_margin.setDecimals(2)
        self._spider_margin.setToolTip("Page margin around the charts (inches)")
        self._spider_label_sz = QDoubleSpinBox()
        self._spider_label_sz.setRange(4, 18)
        self._spider_label_sz.setValue(8)
        self._spider_label_sz.setSingleStep(0.5)
        self._spider_label_sz.setDecimals(1)
        self._spider_label_angle = QDoubleSpinBox()
        self._spider_label_angle.setRange(-90, 90)
        self._spider_label_angle.setValue(0)
        self._spider_label_angle.setSingleStep(5)
        self._spider_label_angle.setDecimals(0)
        self._spider_label_angle.setToolTip("Extra rotation added to radial label orientation (degrees)")
        self._spider_label_pad = QDoubleSpinBox()
        self._spider_label_pad.setRange(5, 60)
        self._spider_label_pad.setValue(18)
        self._spider_label_pad.setSingleStep(2)
        self._spider_label_pad.setDecimals(0)
        self._spider_label_pad.setToolTip("Distance between labels and chart edge")
        self._spider_cbar_shrink = QDoubleSpinBox()
        self._spider_cbar_shrink.setRange(0.2, 1.0)
        self._spider_cbar_shrink.setValue(0.75)
        self._spider_cbar_shrink.setSingleStep(0.05)
        self._spider_cbar_shrink.setDecimals(2)
        self._spider_cbar_lbl = QLineEdit("Gradient Scale")
        self._spider_cbar_lbl.setPlaceholderText("Colorbar label")
        self._spider_show_vals = QCheckBox("Show values on bars")
        self._spider_show_vals.setChecked(False)
        self._spider_show_vals.setToolTip("Display value ± SE at the end of each bar")
        self._spider_sort = QComboBox()
        self._spider_sort.addItems(["Alphabetical", "pK ↓ (strongest)",
                                     "pK ↑ (weakest)", "Host", "Dye", "Guest"])
        self._spider_group = QComboBox()
        self._spider_group.addItems(["None (single chart)", "Host", "Dye", "Guest"])
        sfl.addRow("Y-axis:",      self._spider_yscale)
        sfl.addRow("Colormap:",    self._spider_cmap)
        sfl.addRow(self._spider_reverse_chk)
        sfl.addRow("Fill alpha:",  self._spider_fill_alpha)
        sfl.addRow("Line width:",  self._spider_line_w)
        sfl.addRow("Line style:",  self._spider_line_style)
        sfl.addRow("Chart size:",  self._spider_fig_sz)
        sfl.addRow("Margin:",      self._spider_margin)
        sfl.addRow("Label size:",  self._spider_label_sz)
        sfl.addRow("Label angle:", self._spider_label_angle)
        sfl.addRow("Label dist:",  self._spider_label_pad)
        sfl.addRow("Cbar shrink:", self._spider_cbar_shrink)
        sfl.addRow("Cbar label:",  self._spider_cbar_lbl)
        sfl.addRow(self._spider_show_vals)
        sfl.addRow("Sort bars:",   self._spider_sort)
        sfl.addRow("Split by:",    self._spider_group)
        sv.addLayout(sfl)
        sv.addWidget(self._spider_line_col)
        cl.addWidget(self._gb_spider)
        self._gb_spider.setVisible(False)

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
        self._axis_fsz = QDoubleSpinBox()
        self._axis_fsz.setRange(4, 18)
        self._axis_fsz.setValue(9)
        self._axis_fsz.setSingleStep(0.5)
        self._axis_fsz.setDecimals(1)
        self._tick_fsz = QDoubleSpinBox()
        self._tick_fsz.setRange(3, 16)
        self._tick_fsz.setValue(8)
        self._tick_fsz.setSingleStep(0.5)
        self._tick_fsz.setDecimals(1)
        ff_lay.addRow("Family:", self._fam)
        ff_lay.addRow("Size:",   self._fsz)
        ff_lay.addRow("Axis labels:", self._axis_fsz)
        ff_lay.addRow("Tick numbers:", self._tick_fsz)
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
        self._bar_rb.toggled.connect(self._on_type_changed)
        self._spider_rb.toggled.connect(self._on_type_changed)
        return scroll

    def _on_type_changed(self, _checked=None):
        self._gb_heat.setVisible(self._heat_rb.isChecked())
        self._gb_bar.setVisible(self._bar_rb.isChecked())
        self._gb_spider.setVisible(self._spider_rb.isChecked())

    # ── File management ────────────────────────────────────────────────────

    def _on_ki_pivot_changed(self, text):
        per_host = text == "Dye vs Guest (per Host)"
        self._heat_ki_host.setVisible(per_host)
        self._heat_ki_host_lbl.setVisible(per_host)

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

    def _rebuild_combined_traffic_colors(self):
        layout = self._traffic_colors_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._traffic_color_rows.clear()
        n_steps = self._traffic_steps_spin.value()
        cmap_name = self._cmap_combo.currentText()
        _base_name = cmap_name[:-2] if cmap_name.endswith("_r") else cmap_name
        _base = matplotlib.colormaps.get_cmap(_base_name)
        rev = self._cmap_reverse_chk.isChecked()
        for i in range(n_steps):
            frac = i / max(n_steps - 1, 1)
            if rev:
                frac = 1.0 - frac
            rgba = _base(frac)
            hex_c = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            row = ColorPickerRow(f"Band {i + 1}:", hex_c)
            layout.addWidget(row)
            self._traffic_color_rows.append(row)

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
        if "Kd" not in sel:  kd = None;  kd_all = None
        if "Ki" not in sel:  ki = None;  ki_all = None

        if ki is not None and "Host" in ki.columns:
            prev = self._heat_ki_host.currentText()
            self._heat_ki_host.blockSignals(True)
            self._heat_ki_host.clear()
            self._heat_ki_host.addItems(sorted(ki["Host"].dropna().unique()))
            if prev:
                idx = self._heat_ki_host.findText(prev)
                if idx >= 0:
                    self._heat_ki_host.setCurrentIndex(idx)
            self._heat_ki_host.blockSignals(False)

        n_kd = 0 if kd is None else len(kd)
        n_ki = 0 if ki is None else len(ki)
        if n_kd == 0 and n_ki == 0:
            self._status_lbl.setText("No data found for this filter/selection.")
            self._status_lbl.setStyleSheet("color: red;")
            return

        fs = self._fsz.value();  ff = self._fam.currentText()
        afs = self._axis_fsz.value()
        tfs = self._tick_fsz.value()
        try:
            if self._heat_rb.isChecked():
                figs = self._render_heatmaps(kd, ki, fs, ff, afs, tfs,
                                             kd_all=kd_all, ki_all=ki_all)
            elif self._bar_rb.isChecked():
                figs = self._render_barcharts(kd, ki, fs, ff, afs, tfs)
            else:
                figs = self._render_spiders(kd, ki, fs, ff, afs, tfs)
            for old_fig in self._current_figs:
                old_fig.clear()
            self._current_figs = figs

            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setAlignment(Qt.AlignHCenter)
            for fig in figs:
                canvas = FigureCanvas(fig)
                dpi = fig.get_dpi()
                w_px = int(fig.get_figwidth() * dpi)
                h_px = int(fig.get_figheight() * dpi)
                canvas.setFixedSize(w_px, h_px)
                canvas.draw()
                vl.addWidget(canvas)
            container.adjustSize()
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

    def _prep_pivot(self, df, val_col, row_col, col_col, all_df=None):
        import pandas as pd
        attempt_src = all_df if all_df is not None else df
        attempted = (attempt_src.assign(_one=1)
                     .pivot_table(index=row_col, columns=col_col,
                                  values="_one", aggfunc="sum")
                     .fillna(0) > 0)
        good = df[df[val_col].notna() & (df[val_col] > 0) & (df[val_col] < 1e6)]
        pivot = good.pivot_table(index=row_col, columns=col_col,
                                 values=val_col, aggfunc="mean")
        return pivot, attempted

    def _render_heatmaps(self, kd_df, ki_df, fs, ff, afs=9, tfs=8,
                         kd_all=None, ki_all=None):
        import pandas as pd
        cmap       = self._cmap_combo.currentText()
        annot_mode = self._annot_combo.currentText()
        sort_mode  = self._sort_combo.currentText()
        shared     = self._shared_chk.isChecked()
        grey_mode  = self._grey_combo.currentText()
        tile_w     = self._tile_w_spin.value()
        tile_h     = self._tile_h_spin.value()
        override_w = self._heat_fig_w.value()
        override_h = self._heat_fig_h.value()
        reverse_cmap = self._cmap_reverse_chk.isChecked()
        portrait   = self._orientation_combo.currentText() == "Portrait"
        layout_mode = self._heat_layout_combo.currentText()
        page_cols   = self._heat_page_cols.value()

        if reverse_cmap:
            cmap = cmap + "_r"

        def _apply_grey_expand(p, m, se_p, grey_mode):
            """Expand or clip pivot/mask based on grey mode."""
            if "all attempted" in grey_mode.lower():
                all_rows = sorted(set(p.index) | set(m.index))
                all_cols = sorted(set(p.columns) | set(m.columns))
                p = p.reindex(index=all_rows, columns=all_cols)
                m = m.reindex(index=all_rows, columns=all_cols).fillna(False)
                if se_p is not None:
                    se_p = se_p.reindex(index=all_rows, columns=all_cols)
            else:
                m = m.reindex(index=p.index, columns=p.columns).fillna(False)
                if se_p is not None:
                    se_p = se_p.reindex(index=p.index, columns=p.columns)
            return p, m, se_p

        def _apply_sort(p, m, se_p, sort_mode):
            if "median" in sort_mode.lower():
                p = self._sort_pivot(p)
            elif "Alpha" in sort_mode:
                p = p.sort_index()
            elif "Dye" in sort_mode or "Guest" in sort_mode:
                first_col_vals = p.iloc[:, 0].fillna(np.inf)
                p = p.loc[first_col_vals.sort_values().index]
            m = m.reindex(index=p.index, columns=p.columns).fillna(False)
            if se_p is not None:
                se_p = se_p.reindex(index=p.index, columns=p.columns)
            return p, m, se_p

        entries = []   # (label, pivot, attempted, xlabel, ylabel, se_pivot)
        if kd_df is not None and not kd_df.empty:
            p, m = self._prep_pivot(kd_df, "Kd", "Host", "Dye", all_df=kd_all)
            se_p = None
            if "Kd_SE" in kd_df.columns:
                good = kd_df[kd_df["Kd"].notna() & (kd_df["Kd"] > 0) & (kd_df["Kd"] < 1e6)]
                se_p = good.pivot_table(index="Host", columns="Dye",
                                        values="Kd_SE", aggfunc="mean")
            if not p.empty:
                p, m, se_p = _apply_grey_expand(p, m, se_p, grey_mode)
                p, m, se_p = _apply_sort(p, m, se_p, sort_mode)
                xl, yl = "Dye", "Host"
                if portrait:
                    p = p.T
                    m = m.T
                    if se_p is not None:
                        se_p = se_p.T
                    xl, yl = yl, xl
                entries.append(("Direct Binding — Kd (Host–Dye)", p, m, xl, yl, se_p))
        if ki_df is not None and not ki_df.empty:
            ki_pivot_mode = self._heat_ki_pivot.currentText()
            if ki_pivot_mode == "Dye vs Guest (per Host)":
                host = self._heat_ki_host.currentText()
                ki_work = ki_df.copy()
                if host:
                    ki_work = ki_work[ki_work["Host"] == host]
                ki_all_work = ki_all.copy() if ki_all is not None else None
                if ki_all_work is not None and host:
                    ki_all_work = ki_all_work[ki_all_work["Host"] == host]
                row_name, col_name = "Dye", "Guest"
                p, m = self._prep_pivot(ki_work, "Ki_uM", row_name, col_name,
                                        all_df=ki_all_work)
                se_p = None
                if "Ki_err_uM" in ki_work.columns:
                    good = ki_work[ki_work["Ki_uM"].notna() & (ki_work["Ki_uM"] > 0) & (ki_work["Ki_uM"] < 1e6)]
                    se_p = good.pivot_table(index=row_name, columns=col_name,
                                            values="Ki_err_uM", aggfunc="mean")
                if not p.empty:
                    p, m, se_p = _apply_grey_expand(p, m, se_p, grey_mode)
                    p, m, se_p = _apply_sort(p, m, se_p, sort_mode)
                    xl, yl = col_name, row_name
                    if portrait:
                        p = p.T
                        m = m.T
                        if se_p is not None:
                            se_p = se_p.T
                        xl, yl = yl, xl
                    title = f"Competition Binding — Ki ({host})" if host else "Competition Binding — Ki"
                    entries.append((title, p, m, xl, yl, se_p))
            else:
                ki_work = ki_df.copy()
                ki_all_work = None
                if "Dye" in ki_work.columns:
                    ki_work["Guest | Dye"] = ki_work["Guest"] + " | " + ki_work["Dye"]
                    col_name = "Guest | Dye"
                    if ki_all is not None:
                        ki_all_work = ki_all.copy()
                        ki_all_work["Guest | Dye"] = ki_all_work["Guest"] + " | " + ki_all_work["Dye"]
                else:
                    col_name = "Guest"
                    ki_all_work = ki_all
                p, m = self._prep_pivot(ki_work, "Ki_uM", "Host", col_name,
                                        all_df=ki_all_work)
                se_p = None
                if "Ki_err_uM" in ki_work.columns:
                    good = ki_work[ki_work["Ki_uM"].notna() & (ki_work["Ki_uM"] > 0) & (ki_work["Ki_uM"] < 1e6)]
                    se_p = good.pivot_table(index="Host", columns=col_name,
                                            values="Ki_err_uM", aggfunc="mean")
                if not p.empty:
                    p, m, se_p = _apply_grey_expand(p, m, se_p, grey_mode)
                    p, m, se_p = _apply_sort(p, m, se_p, sort_mode)
                    xl, yl = col_name, "Host"
                    if portrait:
                        p = p.T
                        m = m.T
                        if se_p is not None:
                            se_p = se_p.T
                        xl, yl = yl, xl
                    entries.append(("Competition Binding — Ki (Host–Guest)", p, m, xl, yl, se_p))
        if not entries:
            return []

        # ── Split / Paginate based on layout mode ──
        if "Split" in layout_mode:
            split_by = ("Host" if "Host" in layout_mode
                        else "Dye" if "Dye" in layout_mode
                        else "Guest")
            expanded = []
            for (title, pivot, mask, xlabel, ylabel, se_pivot) in entries:
                if split_by == ylabel:
                    for val in sorted(pivot.index):
                        sub_p = pivot.loc[[val]]
                        sub_m = mask.reindex(index=[val], columns=pivot.columns).fillna(False)
                        sub_se = (se_pivot.reindex(index=[val], columns=pivot.columns)
                                  if se_pivot is not None else None)
                        expanded.append((f"{title}  —  {split_by}: {val}",
                                         sub_p, sub_m, xlabel, ylabel, sub_se))
                elif split_by in (xlabel,) or split_by.lower() in xlabel.lower():
                    for val in sorted(pivot.columns):
                        sub_p = pivot[[val]]
                        sub_m = mask.reindex(index=pivot.index, columns=[val]).fillna(False)
                        sub_se = (se_pivot.reindex(index=pivot.index, columns=[val])
                                  if se_pivot is not None else None)
                        expanded.append((f"{title}  —  {split_by}: {val}",
                                         sub_p, sub_m, xlabel, ylabel, sub_se))
                else:
                    expanded.append((title, pivot, mask, xlabel, ylabel, se_pivot))
            entries = expanded

        elif "Paginated" in layout_mode:
            expanded = []
            for (title, pivot, mask, xlabel, ylabel, se_pivot) in entries:
                nc = pivot.shape[1]
                if nc <= page_cols:
                    expanded.append((title, pivot, mask, xlabel, ylabel, se_pivot))
                else:
                    for start in range(0, nc, page_cols):
                        end = min(start + page_cols, nc)
                        cols_slice = pivot.columns[start:end]
                        sub_p = pivot[cols_slice]
                        sub_m = mask.reindex(columns=cols_slice).fillna(False)
                        sub_se = (se_pivot[cols_slice]
                                  if se_pivot is not None else None)
                        pg = start // page_cols + 1
                        expanded.append((f"{title}  (page {pg})",
                                         sub_p, sub_m, xlabel, ylabel, sub_se))
            entries = expanded

        traffic = self._traffic_chk.isChecked()

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
            for _, p, _, _, _, _ in entries:
                lv = np.log10(p.replace(0, np.nan)).values.flatten()
                lv = lv[~np.isnan(lv)]
                if len(lv):
                    scales.append((float(np.floor(np.nanpercentile(lv, 2))),
                                   float(np.ceil(np.nanpercentile(lv, 98)))))
                else:
                    scales.append((0.0, 1.0))

        if traffic and self._traffic_range_chk.isChecked():
            _mv, _mx = self._traffic_vmin_spin.value(), self._traffic_vmax_spin.value()
            scales = [(_mv, _mx)] * len(scales)

        figs = []
        for (title, pivot, mask, xlabel, ylabel, se_pivot), (vmin_, vmax_) in zip(entries, scales):
            nr, nc = pivot.shape
            heat_w = nc * tile_w
            heat_h = nr * tile_h
            fig_w = override_w if override_w > 0 else heat_w + 3.0
            fig_h = override_h if override_h > 0 else heat_h + 2.0
            fig = Figure(figsize=(fig_w, fig_h))
            ax  = fig.add_subplot(111)
            ax.set_facecolor("white")
            log_df = np.log10(pivot.replace(0, np.nan))

            if traffic:
                n_steps = self._traffic_steps_spin.value()
                t_bounds = np.arange(np.floor(vmin_), np.ceil(vmax_) + 1, 1.0)
                if len(t_bounds) - 1 > n_steps:
                    t_bounds = np.linspace(np.floor(vmin_), np.ceil(vmax_),
                                           n_steps + 1)
                n_colors = len(t_bounds) - 1
                if (self._traffic_colors_chk.isChecked()
                        and self._traffic_color_rows):
                    _cols = []
                    for _cr in self._traffic_color_rows[:n_colors]:
                        _c = QColor(_cr.color)
                        _cols.append((_c.redF(), _c.greenF(), _c.blueF(), 1.0))
                    while len(_cols) < n_colors:
                        _cols.append((0.5, 0.5, 0.5, 1.0))
                else:
                    _base_cmap_name = cmap[:-2] if cmap.endswith("_r") else cmap
                    _base = matplotlib.colormaps.get_cmap(_base_cmap_name)
                    _cols = [_base(i / max(n_colors - 1, 1)) for i in range(n_colors)]
                    if cmap.endswith("_r"):
                        _cols = list(reversed(_cols))
                _disc_cmap = ListedColormap(_cols)
                _norm = BoundaryNorm(t_bounds, _disc_cmap.N)
                sns.heatmap(
                    log_df, ax=ax, mask=log_df.isna(),
                    cmap=_disc_cmap, norm=_norm,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"shrink": 0.7, "pad": 0.02},
                    annot=False, square=False)
            else:
                t_bounds = None
                sns.heatmap(
                    log_df, ax=ax, mask=log_df.isna(),
                    cmap=cmap, vmin=vmin_, vmax=vmax_,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"shrink": 0.7, "pad": 0.02},
                    annot=False, square=False)
            ax.set_aspect(tile_h / tile_w)

            if "grey" in grey_mode.lower():
                failed = mask & log_df.isna()
                for i in range(nr):
                    for j in range(nc):
                        if failed.iloc[i, j]:
                            ax.add_patch(mpatches.Rectangle(
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
                                txt = _fmt_uM(val)
                                if pd.notna(se_val) and se_val > 0:
                                    txt += f"\n±{_fmt_uM(se_val)}"
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
                                    fontsize=tfs, fontfamily=ff, color=tc)

            cbar = ax.collections[0].colorbar
            if traffic and t_bounds is not None:
                cbar.set_ticks(t_bounds)
                cbar.set_ticklabels(
                    [r"$10^{%g}$" % b for b in t_bounds], fontsize=tfs)
            else:
                ticks = np.arange(int(vmin_), int(vmax_) + 1)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([r"$10^{%d}$" % t for t in ticks],
                                    fontsize=tfs)
            cbar.set_label("µM", fontsize=afs, labelpad=4)

            ax.set_title(title, fontsize=fs + 2, fontweight="bold",
                         fontfamily=ff, pad=8)
            ax.set_xlabel(xlabel, fontsize=afs, fontfamily=ff)
            ax.set_ylabel(ylabel, fontsize=afs, fontfamily=ff)
            for lbl in ax.get_yticklabels():
                lbl.set_rotation(0); lbl.set_fontsize(tfs); lbl.set_fontfamily(ff)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45); lbl.set_ha("right")
                lbl.set_fontsize(tfs); lbl.set_fontfamily(ff)

            fig.tight_layout()
            figs.append(fig)
        return figs

    # ── Bar chart ──────────────────────────────────────────────────────────

    def _render_barcharts(self, kd_df, ki_df, fs, ff, afs=9, tfs=8):
        figs = []
        if kd_df is not None and not kd_df.empty:
            f = self._bar_one(kd_df, "Kd", "pKd",
                              "pKd = −log₁₀(Kd/M)", "Direct Binding — pKd",
                              "Dye", fs, ff, afs, tfs)
            if f: figs.append(f)
        if ki_df is not None and not ki_df.empty:
            f = self._bar_one(ki_df, "Ki_uM", "pKi",
                              "pKi = −log₁₀(Ki/M)", "Competition Binding — pKi",
                              "Guest", fs, ff, afs, tfs)
            if f: figs.append(f)
        return figs

    def _bar_one(self, df, val_col, pv, y_label, title, secondary_col, fs, ff, afs=9, tfs=8):
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
                           fontsize=tfs, fontfamily=ff)
        ax.set_ylabel(used_ylabel, fontsize=afs, fontfamily=ff)
        ax.set_title(title, fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        ax.tick_params(axis="y", labelsize=tfs)
        ax.grid(axis="y", alpha=0.35, zorder=0);  ax.set_axisbelow(True)
        if leg:
            ax.legend(handles=leg, title=grp, fontsize=tfs,
                      title_fontsize=tfs,
                      bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        return fig

    # ── Spider ─────────────────────────────────────────────────────────────

    def _render_spiders(self, kd_df, ki_df, fs, ff, afs=9, tfs=8):
        from matplotlib.colors import Normalize
        from matplotlib.ticker import MaxNLocator

        yscale      = self._spider_yscale.currentText()
        cmap_name   = self._spider_cmap.currentText()
        if self._spider_reverse_chk.isChecked():
            cmap_name += "_r"
        fill_a      = self._spider_fill_alpha.value()
        lw          = self._spider_line_w.value()
        ls          = self._spider_line_style.currentText()
        line_col    = self._spider_line_col.color
        cell_sz     = self._spider_fig_sz.value()
        margin      = self._spider_margin.value()
        cbar_lbl    = self._spider_cbar_lbl.text()
        split_by    = self._spider_group.currentText()
        sort_txt    = self._spider_sort.currentText()
        label_sz    = self._spider_label_sz.value()
        label_angle = self._spider_label_angle.value()
        label_pad   = self._spider_label_pad.value()
        cbar_shrink = self._spider_cbar_shrink.value()
        show_vals   = self._spider_show_vals.isChecked()

        def _sort_agg(agg, grp_df):
            if "↓" in sort_txt:
                return agg.sort_values(ascending=False)
            elif "↑" in sort_txt:
                return agg.sort_values(ascending=True)
            elif sort_txt in ("Host", "Dye", "Guest"):
                if sort_txt in grp_df.columns:
                    order = (grp_df.groupby("_bar")[sort_txt].first()
                             .sort_values().index)
                    return agg.reindex(order).dropna()
            return agg.sort_index()

        figs = []
        for df, mode in [(kd_df, "kd"), (ki_df, "ki")]:
            if df is None or df.empty:
                continue
            vc  = "Kd" if mode == "kd" else "Ki_uM"
            sc  = "Kd_SE" if mode == "kd" else "Ki_err_uM"
            pv  = "pKd" if mode == "kd" else "pKi"
            df  = df.copy()
            df[pv] = 6.0 - np.log10(df[vc].clip(lower=1e-12))
            df["_p_SE"] = (df[sc] / (df[vc].clip(lower=1e-12) * np.log(10))
                           if sc in df.columns else np.nan)

            if "pK" in yscale:
                df["_y"]  = df[pv]
                df["_se"] = df["_p_SE"]
            elif "Raw" in yscale:
                df["_y"]  = df[vc]
                df["_se"] = df[sc] if sc in df.columns else np.nan
            else:
                df["_y"]  = np.log10(df[vc].clip(lower=1e-12))
                df["_se"] = np.nan

            def _bar_lbl(row):
                if mode == "kd":
                    parts = []
                    if split_by != "Host": parts.append(str(row["Host"]))
                    if split_by != "Dye" and "Dye" in row.index:
                        parts.append(str(row["Dye"]))
                    return " | ".join(parts) if parts else str(row["Host"])
                else:
                    parts = []
                    if split_by != "Guest" and "Guest" in row.index:
                        parts.append(str(row["Guest"]))
                    hd = []
                    if split_by != "Host": hd.append(str(row["Host"]))
                    if split_by != "Dye" and "Dye" in row.index:
                        hd.append(str(row["Dye"]))
                    if hd: parts.append(" | ".join(hd))
                    return " / ".join(parts) if parts else str(row["Host"])

            df["_bar"] = df.apply(_bar_lbl, axis=1)

            if split_by == "None (single chart)" or split_by not in df.columns:
                groups = [("", df)]
            else:
                groups = [(str(name), grp.reset_index(drop=True))
                          for name, grp in df.groupby(split_by, sort=True)]

            n_groups = len(groups)
            if n_groups == 0:
                continue

            has_titles = n_groups > 1
            cols_g = min(n_groups, 3)
            rows_g = int(np.ceil(n_groups / cols_g))

            title_extra = 1.0 if has_titles else 0.0
            cbar_room = 3.0
            chart_w = cell_sz * cols_g
            chart_h = (cell_sz + title_extra) * rows_g
            fig_w = margin + chart_w + cbar_room + margin
            fig_h = margin + chart_h + margin
            fig = Figure(figsize=(fig_w, fig_h))

            left_f   = margin / fig_w
            right_f  = (margin + chart_w) / fig_w
            bottom_f = margin / fig_h
            top_f    = 1.0 - margin / fig_h

            gs = GridSpec(rows_g, cols_g, figure=fig,
                          left=left_f, right=right_f,
                          bottom=bottom_f, top=top_f,
                          hspace=0.95 if has_titles else 0.55,
                          wspace=0.55)

            all_y = df["_y"].dropna()
            if all_y.empty:
                continue
            vmin_val = float(all_y.min())
            vmax_val = float(all_y.max())
            if vmax_val == vmin_val:
                vmax_val = vmin_val + 1.0
            colormap = matplotlib.colormaps.get_cmap(cmap_name)
            norm = Normalize(vmin=vmin_val, vmax=vmax_val)

            for idx, (group_name, grp_df) in enumerate(groups):
                agg_y  = grp_df.groupby("_bar")["_y"].mean()
                agg_se = grp_df.groupby("_bar")["_se"].mean()
                agg_y  = _sort_agg(agg_y, grp_df)
                agg_se = agg_se.reindex(agg_y.index)
                labels = list(agg_y.index)
                values = list(agg_y.values)
                ses    = list(agg_se.values)
                n_vars = len(labels)

                ri, ci = divmod(idx, cols_g)
                ax = fig.add_subplot(gs[ri, ci], polar=True)
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_axisbelow(True)
                ax.grid(True, alpha=0.3, zorder=0)

                if n_vars < 1:
                    ax.set_visible(False)
                    continue

                angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
                bar_width = (2 * np.pi / n_vars * 0.85) if n_vars > 1 else (np.pi / 2)

                ax.bar(angles, values, width=bar_width, bottom=0,
                       color=[colormap(norm(v)) for v in values],
                       alpha=fill_a, edgecolor=line_col, linewidth=lw,
                       linestyle=ls, zorder=3)

                ax.set_xticks(angles)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                r_max = ax.get_ylim()[1]
                label_r = r_max * (1.0 + label_pad / 50.0)
                for angle, label_text in zip(angles, labels):
                    angle_deg = np.degrees(angle) % 360
                    if angle_deg < 10 or angle_deg > 350:
                        ha, va = "center", "bottom"
                    elif 170 < angle_deg < 190:
                        ha, va = "center", "top"
                    elif angle_deg < 180:
                        ha, va = "left", "center"
                    else:
                        ha, va = "right", "center"
                    ax.text(angle, label_r, label_text, ha=ha, va=va,
                            fontsize=label_sz, fontfamily=ff,
                            rotation=label_angle, clip_on=False, zorder=4)

                if show_vals:
                    for angle, val, se in zip(angles, values, ses):
                        txt = f"{val:.2f}"
                        if not np.isnan(se) and se > 0:
                            txt += f" ± {se:.2f}"
                        angle_deg = np.degrees(angle) % 360
                        text_rot = 90 - angle_deg
                        if 90 < angle_deg < 270:
                            text_rot += 180
                        ax.text(angle, val * 0.85, txt, ha="center", va="center",
                                fontsize=tfs * 0.7, fontfamily=ff, fontweight="bold",
                                rotation=text_rot, clip_on=False, zorder=5)

                if group_name:
                    pos = ax.get_position()
                    title_y = pos.y1 + 0.8 / fig_h
                    fig.text(pos.x0 + pos.width / 2, title_y,
                             group_name, ha="center", va="bottom",
                             fontsize=fs, fontweight="bold", fontfamily=ff)

            for idx in range(n_groups, rows_g * cols_g):
                ri, ci = divmod(idx, cols_g)
                fig.add_subplot(gs[ri, ci]).set_visible(False)

            cbar_left   = 1.0 - (margin + 1.2) / fig_w
            cbar_height = (top_f - bottom_f) * cbar_shrink
            cbar_bot    = bottom_f + (top_f - bottom_f - cbar_height) / 2
            cbar_width  = 0.02
            cax = fig.add_axes([cbar_left, cbar_bot, cbar_width, cbar_height])

            used_cbar_lbl = cbar_lbl or (
                "pKd  (−log₁₀ Kd/M)" if mode == "kd" else "pKi  (−log₁₀ Ki/M)")
            sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            ticks = np.linspace(vmin_val, vmax_val, 6)
            cbar.set_ticks(ticks)
            tick_labels = []
            for t in ticks:
                if "pK" in yscale:
                    um = 10**(6 - t)
                    tick_labels.append(f"{t:.1f}  ({um:.1e} µM)")
                elif "Raw" in yscale:
                    tick_labels.append(f"{t:.2f} µM")
                else:
                    um = 10**t
                    tick_labels.append(f"{t:.1f}  ({um:.1e} µM)")
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=tfs)
            cbar.set_label(used_cbar_lbl, fontsize=afs)

            figs.append(fig)
        return figs

    # ── Export ─────────────────────────────────────────────────────────────

    def _export(self):
        if not self._current_figs:
            QMessageBox.warning(self, "Nothing to export",
                                "Generate charts first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "summary_plots_export.pdf",
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

        btn_heat = QPushButton("Summary Plots  (from results)")
        btn_heat.setFixedHeight(52)
        btn_heat.setStyleSheet("font-size: 14px;")
        btn_heat.setToolTip(
            "Load pipeline output files (binding_fit_results.xlsx /\n"
            "competitive_fit_results.xlsx) or Kd_Tables_Gen format\n"
            "and export pKd / pKi summary plots as PDF.")
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


def _close_old_canvas(scroll: QScrollArea):
    old = scroll.widget()
    if old is not None and isinstance(old, FigureCanvas):
        fig = old.figure
        fig.clear()
        del fig


class GridCanvas(FigureCanvas):
    plot_clicked = Signal(int)

    def __init__(self, items: list, color_data: str = None, color_fit: str = None,
                 color_resid: str = None, show_residuals: bool = False,
                 title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "normal", title_fontstyle: str = "normal",
                 axis_fontsize: float = 14, tick_fontsize: float = 12,
                 layout_cfg: dict = None, dpi: int = 100, parent=None,
                 normalise_y: bool = False, sci_notation_y: bool = False,
                 lw_fit: float = 2.0, ms_data: float = 4.0,
                 lw_errorbar: float = 1.5, ms_resid: float = 2.5,
                 x_min: float = None, x_max: float = None,
                 x_tick_step: float = None, show_excluded: bool = True):
        self._items = items
        lc   = layout_cfg or {}
        n    = max(len(items), 1)
        cols = lc.get("pdf_cols", min(int(np.ceil(np.sqrt(n))), 4))
        rows = int(np.ceil(n / cols))
        cd   = color_data  or pl_fda.PLOT_COLOR
        cf   = color_fit   or pl_fda.PLOT_COLOR
        cr   = color_resid or cd
        tkw  = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                    title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                    axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
                    normalise_y=normalise_y, sci_notation_y=sci_notation_y,
                    lw_fit=lw_fit, ms_data=ms_data,
                    lw_errorbar=lw_errorbar, ms_resid=ms_resid,
                    x_min=x_min, x_max=x_max, x_tick_step=x_tick_step, show_excluded=show_excluded)
        cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
        fl = layout_utils.figure_layout(lc, rows, cols, cl.row_h_in)

        fig = Figure(figsize=(fl.fig_w, fl.fig_h), dpi=dpi)
        outer = GridSpec(rows, cols, figure=fig, **fl.outer_kwargs())
        super().__init__(fig)
        self._axes = []
        for i, entry in enumerate(items):
            ri, ci = divmod(i, cols)
            if show_residuals:
                igs = outer[ri, ci].subgridspec(
                    cl.sub_rows, 1, height_ratios=cl.height_ratios,
                    hspace=cl.resid_gap)
                ax   = fig.add_subplot(igs[0])
                ax_r = fig.add_subplot(igs[1], sharex=ax)
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                                   color_resid=cr, ax_resid=ax_r, **tkw)
            else:
                ax = fig.add_subplot(outer[ri, ci])
                pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                                   color_resid=cr, **tkw)
            self._axes.append(ax)

        dpi_v = fig.dpi
        self.setMinimumSize(int(fl.fig_w * dpi_v), int(fl.fig_h * dpi_v))
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
                   title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "normal", title_fontstyle: str = "normal",
                   axis_fontsize: float = 14, tick_fontsize: float = 12,
                   layout_cfg: dict = None,
                   normalise_y: bool = False, sci_notation_y: bool = False,
                   lw_fit: float = 2.0, ms_data: float = 4.0,
                   lw_errorbar: float = 1.5, ms_resid: float = 2.5,
                   x_min: float = None, x_max: float = None,
                   x_tick_step: float = None, show_excluded: bool = True):
        lc = layout_cfg or {}
        cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
        fl = layout_utils.figure_layout(lc, 1, 1, cl.row_h_in)
        self._fig.set_size_inches(fl.fig_w, fl.fig_h)
        self._fig.clear()
        cd = color_data  or pl_fda.PLOT_COLOR
        cf = color_fit   or pl_fda.PLOT_COLOR
        cr = color_resid or cd
        tkw = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                   title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                   axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
                   normalise_y=normalise_y, sci_notation_y=sci_notation_y,
                   lw_fit=lw_fit, ms_data=ms_data,
                   lw_errorbar=lw_errorbar, ms_resid=ms_resid,
                   x_min=x_min, x_max=x_max, x_tick_step=x_tick_step, show_excluded=show_excluded)
        gs = GridSpec(cl.sub_rows, 1, figure=self._fig, height_ratios=cl.height_ratios,
                      hspace=cl.resid_gap, **fl.single_kwargs())
        if show_residuals:
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
            ax.tick_params(labelbottom=False)
        else:
            ax   = self._fig.add_subplot(gs[0])
            ax_r = None
        pl_fda.render_plot(ax, entry, color_data=cd, color_fit=cf,
                           color_resid=cr, ax_resid=ax_r, **tkw)
        dpi_v = self._fig.dpi
        w_px, h_px = int(fl.fig_w * dpi_v), int(fl.fig_h * dpi_v)
        self.setMinimumSize(w_px, h_px)
        self.setFixedSize(w_px, h_px)
        self.draw()


class DirectPlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pass_items       = []
        self._fail_items       = []
        self._current_items    = []
        self._current_idx      = 0
        self._color_data       = pl_fda.PLOT_COLOR_DATA
        self._color_fit        = pl_fda.PLOT_COLOR
        self._color_resid      = pl_fda.PLOT_COLOR_DATA
        self._show_residuals   = False
        self._layout_cfg       = {}
        self._title_fontsize   = 12.0
        self._title_fontfamily = "sans-serif"
        self._title_fontweight = "normal"
        self._title_fontstyle  = "normal"
        self._axis_fontsize    = 14.0
        self._tick_fontsize    = 12.0
        self._normalise_y      = False
        self._sci_notation_y   = False
        self._lw_fit           = 2.0
        self._ms_data          = 4.0
        self._lw_errorbar      = 1.5
        self._ms_resid         = 2.5
        self._x_min            = None
        self._x_max            = None
        self._x_tick_step      = None
        self._show_excluded    = True

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
             axis_fontsize: float = None, tick_fontsize: float = None,
             layout_cfg: dict = None,
             normalise_y: bool = None, sci_notation_y: bool = None,
             lw_fit: float = None, ms_data: float = None,
             lw_errorbar: float = None, ms_resid: float = None,
             x_min: float = None, x_max: float = None,
             x_tick_step: float = None, show_excluded: bool = True):
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
        if axis_fontsize    is not None: self._axis_fontsize    = axis_fontsize
        if tick_fontsize    is not None: self._tick_fontsize    = tick_fontsize
        if layout_cfg       is not None: self._layout_cfg       = layout_cfg
        if normalise_y      is not None: self._normalise_y      = normalise_y
        if sci_notation_y   is not None: self._sci_notation_y   = sci_notation_y
        if lw_fit           is not None: self._lw_fit           = lw_fit
        if ms_data          is not None: self._ms_data          = ms_data
        if lw_errorbar      is not None: self._lw_errorbar      = lw_errorbar
        if ms_resid         is not None: self._ms_resid         = ms_resid
        self._x_min       = x_min
        self._x_max       = x_max
        self._x_tick_step = x_tick_step
        self._show_excluded = show_excluded

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
                    title_fontstyle=self._title_fontstyle,
                    axis_fontsize=self._axis_fontsize,
                    tick_fontsize=self._tick_fontsize,
                    normalise_y=self._normalise_y,
                    sci_notation_y=self._sci_notation_y,
                    lw_fit=self._lw_fit, ms_data=self._ms_data,
                    lw_errorbar=self._lw_errorbar, ms_resid=self._ms_resid,
                    x_min=self._x_min, x_max=self._x_max,
                    x_tick_step=self._x_tick_step, show_excluded=self._show_excluded)

    def _build_grids(self):
        host  = self._current_host()
        plate = self._current_plate()
        pass_items = self._items_for_filters(self._pass_items, host, plate)
        fail_items = self._items_for_filters(self._fail_items, host, plate)
        tkw = self._tkw()
        no_pass = "  No PASS results for this selection."
        no_fail = "  No FAIL results for this selection."
        lc = self._layout_cfg or None
        _close_old_canvas(self._grid_scroll_pass)
        _close_old_canvas(self._grid_scroll_fail)
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
            self._jump.addItem(f"[PASS] {entry.get('host', '?')}-{entry.get('dye', '?')}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry.get('host', '?')}-{entry.get('dye', '?')}")

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
                 title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "normal", title_fontstyle: str = "normal",
                 axis_fontsize: float = 14, tick_fontsize: float = 12,
                 layout_cfg: dict = None, dpi: int = 100, parent=None,
                 normalise_y: bool = False, sci_notation_y: bool = False,
                 lw_fit: float = 2.0, ms_data: float = 4.0,
                 lw_errorbar: float = 1.5, ms_resid: float = 2.5,
                 x_min: float = None, x_max: float = None,
                 x_tick_step: float = None, show_excluded: bool = True):
        self._items = items
        lc   = layout_cfg or {}
        n    = max(len(items), 1)
        cols = lc.get("pdf_cols", min(int(np.ceil(np.sqrt(n))), 4))
        rows = int(np.ceil(n / cols))
        cf   = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd   = color_data  or pl_ki.PLOT_COLOR_DATA
        cr   = color_resid or cd
        tkw  = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                    title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                    axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
                    normalise_y=normalise_y, sci_notation_y=sci_notation_y,
                    lw_fit=lw_fit, ms_data=ms_data,
                    lw_errorbar=lw_errorbar, ms_resid=ms_resid,
                    x_min=x_min, x_max=x_max, x_tick_step=x_tick_step, show_excluded=show_excluded)
        cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
        fl = layout_utils.figure_layout(lc, rows, cols, cl.row_h_in)

        fig = Figure(figsize=(fl.fig_w, fl.fig_h), dpi=dpi)
        outer = GridSpec(rows, cols, figure=fig, **fl.outer_kwargs())
        super().__init__(fig)
        self._axes = []
        for i, entry in enumerate(items):
            ri, ci = divmod(i, cols)
            if show_residuals:
                igs = outer[ri, ci].subgridspec(
                    cl.sub_rows, 1, height_ratios=cl.height_ratios,
                    hspace=cl.resid_gap)
                ax   = fig.add_subplot(igs[0])
                ax_r = fig.add_subplot(igs[1], sharex=ax)
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, ax_resid=ax_r, **tkw)
            else:
                ax = fig.add_subplot(outer[ri, ci])
                pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                                     color_resid=cr, **tkw)
            self._axes.append(ax)

        dpi_v = fig.dpi
        self.setMinimumSize(int(fl.fig_w * dpi_v), int(fl.fig_h * dpi_v))
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
                   title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "normal", title_fontstyle: str = "normal",
                   axis_fontsize: float = 14, tick_fontsize: float = 12,
                   layout_cfg: dict = None,
                   normalise_y: bool = False, sci_notation_y: bool = False,
                   lw_fit: float = 2.0, ms_data: float = 4.0,
                   lw_errorbar: float = 1.5, ms_resid: float = 2.5,
                   x_min: float = None, x_max: float = None,
                   x_tick_step: float = None, show_excluded: bool = True):
        lc = layout_cfg or {}
        cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
        fl = layout_utils.figure_layout(lc, 1, 1, cl.row_h_in)
        self._fig.set_size_inches(fl.fig_w, fl.fig_h)
        self._fig.clear()
        cf = color_fit   or pl_ki.PLOT_COLOR_FIT
        cd = color_data  or pl_ki.PLOT_COLOR_DATA
        cr = color_resid or cd
        gs = GridSpec(cl.sub_rows, 1, figure=self._fig, height_ratios=cl.height_ratios,
                      hspace=cl.resid_gap, **fl.single_kwargs())
        if show_residuals:
            ax   = self._fig.add_subplot(gs[0])
            ax_r = self._fig.add_subplot(gs[1], sharex=ax)
            ax.tick_params(labelbottom=False)
        else:
            ax   = self._fig.add_subplot(gs[0])
            ax_r = None
        pl_ki.render_plot_ki(ax, entry, color_fit=cf, color_data=cd,
                             color_resid=cr, ax_resid=ax_r,
                             title_fontsize=title_fontsize,
                             title_fontfamily=title_fontfamily,
                             title_fontweight=title_fontweight,
                             title_fontstyle=title_fontstyle,
                             axis_fontsize=axis_fontsize,
                             tick_fontsize=tick_fontsize,
                             normalise_y=normalise_y,
                             sci_notation_y=sci_notation_y,
                             lw_fit=lw_fit, ms_data=ms_data,
                             lw_errorbar=lw_errorbar, ms_resid=ms_resid,
                             x_min=x_min, x_max=x_max, x_tick_step=x_tick_step, show_excluded=show_excluded)
        dpi_v = self._fig.dpi
        w_px, h_px = int(fl.fig_w * dpi_v), int(fl.fig_h * dpi_v)
        self.setMinimumSize(w_px, h_px)
        self.setFixedSize(w_px, h_px)
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
        self._title_fontsize   = 12.0
        self._title_fontfamily = "sans-serif"
        self._title_fontweight = "normal"
        self._title_fontstyle  = "normal"
        self._axis_fontsize    = 14.0
        self._tick_fontsize    = 12.0
        self._normalise_y      = False
        self._sci_notation_y   = False
        self._lw_fit           = 2.0
        self._ms_data          = 4.0
        self._lw_errorbar      = 1.5
        self._ms_resid         = 2.5
        self._x_min            = None
        self._x_max            = None
        self._x_tick_step      = None
        self._show_excluded    = True

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
             axis_fontsize: float = None, tick_fontsize: float = None,
             layout_cfg: dict = None,
             normalise_y: bool = None, sci_notation_y: bool = None,
             lw_fit: float = None, ms_data: float = None,
             lw_errorbar: float = None, ms_resid: float = None,
             x_min: float = None, x_max: float = None,
             x_tick_step: float = None, show_excluded: bool = True):
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
        if axis_fontsize    is not None: self._axis_fontsize    = axis_fontsize
        if tick_fontsize    is not None: self._tick_fontsize    = tick_fontsize
        if layout_cfg       is not None: self._layout_cfg       = layout_cfg
        if normalise_y      is not None: self._normalise_y      = normalise_y
        if sci_notation_y   is not None: self._sci_notation_y   = sci_notation_y
        if lw_fit           is not None: self._lw_fit           = lw_fit
        if ms_data          is not None: self._ms_data          = ms_data
        if lw_errorbar      is not None: self._lw_errorbar      = lw_errorbar
        if ms_resid         is not None: self._ms_resid         = ms_resid
        self._x_min       = x_min
        self._x_max       = x_max
        self._x_tick_step = x_tick_step
        self._show_excluded = show_excluded

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
                   title_fontstyle=self._title_fontstyle,
                   axis_fontsize=self._axis_fontsize,
                   tick_fontsize=self._tick_fontsize,
                   normalise_y=self._normalise_y,
                   sci_notation_y=self._sci_notation_y,
                   lw_fit=self._lw_fit, ms_data=self._ms_data,
                   lw_errorbar=self._lw_errorbar, ms_resid=self._ms_resid,
                   x_min=self._x_min, x_max=self._x_max,
                   x_tick_step=self._x_tick_step, show_excluded=self._show_excluded)
        lc = self._layout_cfg or None
        no_pass_lbl = "  No PASS results for this selection."
        no_fail_lbl = "  No FAIL results for this selection."
        _close_old_canvas(self._grid_scroll_pass)
        _close_old_canvas(self._grid_scroll_fail)
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
            self._jump.addItem(f"[PASS] {entry.get('host', '?')} | {entry.get('dye', '?')} | {entry.get('guest', '?')}")
        for entry in self._fail_items:
            self._jump.addItem(f"[FAIL] {entry.get('host', '?')} | {entry.get('dye', '?')} | {entry.get('guest', '?')}")

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
                          title_fontstyle=self._title_fontstyle,
                          axis_fontsize=self._axis_fontsize,
                          tick_fontsize=self._tick_fontsize,
                          normalise_y=self._normalise_y,
                          sci_notation_y=self._sci_notation_y,
                          lw_fit=self._lw_fit, ms_data=self._ms_data,
                          lw_errorbar=self._lw_errorbar, ms_resid=self._ms_resid,
                          x_min=self._x_min, x_max=self._x_max,
                          x_tick_step=self._x_tick_step, show_excluded=self._show_excluded)
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
                              title_fontstyle=self._title_fontstyle,
                              axis_fontsize=self._axis_fontsize,
                              tick_fontsize=self._tick_fontsize,
                              normalise_y=self._normalise_y,
                              sci_notation_y=self._sci_notation_y,
                              lw_fit=self._lw_fit, ms_data=self._ms_data,
                              lw_errorbar=self._lw_errorbar, ms_resid=self._ms_resid,
                              x_min=self._x_min, x_max=self._x_max,
                              x_tick_step=self._x_tick_step, show_excluded=self._show_excluded)
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


# ── Summary Plots tab  (pKd / pKi bar-chart, heatmap & spider) ───────────────

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

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # ── left controls ─────────────────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QScrollArea.NoFrame)
        ctrl_scroll.setMinimumWidth(240)
        ctrl = QWidget()
        ctrl_scroll.setWidget(ctrl)
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
        self._bar_rb    = QRadioButton("Bar chart")
        self._heat_rb   = QRadioButton("Heatmap")
        self._spider_rb = QRadioButton("Spider chart")
        self._bar_rb.setChecked(True)
        tl.addWidget(self._bar_rb)
        tl.addWidget(self._heat_rb)
        tl.addWidget(self._spider_rb)
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
            self._heat_col_combo.addItems(["Host | Dye", "Host", "Dye",
                                           "Dye vs Guest (per Host)"])
        self._heat_host_combo = QComboBox()
        self._heat_host_combo.setVisible(False)
        self._heat_host_lbl = QLabel("Host:")
        self._heat_host_lbl.setVisible(False)
        self._heat_col_combo.currentTextChanged.connect(self._on_heat_col_changed)
        self._annot_fmt = QComboBox()
        self._annot_fmt.addItems(["None", ".2f", ".1f", ".3f"])
        self._heat_tile_w = QDoubleSpinBox()
        self._heat_tile_w.setRange(0.3, 5.0)
        self._heat_tile_w.setValue(1.2)
        self._heat_tile_w.setSingleStep(0.1)
        self._heat_tile_w.setDecimals(1)
        self._heat_tile_h = QDoubleSpinBox()
        self._heat_tile_h.setRange(0.3, 5.0)
        self._heat_tile_h.setValue(0.8)
        self._heat_tile_h.setSingleStep(0.1)
        self._heat_tile_h.setDecimals(1)
        self._heat_reverse_chk = QCheckBox("Reverse colormap")
        self._heat_reverse_chk.setChecked(False)
        self._heat_orient = QComboBox()
        self._heat_orient.addItems(["Landscape", "Portrait"])
        hf.addRow("Colormap:",   self._cmap)
        if self._mode == "ki":
            hf.addRow("Columns:", self._heat_col_combo)
            hf.addRow(self._heat_host_lbl, self._heat_host_combo)
        hf.addRow("Annotation:", self._annot_fmt)
        hf.addRow("Tile width:",  self._heat_tile_w)
        hf.addRow("Tile height:", self._heat_tile_h)
        hf.addRow(self._heat_reverse_chk)
        hf.addRow("Orientation:", self._heat_orient)
        self._heat_traffic_chk = QCheckBox("Traffic light (discrete bands)")
        self._heat_traffic_chk.setChecked(False)
        self._heat_traffic_steps_lbl = QLabel("Steps:")
        self._heat_traffic_steps_spin = QSpinBox()
        self._heat_traffic_steps_spin.setRange(2, 20)
        self._heat_traffic_steps_spin.setValue(7)
        self._heat_traffic_steps_spin.setToolTip("Number of discrete colour bands")
        self._heat_traffic_steps_lbl.setVisible(False)
        self._heat_traffic_steps_spin.setVisible(False)
        self._heat_traffic_chk.stateChanged.connect(
            lambda s: (self._heat_traffic_steps_lbl.setVisible(bool(s)),
                       self._heat_traffic_steps_spin.setVisible(bool(s)),
                       self._heat_traffic_range_chk.setVisible(bool(s)),
                       self._heat_traffic_colors_chk.setVisible(bool(s))))
        hf.addRow(self._heat_traffic_chk)
        hf.addRow(self._heat_traffic_steps_lbl, self._heat_traffic_steps_spin)
        self._heat_traffic_range_chk = QCheckBox("Manual range")
        self._heat_traffic_range_chk.setChecked(False)
        self._heat_traffic_range_chk.setVisible(False)
        _pk_unit = "pKd" if self._mode == "kd" else "pKi"
        self._heat_traffic_vmin_lbl = QLabel(f"Min ({_pk_unit}):")
        self._heat_traffic_vmin_spin = QDoubleSpinBox()
        self._heat_traffic_vmin_spin.setRange(0.0, 15.0)
        self._heat_traffic_vmin_spin.setValue(4.0)
        self._heat_traffic_vmin_spin.setDecimals(1)
        self._heat_traffic_vmin_spin.setSingleStep(0.5)
        self._heat_traffic_vmax_lbl = QLabel(f"Max ({_pk_unit}):")
        self._heat_traffic_vmax_spin = QDoubleSpinBox()
        self._heat_traffic_vmax_spin.setRange(0.0, 15.0)
        self._heat_traffic_vmax_spin.setValue(9.0)
        self._heat_traffic_vmax_spin.setDecimals(1)
        self._heat_traffic_vmax_spin.setSingleStep(0.5)
        for _w in (self._heat_traffic_vmin_lbl, self._heat_traffic_vmin_spin,
                   self._heat_traffic_vmax_lbl, self._heat_traffic_vmax_spin):
            _w.setVisible(False)
        self._heat_traffic_range_chk.stateChanged.connect(
            lambda s: [_w.setVisible(bool(s)) for _w in (
                self._heat_traffic_vmin_lbl, self._heat_traffic_vmin_spin,
                self._heat_traffic_vmax_lbl, self._heat_traffic_vmax_spin)])
        hf.addRow(self._heat_traffic_range_chk)
        hf.addRow(self._heat_traffic_vmin_lbl, self._heat_traffic_vmin_spin)
        hf.addRow(self._heat_traffic_vmax_lbl, self._heat_traffic_vmax_spin)
        self._heat_traffic_colors_chk = QCheckBox("Custom band colours")
        self._heat_traffic_colors_chk.setChecked(False)
        self._heat_traffic_colors_chk.setVisible(False)
        self._heat_traffic_colors_widget = QWidget()
        _htcvbox = QVBoxLayout(self._heat_traffic_colors_widget)
        _htcvbox.setContentsMargins(0, 0, 0, 0)
        _htcvbox.setSpacing(2)
        self._heat_traffic_color_rows = []
        self._heat_traffic_colors_widget.setVisible(False)
        self._heat_traffic_colors_chk.stateChanged.connect(
            lambda s: (self._heat_traffic_colors_widget.setVisible(bool(s)),
                       self._rebuild_permode_traffic_colors() if bool(s) else None))
        self._heat_traffic_steps_spin.valueChanged.connect(
            lambda _: self._rebuild_permode_traffic_colors()
                      if self._heat_traffic_colors_chk.isChecked() else None)
        hf.addRow(self._heat_traffic_colors_chk)
        hf.addRow(self._heat_traffic_colors_widget)
        cl.addWidget(self._gb_heat)
        self._gb_heat.setVisible(False)

        # Spider options ──────────────────────────────────────────────────
        self._gb_spider = QGroupBox("Spider Options")
        sv = QVBoxLayout(self._gb_spider)
        sf = QFormLayout()
        self._spider_yscale = QComboBox()
        self._spider_yscale.addItems(["pK  (−log₁₀ K/M)", "Raw µM", "log₁₀(K/µM)"])
        self._spider_cmap = QComboBox()
        self._spider_cmap.addItems(["Blues", "viridis", "plasma", "magma",
                                     "inferno", "coolwarm", "RdYlGn", "YlOrRd"])
        self._spider_reverse_chk = QCheckBox("Reverse colormap")
        self._spider_reverse_chk.setChecked(False)
        self._spider_fill_alpha = QDoubleSpinBox()
        self._spider_fill_alpha.setRange(0.0, 1.0)
        self._spider_fill_alpha.setValue(0.9)
        self._spider_fill_alpha.setSingleStep(0.05)
        self._spider_fill_alpha.setDecimals(2)
        self._spider_line_w = QDoubleSpinBox()
        self._spider_line_w.setRange(0.1, 5.0)
        self._spider_line_w.setValue(0.5)
        self._spider_line_w.setSingleStep(0.5)
        self._spider_line_w.setDecimals(1)
        self._spider_line_style = QComboBox()
        self._spider_line_style.addItems(["solid", "dotted", "dashed", "dashdot"])
        self._spider_line_col = ColorPickerRow("Outline:", "#000000")
        self._spider_fig_sz = QDoubleSpinBox()
        self._spider_fig_sz.setRange(4.0, 16.0)
        self._spider_fig_sz.setValue(7.0)
        self._spider_fig_sz.setSingleStep(0.5)
        self._spider_fig_sz.setDecimals(1)
        self._spider_margin = QDoubleSpinBox()
        self._spider_margin.setRange(0.5, 6.0)
        self._spider_margin.setValue(2.0)
        self._spider_margin.setSingleStep(0.25)
        self._spider_margin.setDecimals(2)
        self._spider_margin.setToolTip("Page margin around the charts (inches)")
        self._spider_label_sz = QDoubleSpinBox()
        self._spider_label_sz.setRange(4, 18)
        self._spider_label_sz.setValue(8)
        self._spider_label_sz.setSingleStep(0.5)
        self._spider_label_sz.setDecimals(1)
        self._spider_label_angle = QDoubleSpinBox()
        self._spider_label_angle.setRange(-90, 90)
        self._spider_label_angle.setValue(0)
        self._spider_label_angle.setSingleStep(5)
        self._spider_label_angle.setDecimals(0)
        self._spider_label_angle.setToolTip("Extra rotation added to radial label orientation (degrees)")
        self._spider_label_pad = QDoubleSpinBox()
        self._spider_label_pad.setRange(5, 60)
        self._spider_label_pad.setValue(18)
        self._spider_label_pad.setSingleStep(2)
        self._spider_label_pad.setDecimals(0)
        self._spider_label_pad.setToolTip("Distance between labels and chart edge")
        self._spider_cbar_shrink = QDoubleSpinBox()
        self._spider_cbar_shrink.setRange(0.2, 1.0)
        self._spider_cbar_shrink.setValue(0.75)
        self._spider_cbar_shrink.setSingleStep(0.05)
        self._spider_cbar_shrink.setDecimals(2)
        self._spider_cbar_lbl = QLineEdit("Gradient Scale")
        self._spider_cbar_lbl.setPlaceholderText("Colorbar label")
        self._spider_show_vals = QCheckBox("Show values on bars")
        self._spider_show_vals.setChecked(False)
        self._spider_show_vals.setToolTip("Display value ± SE at the end of each bar")
        self._spider_sort = QComboBox()
        if self._mode == "kd":
            self._spider_sort.addItems(["Alphabetical", "pK ↓ (strongest)",
                                         "pK ↑ (weakest)", "Host", "Dye"])
        else:
            self._spider_sort.addItems(["Alphabetical", "pK ↓ (strongest)",
                                         "pK ↑ (weakest)", "Host", "Dye", "Guest"])
        self._spider_group = QComboBox()
        if self._mode == "kd":
            self._spider_group.addItems(["None (single chart)", "Host", "Dye"])
        else:
            self._spider_group.addItems(["None (single chart)", "Host", "Dye", "Guest"])
        sf.addRow("Y-axis:",       self._spider_yscale)
        sf.addRow("Colormap:",     self._spider_cmap)
        sf.addRow(self._spider_reverse_chk)
        sf.addRow("Fill alpha:",   self._spider_fill_alpha)
        sf.addRow("Line width:",   self._spider_line_w)
        sf.addRow("Line style:",   self._spider_line_style)
        sf.addRow("Chart size:",   self._spider_fig_sz)
        sf.addRow("Margin:",       self._spider_margin)
        sf.addRow("Label size:",   self._spider_label_sz)
        sf.addRow("Label angle:",  self._spider_label_angle)
        sf.addRow("Label dist:",   self._spider_label_pad)
        sf.addRow("Cbar shrink:",  self._spider_cbar_shrink)
        sf.addRow("Cbar label:",   self._spider_cbar_lbl)
        sf.addRow(self._spider_show_vals)
        sf.addRow("Sort bars:",    self._spider_sort)
        sf.addRow("Split by:",     self._spider_group)
        sv.addLayout(sf)
        sv.addWidget(self._spider_line_col)
        cl.addWidget(self._gb_spider)
        self._gb_spider.setVisible(False)

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
        self._axis_fsz_sum = QDoubleSpinBox()
        self._axis_fsz_sum.setRange(4, 18)
        self._axis_fsz_sum.setValue(9)
        self._axis_fsz_sum.setSingleStep(0.5)
        self._axis_fsz_sum.setDecimals(1)
        self._tick_fsz_sum = QDoubleSpinBox()
        self._tick_fsz_sum.setRange(3, 16)
        self._tick_fsz_sum.setValue(8)
        self._tick_fsz_sum.setSingleStep(0.5)
        self._tick_fsz_sum.setDecimals(1)
        ff_lay.addRow("Family:", self._fam)
        ff_lay.addRow("Size:",   self._fsz)
        ff_lay.addRow("Axis labels:", self._axis_fsz_sum)
        ff_lay.addRow("Tick numbers:", self._tick_fsz_sum)
        cl.addWidget(gb_font)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._render)
        cl.addWidget(refresh_btn)

        export_btn = QPushButton("Export…")
        export_btn.setToolTip("Save the current chart as PDF, PNG or SVG")
        export_btn.clicked.connect(self._export)
        cl.addWidget(export_btn)

        cl.addStretch()
        splitter.addWidget(ctrl_scroll)

        # ── right: scrollable canvas ──────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        splitter.addWidget(self._scroll)
        splitter.setSizes([260, 740])

        # signals
        self._bar_rb.toggled.connect(self._on_type)
        self._heat_rb.toggled.connect(self._on_type)
        self._spider_rb.toggled.connect(self._on_type)
        self._filt.currentIndexChanged.connect(self._render)

    def _on_type(self, _checked=None):
        self._gb_bar.setVisible(self._bar_rb.isChecked())
        self._gb_heat.setVisible(self._heat_rb.isChecked())
        self._gb_spider.setVisible(self._spider_rb.isChecked())
        self._render()

    def _on_heat_col_changed(self, text):
        per_host = text == "Dye vs Guest (per Host)"
        self._heat_host_combo.setVisible(per_host)
        self._heat_host_lbl.setVisible(per_host)

    # ── public ─────────────────────────────────────────────────────────────────
    def load(self, df):
        self._df = df
        if self._mode == "ki" and df is not None and "Host" in df.columns:
            prev = self._heat_host_combo.currentText()
            self._heat_host_combo.blockSignals(True)
            self._heat_host_combo.clear()
            self._heat_host_combo.addItems(sorted(df["Host"].dropna().unique()))
            if prev:
                idx = self._heat_host_combo.findText(prev)
                if idx >= 0:
                    self._heat_host_combo.setCurrentIndex(idx)
            self._heat_host_combo.blockSignals(False)
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
        afs = self._axis_fsz_sum.value()
        tfs = self._tick_fsz_sum.value()
        if self._current_fig is not None:
            self._current_fig.clear()
            self._current_fig = None
        try:
            if self._bar_rb.isChecked():
                fig = self._bar_chart(df, pv, fs, ff, afs, tfs)
            elif self._heat_rb.isChecked():
                fig = self._heatmap(df, pv, fs, ff, afs, tfs)
            else:
                fig = self._spider_chart(df, pv, fs, ff, afs, tfs)
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

    def _bar_chart(self, df, pv, fs, ff, afs=9, tfs=8):
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
                           fontsize=tfs, fontfamily=ff)
        ax.set_ylabel(used_ylabel, fontsize=afs, fontfamily=ff)
        ax.set_title(f"{'pKd' if self._mode == 'kd' else 'pKi'} — summary",
                     fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        ax.tick_params(axis="y", labelsize=tfs)
        ax.grid(axis="y", alpha=0.35, zorder=0);  ax.set_axisbelow(True)
        if leg:
            ax.legend(handles=leg, title=grp, fontsize=tfs,
                      title_fontsize=tfs,
                      bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
        fig.tight_layout()
        return fig

    def _rebuild_permode_traffic_colors(self):
        layout = self._heat_traffic_colors_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._heat_traffic_color_rows.clear()
        n_steps = self._heat_traffic_steps_spin.value()
        cmap_name = self._cmap.currentText()
        _base_name = cmap_name[:-2] if cmap_name.endswith("_r") else cmap_name
        _base = matplotlib.colormaps.get_cmap(_base_name)
        rev = self._heat_reverse_chk.isChecked()
        for i in range(n_steps):
            frac = i / max(n_steps - 1, 1)
            if rev:
                frac = 1.0 - frac
            rgba = _base(frac)
            hex_c = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            row = ColorPickerRow(f"Band {i + 1}:", hex_c)
            layout.addWidget(row)
            self._heat_traffic_color_rows.append(row)

    def _heatmap(self, df, pv, fs, ff, afs=9, tfs=8):
        cmap     = self._cmap.currentText()
        fmt      = self._annot_fmt.currentText()
        annot    = fmt != "None"
        tile_w   = self._heat_tile_w.value()
        tile_h   = self._heat_tile_h.value()
        reverse  = self._heat_reverse_chk.isChecked()
        portrait = self._heat_orient.currentText() == "Portrait"

        if reverse:
            cmap = cmap + "_r"

        if self._mode == "kd":
            row_col = "Host";  col_col = "Dye"
        else:
            cm = self._heat_col_combo.currentText()
            if cm == "Dye vs Guest (per Host)":
                host = self._heat_host_combo.currentText()
                if host:
                    df = df[df["Host"] == host].copy()
                row_col = "Dye";  col_col = "Guest"
            elif cm == "Host | Dye":
                df = df.copy()
                df["_hd"] = df["Host"] + " | " + df["Dye"]
                row_col = "Guest";  col_col = "_hd"
            else:
                row_col = "Guest";  col_col = cm

        pivot = df.groupby([row_col, col_col])[pv].mean().unstack(col_col)
        xlabel = col_col.replace("_hd", "Host | Dye")
        ylabel = row_col

        if portrait:
            pivot = pivot.T
            xlabel, ylabel = ylabel, xlabel

        nr, nc = pivot.shape
        heat_w = nc * tile_w
        heat_h = nr * tile_h
        fig_w = heat_w + 3.0
        fig_h = heat_h + 2.0
        fig = Figure(figsize=(fig_w, fig_h))
        ax  = fig.add_subplot(111)

        data = pivot.values.astype(float)
        vmin = float(np.nanmin(data)) if not np.all(np.isnan(data)) else 0.0
        vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0

        traffic = self._heat_traffic_chk.isChecked()
        if traffic and self._heat_traffic_range_chk.isChecked():
            vmin = self._heat_traffic_vmin_spin.value()
            vmax = self._heat_traffic_vmax_spin.value()

        if traffic:
            n_steps = self._heat_traffic_steps_spin.value()
            t_bounds = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1.0)
            if len(t_bounds) - 1 > n_steps:
                t_bounds = np.linspace(np.floor(vmin), np.ceil(vmax),
                                       n_steps + 1)
            n_colors = len(t_bounds) - 1
            if (self._heat_traffic_colors_chk.isChecked()
                    and self._heat_traffic_color_rows):
                _cols = []
                for _cr in self._heat_traffic_color_rows[:n_colors]:
                    _c = QColor(_cr.color)
                    _cols.append((_c.redF(), _c.greenF(), _c.blueF(), 1.0))
                while len(_cols) < n_colors:
                    _cols.append((0.5, 0.5, 0.5, 1.0))
            else:
                _base_name = cmap[:-2] if cmap.endswith("_r") else cmap
                _base = matplotlib.colormaps.get_cmap(_base_name)
                _cols = [_base(i / max(n_colors - 1, 1)) for i in range(n_colors)]
                if cmap.endswith("_r"):
                    _cols = list(reversed(_cols))
            _disc_cmap = ListedColormap(_cols)
            _norm = BoundaryNorm(t_bounds, _disc_cmap.N)
            im = ax.imshow(data, cmap=_disc_cmap, norm=_norm, aspect="auto")
        else:
            t_bounds = None
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_aspect(tile_h / tile_w)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right",
                           fontsize=tfs, fontfamily=ff)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(pivot.index, fontsize=tfs, fontfamily=ff)

        if annot:
            span = max(vmax - vmin, 1e-9)
            for i in range(nr):
                for j in range(nc):
                    v = data[i, j]
                    if not np.isnan(v):
                        txt_c = "white" if (v - vmin) / span > 0.5 else "black"
                        ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                                fontsize=tfs, fontfamily=ff, color=txt_c)

        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        if traffic and t_bounds is not None:
            cb.set_ticks(t_bounds)
            cb.set_ticklabels([f"{b:.4g}" for b in t_bounds])
        cb.set_label(self._y_label(), fontsize=afs, fontfamily=ff)
        cb.ax.tick_params(labelsize=tfs)
        ax.set_xlabel(xlabel, fontsize=afs, fontfamily=ff)
        ax.set_ylabel(ylabel, fontsize=afs, fontfamily=ff)
        _pk = 'pKd' if self._mode == 'kd' else 'pKi'
        _title = f"{_pk} — heatmap"
        if (self._mode == "ki"
                and self._heat_col_combo.currentText() == "Dye vs Guest (per Host)"):
            host = self._heat_host_combo.currentText()
            if host:
                _title = f"{_pk} — {host}"
        ax.set_title(_title, fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        fig.tight_layout()
        return fig

    def _spider_bar_label(self, row, split_by):
        if self._mode == "kd":
            parts = []
            if split_by != "Host":
                parts.append(str(row["Host"]))
            if split_by != "Dye":
                parts.append(str(row["Dye"]))
            lbl = " | ".join(parts) if parts else str(row["Host"])
            conc = row.get("Dye_Concentration", "")
            if conc != "":
                lbl += f" [{conc} µM]"
            return lbl
        else:
            parts = []
            if split_by != "Guest" and "Guest" in row.index:
                parts.append(str(row["Guest"]))
            host_dye = []
            if split_by != "Host":
                host_dye.append(str(row["Host"]))
            if split_by != "Dye" and "Dye" in row.index:
                host_dye.append(str(row["Dye"]))
            if host_dye:
                parts.append(" | ".join(host_dye))
            return " / ".join(parts) if parts else str(row["Host"])

    def _spider_sort_agg(self, agg, grp_df, sort_txt):
        if "↓" in sort_txt:
            agg = agg.sort_values(ascending=False)
        elif "↑" in sort_txt:
            agg = agg.sort_values(ascending=True)
        elif sort_txt in ("Host", "Dye", "Guest"):
            col = sort_txt
            if col in grp_df.columns:
                order = (grp_df.groupby("_bar")[col].first()
                         .sort_values().index)
                agg = agg.reindex(order).dropna()
        else:
            agg = agg.sort_index()
        return agg

    def _spider_chart(self, df, pv, fs, ff, afs=9, tfs=8):
        from matplotlib.colors import Normalize
        from matplotlib.ticker import MaxNLocator

        yscale      = self._spider_yscale.currentText()
        cmap_name   = self._spider_cmap.currentText()
        if self._spider_reverse_chk.isChecked():
            cmap_name += "_r"
        fill_a      = self._spider_fill_alpha.value()
        lw          = self._spider_line_w.value()
        ls          = self._spider_line_style.currentText()
        line_col    = self._spider_line_col.color
        cell_sz     = self._spider_fig_sz.value()
        margin      = self._spider_margin.value()
        cbar_lbl    = self._spider_cbar_lbl.text() or self._y_label()
        split_by    = self._spider_group.currentText()
        sort_txt    = self._spider_sort.currentText()
        label_sz    = self._spider_label_sz.value()
        label_angle = self._spider_label_angle.value()
        label_pad   = self._spider_label_pad.value()
        cbar_shrink = self._spider_cbar_shrink.value()
        show_vals   = self._spider_show_vals.isChecked()

        vc = self._val_col();  sc = self._se_col()
        df = df.copy()

        if "pK" in yscale:
            df["_y"]  = df[pv]
            df["_se"] = df["p_SE"] if "p_SE" in df.columns else np.nan
        elif "Raw" in yscale:
            df["_y"]  = df[vc]
            df["_se"] = df[sc] if sc in df.columns else np.nan
        else:
            df["_y"]  = np.log10(df[vc].clip(lower=1e-12))
            df["_se"] = np.nan

        df["_bar"] = df.apply(lambda r: self._spider_bar_label(r, split_by), axis=1)

        if split_by == "None (single chart)" or split_by not in df.columns:
            groups = [("", df)]
        else:
            groups = [(str(name), grp.reset_index(drop=True))
                      for name, grp in df.groupby(split_by, sort=True)]

        n_groups = len(groups)
        if n_groups == 0:
            fig = Figure(figsize=(cell_sz, cell_sz))
            fig.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=fs)
            return fig

        has_titles = n_groups > 1
        cols = min(n_groups, 3)
        rows = int(np.ceil(n_groups / cols))

        title_extra = 1.0 if has_titles else 0.0
        cbar_room = 3.0
        chart_w = cell_sz * cols
        chart_h = (cell_sz + title_extra) * rows
        fig_w = margin + chart_w + cbar_room + margin
        fig_h = margin + chart_h + margin

        fig = Figure(figsize=(fig_w, fig_h))

        left_f   = margin / fig_w
        right_f  = (margin + chart_w) / fig_w
        bottom_f = margin / fig_h
        top_f    = 1.0 - margin / fig_h

        gs = GridSpec(rows, cols, figure=fig,
                      left=left_f, right=right_f,
                      bottom=bottom_f, top=top_f,
                      hspace=0.95 if has_titles else 0.55,
                      wspace=0.55)

        all_y = df["_y"].dropna()
        if all_y.empty:
            fig.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=fs)
            return fig
        vmin_val = float(all_y.min())
        vmax_val = float(all_y.max())
        if vmax_val == vmin_val:
            vmax_val = vmin_val + 1.0
        colormap = matplotlib.colormaps.get_cmap(cmap_name)
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)

        for idx, (group_name, grp_df) in enumerate(groups):
            agg_y  = grp_df.groupby("_bar")["_y"].mean()
            agg_se = grp_df.groupby("_bar")["_se"].mean()
            agg_y  = self._spider_sort_agg(agg_y, grp_df, sort_txt)
            agg_se = agg_se.reindex(agg_y.index)
            labels = list(agg_y.index)
            values = list(agg_y.values)
            ses    = list(agg_se.values)
            n_vars = len(labels)

            ri, ci = divmod(idx, cols)
            ax = fig.add_subplot(gs[ri, ci], polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3, zorder=0)

            if n_vars < 1:
                ax.set_visible(False)
                continue

            angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
            bar_width = (2 * np.pi / n_vars * 0.85) if n_vars > 1 else (np.pi / 2)

            ax.bar(angles, values, width=bar_width, bottom=0,
                   color=[colormap(norm(v)) for v in values],
                   alpha=fill_a, edgecolor=line_col, linewidth=lw,
                   linestyle=ls, zorder=3)

            ax.set_xticks(angles)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            r_max = ax.get_ylim()[1]
            label_r = r_max * (1.0 + label_pad / 50.0)
            for angle, label_text in zip(angles, labels):
                angle_deg = np.degrees(angle) % 360
                if angle_deg < 10 or angle_deg > 350:
                    ha, va = "center", "bottom"
                elif 170 < angle_deg < 190:
                    ha, va = "center", "top"
                elif angle_deg < 180:
                    ha, va = "left", "center"
                else:
                    ha, va = "right", "center"
                ax.text(angle, label_r, label_text, ha=ha, va=va,
                        fontsize=label_sz, fontfamily=ff,
                        rotation=label_angle, clip_on=False, zorder=4)

            if show_vals:
                for angle, val, se in zip(angles, values, ses):
                    txt = f"{val:.2f}"
                    if not np.isnan(se) and se > 0:
                        txt += f" ± {se:.2f}"
                    angle_deg = np.degrees(angle) % 360
                    text_rot = 90 - angle_deg
                    if 90 < angle_deg < 270:
                        text_rot += 180
                    ax.text(angle, val * 0.85, txt, ha="center", va="center",
                            fontsize=tfs * 0.7, fontfamily=ff, fontweight="bold",
                            rotation=text_rot, clip_on=False, zorder=5)

            if group_name:
                pos = ax.get_position()
                title_y = pos.y1 + 0.8 / fig_h
                fig.text(pos.x0 + pos.width / 2, title_y,
                         group_name, ha="center", va="bottom",
                         fontsize=fs, fontweight="bold", fontfamily=ff)

        for idx in range(n_groups, rows * cols):
            ri, ci = divmod(idx, cols)
            fig.add_subplot(gs[ri, ci]).set_visible(False)

        cbar_left   = 1.0 - (margin + 1.2) / fig_w
        cbar_height = (top_f - bottom_f) * cbar_shrink
        cbar_bot    = bottom_f + (top_f - bottom_f - cbar_height) / 2
        cbar_width  = 0.02
        cax = fig.add_axes([cbar_left, cbar_bot, cbar_width, cbar_height])

        sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        ticks = np.linspace(vmin_val, vmax_val, 6)
        cbar.set_ticks(ticks)
        tick_labels = []
        for t in ticks:
            if "pK" in yscale:
                um = 10**(6 - t)
                tick_labels.append(f"{t:.1f}  ({um:.1e} µM)")
            elif "Raw" in yscale:
                tick_labels.append(f"{t:.2f} µM")
            else:
                um = 10**t
                tick_labels.append(f"{t:.1f}  ({um:.1e} µM)")
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=tfs)
        cbar.set_label(cbar_lbl, fontsize=afs)

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
        scroll.setMinimumWidth(300)
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
        self._buffer_map_chk = QCheckBox("Use buffer map")
        self._buffer_map_chk.setChecked(False)
        self._buffer_map_chk.setToolTip(
            "When ticked, expects a buffer_map/ folder inside the\n"
            "experiment folder containing Echo mapping .xlsx files.\n"
            "The Buffer label is used to distinguish the same\n"
            "Host-Dye combinations tested in different buffers.")
        ff.addRow(self._buffer_map_chk)
        self._multi_chrom_chk = QCheckBox("Multi-chromatic plates")
        self._multi_chrom_chk.setChecked(False)
        self._multi_chrom_chk.setToolTip(
            "Tick only if a single plate carries more than one dye. When\n"
            "ticked, each raw file's own filter-settings header is matched\n"
            "against the Chromatic DB above to target which chromatic block\n"
            "belongs to which dye — mismatches with the dye mapping are\n"
            "dropped.\n\n"
            "Left unticked (default), the Chromatic DB is not consulted:\n"
            "every plate is assumed single-dye, only its first chromatic\n"
            "block is used, and the dye comes straight from the dye mapping.\n"
            "A mismatch between the mapping and a manually renamed chromatic\n"
            "block only produces a warning in this mode.")
        ff.addRow(self._multi_chrom_chk)
        self._exceptions_chk = QCheckBox("Use exceptions file")
        self._exceptions_chk.setChecked(False)
        self._exceptions_chk.setToolTip(
            "When ticked, expects an exceptions_map/ folder inside the\n"
            "experiment folder containing Echo 'Exceptions' reports (.csv,\n"
            "columns 'Destination Plate Name' / 'Destination Well').\n"
            "Every well listed there had a failed dispense — it is excluded\n"
            "from curve fitting and shown as a black cross on the plots\n"
            "('Show excluded' toggle in Plot Appearance controls visibility).")
        ff.addRow(self._exceptions_chk)
        hint = QLabel("Raw/  dye_map/  host_map/  blank_map/\n"
                      "buffer_map/ (optional, tick above)\n"
                      "exceptions_map/ (optional, tick above)\n"
                      "← expected inside experiment folder\n\n"
                      "Chromatic DB: optional Dye/Filter lookup table\n"
                      "(folder of .xlsx, columns Dye/Filter) used to\n"
                      "auto-resolve multichromatic raw files by their\n"
                      "own filter-settings header — only used when\n"
                      "'Multi-chromatic plates' above is ticked.")
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
        self._cross_conc_alpha_spin.setEnabled(False)
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

        box_col = QGroupBox("Plot Appearance")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", pl_fda.PLOT_COLOR_DATA)
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", pl_fda.PLOT_COLOR)
        self._color_resid_row = ColorPickerRow("Residuals:  ", pl_fda.PLOT_COLOR_DATA)
        self._color_data_row.color_changed.connect(self._on_color_changed)
        self._color_fit_row.color_changed.connect(self._on_color_changed)
        self._color_resid_row.color_changed.connect(self._on_color_changed)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        sz_form = QFormLayout()
        self._sz_fit = QDoubleSpinBox(); self._sz_fit.setRange(0.5, 8.0); self._sz_fit.setValue(2.0); self._sz_fit.setSingleStep(0.5); self._sz_fit.setDecimals(1)
        self._sz_data = QDoubleSpinBox(); self._sz_data.setRange(1.0, 12.0); self._sz_data.setValue(4.0); self._sz_data.setSingleStep(0.5); self._sz_data.setDecimals(1)
        self._sz_errorbar = QDoubleSpinBox(); self._sz_errorbar.setRange(0.5, 6.0); self._sz_errorbar.setValue(1.5); self._sz_errorbar.setSingleStep(0.5); self._sz_errorbar.setDecimals(1)
        self._sz_resid = QDoubleSpinBox(); self._sz_resid.setRange(0.5, 8.0); self._sz_resid.setValue(2.5); self._sz_resid.setSingleStep(0.5); self._sz_resid.setDecimals(1)
        self._sz_fit.valueChanged.connect(self._on_color_changed)
        self._sz_data.valueChanged.connect(self._on_color_changed)
        self._sz_errorbar.valueChanged.connect(self._on_color_changed)
        self._sz_resid.valueChanged.connect(self._on_color_changed)
        sz_form.addRow("Fit curve lw:", self._sz_fit)
        sz_form.addRow("Data point size:", self._sz_data)
        sz_form.addRow("Error bar lw:", self._sz_errorbar)
        sz_form.addRow("Residual size:", self._sz_resid)
        cl.addLayout(sz_form)
        layout.addWidget(box_col)

        box_tf = QGroupBox("Title Font")
        tf_lay = QFormLayout(box_tf)
        self._title_fam = QComboBox()
        self._title_fam.addItems([
            "sans-serif", "serif", "monospace",
            "DejaVu Sans", "Arial", "Helvetica"])
        self._title_fsz = QDoubleSpinBox()
        self._title_fsz.setRange(5, 18)
        self._title_fsz.setValue(11)
        self._title_fsz.setSingleStep(0.5)
        self._title_fsz.setDecimals(1)
        self._title_style = QComboBox()
        self._title_style.addItems(["Normal", "Bold", "Italic", "Bold Italic"])
        self._title_fam.currentIndexChanged.connect(self._on_title_font_changed)
        self._title_fsz.valueChanged.connect(self._on_title_font_changed)
        self._title_style.currentIndexChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        self._axis_fsz = QDoubleSpinBox()
        self._axis_fsz.setRange(4, 18)
        self._axis_fsz.setValue(10)
        self._axis_fsz.setSingleStep(0.5)
        self._axis_fsz.setDecimals(1)
        self._axis_fsz.valueChanged.connect(self._on_title_font_changed)
        self._tick_fsz = QDoubleSpinBox()
        self._tick_fsz.setRange(3, 16)
        self._tick_fsz.setValue(10)
        self._tick_fsz.setSingleStep(0.5)
        self._tick_fsz.setDecimals(1)
        self._tick_fsz.valueChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Style:",       self._title_style)
        tf_lay.addRow("Axis labels:", self._axis_fsz)
        tf_lay.addRow("Tick numbers:", self._tick_fsz)
        layout.addWidget(box_tf)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._normalise_chk = QCheckBox("Normalise Y-axis (0–1)")
        self._normalise_chk.setChecked(False)
        self._normalise_chk.stateChanged.connect(self._on_display_changed)
        self._sci_notation_chk = QCheckBox("Scientific notation (Y-axis)")
        self._sci_notation_chk.setChecked(False)
        self._sci_notation_chk.stateChanged.connect(self._on_display_changed)
        self._xaxis_manual_chk = QCheckBox("Manual X-axis range")
        self._xaxis_manual_chk.setChecked(False)
        self._xaxis_manual_chk.setToolTip(
            "Override the automatic (data-range) X-axis limits and tick\n"
            "spacing on every curve plot — main view, PDF preview, and\n"
            "PDF/PNG export.")
        self._xaxis_manual_chk.stateChanged.connect(self._on_display_changed)
        xf = QFormLayout()
        self._x_min_spin = QDoubleSpinBox()
        self._x_min_spin.setRange(-1e6, 1e6)
        self._x_min_spin.setDecimals(3)
        self._x_min_spin.setSingleStep(0.5)
        self._x_min_spin.setValue(0.0)
        self._x_min_spin.setEnabled(False)
        self._x_max_spin = QDoubleSpinBox()
        self._x_max_spin.setRange(-1e6, 1e6)
        self._x_max_spin.setDecimals(3)
        self._x_max_spin.setSingleStep(0.5)
        self._x_max_spin.setValue(10.0)
        self._x_max_spin.setEnabled(False)
        self._x_tick_spin = QDoubleSpinBox()
        self._x_tick_spin.setRange(0.0, 1e6)
        self._x_tick_spin.setDecimals(3)
        self._x_tick_spin.setSingleStep(0.5)
        self._x_tick_spin.setValue(0.0)
        self._x_tick_spin.setSpecialValueText("Auto")
        self._x_tick_spin.setEnabled(False)
        for w in (self._x_min_spin, self._x_max_spin, self._x_tick_spin):
            w.valueChanged.connect(self._on_display_changed)
        self._xaxis_manual_chk.toggled.connect(self._x_min_spin.setEnabled)
        self._xaxis_manual_chk.toggled.connect(self._x_max_spin.setEnabled)
        self._xaxis_manual_chk.toggled.connect(self._x_tick_spin.setEnabled)
        xf.addRow("X min:",      self._x_min_spin)
        xf.addRow("X max:",      self._x_max_spin)
        xf.addRow("X tick step:", self._x_tick_spin)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(True)
        self._resid_chk.setToolTip("Show residual sub-panel beneath each binding curve")
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._show_excluded_chk = QCheckBox("Show excluded (dispense failed) points")
        self._show_excluded_chk.setChecked(True)
        self._show_excluded_chk.setToolTip(
            "Show wells flagged by the exceptions file as black crosses.\n"
            "They are always excluded from fitting regardless of this toggle —\n"
            "this only controls whether they're drawn on the plots.")
        self._show_excluded_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On Save: write one PDF per curve into\n"
            "results/plots/individual/PASS|FAIL/")
        self._summary_chk = QCheckBox("Show summary plots")
        self._summary_chk.setChecked(True)
        self._summary_chk.setToolTip("pKd bar-chart / heatmap / spider summary plots")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._normalise_chk)
        dl.addWidget(self._sci_notation_chk)
        dl.addWidget(self._xaxis_manual_chk)
        dl.addLayout(xf)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._show_excluded_chk)
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
        self._tabs.addTab(self._summary_tab, "Summary Plots")   # 6

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

    def _get_config(self) -> dict:
        model_map = {
            "Auto (AICc)":             "auto",
            "One-site only":           "one_site",
            "Quadratic only":          "quadratic",
            "One-site (quenching only)": "stern_volmer",  # internal key kept for compatibility
        }
        fw, fs = _parse_title_style(self._title_style.currentText())
        base = self._input_row.path
        return {
            "raw_folder":            os.path.join(base, "Raw"),
            "dye_folder":            os.path.join(base, "dye_map"),
            "host_folder":           os.path.join(base, "host_map"),
            "blank_folder":          os.path.join(base, "blank_map"),
            "buffer_folder":         os.path.join(base, "buffer_map") if self._buffer_map_chk.isChecked() else None,
            "chromatic_folder":      self._chrom_db_row.path,
            "multi_chromatic":       self._multi_chrom_chk.isChecked(),
            "exceptions_folder":     (os.path.join(base, "exceptions_map")
                                       if self._exceptions_chk.isChecked() else None),
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
            "axis_fontsize":         self._axis_fsz.value(),
            "tick_fontsize":         self._tick_fsz.value(),
            "normalise_y":           self._normalise_chk.isChecked(),
            "sci_notation_y":        self._sci_notation_chk.isChecked(),
            "lw_fit":                self._sz_fit.value(),
            "ms_data":               self._sz_data.value(),
            "lw_errorbar":           self._sz_errorbar.value(),
            "ms_resid":              self._sz_resid.value(),
            "x_min":                 (self._x_min_spin.value()
                                       if self._xaxis_manual_chk.isChecked() else None),
            "x_max":                 (self._x_max_spin.value()
                                       if self._xaxis_manual_chk.isChecked() else None),
            "x_tick_step":           (self._x_tick_spin.value()
                                       if self._xaxis_manual_chk.isChecked()
                                       and self._x_tick_spin.value() > 0 else None),
            "show_excluded":         self._show_excluded_chk.isChecked(),
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_fda.PASS_R2_DEFAULT)
        self._kd_lo_spin.setValue(pl_fda.KD_RANGE_FACTOR_LO_DEFAULT)
        self._kd_hi_spin.setValue(pl_fda.KD_RANGE_FACTOR_HI_DEFAULT)
        self._grubbs_spin.setValue(0.05)
        self._cross_conc_chk.setChecked(False)   # default OFF per pipeline v4
        self._cross_conc_alpha_spin.setValue(0.01)
        self._model_combo.setCurrentIndex(0)
        self._buffer_map_chk.setChecked(False)
        self._multi_chrom_chk.setChecked(False)
        self._exceptions_chk.setChecked(False)
        self._color_data_row.set_color(pl_fda.PLOT_COLOR_DATA)
        self._color_fit_row.set_color(pl_fda.PLOT_COLOR)
        self._color_resid_row.set_color(pl_fda.PLOT_COLOR_DATA)
        self._sz_fit.setValue(2.0)
        self._sz_data.setValue(4.0)
        self._sz_errorbar.setValue(1.5)
        self._sz_resid.setValue(2.5)
        self._title_fsz.setValue(12)
        self._title_fam.setCurrentText("sans-serif")
        self._title_style.setCurrentText("Normal")
        self._axis_fsz.setValue(10)
        self._tick_fsz.setValue(10)
        self._normalise_chk.setChecked(False)
        self._sci_notation_chk.setChecked(False)
        self._xaxis_manual_chk.setChecked(False)
        self._x_min_spin.setValue(0.0)
        self._x_max_spin.setValue(10.0)
        self._x_tick_spin.setValue(0.0)
        self._show_excluded_chk.setChecked(True)
        self._resid_chk.setChecked(False)
        self._ind_export_chk.setChecked(False)
        self._summary_chk.setChecked(True)
        if self._state.plot_data:
            self._refresh_results()

    def _schedule_refresh(self):
        if not self._state.plot_data:
            return
        if not hasattr(self, "_refresh_timer"):
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._refresh_results)
        self._refresh_timer.start(150)

    def _on_color_changed(self, _hex):
        self._schedule_refresh()

    def _on_display_changed(self, _state):
        self._schedule_refresh()

    def _on_title_font_changed(self, _=None):
        self._schedule_refresh()

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
                         args=(cfg["dye_folder"], cfg["host_folder"]),
                         kwargs={"buffer_folder": cfg["buffer_folder"]})
        elif self._stage == 1:
            self._launch(pl_fda.load_plates,
                         args=(cfg["raw_folder"],),
                         kwargs={"chromatic_folder": cfg["chromatic_folder"],
                                 "multi_chromatic": cfg["multi_chromatic"]})
        elif self._stage == 2:
            self._launch(pl_fda.merge_blanks,
                         args=(self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"]),
                         kwargs={"chromatic_folder": cfg["chromatic_folder"],
                                 "multi_chromatic": cfg["multi_chromatic"],
                                 "exceptions_folder": cfg["exceptions_folder"]})
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
        self._worker.start()
        self._set_buttons_enabled()

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
            self._schedule_refresh()

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
                             axis_fontsize=cfg["axis_fontsize"],
                             tick_fontsize=cfg["tick_fontsize"],
                             layout_cfg=self._layout_panel.get_cfg(),
                             normalise_y=cfg["normalise_y"],
                             sci_notation_y=cfg["sci_notation_y"],
                             lw_fit=cfg["lw_fit"], ms_data=cfg["ms_data"],
                             lw_errorbar=cfg["lw_errorbar"], ms_resid=cfg["ms_resid"],
                             x_min=cfg["x_min"], x_max=cfg["x_max"],
                             x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
        self._tabs.setTabEnabled(7, True)
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)
        self._preview_dirty = True

    def _on_layout_changed(self):
        self._preview_dirty = True
        self._schedule_refresh()

    def _on_tab_changed(self, idx):
        if self._tabs.tabText(idx) == "PDF Preview" and self._state.plot_data:
            if getattr(self, "_preview_dirty", True):
                self._refresh_preview()
                self._preview_dirty = False

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
                                axis_fontsize=cfg["axis_fontsize"],
                                tick_fontsize=cfg["tick_fontsize"],
                                layout_cfg=lc, dpi=100,
                                lw_fit=cfg["lw_fit"], ms_data=cfg["ms_data"],
                                lw_errorbar=cfg["lw_errorbar"], ms_resid=cfg["ms_resid"],
                                x_min=cfg["x_min"], x_max=cfg["x_max"],
                                x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
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
        state       = self._state

        normalise   = cfg["normalise_y"]
        sci_notn    = cfg["sci_notation_y"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_fda.save_outputs(state, cfg["output_folder"],
                                       cfg["r2_threshold"], progress_cb,
                                       plot_color_data=color_data,
                                       plot_color_fit=color_fit,
                                       plot_color_resid=color_resid,
                                       show_residuals=show_resid,
                                       export_individual=export_ind,
                                       layout_cfg=lc,
                                       title_fontsize=cfg["title_fontsize"],
                                       title_fontfamily=cfg["title_fontfamily"],
                                       title_fontweight=cfg["title_fontweight"],
                                       title_fontstyle=cfg["title_fontstyle"],
                                       axis_fontsize=cfg["axis_fontsize"],
                                       tick_fontsize=cfg["tick_fontsize"],
                                       input_folder=cfg["input_folder"],
                                       normalise_y=normalise,
                                       sci_notation_y=sci_notn,
                                       lw_fit=cfg["lw_fit"],
                                       ms_data=cfg["ms_data"],
                                       lw_errorbar=cfg["lw_errorbar"],
                                       ms_resid=cfg["ms_resid"],
                                       x_min=cfg["x_min"], x_max=cfg["x_max"],
                                       x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
        self._worker = StageWorker(_save_fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_save_done)
        self._worker.error.connect(lambda tb: self._log.append(f"\n[ERROR]\n{tb}"))
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Saving…")
        self._worker.start()

    def _on_save_done(self, result):
        self._save_btn.setEnabled(True)
        self._set_buttons_enabled()
        if isinstance(result, Exception):
            self._status_lbl.setText("Save failed — see log")
            return
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
        scroll.setMinimumWidth(300)
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
        self._buffer_map_chk = QCheckBox("Use buffer map")
        self._buffer_map_chk.setChecked(False)
        self._buffer_map_chk.setToolTip(
            "When ticked, expects a buffer_map/ folder inside the\n"
            "experiment folder containing Echo mapping .xlsx files.\n"
            "The Buffer label is used to distinguish the same\n"
            "Host-Dye-Guest combinations tested in different buffers.")
        ff.addRow(self._buffer_map_chk)
        self._exceptions_chk = QCheckBox("Use exceptions file")
        self._exceptions_chk.setChecked(False)
        self._exceptions_chk.setToolTip(
            "When ticked, expects an exceptions_map/ folder inside the\n"
            "experiment folder containing Echo 'Exceptions' reports (.csv,\n"
            "columns 'Destination Plate Name' / 'Destination Well').\n"
            "Every well listed there had a failed dispense — it is excluded\n"
            "from curve fitting and shown as a black cross on the plots\n"
            "('Show excluded' toggle in Plot Appearance controls visibility).")
        ff.addRow(self._exceptions_chk)
        hint = QLabel("Experiment must contain:\nRaw/ dye_map/ host_map/\nguest_map/ blank_map/\n"
                      "buffer_map/ (optional, tick above)\n"
                      "exceptions_map/ (optional, tick above)\n\n"
                      "Chromatic DB: optional Dye/Filter lookup table\n"
                      "used to auto-resolve chromatic blocks from each\n"
                      "raw file's own filter-settings header — only used\n"
                      "when 'Multi-chromatic plates' below is ticked.")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        ff.addRow(hint)
        layout.addWidget(box_f)

        box_c = QGroupBox("Chromatic → Dye (manual override)")
        cc    = QVBoxLayout(box_c)
        self._multi_chrom_chk = QCheckBox("Multi-chromatic plates")
        self._multi_chrom_chk.setChecked(False)
        self._multi_chrom_chk.setToolTip(
            "Tick only if a single plate carries more than one dye. When\n"
            "ticked, each raw file's own filter-settings header is matched\n"
            "against the Chromatic DB above to target which chromatic block\n"
            "belongs to which dye; entries below override the auto-detected\n"
            "mapping (e.g. a dye not yet in the Chromatic DB, or a raw file\n"
            "with no filter-settings header). Use the 'Plate (substring)'\n"
            "column to target specific plates; leave it empty for a global\n"
            "override. A mismatch between the resolved chromatic and the\n"
            "dye mapping drops those rows.\n\n"
            "Left unticked (default), the Chromatic DB is not consulted:\n"
            "every plate is assumed single-dye, only its first chromatic\n"
            "block is loaded, and the dye comes straight from the dye\n"
            "mapping.")
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

        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["Auto (AICc)", "Standard", "Hill Slope", "Morrison", "Biphasic"])
        self._model_combo.setToolTip(
            "Auto (AICc): tries every applicable model (Standard, Hill Slope,\n"
            "Morrison when Ki is within the Morrison gate above) and keeps\n"
            "the best by AICc. Biphasic is excluded from Auto by default.\n\n"
            "Forcing a model (Standard/Hill Slope/Morrison/Biphasic) fits\n"
            "only that model — Morrison bypasses the gate above (still needs\n"
            "a Host concentration); Biphasic bypasses its default disable.")
        self._model_combo.currentTextChanged.connect(self._on_model_combo_changed)

        self._morrison_degenerate_chk = QCheckBox("Show degenerate Morrison fits (diagnostic)")
        self._morrison_degenerate_chk.setEnabled(False)
        self._morrison_degenerate_chk.setToolTip(
            "Only applies when Model (above) is forced to Morrison.\n\n"
            "A forced Morrison fit is normally discarded — the curve shows\n"
            "nothing at all, not even a FAIL — when it lands in the\n"
            "numerically unidentifiable stoichiometric/step-function limit:\n"
            "  • Ki_fit < 5% of [Host] (see morrison_ki_frac), or\n"
            "  • the 10-90% transition spans < 1 log unit\n"
            "In that regime the fitted Ki no longer meaningfully\n"
            "constrains the curve shape, so it isn't reported.\n\n"
            "Check this to plot those curves anyway for scientific\n"
            "comparison. They're titled in orange and tagged\n"
            "'degenerate, Ki unreliable' — treat the Ki as illustrative,\n"
            "not a real measurement.")

        self._autofl_spin = QDoubleSpinBox()
        self._autofl_spin.setRange(0.1, 50.0)
        self._autofl_spin.setSingleStep(0.1)
        self._autofl_spin.setValue(pl_ki.GUEST_AUTOFL_FACTOR)
        self._autofl_spin.setDecimals(2)
        self._autofl_spin.valueChanged.connect(self._on_r2_changed)
        self._autofl_spin.setToolTip(
            "FAIL a curve if its guest-only wells' median fluorescence\n"
            "exceeds this factor × the dye-blank mean, on the SAME plate\n"
            "(and buffer, if used) as that curve — flags guests whose own\n"
            "fluorescence may be confounding the FI-F0 signal.\n\n"
            "Default 1.0 = any guest signal above the dye blank fails.\n"
            "This is a tight, no-margin gate — raise it (e.g. 1.5–3) if\n"
            "curves are being failed for autofluorescence that look fine\n"
            "otherwise (good R², well-determined Ki).")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Morrison gate:", self._morrison_spin)
        pf.addRow("Model:",        self._model_combo)
        pf.addRow("",              self._morrison_degenerate_chk)
        pf.addRow("Guest autofl. gate:", self._autofl_spin)
        layout.addWidget(box_p)

        box_col = QGroupBox("Plot Appearance")
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
        sz_form = QFormLayout()
        self._sz_fit = QDoubleSpinBox(); self._sz_fit.setRange(0.5, 8.0); self._sz_fit.setValue(2.0); self._sz_fit.setSingleStep(0.5); self._sz_fit.setDecimals(1)
        self._sz_data = QDoubleSpinBox(); self._sz_data.setRange(1.0, 12.0); self._sz_data.setValue(4.0); self._sz_data.setSingleStep(0.5); self._sz_data.setDecimals(1)
        self._sz_errorbar = QDoubleSpinBox(); self._sz_errorbar.setRange(0.5, 6.0); self._sz_errorbar.setValue(1.5); self._sz_errorbar.setSingleStep(0.5); self._sz_errorbar.setDecimals(1)
        self._sz_resid = QDoubleSpinBox(); self._sz_resid.setRange(0.5, 8.0); self._sz_resid.setValue(2.5); self._sz_resid.setSingleStep(0.5); self._sz_resid.setDecimals(1)
        self._sz_fit.valueChanged.connect(self._on_color_changed)
        self._sz_data.valueChanged.connect(self._on_color_changed)
        self._sz_errorbar.valueChanged.connect(self._on_color_changed)
        self._sz_resid.valueChanged.connect(self._on_color_changed)
        sz_form.addRow("Fit curve lw:", self._sz_fit)
        sz_form.addRow("Data point size:", self._sz_data)
        sz_form.addRow("Error bar lw:", self._sz_errorbar)
        sz_form.addRow("Residual size:", self._sz_resid)
        cl.addLayout(sz_form)
        layout.addWidget(box_col)

        box_tf = QGroupBox("Title Font")
        tf_lay = QFormLayout(box_tf)
        self._title_fam = QComboBox()
        self._title_fam.addItems([
            "sans-serif", "serif", "monospace",
            "DejaVu Sans", "Arial", "Helvetica"])
        self._title_fsz = QDoubleSpinBox()
        self._title_fsz.setRange(5, 18)
        self._title_fsz.setValue(11)
        self._title_fsz.setSingleStep(0.5)
        self._title_fsz.setDecimals(1)
        self._title_style = QComboBox()
        self._title_style.addItems(["Normal", "Bold", "Italic", "Bold Italic"])
        self._title_fam.currentIndexChanged.connect(self._on_title_font_changed)
        self._title_fsz.valueChanged.connect(self._on_title_font_changed)
        self._title_style.currentIndexChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        self._axis_fsz = QDoubleSpinBox()
        self._axis_fsz.setRange(4, 18)
        self._axis_fsz.setValue(10)
        self._axis_fsz.setSingleStep(0.5)
        self._axis_fsz.setDecimals(1)
        self._axis_fsz.valueChanged.connect(self._on_title_font_changed)
        self._tick_fsz = QDoubleSpinBox()
        self._tick_fsz.setRange(3, 16)
        self._tick_fsz.setValue(10)
        self._tick_fsz.setSingleStep(0.5)
        self._tick_fsz.setDecimals(1)
        self._tick_fsz.valueChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Style:",       self._title_style)
        tf_lay.addRow("Axis labels:", self._axis_fsz)
        tf_lay.addRow("Tick numbers:", self._tick_fsz)
        layout.addWidget(box_tf)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._normalise_chk = QCheckBox("Normalise Y-axis (0–1)")
        self._normalise_chk.setChecked(False)
        self._normalise_chk.stateChanged.connect(self._on_display_changed)
        self._sci_notation_chk = QCheckBox("Scientific notation (Y-axis)")
        self._sci_notation_chk.setChecked(False)
        self._sci_notation_chk.stateChanged.connect(self._on_display_changed)
        self._xaxis_manual_chk = QCheckBox("Manual X-axis range")
        self._xaxis_manual_chk.setChecked(False)
        self._xaxis_manual_chk.setToolTip(
            "Override the automatic (data-range) X-axis limits and tick\n"
            "spacing on every curve plot — main view, PDF preview, and\n"
            "PDF/PNG export. X is log10([Guest]/µM), so a tick step of 1\n"
            "means one order of magnitude.")
        self._xaxis_manual_chk.stateChanged.connect(self._on_display_changed)
        xf = QFormLayout()
        self._x_min_spin = QDoubleSpinBox()
        self._x_min_spin.setRange(-1e6, 1e6)
        self._x_min_spin.setDecimals(3)
        self._x_min_spin.setSingleStep(0.5)
        self._x_min_spin.setValue(-3.0)
        self._x_min_spin.setEnabled(False)
        self._x_max_spin = QDoubleSpinBox()
        self._x_max_spin.setRange(-1e6, 1e6)
        self._x_max_spin.setDecimals(3)
        self._x_max_spin.setSingleStep(0.5)
        self._x_max_spin.setValue(3.0)
        self._x_max_spin.setEnabled(False)
        self._x_tick_spin = QDoubleSpinBox()
        self._x_tick_spin.setRange(0.0, 1e6)
        self._x_tick_spin.setDecimals(3)
        self._x_tick_spin.setSingleStep(0.5)
        self._x_tick_spin.setValue(0.0)
        self._x_tick_spin.setSpecialValueText("Auto")
        self._x_tick_spin.setEnabled(False)
        for w in (self._x_min_spin, self._x_max_spin, self._x_tick_spin):
            w.valueChanged.connect(self._on_display_changed)
        self._xaxis_manual_chk.toggled.connect(self._x_min_spin.setEnabled)
        self._xaxis_manual_chk.toggled.connect(self._x_max_spin.setEnabled)
        self._xaxis_manual_chk.toggled.connect(self._x_tick_spin.setEnabled)
        xf.addRow("X min (log10):",      self._x_min_spin)
        xf.addRow("X max (log10):",      self._x_max_spin)
        xf.addRow("X tick step:", self._x_tick_spin)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(True)
        self._resid_chk.setToolTip("Show residual sub-panel beneath each binding curve")
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._show_excluded_chk = QCheckBox("Show excluded (dispense failed) points")
        self._show_excluded_chk.setChecked(True)
        self._show_excluded_chk.setToolTip(
            "Show wells flagged by the exceptions file as black crosses.\n"
            "They are always excluded from fitting regardless of this toggle —\n"
            "this only controls whether they're drawn on the plots.")
        self._show_excluded_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On Save: write one PDF per curve into\n"
            "results/plots/individual/PASS|FAIL/")
        self._summary_chk = QCheckBox("Show summary plots")
        self._summary_chk.setChecked(True)
        self._summary_chk.setToolTip("pKi bar-chart / heatmap / spider summary plots")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._normalise_chk)
        dl.addWidget(self._sci_notation_chk)
        dl.addWidget(self._xaxis_manual_chk)
        dl.addLayout(xf)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._show_excluded_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

        box_qc = QGroupBox("QC1 Y-axis (Control Wells)")
        qcl = QFormLayout(box_qc)
        self._qc_y_mode_combo = QComboBox()
        self._qc_y_mode_combo.addItems(["All points", "Tukey fence (robust)", "Manual"])
        self._qc_y_mode_combo.setToolTip(
            "All points (default): every point visible, plain auto-scaled axis.\n\n"
            "Tukey fence: bounds the Y-axis to the IQR fence, robust to a few\n"
            "very fluorescent guests stretching the range and flattening\n"
            "everything else — those points are still plotted, just clipped\n"
            "at the edges.\n\n"
            "Manual: set your own Y min/max below (Tukey fence fills in\n"
            "whichever side you leave blank).")
        self._qc_y_min_spin = QDoubleSpinBox()
        self._qc_y_min_spin.setRange(-1e9, 1e9)
        self._qc_y_min_spin.setDecimals(1)
        self._qc_y_min_spin.setSingleStep(1000.0)
        self._qc_y_min_spin.setValue(0.0)
        self._qc_y_min_spin.setEnabled(False)
        self._qc_y_max_spin = QDoubleSpinBox()
        self._qc_y_max_spin.setRange(-1e9, 1e9)
        self._qc_y_max_spin.setDecimals(1)
        self._qc_y_max_spin.setSingleStep(1000.0)
        self._qc_y_max_spin.setValue(100000.0)
        self._qc_y_max_spin.setEnabled(False)
        self._qc_y_mode_combo.currentTextChanged.connect(
            lambda text: self._qc_y_min_spin.setEnabled(text == "Manual"))
        self._qc_y_mode_combo.currentTextChanged.connect(
            lambda text: self._qc_y_max_spin.setEnabled(text == "Manual"))
        self._qc_y_mode_combo.currentTextChanged.connect(self._schedule_qc_refresh)
        self._qc_y_min_spin.valueChanged.connect(self._schedule_qc_refresh)
        self._qc_y_max_spin.valueChanged.connect(self._schedule_qc_refresh)
        qcl.addRow("Range:", self._qc_y_mode_combo)
        qcl.addRow("Y min:", self._qc_y_min_spin)
        qcl.addRow("Y max:", self._qc_y_max_spin)
        layout.addWidget(box_qc)

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
        self._tabs.addTab(self._summary_tab, "Summary Plots")   # 7

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

    def _get_config(self) -> dict:
        fw, fs = _parse_title_style(self._title_style.currentText())
        base = self._input_row.path
        return {
            "dye_folder":         os.path.join(base, "dye_map"),
            "host_folder":        os.path.join(base, "host_map"),
            "guest_folder":       os.path.join(base, "guest_map"),
            "blank_folder":       os.path.join(base, "blank_map"),
            "buffer_folder":      os.path.join(base, "buffer_map") if self._buffer_map_chk.isChecked() else None,
            "raw_folder":         os.path.join(base, "Raw"),
            "kd_folder":          self._kd_row.path,
            "chromatic_folder":   self._chrom_db_row.path,
            "output_folder":      self._output_row.path,
            "input_folder":       base,
            "multi_chromatic":    self._multi_chrom_chk.isChecked(),
            "exceptions_folder":  (os.path.join(base, "exceptions_map")
                                    if self._exceptions_chk.isChecked() else None),
            "chromatic_to_dye":   (self._chrom_map.mapping
                                      if self._multi_chrom_chk.isChecked() else {}),
            "r2_threshold":       self._r2_spin.value(),
            "grubbs_alpha":       self._grubbs_spin.value(),
            "morrison_gate":      self._morrison_spin.value(),
            "autofl_factor":      self._autofl_spin.value(),
            "model_preference":   {"Auto (AICc)": "auto", "Standard": "standard",
                                    "Hill Slope": "hillslope", "Morrison": "morrison",
                                    "Biphasic": "biphasic"}[self._model_combo.currentText()],
            "morrison_allow_degenerate": self._morrison_degenerate_chk.isChecked(),
            "color_data":         self._color_data_row.color,
            "color_fit":          self._color_fit_row.color,
            "color_resid":        self._color_resid_row.color,
            "show_residuals":     self._resid_chk.isChecked(),
            "export_individual":  self._ind_export_chk.isChecked(),
            "title_fontsize":     self._title_fsz.value(),
            "title_fontfamily":   self._title_fam.currentText(),
            "title_fontweight":   fw,
            "title_fontstyle":    fs,
            "axis_fontsize":      self._axis_fsz.value(),
            "tick_fontsize":      self._tick_fsz.value(),
            "normalise_y":        self._normalise_chk.isChecked(),
            "sci_notation_y":     self._sci_notation_chk.isChecked(),
            "lw_fit":             self._sz_fit.value(),
            "ms_data":            self._sz_data.value(),
            "lw_errorbar":        self._sz_errorbar.value(),
            "ms_resid":           self._sz_resid.value(),
            "x_min":              (self._x_min_spin.value()
                                    if self._xaxis_manual_chk.isChecked() else None),
            "x_max":              (self._x_max_spin.value()
                                    if self._xaxis_manual_chk.isChecked() else None),
            "x_tick_step":        (self._x_tick_spin.value()
                                    if self._xaxis_manual_chk.isChecked()
                                    and self._x_tick_spin.value() > 0 else None),
            "show_excluded":      self._show_excluded_chk.isChecked(),
            "qc1_y_mode":         {"All points": "all", "Tukey fence (robust)": "tukey",
                                    "Manual": "manual"}[self._qc_y_mode_combo.currentText()],
            "qc1_y_min":          (self._qc_y_min_spin.value()
                                    if self._qc_y_mode_combo.currentText() == "Manual" else None),
            "qc1_y_max":          (self._qc_y_max_spin.value()
                                    if self._qc_y_mode_combo.currentText() == "Manual" else None),
        }

    def _reset_params(self):
        self._r2_spin.setValue(pl_ki.PASS_R2_DEFAULT_KI)
        self._grubbs_spin.setValue(0.05)
        self._morrison_spin.setValue(pl_ki.MORRISON_GATE_DEFAULT)
        self._autofl_spin.setValue(pl_ki.GUEST_AUTOFL_FACTOR)
        self._model_combo.setCurrentIndex(0)
        self._morrison_degenerate_chk.setChecked(False)
        self._color_data_row.set_color(pl_ki.PLOT_COLOR_DATA)
        self._color_fit_row.set_color(pl_ki.PLOT_COLOR_FIT)
        self._color_resid_row.set_color(pl_ki.PLOT_COLOR_DATA)
        self._sz_fit.setValue(2.0)
        self._sz_data.setValue(4.0)
        self._sz_errorbar.setValue(1.5)
        self._sz_resid.setValue(2.5)
        self._multi_chrom_chk.setChecked(False)
        self._buffer_map_chk.setChecked(False)
        self._exceptions_chk.setChecked(False)
        self._normalise_chk.setChecked(False)
        self._sci_notation_chk.setChecked(False)
        self._xaxis_manual_chk.setChecked(False)
        self._x_min_spin.setValue(-3.0)
        self._x_max_spin.setValue(3.0)
        self._x_tick_spin.setValue(0.0)
        self._show_excluded_chk.setChecked(True)
        self._resid_chk.setChecked(False)
        self._ind_export_chk.setChecked(False)
        self._summary_chk.setChecked(True)
        self._title_fsz.setValue(12)
        self._title_fam.setCurrentText("sans-serif")
        self._title_style.setCurrentText("Normal")
        self._axis_fsz.setValue(10)
        self._tick_fsz.setValue(10)
        self._qc_y_mode_combo.setCurrentText("All points")
        self._qc_y_min_spin.setValue(0.0)
        self._qc_y_max_spin.setValue(100000.0)
        if self._state.fit_results:
            self._refresh_results()

    def _schedule_qc_refresh(self):
        if self._state.fi_df is None:
            return
        if not hasattr(self, "_qc_refresh_timer"):
            self._qc_refresh_timer = QTimer(self)
            self._qc_refresh_timer.setSingleShot(True)
            self._qc_refresh_timer.timeout.connect(self._refresh_qc)
        self._qc_refresh_timer.start(150)

    def _refresh_qc(self):
        cfg = self._get_config()
        qc = pl_ki.make_qc_figures_ki(self._state.fi_df, progress_cb=self._log.append,
                                      qc1_y_mode=cfg["qc1_y_mode"],
                                      qc1_y_min=cfg["qc1_y_min"], qc1_y_max=cfg["qc1_y_max"])
        self._state.qc_figures = qc
        self._qc_tab.load(qc)

    def _schedule_refresh(self):
        if not self._state.fit_results:
            return
        if not hasattr(self, "_refresh_timer"):
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._refresh_results)
        self._refresh_timer.start(150)

    def _on_color_changed(self, _hex):
        self._schedule_refresh()

    def _on_display_changed(self, _state):
        self._schedule_refresh()

    def _on_title_font_changed(self, _=None):
        self._schedule_refresh()

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
                               cfg["guest_folder"], cfg["kd_folder"],
                               buffer_folder=cfg["buffer_folder"]))
        elif self._stage == 1:
            self._launch(_wrap(pl_ki.load_plates_ki,
                               cfg["raw_folder"], cfg["chromatic_to_dye"],
                               chromatic_folder=cfg["chromatic_folder"],
                               multi_chromatic=cfg["multi_chromatic"]))
        elif self._stage == 2:
            self._launch(_wrap(pl_ki.merge_ki,
                               self._state.fluorescence,
                               self._state.merged_mapping,
                               cfg["blank_folder"],
                               chromatic_folder=cfg["chromatic_folder"],
                               multi_chromatic=cfg["multi_chromatic"],
                               exceptions_folder=cfg["exceptions_folder"]))
        elif self._stage == 3:
            self._launch(_wrap(pl_ki.subtract_background_ki,
                               self._state.merged))
        elif self._stage == 4:
            self._launch(_wrap(pl_ki.fit_curves_ki,
                               self._state.fi_df, self._state.hot_df,
                               grubbs_alpha=cfg["grubbs_alpha"],
                               morrison_gate=cfg["morrison_gate"],
                               model_preference=cfg["model_preference"],
                               morrison_allow_degenerate=cfg["morrison_allow_degenerate"]))

    def _launch(self, fn):
        self._worker = StageWorker(fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_stage_done)
        self._worker.error.connect(lambda tb: (self._log.append(f"\n[ERROR]\n{tb}"),
                                               QMessageBox.critical(self, "Error", "See log.")))
        stage_lbl = STAGE_NAMES_KI[self._stage] if self._stage < len(STAGE_NAMES_KI) else "?"
        self._status_lbl.setText(f"Running: {stage_lbl}…")
        self._worker.start()
        self._set_buttons_enabled()

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
                cfg = self._get_config()
                qc = pl_ki.make_qc_figures_ki(result["fi_df"], progress_cb=self._log.append,
                                              qc1_y_mode=cfg["qc1_y_mode"],
                                              qc1_y_min=cfg["qc1_y_min"], qc1_y_max=cfg["qc1_y_max"])
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
            self._schedule_refresh()

    def _on_grubbs_changed(self, _value):
        if self._state.fit_results:
            QMessageBox.information(
                self, "Re-run required",
                "Grubbs α affects outlier removal during curve fitting.\n"
                "Please re-run from the Competitive Fitting stage to apply the change.")

    def _on_model_combo_changed(self, text):
        is_morrison = (text == "Morrison")
        self._morrison_degenerate_chk.setEnabled(is_morrison)
        if not is_morrison:
            self._morrison_degenerate_chk.setChecked(False)

    def _refresh_results(self):
        cfg = self._get_config()
        df  = pl_ki.apply_thresholds_ki(
            self._state.fit_results, self._state.plot_data, cfg["r2_threshold"],
            fi_df=self._state.fi_df, autofl_factor=cfg["autofl_factor"])
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
                             axis_fontsize=cfg["axis_fontsize"],
                             tick_fontsize=cfg["tick_fontsize"],
                             layout_cfg=self._layout_panel.get_cfg(),
                             normalise_y=cfg["normalise_y"],
                             sci_notation_y=cfg["sci_notation_y"],
                             lw_fit=cfg["lw_fit"], ms_data=cfg["ms_data"],
                             lw_errorbar=cfg["lw_errorbar"], ms_resid=cfg["ms_resid"],
                             x_min=cfg["x_min"], x_max=cfg["x_max"],
                             x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
        self._tabs.setTabEnabled(8, True)
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)
        self._preview_dirty = True

    def _on_layout_changed(self):
        self._preview_dirty = True
        self._schedule_refresh()

    def _on_tab_changed(self, idx):
        if self._tabs.tabText(idx) == "PDF Preview" and self._state.plot_data:
            if getattr(self, "_preview_dirty", True):
                self._refresh_preview()
                self._preview_dirty = False

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
                                    axis_fontsize=cfg["axis_fontsize"],
                                    tick_fontsize=cfg["tick_fontsize"],
                                    layout_cfg=lc, dpi=100,
                                    lw_fit=cfg["lw_fit"], ms_data=cfg["ms_data"],
                                    lw_errorbar=cfg["lw_errorbar"], ms_resid=cfg["ms_resid"],
                                    x_min=cfg["x_min"], x_max=cfg["x_max"],
                                    x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
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
        state       = self._state
        normalise   = cfg["normalise_y"]
        sci_notn    = cfg["sci_notation_y"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_ki.save_outputs_ki(state, cfg["output_folder"],
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
                                         axis_fontsize=cfg["axis_fontsize"],
                                         tick_fontsize=cfg["tick_fontsize"],
                                         layout_cfg=lc,
                                         input_folder=cfg["input_folder"],
                                         normalise_y=normalise,
                                         sci_notation_y=sci_notn,
                                         lw_fit=cfg["lw_fit"],
                                         ms_data=cfg["ms_data"],
                                         lw_errorbar=cfg["lw_errorbar"],
                                         ms_resid=cfg["ms_resid"],
                                         x_min=cfg["x_min"], x_max=cfg["x_max"],
                                         x_tick_step=cfg["x_tick_step"], show_excluded=cfg["show_excluded"])
        self._worker = StageWorker(_save_fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_save_done)
        self._worker.error.connect(lambda tb: self._log.append(f"\n[ERROR]\n{tb}"))
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Saving…")
        self._worker.start()

    def _on_save_done(self, result):
        self._save_btn.setEnabled(True)
        self._set_buttons_enabled()
        if isinstance(result, Exception):
            self._status_lbl.setText("Save failed — see log")
            return
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
        scroll.setMinimumWidth(300)
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

        box_col = QGroupBox("Plot Appearance")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", pl_fda.PLOT_COLOR_DATA)
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", pl_fda.PLOT_COLOR)
        self._color_resid_row = ColorPickerRow("Residuals:  ", pl_fda.PLOT_COLOR_DATA)
        self._color_data_row.color_changed.connect(self._on_color_changed)
        self._color_fit_row.color_changed.connect(self._on_color_changed)
        self._color_resid_row.color_changed.connect(self._on_color_changed)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        sz_form = QFormLayout()
        self._sz_fit = QDoubleSpinBox(); self._sz_fit.setRange(0.5, 8.0); self._sz_fit.setValue(2.0); self._sz_fit.setSingleStep(0.5); self._sz_fit.setDecimals(1)
        self._sz_data = QDoubleSpinBox(); self._sz_data.setRange(1.0, 12.0); self._sz_data.setValue(4.0); self._sz_data.setSingleStep(0.5); self._sz_data.setDecimals(1)
        self._sz_errorbar = QDoubleSpinBox(); self._sz_errorbar.setRange(0.5, 6.0); self._sz_errorbar.setValue(1.5); self._sz_errorbar.setSingleStep(0.5); self._sz_errorbar.setDecimals(1)
        self._sz_resid = QDoubleSpinBox(); self._sz_resid.setRange(0.5, 8.0); self._sz_resid.setValue(2.5); self._sz_resid.setSingleStep(0.5); self._sz_resid.setDecimals(1)
        self._sz_fit.valueChanged.connect(self._on_color_changed)
        self._sz_data.valueChanged.connect(self._on_color_changed)
        self._sz_errorbar.valueChanged.connect(self._on_color_changed)
        self._sz_resid.valueChanged.connect(self._on_color_changed)
        sz_form.addRow("Fit curve lw:", self._sz_fit)
        sz_form.addRow("Data point size:", self._sz_data)
        sz_form.addRow("Error bar lw:", self._sz_errorbar)
        sz_form.addRow("Residual size:", self._sz_resid)
        cl.addLayout(sz_form)
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

        self._spectra_grouping_combo = QComboBox()
        self._spectra_grouping_combo.addItems(["Per Host", "All Hosts by Dye"])
        self._spectra_grouping_combo.currentIndexChanged.connect(self._on_spectra_display_changed)
        spec_form.addRow("Grouping:", self._spectra_grouping_combo)
        self._spectra_fig_w = QDoubleSpinBox()
        self._spectra_fig_w.setRange(4.0, 24.0)
        self._spectra_fig_w.setValue(11.0)
        self._spectra_fig_w.setSingleStep(0.5)
        self._spectra_fig_w.setDecimals(1)
        self._spectra_fig_w.setSuffix(" in")
        self._spectra_fig_w.valueChanged.connect(self._on_spectra_display_changed)
        self._spectra_fig_h = QDoubleSpinBox()
        self._spectra_fig_h.setRange(2.0, 18.0)
        self._spectra_fig_h.setValue(5.0)
        self._spectra_fig_h.setSingleStep(0.5)
        self._spectra_fig_h.setDecimals(1)
        self._spectra_fig_h.setSuffix(" in")
        self._spectra_fig_h.valueChanged.connect(self._on_spectra_display_changed)
        spec_form.addRow("Fig width:", self._spectra_fig_w)
        spec_form.addRow("Fig height:", self._spectra_fig_h)
        sl.addLayout(spec_form)

        self._conc_show_all_chk = QCheckBox("Show all concentrations")
        self._conc_show_all_chk.setChecked(True)
        self._conc_show_all_chk.toggled.connect(self._on_conc_show_all_toggled)
        sl.addWidget(self._conc_show_all_chk)

        conc_btn_row = QHBoxLayout()
        self._conc_select_all_btn = QPushButton("Select all")
        self._conc_deselect_all_btn = QPushButton("Deselect all")
        self._conc_select_all_btn.clicked.connect(self._on_conc_select_all)
        self._conc_deselect_all_btn.clicked.connect(self._on_conc_deselect_all)
        conc_btn_row.addWidget(self._conc_select_all_btn)
        conc_btn_row.addWidget(self._conc_deselect_all_btn)
        self._conc_btn_widget = QWidget()
        self._conc_btn_widget.setLayout(conc_btn_row)
        self._conc_btn_widget.setVisible(False)
        sl.addWidget(self._conc_btn_widget)

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
        self._override_lbl = QLabel("Per-concentration colour overrides:")
        self._override_lbl.setStyleSheet("color: grey; font-size: 10px;")
        sl.addWidget(self._override_lbl)
        sl.addWidget(self._override_scroll)

        opacity_hdr = QHBoxLayout()
        self._conc_opacity_lbl = QLabel("Per-concentration opacity:")
        self._conc_opacity_lbl.setStyleSheet("color: grey; font-size: 10px;")
        self._conc_opacity_lbl.setVisible(False)
        self._conc_opacity_reset_btn = QPushButton("Reset")
        self._conc_opacity_reset_btn.setFixedWidth(48)
        self._conc_opacity_reset_btn.setVisible(False)
        self._conc_opacity_reset_btn.clicked.connect(self._on_conc_opacity_reset)
        opacity_hdr.addWidget(self._conc_opacity_lbl)
        opacity_hdr.addWidget(self._conc_opacity_reset_btn)
        sl.addLayout(opacity_hdr)
        self._conc_opacity_scroll = QScrollArea()
        self._conc_opacity_scroll.setWidgetResizable(True)
        self._conc_opacity_scroll.setMaximumHeight(120)
        self._conc_opacity_inner = QWidget()
        self._conc_opacity_layout = QFormLayout(self._conc_opacity_inner)
        self._conc_opacity_layout.setContentsMargins(2, 2, 2, 2)
        self._conc_opacity_layout.setSpacing(2)
        self._conc_opacity_scroll.setWidget(self._conc_opacity_inner)
        self._conc_opacity_scroll.setVisible(False)
        self._conc_opacity_spins: list[tuple[float, QDoubleSpinBox]] = []
        sl.addWidget(self._conc_opacity_scroll)

        self._host_color_lbl = QLabel("Per-host colours:")
        self._host_color_lbl.setStyleSheet("color: grey; font-size: 10px;")
        self._host_color_lbl.setVisible(False)
        sl.addWidget(self._host_color_lbl)
        self._host_color_scroll = QScrollArea()
        self._host_color_scroll.setWidgetResizable(True)
        self._host_color_scroll.setMaximumHeight(140)
        self._host_color_inner = QWidget()
        self._host_color_layout = QVBoxLayout(self._host_color_inner)
        self._host_color_layout.setContentsMargins(2, 2, 2, 2)
        self._host_color_layout.setSpacing(2)
        self._host_color_scroll.setWidget(self._host_color_inner)
        self._host_color_scroll.setVisible(False)
        self._host_color_rows: list[tuple[str, ColorPickerRow]] = []
        sl.addWidget(self._host_color_scroll)

        layout.addWidget(box_spec)

        box_tf = QGroupBox("Title Font")
        tf_lay = QFormLayout(box_tf)
        self._title_fam = QComboBox()
        self._title_fam.addItems([
            "sans-serif", "serif", "monospace",
            "DejaVu Sans", "Arial", "Helvetica"])
        self._title_fsz = QDoubleSpinBox()
        self._title_fsz.setRange(5, 18)
        self._title_fsz.setValue(11)
        self._title_fsz.setSingleStep(0.5)
        self._title_fsz.setDecimals(1)
        self._title_style = QComboBox()
        self._title_style.addItems(["Normal", "Bold", "Italic", "Bold Italic"])
        self._title_fam.currentIndexChanged.connect(self._on_title_font_changed)
        self._title_fsz.valueChanged.connect(self._on_title_font_changed)
        self._title_style.currentIndexChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        self._axis_fsz = QDoubleSpinBox()
        self._axis_fsz.setRange(4, 18)
        self._axis_fsz.setValue(10)
        self._axis_fsz.setSingleStep(0.5)
        self._axis_fsz.setDecimals(1)
        self._axis_fsz.valueChanged.connect(self._on_title_font_changed)
        self._tick_fsz = QDoubleSpinBox()
        self._tick_fsz.setRange(3, 16)
        self._tick_fsz.setValue(10)
        self._tick_fsz.setSingleStep(0.5)
        self._tick_fsz.setDecimals(1)
        self._tick_fsz.valueChanged.connect(self._on_title_font_changed)
        tf_lay.addRow("Style:",       self._title_style)
        tf_lay.addRow("Axis labels:", self._axis_fsz)
        tf_lay.addRow("Tick numbers:", self._tick_fsz)
        layout.addWidget(box_tf)

        box_disp = QGroupBox("Display")
        dl       = QVBoxLayout(box_disp)
        self._normalise_chk = QCheckBox("Normalise Y-axis (0–1)")
        self._normalise_chk.setChecked(False)
        self._normalise_chk.stateChanged.connect(self._on_display_changed)
        self._sci_notation_chk = QCheckBox("Scientific notation (Y-axis)")
        self._sci_notation_chk.setChecked(False)
        self._sci_notation_chk.stateChanged.connect(self._on_display_changed)
        self._resid_chk = QCheckBox("Show residual plots")
        self._resid_chk.setChecked(True)
        self._resid_chk.setToolTip("Show residual sub-panel beneath each binding curve")
        self._resid_chk.stateChanged.connect(self._on_display_changed)
        self._ind_export_chk = QCheckBox("Export individual PDFs")
        self._ind_export_chk.setChecked(False)
        self._ind_export_chk.setToolTip(
            "On Save: write one PDF per curve into\n"
            "results/plots/individual/PASS|FAIL/")
        self._ind_spectra_export_chk = QCheckBox("Export individual spectra PDFs")
        self._ind_spectra_export_chk.setChecked(False)
        self._ind_spectra_export_chk.setToolTip(
            "On Save: write one PDF per spectrum into\n"
            "results/plots/spectra_individual/")
        self._overlay_dye_chk = QCheckBox("Overlay fits by dye")
        self._overlay_dye_chk.setChecked(False)
        self._overlay_dye_chk.setToolTip(
            "Group all hosts for the same dye on one plot,\n"
            "each host in a distinct colour")
        self._overlay_dye_chk.stateChanged.connect(self._on_display_changed)
        self._summary_chk = QCheckBox("Show summary plots")
        self._summary_chk.setChecked(True)
        self._summary_chk.setToolTip("pKd bar-chart / heatmap / spider summary plots")
        self._summary_chk.stateChanged.connect(self._on_summary_toggle)
        dl.addWidget(self._normalise_chk)
        dl.addWidget(self._sci_notation_chk)
        dl.addWidget(self._resid_chk)
        dl.addWidget(self._overlay_dye_chk)
        dl.addWidget(self._ind_export_chk)
        dl.addWidget(self._ind_spectra_export_chk)
        dl.addWidget(self._summary_chk)
        layout.addWidget(box_disp)

        self._layout_panel = _LayoutPanel()
        self._layout_panel.changed.connect(self._on_layout_changed)
        layout.addWidget(self._layout_panel)

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
        self._tabs.addTab(self._summary_tab, "Summary Plots")      # 6

        self._overlay_scroll = QScrollArea()
        self._overlay_scroll.setWidgetResizable(False)
        self._tabs.addTab(self._overlay_scroll, "Overlay by Dye")  # 7

        for i in range(self._tabs.count()):
            self._tabs.setTabEnabled(i, False)
        self._tabs.setTabVisible(7, False)
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
        fw, fs = _parse_title_style(self._title_style.currentText())
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
            "export_individual_spectra": self._ind_spectra_export_chk.isChecked(),
            "overlay_fits_by_dye": self._overlay_dye_chk.isChecked(),
            "spectra_fig_w": self._spectra_fig_w.value(),
            "spectra_fig_h": self._spectra_fig_h.value(),
            "spectra_palette": self._spectra_palette_combo.currentText(),
            "spectra_display": display_map.get(self._spectra_display_combo.currentText(),
                                               "side_by_side"),
            "spectra_color_overrides": color_overrides if color_overrides else None,
            "host_color_overrides": {h: row.color for h, row in self._host_color_rows},
            "conc_opacity_overrides": {c: spin.value() for c, spin in self._conc_opacity_spins},
            "selected_concs": selected_concs,
            "spectra_grouping": ("by_dye" if self._spectra_grouping_combo.currentIndex() == 1
                                 else "per_host"),
            "per_dye_wl":     self._per_dye_wl_chk.isChecked(),
            "wavelength_map":  {dye: spin.value()
                                for dye, spin in self._per_dye_wl_spins.items()},
            "title_fontsize":        self._title_fsz.value(),
            "title_fontfamily":      self._title_fam.currentText(),
            "title_fontweight":      fw,
            "title_fontstyle":       fs,
            "axis_fontsize":         self._axis_fsz.value(),
            "tick_fontsize":         self._tick_fsz.value(),
            "normalise_y":           self._normalise_chk.isChecked(),
            "sci_notation_y":        self._sci_notation_chk.isChecked(),
            "lw_fit":                self._sz_fit.value(),
            "ms_data":               self._sz_data.value(),
            "lw_errorbar":           self._sz_errorbar.value(),
            "ms_resid":              self._sz_resid.value(),
        }

    def _schedule_refresh(self):
        if not self._state.plot_data:
            return
        if not hasattr(self, "_refresh_timer"):
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._refresh_results)
        self._refresh_timer.start(150)

    def _on_color_changed(self, _hex):
        self._schedule_refresh()

    def _on_display_changed(self, _state):
        self._schedule_refresh()

    def _on_spectra_display_changed(self, _index=None):
        by_dye = self._spectra_grouping_combo.currentIndex() == 1
        has_hosts = len(self._host_color_rows) > 0
        has_opacities = len(self._conc_opacity_spins) > 0
        self._host_color_lbl.setVisible(by_dye and has_hosts)
        self._host_color_scroll.setVisible(by_dye and has_hosts)
        self._conc_opacity_lbl.setVisible(by_dye and has_opacities)
        self._conc_opacity_scroll.setVisible(by_dye and has_opacities)
        self._conc_opacity_reset_btn.setVisible(by_dye and has_opacities)
        self._override_lbl.setVisible(not by_dye)
        self._override_scroll.setVisible(not by_dye and len(self._override_rows) > 0)
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_show_all_toggled(self, checked):
        self._conc_scroll.setVisible(not checked)
        self._conc_btn_widget.setVisible(not checked)
        for _c, chk in self._conc_checks:
            chk.setEnabled(not checked)
            if checked:
                chk.setChecked(True)
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_select_all(self):
        for _c, chk in self._conc_checks:
            chk.setChecked(True)

    def _on_conc_deselect_all(self):
        for _c, chk in self._conc_checks:
            chk.setChecked(False)

    def _on_conc_check_changed(self, _state):
        if self._state.fi_df is not None and not self._conc_show_all_chk.isChecked():
            self._regenerate_spectra()

    def _on_override_color_changed(self, _hex):
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_host_color_changed(self, _hex):
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_opacity_changed(self, _value):
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_conc_opacity_reset(self):
        n = len(self._conc_opacity_spins)
        for i, (_c, spin) in enumerate(self._conc_opacity_spins):
            spin.blockSignals(True)
            alpha = 1.0 if n <= 1 else round(0.3 + 0.7 * i / (n - 1), 2)
            spin.setValue(alpha)
            spin.blockSignals(False)
        if self._state.fi_df is not None:
            self._regenerate_spectra()

    def _on_title_font_changed(self, _=None):
        self._schedule_refresh()

    def _on_layout_changed(self):
        self._schedule_refresh()

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
        by_dye = self._spectra_grouping_combo.currentIndex() == 1
        self._override_lbl.setVisible(not by_dye)
        self._override_scroll.setVisible(not by_dye and len(concs) > 0)

        # Per-concentration opacity spinboxes (for by_dye mode)
        for _c, spin in self._conc_opacity_spins:
            self._conc_opacity_layout.removeRow(spin)
        self._conc_opacity_spins.clear()
        n_concs = len(concs)
        for i, c in enumerate(concs):
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            default_alpha = 1.0 if n_concs <= 1 else (0.3 + 0.7 * i / (n_concs - 1))
            spin.setValue(round(default_alpha, 2))
            spin.valueChanged.connect(self._on_conc_opacity_changed)
            self._conc_opacity_layout.addRow(f"{c:g} µM:", spin)
            self._conc_opacity_spins.append((c, spin))
        self._conc_opacity_lbl.setVisible(by_dye and n_concs > 0)
        self._conc_opacity_scroll.setVisible(by_dye and n_concs > 0)
        self._conc_opacity_reset_btn.setVisible(by_dye and n_concs > 0)

        # Per-host colour rows
        for _h, row in self._host_color_rows:
            self._host_color_layout.removeWidget(row)
            row.deleteLater()
        self._host_color_rows.clear()
        hosts = sorted(fi_df["Host"].dropna().unique())
        for i, h in enumerate(hosts):
            default = pl_spec._DEFAULT_HOST_COLORS[i % len(pl_spec._DEFAULT_HOST_COLORS)]
            row = ColorPickerRow(f"{h}:", default)
            row.color_changed.connect(self._on_host_color_changed)
            self._host_color_layout.addWidget(row)
            self._host_color_rows.append((h, row))
        by_dye = self._spectra_grouping_combo.currentIndex() == 1
        self._host_color_lbl.setVisible(by_dye and len(hosts) > 0)
        self._host_color_scroll.setVisible(by_dye and len(hosts) > 0)

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
                host_color_overrides=cfg["host_color_overrides"],
                conc_opacity_overrides=cfg["conc_opacity_overrides"],
                display_mode=cfg["spectra_display"],
                selected_concs=cfg["selected_concs"],
                grouping=cfg["spectra_grouping"],
                fig_w=cfg["spectra_fig_w"],
                fig_h=cfg["spectra_fig_h"],
                title_fontsize=cfg["title_fontsize"],
                axis_fontsize=cfg["axis_fontsize"],
                tick_fontsize=cfg["tick_fontsize"],
                sci_notation_y=cfg["sci_notation_y"])
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
        self._worker.start()
        self._set_buttons_enabled()

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
                    host_color_overrides=cfg["host_color_overrides"],
                    conc_opacity_overrides=cfg["conc_opacity_overrides"],
                    display_mode=cfg["spectra_display"],
                    selected_concs=cfg["selected_concs"],
                    grouping=cfg["spectra_grouping"],
                    fig_w=cfg["spectra_fig_w"],
                    fig_h=cfg["spectra_fig_h"],
                    title_fontsize=cfg["title_fontsize"],
                    axis_fontsize=cfg["axis_fontsize"],
                    tick_fontsize=cfg["tick_fontsize"],
                    sci_notation_y=cfg["sci_notation_y"])
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
        self._worker.start()
        self._set_buttons_enabled()

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
        lc = self._layout_panel.get_cfg()
        self._plots_tab.load(pass_items, fail_items,
                             color_data=cfg["color_data"], color_fit=cfg["color_fit"],
                             color_resid=cfg["color_resid"],
                             show_residuals=cfg["show_residuals"],
                             title_fontsize=cfg["title_fontsize"],
                             title_fontfamily=cfg["title_fontfamily"],
                             title_fontweight=cfg["title_fontweight"],
                             title_fontstyle=cfg["title_fontstyle"],
                             axis_fontsize=cfg["axis_fontsize"],
                             tick_fontsize=cfg["tick_fontsize"],
                             layout_cfg=lc,
                             normalise_y=cfg["normalise_y"],
                             sci_notation_y=cfg["sci_notation_y"],
                             lw_fit=cfg["lw_fit"], ms_data=cfg["ms_data"],
                             lw_errorbar=cfg["lw_errorbar"], ms_resid=cfg["ms_resid"])
        overlay_on = cfg["overlay_fits_by_dye"]
        self._tabs.setTabVisible(7, overlay_on)
        if overlay_on:
            overlay_fig, ov_fw, ov_fh = self._render_overlay_by_dye(
                self._state.plot_data, cfg, lc)
            self._state.overlay_figures = [("Overlay by Dye", overlay_fig)]
            _close_old_canvas(self._overlay_scroll)
            canvas = FigureCanvas(overlay_fig)
            dpi_v = overlay_fig.dpi
            canvas.resize(int(ov_fw * dpi_v), int(ov_fh * dpi_v))
            canvas.draw()
            self._overlay_scroll.setWidget(canvas)
            self._tabs.setTabEnabled(7, True)
        else:
            self._state.overlay_figures = []
        if self._summary_chk.isChecked() and not df.empty:
            self._summary_tab.load(df)

    def _render_overlay_by_dye(self, plot_data, cfg, lc):
        import pandas as pd
        from collections import defaultdict
        by_dye = defaultdict(list)
        for e in plot_data:
            by_dye[e["dye"]].append(e)

        host_colors = {}
        for h, row in self._host_color_rows:
            host_colors[h] = row.color

        dyes  = sorted(by_dye.keys())
        n     = max(len(dyes), 1)
        cw    = lc.get("cell_w", 3.0)
        ch    = lc.get("cell_h", 2.5)
        cols  = lc.get("pdf_cols", min(int(np.ceil(np.sqrt(n))), 4))
        rows  = int(np.ceil(n / cols))
        hs_in = lc.get("hspace", 0.25)
        ws_in = lc.get("wspace", 0.35)
        _ML, _MR, _MT, _MB = 1.2, 0.4, 0.9, 0.9

        cfg_fig_w = lc.get("fig_w", None)
        if cfg_fig_w:
            grid_w = cfg_fig_w - _ML - _MR
        else:
            grid_w = cols * cw + max(cols - 1, 0) * ws_in
        grid_h = rows * ch + max(rows - 1, 0) * hs_in
        fig_w  = grid_w + _ML + _MR
        fig_h  = grid_h + _MT + _MB
        hs_frac = hs_in / ch if ch > 0 else 0.25
        ws_frac = ws_in / cw if cw > 0 else 0.35

        fig   = Figure(figsize=(fig_w, fig_h))
        outer = GridSpec(rows, cols, figure=fig,
                         hspace=hs_frac, wspace=ws_frac,
                         left=_ML / fig_w, right=1 - _MR / fig_w,
                         top=1 - _MT / fig_h, bottom=_MB / fig_h)

        afs   = cfg.get("axis_fontsize", 14)
        tfs   = cfg.get("tick_fontsize", 12)
        t_fs  = cfg.get("title_fontsize", 12)
        t_fw  = cfg.get("title_fontweight", "normal")
        t_fs2 = cfg.get("title_fontstyle", "normal")
        t_ff  = cfg.get("title_fontfamily", "sans-serif")
        normalise = cfg.get("normalise_y", False)
        sci_notn  = cfg.get("sci_notation_y", False)

        for i, dye in enumerate(dyes):
            ri, ci = divmod(i, cols)
            ax = fig.add_subplot(outer[ri, ci])
            entries = by_dye[dye]
            hosts_in_dye = sorted(set(e["host"] for e in entries))
            for idx, h in enumerate(hosts_in_dye):
                if h not in host_colors:
                    from pipeline_spectral import _DEFAULT_HOST_COLORS
                    host_colors[h] = _DEFAULT_HOST_COLORS[idx % len(_DEFAULT_HOST_COLORS)]

            for host in hosts_in_dye:
                h_entries = [e for e in entries if e["host"] == host]
                color = host_colors.get(host, "#333333")
                for e in h_entries:
                    x_cl = e["x_cleaned"]
                    y_cl = e["y_cleaned"]
                    y_plot = y_cl
                    if normalise:
                        y_min = float(np.nanmin(y_cl))
                        y_max = float(np.nanmax(y_cl))
                        y_rng = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
                        y_plot = (y_cl - y_min) / y_rng
                    stats = (pd.DataFrame({"x": x_cl, "y": y_plot})
                             .groupby("x")["y"].agg(["mean", "std"]))
                    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                                fmt="o", color=color, ecolor=color,
                                elinewidth=1.2, markersize=3.5, capsize=2,
                                zorder=3, label=f"{host} (D={e['D_fixed']})")
                    x_line = np.linspace(x_cl.min(), x_cl.max(), 300)
                    y_line = pl_fda._eval_model(
                        e["model_name"], x_line, e["popt"], e["D_fixed"])
                    if normalise:
                        y_line = (y_line - y_min) / y_rng
                    ax.plot(x_line, y_line, color=color, linewidth=1.5, zorder=2)

            ax.set_xlabel("[Host] (µM)", fontsize=afs)
            y_label = "Normalised Intensity" if normalise else "FI – F₀"
            ax.set_ylabel(y_label, fontsize=afs)
            ax.tick_params(labelsize=tfs)
            if sci_notn and not normalise:
                ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
            ax.set_title(f"{dye} — all hosts",
                         fontsize=t_fs, fontweight=t_fw,
                         fontstyle=t_fs2, fontfamily=t_ff)
            ax.legend(fontsize=tfs * 0.65, loc="best")

        for i in range(len(dyes), rows * cols):
            ri, ci = divmod(i, cols)
            fig.add_subplot(outer[ri, ci]).set_visible(False)

        return fig, fig_w, fig_h

    def _on_save(self):
        cfg       = self._get_config()
        lc        = self._layout_panel.get_cfg()
        file_list = pl_spec.preview_save_files_spectral(
            self._state, cfg["output_folder"], export_individual=cfg["export_individual"],
            export_individual_spectra=cfg["export_individual_spectra"],
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
        export_ind_spectra = cfg["export_individual_spectra"]
        state       = self._state

        normalise   = cfg["normalise_y"]
        sci_notn    = cfg["sci_notation_y"]

        def _save_fn(*a, progress_cb=None, **kw):
            return pl_spec.save_outputs_spectral(
                state, cfg["output_folder"], cfg["r2_threshold"], progress_cb,
                plot_color_data=color_data, plot_color_fit=color_fit,
                plot_color_resid=color_resid, show_residuals=show_resid,
                export_individual=export_ind,
                export_individual_spectra=export_ind_spectra,
                layout_cfg=lc,
                title_fontsize=cfg["title_fontsize"],
                title_fontfamily=cfg["title_fontfamily"],
                title_fontweight=cfg["title_fontweight"],
                title_fontstyle=cfg["title_fontstyle"],
                axis_fontsize=cfg["axis_fontsize"],
                tick_fontsize=cfg["tick_fontsize"],
                normalise_y=normalise,
                sci_notation_y=sci_notn,
                input_folder=cfg["input_folder"],
                lw_fit=cfg["lw_fit"],
                ms_data=cfg["ms_data"],
                lw_errorbar=cfg["lw_errorbar"],
                ms_resid=cfg["ms_resid"])
        self._worker = StageWorker(_save_fn)
        self._worker.progress.connect(lambda m: self._log.append(m))
        self._worker.finished.connect(self._on_save_done)
        self._worker.error.connect(lambda tb: self._log.append(f"\n[ERROR]\n{tb}"))
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Saving…")
        self._worker.start()

    def _on_save_done(self, result):
        self._save_btn.setEnabled(True)
        self._set_buttons_enabled()
        if isinstance(result, Exception):
            self._status_lbl.setText("Save failed — see log")
            return
        self._status_lbl.setText("Saved.")
        if isinstance(result, list):
            self._log.append(f"\nSaved {len(result)} file(s).")
            QMessageBox.information(self, "Saved",
                                    f"{len(result)} file(s) written successfully.")

    def _on_home(self):
        _go_home(self)


# ── Layout panel (shared by DirectMainWindow and CompMainWindow) ───────────────

class _LayoutPanel(QGroupBox):
    """Group box for controlling figure dimensions and spacing.

    Layout model (inches, additive — no parameter affects another):
      cell_w / cell_h  — main curve panel size (fixed regardless of extras)
      resid_h          — residual panel height (added below the curve)
      dots_h           — Ki/Kd dot panel height (added below residuals)
      hspace / wspace  — gap between cells (inches)
      margin_l/r/t/b   — page margins (inches, auto-expand with content)
      pdf_cols         — columns per page
    """
    changed = Signal()
    _DEFS = dict(layout_utils.DEFAULTS)

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

        D = self._DEFS
        self._cell_w    = _d(1.0, 10.0, D["cell_w"],   0.25, 2)
        self._cell_h    = _d(1.0, 10.0, D["cell_h"],   0.25, 2)
        self._resid_h   = _d(0.2, 4.0,  D["resid_h"],  0.1,  2)
        self._dots_h    = _d(0.2, 4.0,  D["dots_h"],   0.1,  2)
        self._pdf_cols  = QSpinBox()
        self._pdf_cols.setRange(1, 8); self._pdf_cols.setValue(D["pdf_cols"])
        self._pdf_cols.valueChanged.connect(self.changed)
        self._hspace    = _d(0.0, 4.0,  D["hspace"])
        self._wspace    = _d(0.0, 4.0,  D["wspace"])
        self._resid_gap = _d(0.0, 0.5,  D["resid_gap"])
        self._margin_l  = _d(0.0, 5.0,  D["margin_l"], 0.1)
        self._margin_r  = _d(0.0, 5.0,  D["margin_r"], 0.1)
        self._margin_t  = _d(0.0, 5.0,  D["margin_t"], 0.1)
        self._margin_b  = _d(0.0, 5.0,  D["margin_b"], 0.1)

        fl.addRow("Cell width (in):",  self._cell_w)
        fl.addRow("Cell height (in):", self._cell_h)
        fl.addRow("Resid. height:",    self._resid_h)
        fl.addRow("Dots height:",      self._dots_h)
        fl.addRow("PDF columns:",      self._pdf_cols)
        fl.addRow("H-space:",          self._hspace)
        fl.addRow("W-space:",          self._wspace)
        fl.addRow("Resid. gap:",       self._resid_gap)
        fl.addRow("Margin left:",      self._margin_l)
        fl.addRow("Margin right:",     self._margin_r)
        fl.addRow("Margin top:",       self._margin_t)
        fl.addRow("Margin bottom:",    self._margin_b)

        rst = QPushButton("Reset layout")
        rst.clicked.connect(self._reset)
        fl.addRow(rst)

    def get_cfg(self) -> dict:
        return {
            "cell_w":    self._cell_w.value(),
            "cell_h":    self._cell_h.value(),
            "resid_h":   self._resid_h.value(),
            "dots_h":    self._dots_h.value(),
            "pdf_cols":  self._pdf_cols.value(),
            "hspace":    self._hspace.value(),
            "wspace":    self._wspace.value(),
            "resid_gap": self._resid_gap.value(),
            "margin_l":  self._margin_l.value(),
            "margin_r":  self._margin_r.value(),
            "margin_t":  self._margin_t.value(),
            "margin_b":  self._margin_b.value(),
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
                 title_fontsize: float = 12, title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "normal", title_fontstyle: str = "normal",
                 axis_fontsize: float = 14, tick_fontsize: float = 12,
                 color_data: str = "#1e4572", color_fit: str = "#6495ED",
                 color_resid: str = None,
                 normalise_y: bool = False, sci_notation_y: bool = False,
                 lw_fit: float = 2.0, ms_data: float = 4.0,
                 lw_errorbar: float = 1.5, ms_resid: float = 2.5,
                 parent=None, layout_cfg: dict = None, **_extra):
        _lc   = layout_cfg or {}
        n     = len(combos)
        ncols = min(ncols, n) if n else 1
        nrows = max(1, -(-n // ncols))
        kw    = dict(show_legend=show_legend, flip_norm=flip_norm,
                     pooled_mode=pooled_mode,
                     title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                     title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                     axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
                     color_data=color_data, color_fit=color_fit, color_resid=color_resid,
                     normalise_y=normalise_y, sci_notation_y=sci_notation_y,
                     lw_fit=lw_fit, ms_data=ms_data,
                     lw_errorbar=lw_errorbar, ms_resid=ms_resid)

        cl  = layout_utils.cell_layout(_lc, show_residuals=show_residuals,
                                        show_dots=show_dots_panel)
        fl  = layout_utils.figure_layout(_lc, nrows, ncols, cl.row_h_in)
        fig = Figure(figsize=(fl.fig_w, fl.fig_h), dpi=dpi_val)
        super().__init__(fig)
        self._axes = []
        outer = GridSpec(nrows, ncols, figure=fig, **fl.outer_kwargs())

        for k, combo in enumerate(combos):
            ri, ci = divmod(k, ncols)
            inner  = outer[ri, ci].subgridspec(
                cl.sub_rows, 1, height_ratios=cl.height_ratios, hspace=cl.resid_gap)
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

    _SINGLE_W = 5.5
    _SINGLE_H = 4.5

    def show_combo(self, combo, runs, mode,
                   show_dots_panel=True, show_residuals=False,
                   flip_norm=False, pooled_mode=False, show_legend=True,
                   title_fontsize=12, title_fontfamily="sans-serif",
                   title_fontweight="normal", title_fontstyle="normal",
                   axis_fontsize=14, tick_fontsize=12,
                   color_data="#1e4d72", color_fit="#5480ed", color_resid=None,
                   normalise_y=False, sci_notation_y=False,
                   lw_fit=2.0, ms_data=4.0,
                   lw_errorbar=1.5, ms_resid=2.5,
                   layout_cfg=None, **_extra):
        lc = layout_cfg or {}
        cl = layout_utils.cell_layout(lc, show_residuals=show_residuals,
                                       show_dots=show_dots_panel)
        fl = layout_utils.figure_layout(lc, 1, 1, cl.row_h_in)
        self._fig.set_size_inches(fl.fig_w, fl.fig_h)
        self._fig.clear()

        gs = GridSpec(cl.sub_rows, 1, figure=self._fig, height_ratios=cl.height_ratios,
                      hspace=cl.resid_gap, **fl.single_kwargs())
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
            axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
            color_data=color_data, color_fit=color_fit, color_resid=color_resid,
            normalise_y=normalise_y, sci_notation_y=sci_notation_y,
            lw_fit=lw_fit, ms_data=ms_data,
            lw_errorbar=lw_errorbar, ms_resid=ms_resid)

        dpi_v = self._fig.dpi
        w_px, h_px = int(fl.fig_w * dpi_v), int(fl.fig_h * dpi_v)
        self.setMinimumSize(w_px, h_px)
        self.setFixedSize(w_px, h_px)
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
        export_xlsx_btn = QPushButton("Export Summary XLSX…")
        home_btn.clicked.connect(self._on_home)
        export_btn.clicked.connect(self._on_export)
        export_ind_btn.clicked.connect(self._on_export_individual)
        export_xlsx_btn.clicked.connect(self._on_export_xlsx)
        self._status_lbl = QLabel("Add fit_results files to begin.")
        for w in (home_btn, QLabel("  |  "), export_btn, export_ind_btn,
                  export_xlsx_btn, QLabel("  "), self._status_lbl):
            toolbar.addWidget(w)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ── left panel ──────────────────────────────────────────────────────
        panel = QWidget()
        panel.setMinimumWidth(340)
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

        # 3. Plot Appearance
        box_col = QGroupBox("Plot Appearance")
        cl      = QVBoxLayout(box_col)
        self._color_data_row  = ColorPickerRow("Data points:", "#1e4572")
        self._color_fit_row   = ColorPickerRow("Fit curve:  ", "#6495ED")
        self._color_resid_row = ColorPickerRow("Residuals:  ", "#1e4572")
        self._color_data_row.color_changed.connect(self._schedule_rebuild)
        self._color_fit_row.color_changed.connect(self._schedule_rebuild)
        self._color_resid_row.color_changed.connect(self._schedule_rebuild)
        cl.addWidget(self._color_data_row)
        cl.addWidget(self._color_fit_row)
        cl.addWidget(self._color_resid_row)
        sz_form = QFormLayout()
        self._sz_fit = QDoubleSpinBox(); self._sz_fit.setRange(0.5, 8.0); self._sz_fit.setValue(2.0); self._sz_fit.setSingleStep(0.5); self._sz_fit.setDecimals(1)
        self._sz_data = QDoubleSpinBox(); self._sz_data.setRange(1.0, 12.0); self._sz_data.setValue(4.0); self._sz_data.setSingleStep(0.5); self._sz_data.setDecimals(1)
        self._sz_errorbar = QDoubleSpinBox(); self._sz_errorbar.setRange(0.5, 6.0); self._sz_errorbar.setValue(1.5); self._sz_errorbar.setSingleStep(0.5); self._sz_errorbar.setDecimals(1)
        self._sz_resid = QDoubleSpinBox(); self._sz_resid.setRange(0.5, 8.0); self._sz_resid.setValue(2.5); self._sz_resid.setSingleStep(0.5); self._sz_resid.setDecimals(1)
        self._sz_fit.valueChanged.connect(self._schedule_rebuild)
        self._sz_data.valueChanged.connect(self._schedule_rebuild)
        self._sz_errorbar.valueChanged.connect(self._schedule_rebuild)
        self._sz_resid.valueChanged.connect(self._schedule_rebuild)
        sz_form.addRow("Fit curve lw:", self._sz_fit)
        sz_form.addRow("Data point size:", self._sz_data)
        sz_form.addRow("Error bar lw:", self._sz_errorbar)
        sz_form.addRow("Residual size:", self._sz_resid)
        cl.addLayout(sz_form)
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
        self._title_fsz.setValue(11)
        self._title_fsz.setSingleStep(0.5)
        self._title_fsz.setDecimals(1)
        self._title_style = QComboBox()
        self._title_style.addItems(["Normal", "Bold", "Italic", "Bold Italic"])
        self._title_fam.currentIndexChanged.connect(self._schedule_rebuild)
        self._title_fsz.valueChanged.connect(self._schedule_rebuild)
        self._title_style.currentIndexChanged.connect(self._schedule_rebuild)
        tf_lay.addRow("Family:", self._title_fam)
        tf_lay.addRow("Size:",   self._title_fsz)
        self._axis_fsz = QDoubleSpinBox()
        self._axis_fsz.setRange(4, 18)
        self._axis_fsz.setValue(10)
        self._axis_fsz.setSingleStep(0.5)
        self._axis_fsz.setDecimals(1)
        self._axis_fsz.valueChanged.connect(self._schedule_rebuild)
        self._tick_fsz = QDoubleSpinBox()
        self._tick_fsz.setRange(3, 16)
        self._tick_fsz.setValue(10)
        self._tick_fsz.setSingleStep(0.5)
        self._tick_fsz.setDecimals(1)
        self._tick_fsz.valueChanged.connect(self._schedule_rebuild)
        tf_lay.addRow("Style:",       self._title_style)
        tf_lay.addRow("Axis labels:", self._axis_fsz)
        tf_lay.addRow("Tick numbers:", self._tick_fsz)
        lay.addWidget(box_tf)

        # 5. Display
        box_p = QGroupBox("Display")
        p_lay = QVBoxLayout(box_p)
        self._normalise_chk = QCheckBox("Normalise Y-axis (0–1)")
        self._normalise_chk.setChecked(False)
        self._normalise_chk.stateChanged.connect(self._schedule_rebuild)
        self._sci_notation_chk = QCheckBox("Scientific notation (Y-axis)")
        self._sci_notation_chk.setChecked(False)
        self._sci_notation_chk.stateChanged.connect(self._schedule_rebuild)
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

        p_lay.addWidget(self._normalise_chk)
        p_lay.addWidget(self._sci_notation_chk)
        p_lay.addWidget(self._dots_chk)
        p_lay.addWidget(self._resid_chk)
        p_lay.addWidget(self._flip_chk)
        p_lay.addWidget(self._legend_chk)
        p_lay.addWidget(self._ind_export_chk)
        p_lay.addLayout(avg_row)
        lay.addWidget(box_p)

        # 6. Figure Layout
        self._layout_panel = _LayoutPanel()
        self._layout_panel.changed.connect(self._schedule_rebuild)
        lay.addWidget(self._layout_panel)

        # 7. Output
        box_out = QGroupBox("Output")
        out_lay = QVBoxLayout(box_out)
        folder_form = QFormLayout()
        self._output_row = FolderRow("")
        folder_form.addRow("Folder:", self._output_row)
        out_lay.addLayout(folder_form)
        self._save_all_btn = QPushButton("Save All (Grid PDF + Individual PDFs + Summary XLSX)")
        self._save_all_btn.setFixedHeight(36)
        self._save_all_btn.clicked.connect(self._on_save_all)
        out_lay.addWidget(self._save_all_btn)
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
                    self._pooled_rb.toggled):
            sig.connect(self._schedule_rebuild)

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

    def _plot_kw(self) -> dict:
        cfg = self._layout_panel.get_cfg()
        fw, fs = _parse_title_style(self._title_style.currentText())
        return dict(
            ncols=cfg["pdf_cols"],
            show_dots_panel=self._dots_chk.isChecked(),
            show_residuals=self._resid_chk.isChecked(),
            flip_norm=self._flip_chk.isChecked(),
            pooled_mode=self._pooled_rb.isChecked(),
            show_legend=self._legend_chk.isChecked(),
            title_fontsize=self._title_fsz.value(),
            title_fontfamily=self._title_fam.currentText(),
            title_fontweight=fw,
            title_fontstyle=fs,
            axis_fontsize=self._axis_fsz.value(),
            tick_fontsize=self._tick_fsz.value(),
            color_data=self._color_data_row.color,
            color_fit=self._color_fit_row.color,
            color_resid=self._color_resid_row.color,
            normalise_y=self._normalise_chk.isChecked(),
            sci_notation_y=self._sci_notation_chk.isChecked(),
            layout_cfg=cfg,
            lw_fit=self._sz_fit.value(),
            ms_data=self._sz_data.value(),
            lw_errorbar=self._sz_errorbar.value(),
            ms_resid=self._sz_resid.value(),
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
        kc    = getattr(self, "_cached_kc", None) or self._all_keys_with_counts()
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
        kc      = getattr(self, "_cached_kc", None) or self._all_keys_with_counts()
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

    def _schedule_rebuild(self, *_):
        if not hasattr(self, "_rebuild_timer"):
            self._rebuild_timer = QTimer(self)
            self._rebuild_timer.setSingleShot(True)
            self._rebuild_timer.timeout.connect(self._rebuild)
        self._rebuild_timer.start(150)

    def _rebuild(self, *_):
        if not self._runs:
            self._plots_tab.load([], [], self._mode, **self._plot_kw())
            self._status_lbl.setText("Add fit_results files to begin.")
            return
        self._cached_kc = self._all_keys_with_counts()
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
        is_ki      = self._mode == "ki"
        val_lbl    = "Ki (µM)" if is_ki else "Kd (µM)"
        _mkey      = "Best_Model" if is_ki else "Model"
        run_hdrs   = []
        for lbl in run_labels:
            run_hdrs += [f"{lbl}\n{val_lbl}", f"{lbl}\nfit SE", f"{lbl}\nR²adj"]
        headers = ["Combination"] + run_hdrs + ["Mean", "SD", "CV%", "n", "Models"]
        if is_ki:
            headers += ["Pooled Ki\n(µM)", "Pooled SE", "Pooled\nR²adj", "Pooled\nModel"]
        self._summary_tbl.setRowCount(len(combos))
        self._summary_tbl.setColumnCount(len(headers))
        self._summary_tbl.setHorizontalHeaderLabels(headers)
        self._summary_tbl.setSortingEnabled(False)

        for ri, combo in enumerate(combos):
            if is_ki:
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
                    v  = float(fr.get("Ki_uM",    np.nan) if is_ki
                               else fr.get("Kd",    np.nan))
                    se = float(fr.get("Ki_err_uM", np.nan) if is_ki
                               else fr.get("Kd_SE", np.nan))
                    r2 = float(fr.get("R2_adj", np.nan))
                except Exception:
                    v = se = r2 = np.nan
                def _g(x, dp=3): return f"{x:.{dp}g}" if not np.isnan(x) else "—"
                self._summary_tbl.setItem(ri, base,     QTableWidgetItem(_fmt_uM(v) if not np.isnan(v) else "—"))
                self._summary_tbl.setItem(ri, base + 1, QTableWidgetItem(_fmt_uM(se) if not np.isnan(se) else "—"))
                self._summary_tbl.setItem(ri, base + 2, QTableWidgetItem(_g(r2, 3)))
                if not np.isnan(v):
                    vals.append(v)

            unique_models   = sorted(set(models_seen))
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

            mdl_item = QTableWidgetItem(model_str)
            self._summary_tbl.setItem(ri, bs + 4, mdl_item)

            if is_ki:
                ps = bs + 5
                pf = _pooled_ki_fit(combo, runs)
                if pf:
                    self._summary_tbl.setItem(ri, ps,     QTableWidgetItem(_fmt_uM(pf["ki"])))
                    self._summary_tbl.setItem(ri, ps + 1, QTableWidgetItem(
                        _fmt_uM(pf["ki_err"]) if not np.isnan(pf["ki_err"]) else "—"))
                    self._summary_tbl.setItem(ri, ps + 2, QTableWidgetItem(f"{pf['r2_adj']:.3f}"))
                    self._summary_tbl.setItem(ri, ps + 3, QTableWidgetItem(pf["model"]))
                else:
                    for j in range(4):
                        self._summary_tbl.setItem(ri, ps + j, QTableWidgetItem("—"))

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
            canvas = RunCompareCanvas(
                combos, runs, self._mode, dpi_val=300, **kw)
            with PdfPages(path) as pdf:
                pdf.savefig(canvas.figure, facecolor="white")
            canvas.figure.clear()
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
        show_dots = kw["show_dots_panel"]
        show_resid = kw["show_residuals"]
        _lc = kw.get("layout_cfg") or {}

        cl = layout_utils.cell_layout(_lc, show_residuals=show_resid, show_dots=show_dots)
        fl = layout_utils.figure_layout(_lc, 1, 1, cl.row_h_in)

        n_written = 0
        try:
            for combo in combos:
                fig = Figure(figsize=(fl.fig_w, fl.fig_h), dpi=300)
                gs = GridSpec(cl.sub_rows, 1, figure=fig, height_ratios=cl.height_ratios,
                              hspace=cl.resid_gap, **fl.single_kwargs())
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
                    axis_fontsize=kw["axis_fontsize"],
                    tick_fontsize=kw["tick_fontsize"],
                    color_data=kw["color_data"], color_fit=kw["color_fit"],
                    color_resid=kw["color_resid"],
                    normalise_y=kw.get("normalise_y", False),
                    sci_notation_y=kw.get("sci_notation_y", False),
                    lw_fit=kw.get("lw_fit", 2.0),
                    ms_data=kw.get("ms_data", 4.0),
                    lw_errorbar=kw.get("lw_errorbar", 1.5),
                    ms_resid=kw.get("ms_resid", 2.5))

                if self._mode == "ki":
                    h, d, g = combo
                    stem = f"{h}_{d}_{g}"
                else:
                    h, d, dc = combo
                    stem = f"{h}_{d}_{dc}uM"
                safe_nm = _re.sub(r"[^\w\-]", "_", stem)
                out_path = os.path.join(ind_dir, f"{safe_nm}.pdf")
                with PdfPages(out_path) as pdf:
                    pdf.savefig(fig, facecolor="white")
                fig.clear()
                n_written += 1
            QMessageBox.information(
                self, "Exported",
                f"{n_written} individual PDF(s) written to:\n{ind_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_export_xlsx(self):
        import pandas as _pd
        from datetime import datetime as _dt
        combos = self._filtered_combos()
        if not combos:
            QMessageBox.warning(self, "Nothing to export",
                                "Build the grid first.")
            return
        out_dir = self._output_row.path
        ts = _dt.now().strftime("%y%m%d_%H%M%S")
        fname = f"{ts}_compare_runs_summary.xlsx"
        default = os.path.join(out_dir, fname) if out_dir else fname
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary XLSX", default,
            "Excel (*.xlsx);;All files (*)")
        if not path:
            return

        runs   = self._active_runs()
        is_ki  = self._mode == "ki"
        val_lbl = "Ki_uM" if is_ki else "Kd_uM"
        _mkey   = "Best_Model" if is_ki else "Model"
        rows: list = []

        for combo in combos:
            if is_ki:
                h, d, g = combo
                combo_str = f"{h} | {d} | {g}"
            else:
                h, d, dc = combo
                combo_str = f"{h} | {d} [{dc} µM]"

            row: dict = {"Combination": combo_str}
            if is_ki:
                row["Host"], row["Dye"], row["Guest"] = combo
            else:
                row["Host"], row["Dye"], row["Dye_Conc"] = combo

            vals: list = []
            models_seen: list = []
            for ci, run in enumerate(runs):
                lbl = run["label"]
                fr  = run["fit"].get(combo)
                if fr is None:
                    continue
                models_seen.append(str(fr.get(_mkey, "?")))
                try:
                    v  = float(fr.get("Ki_uM" if is_ki else "Kd", np.nan))
                    se = float(fr.get("Ki_err_uM" if is_ki else "Kd_SE", np.nan))
                    r2 = float(fr.get("R2_adj", np.nan))
                except Exception:
                    v = se = r2 = np.nan
                row[f"{lbl}_{val_lbl}"]  = v if not np.isnan(v) else None
                row[f"{lbl}_SE"]         = se if not np.isnan(se) else None
                row[f"{lbl}_R2adj"]      = r2 if not np.isnan(r2) else None
                row[f"{lbl}_Model"]      = str(fr.get(_mkey, ""))
                if not np.isnan(v):
                    vals.append(v)

            if vals:
                mean_v = float(np.mean(vals))
                sd_v   = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
                cv_v   = (sd_v / mean_v * 100) if sd_v and mean_v else None
            else:
                mean_v = sd_v = cv_v = None
            row["Mean"]  = mean_v
            row["SD"]    = sd_v
            row["CV%"]   = cv_v
            row["n"]     = len(vals)
            row["Models"] = ", ".join(sorted(set(models_seen))) if models_seen else None

            if is_ki:
                pf = _pooled_ki_fit(combo, runs)
                if pf:
                    row["Pooled_Ki_uM"]   = pf["ki"]
                    row["Pooled_Ki_SE"]   = pf["ki_err"] if not np.isnan(pf["ki_err"]) else None
                    row["Pooled_R2adj"]   = pf["r2_adj"]
                    row["Pooled_Model"]   = pf["model"]

            rows.append(row)

        try:
            df = _pd.DataFrame(rows)
            with _pd.ExcelWriter(path, engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="Summary", index=False)
                ws = w.sheets["Summary"]
                for ci, col in enumerate(df.columns):
                    max_len = max(len(str(col)),
                                  df[col].astype(str).str.len().max())
                    ws.set_column(ci, ci, min(max_len + 2, 30))
            QMessageBox.information(self, "Exported",
                                    f"Summary saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _on_save_all(self):
        import re as _re
        import pandas as _pd
        from datetime import datetime as _dt

        combos = self._filtered_combos()
        if not combos:
            QMessageBox.warning(self, "Nothing to save",
                                "Build the grid first.")
            return
        out_dir = self._output_row.path
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(
                self, "Select output folder")
            if not out_dir:
                return
            self._output_row.edit.setText(out_dir)

        os.makedirs(out_dir, exist_ok=True)
        ts   = _dt.now().strftime("%y%m%d_%H%M%S")
        runs = self._active_runs()
        kw   = self._plot_kw()
        is_ki = self._mode == "ki"
        saved: list = []

        try:
            # ── 1. Grid PDF ──
            grid_path = os.path.join(out_dir, f"{ts}_compare_runs_grid.pdf")
            canvas = RunCompareCanvas(
                combos, runs, self._mode, dpi_val=300, **kw)
            with PdfPages(grid_path) as pdf:
                pdf.savefig(canvas.figure, facecolor="white")
            canvas.figure.clear()
            del canvas
            saved.append(grid_path)

            # ── 2. Individual PDFs ──
            ind_dir = os.path.join(out_dir, "individual")
            os.makedirs(ind_dir, exist_ok=True)
            show_dots  = kw["show_dots_panel"]
            show_resid = kw["show_residuals"]
            _lc2 = kw.get("layout_cfg") or {}
            _cl2 = layout_utils.cell_layout(_lc2, show_residuals=show_resid, show_dots=show_dots)
            _fl2 = layout_utils.figure_layout(_lc2, 1, 1, _cl2.row_h_in)

            for combo in combos:
                fig = Figure(figsize=(_fl2.fig_w, _fl2.fig_h), dpi=300)
                gs  = GridSpec(_cl2.sub_rows, 1, figure=fig,
                               height_ratios=_cl2.height_ratios, hspace=_cl2.resid_gap,
                               **_fl2.single_kwargs())
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
                    axis_fontsize=kw["axis_fontsize"],
                    tick_fontsize=kw["tick_fontsize"],
                    color_data=kw["color_data"], color_fit=kw["color_fit"],
                    color_resid=kw["color_resid"],
                    normalise_y=kw.get("normalise_y", False),
                    sci_notation_y=kw.get("sci_notation_y", False),
                    lw_fit=kw.get("lw_fit", 2.0),
                    ms_data=kw.get("ms_data", 4.0),
                    lw_errorbar=kw.get("lw_errorbar", 1.5),
                    ms_resid=kw.get("ms_resid", 2.5))

                if is_ki:
                    h, d, g = combo
                    stem = f"{h}_{d}_{g}"
                else:
                    h, d, dc = combo
                    stem = f"{h}_{d}_{dc}uM"
                safe_nm = _re.sub(r"[^\w\-]", "_", stem)
                ind_path = os.path.join(ind_dir, f"{ts}_{safe_nm}.pdf")
                with PdfPages(ind_path) as pdf:
                    pdf.savefig(fig, facecolor="white")
                fig.clear()
                saved.append(ind_path)

            # ── 3. Summary XLSX ──
            xlsx_path = os.path.join(out_dir, f"{ts}_compare_runs_summary.xlsx")
            _mkey   = "Best_Model" if is_ki else "Model"
            val_key = "Ki_uM" if is_ki else "Kd"
            se_key  = "Ki_err_uM" if is_ki else "Kd_SE"
            rows_data: list = []

            for combo in combos:
                if is_ki:
                    h, d, g = combo
                    combo_str = f"{h} | {d} | {g}"
                else:
                    h, d, dc = combo
                    combo_str = f"{h} | {d} [{dc} µM]"

                row: dict = {"Combination": combo_str}
                if is_ki:
                    row["Host"], row["Dye"], row["Guest"] = combo
                else:
                    row["Host"], row["Dye"], row["Dye_Conc"] = combo

                vals: list = []
                models_seen: list = []
                for run in runs:
                    lbl = run["label"]
                    fr  = run["fit"].get(combo)
                    if fr is None:
                        continue
                    models_seen.append(str(fr.get(_mkey, "?")))
                    try:
                        v  = float(fr.get(val_key, np.nan))
                        se = float(fr.get(se_key, np.nan))
                        r2 = float(fr.get("R2_adj", np.nan))
                    except Exception:
                        v = se = r2 = np.nan
                    row[f"{lbl}_{val_key}"] = v if not np.isnan(v) else None
                    row[f"{lbl}_SE"]        = se if not np.isnan(se) else None
                    row[f"{lbl}_R2adj"]     = r2 if not np.isnan(r2) else None
                    row[f"{lbl}_Model"]     = str(fr.get(_mkey, ""))
                    if not np.isnan(v):
                        vals.append(v)

                if vals:
                    mean_v = float(np.mean(vals))
                    sd_v   = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
                    cv_v   = (sd_v / mean_v * 100) if sd_v and mean_v else None
                else:
                    mean_v = sd_v = cv_v = None
                row["Mean"]   = mean_v
                row["SD"]     = sd_v
                row["CV%"]    = cv_v
                row["n"]      = len(vals)
                row["Models"] = ", ".join(sorted(set(models_seen))) if models_seen else None

                if is_ki:
                    pf = _pooled_ki_fit(combo, runs)
                    if pf:
                        row["Pooled_Ki_uM"] = pf["ki"]
                        row["Pooled_Ki_SE"] = pf["ki_err"] if not np.isnan(pf["ki_err"]) else None
                        row["Pooled_R2adj"] = pf["r2_adj"]
                        row["Pooled_Model"] = pf["model"]

                rows_data.append(row)

            df = _pd.DataFrame(rows_data)
            with _pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="Summary", index=False)
                ws = w.sheets["Summary"]
                for ci, col in enumerate(df.columns):
                    max_len = max(len(str(col)),
                                  df[col].astype(str).str.len().max())
                    ws.set_column(ci, ci, min(max_len + 2, 30))
            saved.append(xlsx_path)

            QMessageBox.information(
                self, "Save complete",
                f"Saved {len(saved)} file(s) to:\n{out_dir}\n\n"
                f"• Grid PDF\n"
                f"• {len(combos)} individual PDF(s)\n"
                f"• Summary XLSX")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

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
    """Show ModePicker; if user picks a mode, open it alongside the current window."""
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
