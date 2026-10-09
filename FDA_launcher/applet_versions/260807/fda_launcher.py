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
    """Format a Kd/Ki value in µM: fixed-point when compact, scientific when
    very large or very small. A genuinely tight (sub-5nM) Ki would otherwise
    round to a misleading "0.00" under fixed 2-decimal formatting —
    indistinguishable from a failed/zero fit — so small nonzero values get
    scientific notation too, symmetric with the large-value case."""
    if np.isnan(v):
        return "NaN"
    if v != 0 and abs(v) < 0.01:
        return f"{v:.3E}"
    if abs(v) >= 10000:
        return f"{v:.4E}"
    return f"{v:.2f}"


def _traffic_text_color(rgba, override: str = "Auto") -> str:
    """Pick annotation text colour for a traffic-light band.

    `override` is either "Auto" (compute from the band's own colour luminance)
    or an explicit "Black text"/"White text" choice from the user.
    """
    if override == "Black text":
        return "black"
    if override == "White text":
        return "white"
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.5 else "white"


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

    Ki key: (host, dye, guest, host_conc_key)   host_conc_key via pl_ki._hc_key,
            so the same Host|Dye|Guest run at multiple host concentrations in
            the same file/batch stays split into distinct entries instead of
            the later row silently overwriting the earlier one.
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
                       str(r.get("Guest", "")).strip().title(),
                       pl_ki._hc_key(r.get("HostConc_uM", float("nan"))))
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

    Returns {"PASS": {(host,dye,guest,host_conc_key): {"concs": arr, "reps": [arr,...]}},
             "FAIL":  {...}}.
    Each group of 4 columns encodes one host|dye|guest[|host_conc] curve: the
    first column holds guest concentrations, the next three hold FI-F0
    triplicates. host_conc_key comes from pl_ki._hc_key so this lines up with
    the combo keys used elsewhere. Older exports (pre host-conc labeling)
    have only 3 pipe-delimited parts — they still parse, just with the
    NaN-sentinel host-conc key (no concentration disambiguation possible).
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
                if len(parts) in (3, 4):
                    host, dye, guest = parts[0], parts[1], parts[2].title()
                    hc_val = float("nan")
                    if len(parts) == 4:
                        hc_tok = parts[3].split()[0] if parts[3].split() else ""
                        try:
                            hc_val = float(hc_tok)
                        except ValueError:
                            hc_val = float("nan")
                    hc_key = pl_ki._hc_key(hc_val)
                    conc_series = df.iloc[:, i].dropna()
                    n = len(conc_series)
                    concs = conc_series.values.astype(float)
                    # Replicate-column count varies per combo (writer emits
                    # exactly max-replicates-seen columns, not a fixed 3) —
                    # consume every column up to the next "|" header instead
                    # of assuming a fixed 4-column stride, or a combo with
                    # e.g. 2 or 4 replicates misaligns all following combos.
                    j = i + 1
                    while j < len(cols):
                        nxt_clean = _re.sub(r'\.\d+$', '', str(cols[j])).strip()
                        if "|" in nxt_clean:
                            break
                        j += 1
                    reps = [df.iloc[:n, k].values.astype(float) for k in range(i + 1, j)]
                    entries[(host, dye, guest, hc_key)] = {"concs": concs, "reps": reps}
                    i = j
                    continue
            i += 1
        result[sheet] = entries
    return result


def _parse_raw_ki(path: str) -> dict:
    """competitive_data.xlsx → {(host,dye,guest,host_conc_key): {'x_log': arr, 'reps': [arr,…]}}"""
    raw = _parse_run_data(path)
    result: dict = {}
    for sheet_data in raw.values():
        for key, entry in sheet_data.items():
            concs = entry["concs"]
            pos   = concs > 0
            result[key] = {
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
                # Replicate-column count varies per combo (writer emits
                # exactly max-replicates-seen columns, not a fixed 3) —
                # consume every column up to the next header column instead
                # of assuming a fixed 4-column stride.
                j = i + 1
                while j < len(cols):
                    nxt_clean = _re.sub(r'\.\d+$', '', str(cols[j])).strip()
                    if col_re.match(nxt_clean):
                        break
                    j += 1
                reps = [df.iloc[:n, k].values.astype(float) for k in range(i + 1, j)]
                result[(host, dye, dc_str)] = {"x": cs.values.astype(float), "reps": reps}
                i = j
                continue
            i += 1
    return result


def _lookup_raw_combo(raw: dict, combo: tuple):
    """Look up a combo in a run's raw-data dict, with a fallback for Ki
    combos whose host-conc key can't be matched exactly.

    Raw files exported before host-conc got folded into the wide-format
    header (see _parse_run_data) only ever produce the NaN-sentinel host-conc
    key, while fit_results (a separate file, unaffected by that header
    format) still carries the real HostConc_uM — so an exact 4-tuple match
    always misses for those older files even though the data is right there.
    Falls back to a host/dye/guest-only match, but only when it's
    unambiguous (exactly one raw entry for that triple) — with several host
    concentrations for the same triple in one file, there's no way to tell
    them apart post hoc, so we correctly report no match rather than risk
    silently pooling data from the wrong concentration."""
    rd = raw.get(combo)
    if rd is not None or len(combo) != 4:
        return rd
    host, dye, guest = combo[0], combo[1], combo[2]
    candidates = [v for k, v in raw.items() if len(k) == 4 and k[:3] == (host, dye, guest)]
    return candidates[0] if len(candidates) == 1 else None


def _grubbs_mean_sem(norm_reps: list):
    """Per-concentration mean +/- SEM across a run's replicates, Grubbs-
    testing outliers at each concentration first (mirrors the outlier
    rejection pipeline_ki.fit_curves_ki applies to raw replicates before
    fitting) so one bad well doesn't skew the run-level point Compare Runs'
    pooled fit is built from. Only tests when >=3 replicates are present at
    that concentration, same as the main pipeline."""
    arr = np.asarray(norm_reps, dtype=float)  # shape (n_reps, n_conc)
    n_conc = arr.shape[1]
    mean = np.full(n_conc, np.nan)
    sem  = np.zeros(n_conc)
    for c in range(n_conc):
        col = arr[:, c]
        col = col[np.isfinite(col)]
        if len(col) >= 3:
            col = col[pl_ki._grubbs_mask(col)]
        if len(col) == 0:
            continue
        mean[c] = float(np.mean(col))
        sem[c]  = float(np.std(col, ddof=1) / np.sqrt(len(col))) if len(col) > 1 else 0.0
    return mean, sem


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
        elif model == "Wang":
            if np.isnan(HostConc):
                return None, None, "HostConc missing"
            fn, popt = pl_ki._wang_cubic(DyeConc, DyeKd, HostConc), [Top, Bottom, logKi]
        else:
            # Unrecognized/no-longer-supported model (e.g. a pre-existing
            # saved result with Best_Model == "Biphasic", removed from the
            # fitting code) — degrade gracefully to the Standard curve
            # shape rather than erroring.
            if model not in ("Standard",):
                warn = f"Model '{model}' not supported for curve reconstruction — Standard curve shown"
            fn, popt = pl_ki._comp_standard(DyeConc, DyeKd), [Top, Bottom, logKi]

        lo = min(Top, Bottom)
        y_norm = (fn(x, *popt) - lo) / abs(span)
        return x, y_norm, warn
    except Exception as exc:
        return None, None, str(exc)


def _model_disp(fit_row: dict, mkey: str = "Best_Model") -> str:
    """Model name for display, with ' (approx.)' appended when the winning
    Ki model reused the Standard model's Cheng-Prusoff EC50 shift outside
    its exact regime (HillSlope — see Model_Basis in
    pipeline_ki.fit_curves_ki). Falls back to a name-based heuristic for
    saved results predating the Model_Basis column."""
    name = str(fit_row.get(mkey, "?"))
    basis = fit_row.get("Model_Basis")
    if basis is None:
        basis = "approximate_cp_shift" if name == "HillSlope" else "exact"
    return f"{name} (approx.)" if basis == "approximate_cp_shift" else name


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
    """"Normalized Run-Pooling" — the "Compare Runs" pooling procedure.

    Procedure: (1) normalize each run's replicate curve to its own Top/Bottom
    fit, (2) average replicates within each run at each concentration
    (with within-run Grubbs), (3) pool the resulting run-level means across
    runs at each shared concentration (with a second, across-run Grubbs
    pass), then (4) refit a single Ki/model to the pooled, normalized data.

    This is a distinct procedure from (a) a global/shared-parameter fit of
    the raw (non-normalized) data pooled across runs, and (b) a formal
    random-effects meta-analysis of independently-fitted per-run Ki values
    (e.g. inverse-variance weighting). Unlike either of those, this
    procedure does NOT propagate each run's own Top/Bottom fitting
    uncertainty into the pooled fit — each run's normalization anchor
    (its own Top/Bottom) is treated as exact, not as an estimate with its
    own SE. A future enhancement could propagate Top/Bottom SE via a
    weighted normalization, or replace this with a proper mixed-effects
    pooling of the per-run Ki estimates.

    Returns dict with keys: ki, ki_err, r2_adj, model, cp_shift, ec50, popt, fn
    or None if the fit fails or insufficient data.

    Only PASS-status fit rows ever contribute here, regardless of the Status
    filter the caller's `runs` list was built under — a FAIL run showing up
    because the user wants to *look at* it (Status: All/FAIL only) must never
    dilute the pooled fit with an unreliable curve.

    NOT the reported statistic: ki_err here is optimistic (see above) and
    must never be shown/exported as the pooled Ki's uncertainty. Use
    _pooled_ki_simple for that — geometric mean +/- SEM of logKi across
    runs, the standard treatment at n=2-3 independent experiments. The
    Compare-Runs plot itself no longer calls this function either: it
    evaluates the curve at _pooled_ki_simple's already-reported logKi
    (_pooled_ki_display) rather than refitting, so the number in the title
    and the curve on the plot can never disagree. This function is kept
    (not deleted) as a reusable AICc-refit building block should a future
    feature want an actual model-fit visualization of the pooled data.
    """
    from scipy.optimize import curve_fit as _curve_fit

    # Two sequential Grubbs passes, alpha=0.05 each (pl_ki._grubbs_mask's
    # default, used unmodified by both calls below): (1) within each run,
    # test the replicates at each concentration before averaging them down
    # to that run's mean curve; (2) across runs, test the run-level means
    # that land on the same concentration before they're combined into the
    # final fitting dataset. Run-level weighting (one point per run per
    # concentration) is otherwise unchanged.
    all_x, pooled_x, pooled_y = [], [], []
    _by_x: dict = {}
    for run in runs:
        fit_row = run["fit"].get(combo)
        if fit_row is None or fit_row.get("_status") != "PASS":
            continue
        rd = _lookup_raw_combo(run["raw"], combo)
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
                run_mean, _run_sem = _grubbs_mean_sem(norm_reps)
                for xi, yi in zip(xv.tolist(), run_mean.tolist()):
                    if np.isfinite(yi):
                        _by_x.setdefault(round(xi, 6), []).append(yi)
        except Exception:
            continue

    for _xi, _yvals in _by_x.items():
        _yarr = np.asarray(_yvals, dtype=float)
        if len(_yarr) >= 3:
            _yarr = _yarr[pl_ki._grubbs_mask(_yarr)]
        pooled_x.extend([_xi] * len(_yarr))
        pooled_y.extend(_yarr.tolist())

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
    x_min_pool = float(np.min(px))
    y_rng = max(abs(top0 - bot0), 1.0)
    # Bounds mirror pipeline_ki.fit_curves_ki's bounds_std/hillslope bounds —
    # an unbounded curve_fit on the sparser pooled (run-mean) data can run
    # logEC50 off to +/-inf, producing a nonsensical Ki (e.g. ~0) that the
    # bounded single-run fit would never report.
    bounds_std  = ([bot0 - 2*y_rng, bot0 - 2*y_rng, -14],
                   [top0 + 2*y_rng, top0 + 2*y_rng,  8])
    bounds_hill = ([bot0 - 2*y_rng, bot0 - 2*y_rng, -14, 0.1],
                   [top0 + 2*y_rng, top0 + 2*y_rng,  8, 10.0])

    hs_vals, hc_val = [], np.nan
    cp_shift = 0.0
    dc_val = dk_val = np.nan
    for run in runs:
        fr = run["fit"].get(combo)
        if fr and fr.get("_status") == "PASS":
            try: hs_vals.append(abs(float(fr["HillSlope"])))
            except (TypeError, ValueError, KeyError): pass
            try:
                hc = float(fr["HostConc_uM"])
                if not np.isnan(hc): hc_val = hc
            except (TypeError, ValueError, KeyError): pass
            try:
                dc = float(fr["DyeConc_uM"])
                dk = float(fr["DyeKd_uM"])
                if dk > 0:
                    cp_shift = np.log10(1.0 + dc / dk)
                    dc_val, dk_val = dc, dk
            except (TypeError, ValueError, KeyError): pass

    hs0    = float(np.mean(hs_vals)) if hs_vals else 1.0

    def _aicc(n, rss, k):
        if n <= k + 1: return np.inf
        if rss <= 0:   return -np.inf
        return n * np.log(rss / n) + 2.0*k + (2.0*k*(k+1)) / (n - k - 1)

    def _std(x, top, bot, logEC50):
        return bot + (top - bot) / (1.0 + 10.0 ** (x - logEC50))
    def _hill(x, top, bot, logEC50, hs):
        return bot + (top - bot) / (1.0 + 10.0 ** (hs * (x - logEC50)))

    # Preliminary Standard fit — gates Wang exactly as pipeline_ki does:
    # Wang only enters the candidate set when Ki_std < WANG_GATE * [Host].
    _std_Ki   = np.nan
    _std_pfit = None
    try:
        _po_s, _pc_s = _curve_fit(_std, px, py, p0=[top0, bot0, med_x],
                                  bounds=bounds_std, maxfev=5000)
        _std_Ki   = 10.0 ** (float(_po_s[2]) - cp_shift)
        _std_pfit = (_po_s, _pc_s)
    except (RuntimeError, ValueError):
        pass

    _use_wang = (not np.isnan(hc_val) and not np.isnan(dc_val) and not np.isnan(dk_val)
                 and not np.isnan(_std_Ki) and _std_Ki < pl_ki.WANG_GATE_DEFAULT * hc_val)

    candidates = [
        {"name": "Standard",  "fn": _std,  "k": 3, "p0": [top0, bot0, med_x],
         "bounds": bounds_std, "prefit": _std_pfit},
        {"name": "HillSlope", "fn": _hill, "k": 4, "p0": [top0, bot0, med_x, hs0],
         "bounds": bounds_hill},
    ]
    if _use_wang:
        # Wang fits the guest's true Ki directly (unlike Standard/HillSlope's
        # apparent EC50), so it needs the actual dye conc./Kd, not cp_shift.
        # Multiple seeds (as pipeline_ki does) guard against a single bad
        # local minimum on the sparser pooled data — each seed competes for
        # best AICc like any other candidate.
        _wang_fn = pl_ki._wang_cubic(dc_val, dk_val, hc_val)
        for _seed in (x_min_pool, med_x - 1.5, med_x - 0.5, med_x, med_x + 0.5):
            candidates.append({"name": "Wang", "fn": _wang_fn, "k": 3,
                               "p0": [top0, bot0, _seed], "bounds": bounds_std})

    best = None
    for cand in candidates:
        if n_pts <= cand["k"]:
            continue
        prefit = cand.get("prefit")
        if prefit is not None:
            po, pc = prefit
        else:
            try:
                po, pc = _curve_fit(cand["fn"], px, py, p0=cand["p0"],
                                    bounds=cand["bounds"], maxfev=5000)
            except (RuntimeError, ValueError):
                continue
        # A fit that converges onto the logEC/logKi bound wall (-14 or 8,
        # i.e. Ki ~1e-14 or ~1e8 uM) is a numerical artifact of sparse pooled
        # data finding a degenerate near-vertical curve, not a real
        # near-zero/near-infinite Ki — reject it rather than report a
        # physically implausible value.
        _lo_bound, _hi_bound = cand["bounds"][0][2], cand["bounds"][1][2]
        if po[2] <= _lo_bound + 0.5 or po[2] >= _hi_bound - 0.5:
            continue
        # Standard pharmacological QC: an EC50/Ki fitted many decades outside
        # the tested concentration range isn't actually constrained by the
        # data — sparse pooled data can still let AICc pick such a fit over
        # a duller, better-anchored one. Reject any EC/Ki parameter more than
        # 3 log units beyond [min, max] of the concentrations actually tested.
        _ec_idxs = (2,)
        if any(po[_i] < px.min() - 3.0 or po[_i] > px.max() + 3.0 for _i in _ec_idxs):
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
    ki_val = 10.0 ** logEC if best["name"] == "Wang" else 10.0 ** (logEC - cp_shift)
    try:
        se_logEC = float(np.sqrt(best["pcov"][2, 2]))
        ki_err   = ki_val * np.log(10) * se_logEC
    except Exception:
        ki_err = np.nan

    # EC50/IC50 (v2 item 7): the raw fitted midpoint, exposed alongside the
    # pooled Ki because it needs no Cheng-Prusoff correction — see the
    # per-run EC50_uM derivation in pipeline_ki.fit_curves_ki.
    ec50_val = 10.0 ** logEC

    return {"ki": ki_val, "ki_err": ki_err, "r2_adj": best["r2_adj"],
            "model": best["name"], "cp_shift": cp_shift, "ec50": ec50_val,
            "popt": best["popt"], "fn": best["fn"]}


def _pooled_ki_simple(combo: tuple, runs: list, status_filter: str = "PASS"):
    """Standard biochemistry-assay pooling: geometric mean +/- SEM of
    log10(Ki) across independent runs (unweighted — each run counts once,
    the way 'n = independent experiments' is conventionally reported in
    binding-assay papers; see Motulsky & Christopoulos, 'Fitting Models to
    Biological Data'). Appropriate specifically because n is small (2-3):
    does NOT attempt to separately estimate between-run vs within-run
    variance (that needs many more runs to do reliably) — the spread
    across the 2-3 runs is the only, and sufficient, source of pooled
    uncertainty here.

    This is the number that should be reported/exported/plotted anywhere a
    single pooled Ki is shown — NOT _pooled_ki_fit's refit, which treats
    each run's own Top/Bottom normalization as exact and so understates the
    true uncertainty (see _pooled_ki_fit's docstring).

    status_filter (default "PASS") selects which runs' fit rows contribute.
    Passing "FAIL" pools across runs that all failed QC instead — used only
    for the plot-only "consistently failed" visualisation (see
    _render_compare_combo), never for the reported/exported pooled Ki.

    Returns dict: ki, ki_lo, ki_hi, n_runs, individual_ki (list), log_sem,
    note; or None if no runs contribute.
    """
    from scipy.stats import t as _t_dist

    logkis = []
    for run in runs:
        fr = run["fit"].get(combo)
        if fr is not None and fr.get("_status") == status_filter:
            try:
                lk = float(fr["logKi"])
                if np.isfinite(lk):
                    logkis.append(lk)
            except (TypeError, ValueError, KeyError):
                continue

    n = len(logkis)
    if n == 0:
        return None

    logkis = np.asarray(logkis)
    log_mean = float(np.mean(logkis))
    individual_ki = (10.0 ** logkis).tolist()

    if n == 1:
        return {"ki": individual_ki[0], "ki_lo": np.nan, "ki_hi": np.nan,
                "n_runs": 1, "individual_ki": individual_ki, "log_sem": np.nan,
                "note": "single run — no pooling performed"}

    log_sem = float(np.std(logkis, ddof=1) / np.sqrt(n))
    t_crit  = float(_t_dist.ppf(0.975, n - 1))   # n-1 df: 12.7 at n=2, 4.3 at n=3

    return {
        "ki":            10.0 ** log_mean,   # geometric mean
        "ki_lo":         10.0 ** (log_mean - t_crit * log_sem),
        "ki_hi":         10.0 ** (log_mean + t_crit * log_sem),
        "n_runs":        n,
        "individual_ki": individual_ki,
        "log_sem":       log_sem,
        "note": ("wide/unreliable CI at n=2 "
                  if n == 2 else ""),
    }


def _pooled_kd_simple(combo: tuple, runs: list):
    """Kd equivalent of _pooled_ki_simple — same geometric-mean/log-SEM
    pooling, structurally identical, just reading fr["Kd"] (Direct Binding
    fit rows have no logKi field, so it's log-transformed here) instead of
    the already-log fr["logKi"]. See _pooled_ki_simple's docstring for the
    full statistical rationale (Motulsky & Christopoulos; unweighted,
    appropriate at n=2-3).

    Returns dict: kd, kd_lo, kd_hi, n_runs, individual_kd (list), log_sem,
    note; or None if no runs contribute.
    """
    from scipy.stats import t as _t_dist

    logkds = []
    for run in runs:
        fr = run["fit"].get(combo)
        if fr is not None and fr.get("_status") == "PASS":
            try:
                kd = float(fr["Kd"])
                if kd > 0 and np.isfinite(kd):
                    logkds.append(np.log10(kd))
            except (TypeError, ValueError, KeyError):
                continue

    n = len(logkds)
    if n == 0:
        return None

    logkds = np.asarray(logkds)
    log_mean = float(np.mean(logkds))
    individual_kd = (10.0 ** logkds).tolist()

    if n == 1:
        return {"kd": individual_kd[0], "kd_lo": np.nan, "kd_hi": np.nan,
                "n_runs": 1, "individual_kd": individual_kd, "log_sem": np.nan,
                "note": "single run — no pooling performed"}

    log_sem = float(np.std(logkds, ddof=1) / np.sqrt(n))
    t_crit  = float(_t_dist.ppf(0.975, n - 1))   # n-1 df: 12.7 at n=2, 4.3 at n=3

    return {
        "kd":            10.0 ** log_mean,   # geometric mean
        "kd_lo":         10.0 ** (log_mean - t_crit * log_sem),
        "kd_hi":         10.0 ** (log_mean + t_crit * log_sem),
        "n_runs":        n,
        "individual_kd": individual_kd,
        "log_sem":       log_sem,
        "note": ("wide/unreliable CI at n=2 "
                  if n == 2 else ""),
    }


def _build_compare_summary_row(combo: tuple, runs: list, mode: str,
                               cv_flag_threshold: float) -> dict:
    """Build one row of the Compare Runs "..._compare_runs_summary.xlsx"
    export. Single shared implementation for both the standalone "Export
    Summary XLSX" button and the "Save All" combined export — previously
    each had its own copy of this logic, and had drifted out of sync badly
    enough that the standalone button's copy never assigned `row =
    {"Combination": ...}` at all and raised NameError on first use. Keeping
    one copy also means the column order below is the ONLY column order,
    everywhere this export is produced.

    Column order (deliberate, grouped by what a reviewer needs first, not
    the order fields happened to be computed in):
      1. Identity        — Combination, Host, Dye, Guest/Dye_Conc, HostConc_uM
      2. Triage headline — Status, N_PASS, N_FAIL, CV%, Log_CV%, Consistency,
                            Any_flags
      3. Aggregate stats — Mean, SD, Models
      4. Pooled stats    — Pooled_Ki_uM.../Pooled_note,
                            Pooled_FAIL_Ki_uM/_n_runs (Ki mode, plot-only
                            equivalent — see _render_compare_combo's
                            pool_fail_fallback docstring; never populated
                            alongside a real Pooled_Ki_uM, since that only
                            happens when zero PASS runs contributed)
      5. Per-run detail  — {run}_Ki_uM, _SE, _R2adj, _Model, _Status, ... —
                            last, since these are for drilling into one run,
                            not first-glance triage.
    """
    is_ki = mode == "ki"
    if is_ki:
        h, d, g, hc = combo
        _hc_str = f"  ({hc:g} µM host)" if hc != -1.0 else ""
        combo_str = f"{h} | {d} | {g}{_hc_str}"
    else:
        h, d, dc = combo
        combo_str = f"{h} | {d} [{dc} µM]"

    row: dict = {"Combination": combo_str}
    if is_ki:
        row["Host"], row["Dye"], row["Guest"] = h, d, g
        row["HostConc_uM"] = np.nan if hc == -1.0 else hc
    else:
        row["Host"], row["Dye"], row["Dye_Conc"] = combo

    _mkey   = "Best_Model" if is_ki else "Model"
    val_key = "Ki_uM" if is_ki else "Kd"
    se_key  = "Ki_err_uM" if is_ki else "Kd_SE"

    run_cols: dict = {}
    vals: list = []          # PASS-only — feeds Mean/SD/CV%/n
    models_seen: list = []   # PASS-only — feeds the Models column
    statuses_seen: list = []
    flag_notes: list = []    # PASS-only — feeds Any_flags
    for run in runs:
        lbl = run["label"]
        fr  = run["fit"].get(combo)
        if fr is None:
            continue
        _status = str(fr.get("_status", ""))
        statuses_seen.append(_status)
        try:
            v  = float(fr.get(val_key, np.nan))
            se = float(fr.get(se_key, np.nan))
            r2 = float(fr.get("R2_adj", np.nan))
        except Exception:
            v = se = r2 = np.nan
        # Per-run columns always show this run's own data, regardless of
        # status, so a FAIL run picked up by the Status filter can still be
        # inspected here.
        run_cols[f"{lbl}_{val_key}"]  = v if not np.isnan(v) else None
        run_cols[f"{lbl}_SE"]         = se if not np.isnan(se) else None
        run_cols[f"{lbl}_R2adj"]      = r2 if not np.isnan(r2) else None
        run_cols[f"{lbl}_Model"]      = _model_disp(fr, _mkey) if fr.get(_mkey) else ""
        run_cols[f"{lbl}_Status"]     = _status
        run_cols[f"{lbl}_Confidence"] = fr.get("Confidence") or ""
        if is_ki:
            run_cols[f"{lbl}_Hill_flag"] = (bool(fr.get("Hill_flag"))
                if fr.get(_mkey) == "HillSlope" and fr.get("Hill_flag") is not None
                else None)
        else:
            _sa = fr.get("Sign_ambiguous")
            run_cols[f"{lbl}_Sign_ambiguous"] = bool(_sa) if _sa is not None and not (
                isinstance(_sa, float) and np.isnan(_sa)) else None
        if _status == "PASS":
            models_seen.append(_model_disp(fr, _mkey))
            if not np.isnan(v):
                vals.append(v)
            if fr.get("Confidence") and fr.get("Confidence") != "High":
                flag_notes.append("Low confidence")
            if fr.get("Sign_ambiguous"):
                flag_notes.append("Sign ambiguous")
            if fr.get(_mkey) == "HillSlope" and fr.get("Hill_flag"):
                flag_notes.append("Hill slope flag")
            if fr.get("Model_Basis") == "approximate_cp_shift":
                flag_notes.append("Approx. model")

    if vals:
        mean_v = float(np.mean(vals))
        sd_v   = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
        cv_v   = (sd_v / mean_v * 100) if sd_v and mean_v else None
    else:
        mean_v = sd_v = cv_v = None

    # Reported/exported pooled statistic: geometric mean +/- 95% CI of
    # log10(Kd or Ki) across independent runs (_pooled_ki_simple /
    # _pooled_kd_simple), NOT a curve refit (_pooled_ki_fit — plotting only;
    # its refit SE is optimistic, see its docstring). log_sem (log-scale
    # dispersion) drives "Consistency" for BOTH assay types, instead of the
    # linear CV% — Kd/Ki are multiplicative quantities.
    ps_stat = _pooled_ki_simple(combo, runs) if is_ki else _pooled_kd_simple(combo, runs)
    log_cv_pct = np.nan
    if ps_stat and ps_stat["n_runs"] >= 2 and not np.isnan(ps_stat["log_sem"]):
        log_cv_pct = (10.0 ** ps_stat["log_sem"] - 1.0) * 100.0

    # Per-combo overall Status — PASS if any contributing run passed, so
    # Summary Plots (or anything re-loading this export) can tell real FAIL
    # combos apart instead of inferring PASS/FAIL from the Ki value alone (a
    # FAIL row can still carry a finite Ki_uM).
    row["Status"] = ("PASS" if "PASS" in statuses_seen
                      else "FAIL" if statuses_seen else None)
    # N_PASS/N_FAIL: how many of the runs that even attempted this combo
    # landed on each side — distinct from Status (which only says whether
    # *any* run passed) and from the per-run {run}_Status columns (which
    # need opening every run's own column to tally by hand).
    row["N_PASS"] = len(vals)
    row["N_FAIL"] = statuses_seen.count("FAIL")
    row["CV%_linear_reference_only"] = cv_v
    row["Log_CV%"] = log_cv_pct
    # Headline triage flag: inconsistent across runs is a bad hit regardless
    # of any individual run's Status. Driven by log-scale dispersion
    # (Log_CV%, the same number stored above), NOT
    # CV%_linear_reference_only — that column is display-only and is never
    # the one highlighted in the export, so the highlighted column and this
    # text can never disagree.
    row["Consistency"] = (
        f"Inconsistent across runs (log-scale CV%={log_cv_pct:.1f} > "
        f"{cv_flag_threshold:.0f}) — recommend orthogonal retest before follow-up"
        if not np.isnan(log_cv_pct) and log_cv_pct > cv_flag_threshold
        else "")
    row["Any_flags"] = ", ".join(sorted(set(flag_notes))) if flag_notes else ""
    row["Mean"] = mean_v
    row["SD"]   = sd_v
    row["Models"] = ", ".join(sorted(set(models_seen))) if models_seen else None

    if ps_stat:
        _val_key, _lo_key, _hi_key = ("ki", "ki_lo", "ki_hi") if is_ki else ("kd", "kd_lo", "kd_hi")
        _ind_key = "individual_ki" if is_ki else "individual_kd"
        _pfx = "Pooled_Ki" if is_ki else "Pooled_Kd"
        _pval = "Pooled_pKi" if is_ki else "Pooled_pKd"
        lo, hi = ps_stat[_lo_key], ps_stat[_hi_key]
        row[f"{_pfx}_uM"]      = ps_stat[_val_key]
        row[f"{_pfx}_CI_low"]  = None if np.isnan(lo) else lo
        row[f"{_pfx}_CI_high"] = None if np.isnan(hi) else hi
        row[_pval]             = 6.0 - np.log10(ps_stat[_val_key])
        row[f"{_pval}_CI_low"]  = None if np.isnan(hi) else 6.0 - np.log10(hi)
        row[f"{_pval}_CI_high"] = None if np.isnan(lo) else 6.0 - np.log10(lo)
        row["Pooled_n_runs"]        = ps_stat["n_runs"]
        row[f"Pooled_individual_{'Ki' if is_ki else 'Kd'}"] = ", ".join(
            f"{v:.4g}" for v in ps_stat[_ind_key])
        row["Pooled_note"]          = ps_stat["note"]
    elif is_ki:
        # Zero PASS runs — plot-only equivalent of pool_fail_fallback (see
        # _render_compare_combo): a combo failing consistently across every
        # run that attempted it is itself informative. Separate column
        # names so this can never be mistaken for, or overwrite, a real
        # Pooled_Ki_uM — populated only when Pooled_Ki_uM above is absent.
        ps_fail = _pooled_ki_simple(combo, runs, status_filter="FAIL")
        if ps_fail is not None:
            row["Pooled_FAIL_Ki_uM"]    = ps_fail["ki"]
            row["Pooled_FAIL_n_runs"]   = ps_fail["n_runs"]

    row.update(run_cols)
    return row


def _pooled_ki_display(combo: tuple, runs: list, logki_pooled: float,
                        cp_shift: float, status_filter: str = "PASS"):
    """Build normalized per-run data points for plotting a combined Ki
    curve. Does NOT re-fit — draws the caller's already-reported pooled
    logKi (from _pooled_ki_simple), so the figure and the reported number
    can never disagree.

    status_filter (default "PASS") must match whatever status_filter built
    logki_pooled via _pooled_ki_simple — pass "FAIL" for the plot-only
    "consistently failed" curve (see _render_compare_combo).

    Each run's own points are normalized via (r - min(Top,Bottom)) /
    abs(span) — see the per_run loop — which always lands in [0, 1] but can
    either DECREASE or INCREASE with x depending on that run's own
    Top_fit/Bottom_fit sign (curve_fit is free to converge to Top<Bottom for
    a "quenching"-type combo whose real signal rises with x — see
    fit_curves_ki's bounds_std, which do not force Top>Bottom). curve_fn
    below is built with a single fixed (decreasing) shape, so it must be
    inverted whenever the data it's meant to describe actually increases —
    determined here directly from the pooled data's own trend (sign of a
    linear fit through all contributing runs' points), NOT from any one
    run's Top_fit/Bottom_fit: different runs of the same combo can
    individually converge to opposite Top/Bottom signs on noisy/weak-signal
    data, so trusting a single reference run could pick the wrong
    orientation even when the pooled data itself has a clear trend.

    Returns dict: per_run (list of {x, y_norm} per run — for individual
    point overlay when n_runs < 3), pooled_points (mean/sem across
    runs at each shared x — for n_runs >= 3), curve_fn, n_runs,
    show_individual_lines. Or None if no runs contribute.
    """
    per_run = []
    for run in runs:
        fr = run["fit"].get(combo)
        if fr is None or fr.get("_status") != status_filter:
            continue
        rd = _lookup_raw_combo(run["raw"], combo)
        if rd is None:
            continue
        try:
            top, bottom = float(fr["Top_fit"]), float(fr["Bottom_fit"])
        except (TypeError, ValueError, KeyError):
            continue
        span = top - bottom
        if abs(span) < 1e-10:
            continue
        xv = rd["x_log"]
        reps_norm = [(r.astype(float) - min(top, bottom)) / abs(span)
                     for r in rd["reps"] if len(r) == len(xv)]
        if not reps_norm:
            continue
        y_run = np.mean(np.vstack(reps_norm), axis=0)
        per_run.append({"x": xv, "y_norm": y_run, "label": run["label"],
                        "color": run["color"]})

    n_runs = len(per_run)
    if n_runs == 0:
        return None

    by_x: dict = {}
    for run_pts in per_run:
        for xi, yi in zip(run_pts["x"], run_pts["y_norm"]):
            by_x.setdefault(round(float(xi), 6), []).append(yi)
    pooled_points = []
    for xi, yvals in sorted(by_x.items()):
        yarr = np.asarray(yvals)
        sem = (float(np.std(yarr, ddof=1) / np.sqrt(len(yarr)))
               if len(yarr) > 1 else np.nan)
        pooled_points.append({"x": xi, "mean": float(np.mean(yarr)),
                              "sem": sem, "n": len(yarr)})

    # Orientation: fit a simple line through every contributing run's own
    # (x, y_norm) points (not just the per-x means, so this still works when
    # n_runs < 3 and pooled_points isn't used for display) and invert
    # curve_fn if the data's own slope is positive — see docstring above.
    _all_x = np.concatenate([rp["x"] for rp in per_run])
    _all_y = np.concatenate([rp["y_norm"] for rp in per_run])
    invert = False
    if len(_all_x) >= 2 and np.ptp(_all_x) > 0:
        try:
            slope = np.polyfit(_all_x, _all_y, 1)[0]
            invert = slope > 0
        except Exception:
            invert = False

    def curve_fn(x):
        logEC50 = logki_pooled + cp_shift
        base = 1.0 / (1.0 + 10.0 ** (x - logEC50))  # normalized Top=1, Bottom=0
        return (1.0 - base) if invert else base

    return {"per_run": per_run, "pooled_points": pooled_points,
            "curve_fn": curve_fn, "n_runs": n_runs,
            "show_individual_lines": n_runs < 3}


def _build_pooled_ki_summary_df(combos: list, runs: list):
    """Build a SummaryTab(mode='ki')-compatible DataFrame from Compare Runs
    pooled statistics: one row per combo with the pooled Ki (geometric mean
    across independent runs, via _pooled_ki_simple — NOT _pooled_ki_fit's
    refit, which understates uncertainty) plus an "N" column (number of
    independent runs pooled) for display. Only combos with >=1 contributing
    run are included — same convention as fit_curves_ki, which omits curves
    it can't fit rather than reporting a placeholder.

    Ki_err_uM here is the log-scale SEM converted back to a linear-scale SE
    (Ki * ln(10) * log_sem) — NOT the full 95% CI — purely so SummaryTab's
    existing bar-chart error bars (which read a single symmetric SE column)
    render a sensible width. The full, honest asymmetric 95% CI is only
    reliably obtainable from _pooled_ki_simple directly (ki_lo/ki_hi) — use
    that, not this column, anywhere the interval itself needs reporting.
    """
    import pandas as _pd

    rows = []
    for combo in combos:
        h, d, g, hc = combo
        ps = _pooled_ki_simple(combo, runs)
        if ps is None:
            continue
        ki_err_uM = (ps["ki"] * np.log(10) * ps["log_sem"]
                     if not np.isnan(ps.get("log_sem", np.nan)) else np.nan)
        rows.append({
            "Host": h, "Dye": d, "Guest": g,
            "HostConc_uM": np.nan if hc == -1.0 else hc,
            "Ki_uM": ps["ki"], "Ki_err_uM": ki_err_uM,
            "N": ps["n_runs"], "Status": "PASS",
            # Wide/unreliable-CI note (set at n_runs==2 by _pooled_ki_simple)
            # must travel with N wherever N is rendered — see _bar_chart's
            # "n=X" annotation, which appends "⚠" for rows carrying a note.
            "Note": ps.get("note", "") or "",
        })
    return _pd.DataFrame(rows)


def _render_compare_combo(ax_curve, ax_dots, combo: tuple, runs: list, mode: str,
                          show_legend: bool = False, flip_norm: bool = False,
                          pooled_mode: bool = False, ax_resid=None,
                          pool_fail_fallback: bool = True,
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

    pool_fail_fallback (Ki + pooled_mode only, default True): when a combo
    has zero PASS runs, pool across its FAIL runs instead and draw that
    curve/points in a visually distinct (red-toned) style, rather than the
    plain "not enough data" placeholder — a combo failing consistently
    across every run is itself informative (a plausible true negative), and
    otherwise never shows up anywhere. Plot-only: never feeds a numeric Ki
    into any summary/export, since a value derived entirely from failed fits
    (poor R2, out-of-range Ki, etc.) isn't reliable enough to report
    alongside real PASS results. Set False to restore the old placeholder.
    """
    from scipy.optimize import curve_fit as _curve_fit

    ki_vals, ki_errs, dot_labels, dot_colors = [], [], [], []
    ki_vals_pass: list = []   # PASS-only subset — feeds Ki_mean/SD stats and
                              # the dot-panel reference line, so a FAIL run
                              # shown per the Status filter never dilutes them
    legend_lines, legend_labels_l = [], []
    _resid_col = color_resid or color_data

    # Collect x range from raw data across all runs
    all_x: list = []
    for run in runs:
        rd = _lookup_raw_combo(run["raw"], combo)
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

        rd = _lookup_raw_combo(run["raw"], combo)
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
                                # Drawing is deferred to after the loop for Ki
                                # (see _pooled_ki_display below) — the choice
                                # between "each run its own thin dashed line"
                                # (n<3) vs "one pooled mean+SEM per
                                # concentration" (n>=3) can only be made once
                                # every run's contribution is known.
                                pass
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
                                # Grubbs-test replicates at each concentration
                                # before averaging (matches pipeline_ki's own
                                # outlier rejection), so the displayed mean is
                                # the same one the pooled fit is built from.
                                run_mean, run_sem = _grubbs_mean_sem(norm_reps)
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
            if fit_row.get("_status") == "PASS":
                ki_vals_pass.append(ki)
            ki_errs.append(err if not np.isnan(err) else None)
            dot_labels.append(label)
            dot_colors.append(color)

    _pool_ki = _pool_ki_lo = _pool_ki_hi = np.nan
    _pool_n_runs = 0
    _pool_note = ""
    _has_pooled = False
    _pool_fail_ki = np.nan
    _pool_fail_n = 0
    _has_pooled_fail = False

    def _cp_shift_for(status: str) -> float:
        # cp_shift from any one contributing run's DyeConc/DyeKd at the given
        # status — these should agree across runs of the same Host/Dye pair
        # (if they don't, that's a separate data-consistency issue).
        for run in runs:
            fr = run["fit"].get(combo)
            if fr and fr.get("_status") == status:
                try:
                    dc = float(fr["DyeConc_uM"]); dk = float(fr["DyeKd_uM"])
                    if dk > 0:
                        return np.log10(1.0 + dc / dk)
                except (TypeError, ValueError, KeyError):
                    continue
        return 0.0

    def _draw_pooled_ki_disp(disp, curve_color, point_color):
        if disp["show_individual_lines"]:
            # n_runs < 3: each run's own points, unconnected — two
            # points can't honestly produce a mean+SEM worth
            # plotting as such, but a line between them isn't the
            # fitted curve either and shouldn't look like one.
            for rp in disp["per_run"]:
                yv = rp["y_norm"]
                if flip_norm: yv = 1.0 - yv
                ax_curve.plot(rp["x"], yv, ls="none", marker="o",
                             ms=4, color=rp["color"], alpha=0.8, zorder=2)
        else:
            xs    = [p["x"] for p in disp["pooled_points"]]
            means = [p["mean"] for p in disp["pooled_points"]]
            sems  = [p["sem"] for p in disp["pooled_points"]]
            if flip_norm:
                means = [1.0 - m for m in means]
            ax_curve.errorbar(xs, means, yerr=sems, fmt="o", color=point_color,
                              ecolor=point_color, elinewidth=lw_errorbar,
                              markersize=ms_data, capsize=2, alpha=0.8, zorder=2)
        xd = np.linspace(x_lo - 0.5, x_hi + 0.5, 300)
        yd = disp["curve_fn"](xd)
        if flip_norm: yd = 1.0 - yd
        ax_curve.plot(xd, yd, color=curve_color, lw=lw_fit, zorder=3)
        if ax_resid is not None:
            for rp in disp["per_run"]:
                y_pred = disp["curve_fn"](rp["x"])
                yv = rp["y_norm"]
                if flip_norm:
                    yv, y_pred = 1.0 - yv, 1.0 - y_pred
                resid = yv - y_pred
                ax_resid.scatter(rp["x"], resid, color=_resid_col, alpha=0.5,
                                 s=ms_resid**2 * 1.28, linewidths=0, zorder=2)

    # ── pooled-mode, Ki: geometric-mean pooling (_pooled_ki_simple) is the
    # reported statistic; the curve is evaluated at that already-reported
    # logKi (_pooled_ki_display), NOT refit — so the number in the title and
    # the curve on the plot can never disagree. See _pooled_ki_fit's
    # docstring for why a refit's SE is optimistic and should not be
    # reported (it treats each run's own Top/Bottom normalization as exact).
    if pooled_mode and mode == "ki":
        ps = _pooled_ki_simple(combo, runs)
        if ps is not None:
            cp_shift = _cp_shift_for("PASS")
            logki_pooled = np.log10(ps["ki"])
            disp = _pooled_ki_display(combo, runs, logki_pooled, cp_shift)
            if disp is not None:
                _draw_pooled_ki_disp(disp, color_fit, color_data)
            _pool_ki, _pool_ki_lo, _pool_ki_hi = ps["ki"], ps["ki_lo"], ps["ki_hi"]
            _pool_n_runs, _pool_note = ps["n_runs"], ps["note"]
            _has_pooled = True
        elif pool_fail_fallback:
            # Zero PASS runs contributed — before giving up, check whether
            # this combo consistently FAILed across every run that attempted
            # it (rather than never being attempted at all). A combo that
            # fails the same way every time is itself a meaningful result (a
            # plausible true negative), and otherwise never appears anywhere
            # in the app — see _render_compare_combo's docstring. Plot-only:
            # ps_fail never feeds _pool_ki/_pool_n_runs/_has_pooled, which is
            # what the title's "Ki_pooled = ..." line and any
            # summary/export read — only _pool_fail_ki/_has_pooled_fail
            # (below), rendered in a visually distinct red style and only in
            # the "no Ki data" branch of the title.
            ps_fail = _pooled_ki_simple(combo, runs, status_filter="FAIL")
            if ps_fail is not None:
                cp_shift_fail = _cp_shift_for("FAIL")
                logki_pooled_fail = np.log10(ps_fail["ki"])
                disp_fail = _pooled_ki_display(combo, runs, logki_pooled_fail,
                                               cp_shift_fail, status_filter="FAIL")
                if disp_fail is not None:
                    _draw_pooled_ki_disp(disp_fail, "firebrick", "firebrick")
                _pool_fail_ki, _pool_fail_n = ps_fail["ki"], ps_fail["n_runs"]
                _has_pooled_fail = True
            else:
                ax_curve.text(0.5, 0.5, "not enough data\nfor pooled Ki",
                              transform=ax_curve.transAxes, ha="center", va="center",
                              fontsize=6, color="gray")
        else:
            ax_curve.text(0.5, 0.5, "not enough data\nfor pooled Ki",
                          transform=ax_curve.transAxes, ha="center", va="center",
                          fontsize=6, color="gray")

    # ── pooled-mode, Kd: fit a single curve to run-level means (unchanged —
    # this brief only covers the Ki/competitive-binding pooling) ──
    elif pooled_mode and mode != "ki":
        if len(pooled_x) >= 4:
            px = np.asarray(pooled_x)
            py = np.asarray(pooled_y)
            mask = np.isfinite(px) & np.isfinite(py)
            px, py = px[mask], py[mask]
            try:
                def _pool_fn(x, top, kd):
                    return top * x / (kd + x)
                p0 = [max(py), np.median(px)]
                # Bound top >= 0 and kd > 0 (as the per-run one-site fit does)
                # so a sparse/noisy pooled fit can't run away to a negative
                # or ~0 Kd.
                popt, pcov = _curve_fit(_pool_fn, px, py, p0=p0,
                                        bounds=([0, 1e-12], [np.inf, np.inf]),
                                        maxfev=5000)
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
        else:
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
    # PASS-only throughout: the Status filter may be showing/exporting FAIL
    # runs for inspection, but the aggregate numbers reported here (mean,
    # SD, R²adj, model list) must never be diluted by an unreliable fit.
    _mkey = "Best_Model" if mode == "ki" else "Model"
    _models_seen = [_model_disp(run["fit"][combo], _mkey)
                    for run in runs
                    if combo in run["fit"] and run["fit"][combo].get("_status") == "PASS"]
    _unique_models = sorted(set(_models_seen))
    model_str = ", ".join(_unique_models) if _unique_models else "?"

    # Title with stats (matching render_plot_ki format)
    tkw = dict(fontsize=title_fontsize, fontfamily=title_fontfamily,
               fontweight=title_fontweight, fontstyle=title_fontstyle)

    if mode == "ki":
        h, d, g, hc = combo
        _hc_str = f"  ({hc:g} µM host)" if hc != -1.0 else ""
        id_line = f"{h} | {d} | {g}{_hc_str}"
        if ki_vals_pass:
            mean_v = float(np.mean(ki_vals_pass))
            sd_v = float(np.std(ki_vals_pass, ddof=1)) if len(ki_vals_pass) >= 2 else np.nan
            sd_str = f" ± {_fmt_uM(sd_v)}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs
                       if combo in run["fit"] and run["fit"][combo].get("_status") == "PASS"]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Ki_mean = {_fmt_uM(mean_v)}{sd_str} µM (n={len(ki_vals_pass)})  |  {r2_str}"
            if _has_pooled:
                # Geometric mean + 95% CI (_pooled_ki_simple), NOT a refit —
                # this is the same number reported/exported anywhere else
                # for this combo. At n_runs==2 the CI is necessarily very
                # wide (1 df) — surfaced explicitly rather than hidden.
                if np.isnan(_pool_ki_lo) or np.isnan(_pool_ki_hi):
                    ci_str = ""
                else:
                    ci_str = f"  (95% CI {_fmt_uM(_pool_ki_lo)}–{_fmt_uM(_pool_ki_hi)})"
                stats_line += (f"\nKi_pooled = {_fmt_uM(_pool_ki)} µM{ci_str}  "
                               )
                if _pool_note:
                    stats_line += f"\n⚠ {_pool_note}"
        else:
            stats_line = "no Ki data (PASS)"
            if _has_pooled_fail:
                # Plot-only figure: this number is never written to any
                # summary table or export (see pool_fail_fallback's
                # docstring) — it exists only so the red curve/points drawn
                # above have a readable value next to them.
                stats_line += (f"\n⚠ Pooled FAIL Ki ≈ {_fmt_uM(_pool_fail_ki)} µM "
                               f"(n={_pool_fail_n} failed run"
                               f"{'s' if _pool_fail_n != 1 else ''})")
        model_line = f"[{model_str}]"
        ax_curve.set_title(f"{id_line}\n{stats_line}\n{model_line}", **tkw)
    else:
        h, d, dc = combo
        id_line = f"{h} – {d}  [{dc} µM]"
        if ki_vals_pass:  # ki_vals_pass holds Kd values in kd mode
            mean_v = float(np.mean(ki_vals_pass))
            sd_v = float(np.std(ki_vals_pass, ddof=1)) if len(ki_vals_pass) >= 2 else np.nan
            sd_str = f" ± {_fmt_uM(sd_v)}" if not np.isnan(sd_v) else ""
            r2_vals = [float(run["fit"][combo].get("R2_adj", np.nan))
                       for run in runs
                       if combo in run["fit"] and run["fit"][combo].get("_status") == "PASS"]
            r2_vals = [v for v in r2_vals if not np.isnan(v)]
            r2_mean = float(np.mean(r2_vals)) if r2_vals else np.nan
            r2_str = f"R²adj = {r2_mean:.3f}" if not np.isnan(r2_mean) else ""
            stats_line = f"Kd = {_fmt_uM(mean_v)}{sd_str} µM (n={len(ki_vals_pass)})  |  {r2_str}"
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

        # Reference mean/SD band is PASS-only — a FAIL run is still plotted
        # as its own point above (so it can be visually inspected) but must
        # not shift where the mean line/band sits.
        if len(ki_vals_pass) >= 2:
            mean_ki = float(np.nanmean(ki_vals_pass))
            sd_ki   = float(np.nanstd(ki_vals_pass, ddof=1))
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
            "Accepted: pipeline output (PASS/FAIL sheets),\n"
            "Kd_Tables_Gen format (Kd_Direct / Ki_Competition sheets), or\n"
            "a Compare Runs \"..._compare_runs_summary.xlsx\" export")
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
        self._heat_ki_pivot.addItems(["Host vs Guest|Dye", "Dye vs Guest (per Host)",
                                       "Host vs Guest (per Dye)"])
        self._heat_ki_host = QComboBox()
        self._heat_ki_host_lbl = QLabel("Host:")
        self._heat_ki_host.setVisible(False)
        self._heat_ki_host_lbl.setVisible(False)
        self._heat_ki_dye = QComboBox()
        self._heat_ki_dye_lbl = QLabel("Dye:")
        self._heat_ki_dye.setVisible(False)
        self._heat_ki_dye_lbl.setVisible(False)
        self._heat_ki_pivot.currentTextChanged.connect(self._on_ki_pivot_changed)
        hf.addRow("Ki pivot:", self._heat_ki_pivot)
        hf.addRow(self._heat_ki_host_lbl, self._heat_ki_host)
        hf.addRow(self._heat_ki_dye_lbl, self._heat_ki_dye)
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
        self._annot_combo.currentTextChanged.connect(self._on_annot_mode_changed)
        self._on_annot_mode_changed(self._annot_combo.currentText())
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
        per_dye  = text == "Host vs Guest (per Dye)"
        self._heat_ki_host.setVisible(per_host)
        self._heat_ki_host_lbl.setVisible(per_host)
        self._heat_ki_dye.setVisible(per_dye)
        self._heat_ki_dye_lbl.setVisible(per_dye)

    def _on_annot_mode_changed(self, text):
        """Manual-range Min/Max are entered in whatever unit the annotation
        is currently displayed in, so switching Raw/pK/log₁₀ keeps them in sync."""
        self._traffic_vmin_spin.blockSignals(True)
        self._traffic_vmax_spin.blockSignals(True)
        if "Raw" in text:
            self._traffic_vmin_lbl.setText("Min (Ki µM):")
            self._traffic_vmax_lbl.setText("Max (Ki µM):")
            for sp in (self._traffic_vmin_spin, self._traffic_vmax_spin):
                sp.setRange(0.0, 1.0e7)
                sp.setDecimals(2)
                sp.setSingleStep(1.0)
            self._traffic_vmin_spin.setValue(1.0)
            self._traffic_vmax_spin.setValue(100.0)
        elif "pK" in text:
            self._traffic_vmin_lbl.setText("Min (pKi):")
            self._traffic_vmax_lbl.setText("Max (pKi):")
            for sp in (self._traffic_vmin_spin, self._traffic_vmax_spin):
                sp.setRange(0.0, 15.0)
                sp.setDecimals(1)
                sp.setSingleStep(0.5)
            self._traffic_vmin_spin.setValue(4.0)
            self._traffic_vmax_spin.setValue(9.0)
        else:
            self._traffic_vmin_lbl.setText("Min (log₁₀ µM):")
            self._traffic_vmax_lbl.setText("Max (log₁₀ µM):")
            for sp in (self._traffic_vmin_spin, self._traffic_vmax_spin):
                sp.setRange(-6.0, 6.0)
                sp.setDecimals(1)
                sp.setSingleStep(0.5)
            self._traffic_vmin_spin.setValue(-2.0)
            self._traffic_vmax_spin.setValue(2.0)
        self._traffic_vmin_spin.blockSignals(False)
        self._traffic_vmax_spin.blockSignals(False)

    def _traffic_manual_bounds_log10(self):
        """Convert the manual-range Min/Max spin values (entered in the unit
        the annotation mode currently displays) into internal log10(K / µM)."""
        text = self._annot_combo.currentText()
        raw_lo = self._traffic_vmin_spin.value()
        raw_hi = self._traffic_vmax_spin.value()
        if "Raw" in text:
            lo = np.log10(max(raw_lo, 1e-9))
            hi = np.log10(max(raw_hi, 1e-9))
        elif "pK" in text:
            lo = 6.0 - raw_hi
            hi = 6.0 - raw_lo
        else:
            lo, hi = raw_lo, raw_hi
        return min(lo, hi), max(lo, hi)

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
        if "Summary" in sheets:
            try:
                cols = set(xl.parse("Summary", nrows=0).columns)
            except Exception:
                cols = set()
            # Compare Runs' own "..._compare_runs_summary.xlsx" export — a
            # single "Summary" sheet, one row per combo, Pooled_Ki_uM/Mean/SD
            # columns instead of PASS/FAIL rows.
            if "Combination" in cols and ("Guest" in cols or "Dye_Conc" in cols):
                return "compare_summary"
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

    @staticmethod
    def _load_compare_summary(xl):
        """Compare Runs' "..._compare_runs_summary.xlsx" export — one row per
        combo already, with Pooled_Ki_uM/Pooled_Ki_SE (Ki mode) or just
        Mean/SD (Kd mode, which has no pooled-fit column) alongside the raw
        per-run Ki_uM/SE columns. Prefer the properly pooled fit where it
        exists; fall back to the simple cross-run Mean/SD otherwise."""
        import pandas as pd
        try:
            df = xl.parse("Summary")
        except Exception:
            return None, None
        if "Combination" not in df.columns:
            return None, None

        def _num(col):
            return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
                    else pd.Series(np.nan, index=df.index))

        def _status(value_series):
            # Newer exports carry a real per-combo Status column (PASS if any
            # contributing run passed) — use it. Older exports predate that
            # column, so fall back to inferring from the value's range, which
            # can't actually distinguish a real FAIL row from a PASS (a FAIL
            # fit can still carry a finite Ki/Kd) — best-effort only.
            if "Status" in df.columns:
                return (df["Status"].astype(str).str.upper()
                        .map(lambda s: "PASS" if s == "PASS" else "FAIL"))
            return value_series.apply(
                lambda v: "PASS" if pd.notna(v) and 0 < v < 1e6 else "FAIL")

        kd = ki = None
        if "Guest" in df.columns:
            val = _num("Pooled_Ki_uM")
            se  = _num("Pooled_Ki_SE")
            mean_v, sd_v = _num("Mean"), _num("SD")
            val = val.where(val.notna(), mean_v)
            se  = se.where(se.notna(), sd_v)
            out = pd.DataFrame({
                "Host": df.get("Host"), "Dye": df.get("Dye"), "Guest": df.get("Guest"),
                "Ki_uM": val, "Ki_err_uM": se,
            })
            out["Status"] = _status(out["Ki_uM"])
            ki = out
        elif "Dye_Conc" in df.columns:
            out = pd.DataFrame({
                "Host": df.get("Host"), "Dye": df.get("Dye"),
                "Dye_Concentration": _num("Dye_Conc"),
                "Kd": _num("Mean"), "Kd_SE": _num("SD"),
            })
            out["Status"] = _status(out["Kd"])
            kd = out
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
                elif fmt == "compare_summary":
                    kd, ki = self._load_compare_summary(xl)
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
            row.text_combo = QComboBox()
            row.text_combo.addItems(["Auto", "Black text", "White text"])
            row.layout().addWidget(row.text_combo)
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

        if ki is not None and "Dye" in ki.columns:
            prev = self._heat_ki_dye.currentText()
            self._heat_ki_dye.blockSignals(True)
            self._heat_ki_dye.clear()
            self._heat_ki_dye.addItems(sorted(ki["Dye"].dropna().unique()))
            if prev:
                idx = self._heat_ki_dye.findText(prev)
                if idx >= 0:
                    self._heat_ki_dye.setCurrentIndex(idx)
            self._heat_ki_dye.blockSignals(False)

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
        good_mask = df[val_col].notna() & (df[val_col] > 0) & (df[val_col] < 1e6)
        # A FAIL row can still carry a perfectly plausible-looking value (FAIL
        # usually means it missed an R² threshold, not that the number is
        # garbage) — without this check it would be colored as if it passed,
        # defeating "grey out the failed cells".
        if "Status" in df.columns:
            good_mask &= (df["Status"] == "PASS")
        good = df[good_mask]
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
            elif ki_pivot_mode == "Host vs Guest (per Dye)":
                dye = self._heat_ki_dye.currentText()
                ki_work = ki_df.copy()
                if dye:
                    ki_work = ki_work[ki_work["Dye"] == dye]
                ki_all_work = ki_all.copy() if ki_all is not None else None
                if ki_all_work is not None and dye:
                    ki_all_work = ki_all_work[ki_all_work["Dye"] == dye]
                row_name, col_name = "Host", "Guest"
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
                    title = f"Competition Binding — Ki ({dye})" if dye else "Competition Binding — Ki"
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
            _mv, _mx = self._traffic_manual_bounds_log10()
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
                if self._traffic_range_chk.isChecked():
                    # Manual range: always honour the requested step count.
                    t_bounds = np.linspace(vmin_, vmax_, n_steps + 1)
                else:
                    t_bounds = np.arange(np.floor(vmin_), np.ceil(vmax_) + 1, 1.0)
                    if len(t_bounds) - 1 > n_steps:
                        t_bounds = np.linspace(np.floor(vmin_), np.ceil(vmax_),
                                               n_steps + 1)
                n_colors = len(t_bounds) - 1
                if (self._traffic_colors_chk.isChecked()
                        and self._traffic_color_rows):
                    _cols = []
                    _text_overrides = []
                    for _cr in self._traffic_color_rows[:n_colors]:
                        _c = QColor(_cr.color)
                        _cols.append((_c.redF(), _c.greenF(), _c.blueF(), 1.0))
                        _text_overrides.append(
                            _cr.text_combo.currentText()
                            if hasattr(_cr, "text_combo") else "Auto")
                    while len(_cols) < n_colors:
                        _cols.append((0.5, 0.5, 0.5, 1.0))
                        _text_overrides.append("Auto")
                else:
                    _base_cmap_name = cmap[:-2] if cmap.endswith("_r") else cmap
                    _base = matplotlib.colormaps.get_cmap(_base_cmap_name)
                    _cols = [_base(i / max(n_colors - 1, 1)) for i in range(n_colors)]
                    if cmap.endswith("_r"):
                        _cols = list(reversed(_cols))
                    _text_overrides = ["Auto"] * n_colors
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
                            if traffic and t_bounds is not None:
                                band = int(np.clip(
                                    np.searchsorted(t_bounds, lv, side="right") - 1,
                                    0, len(_cols) - 1))
                                tc = _traffic_text_color(_cols[band], _text_overrides[band])
                            else:
                                tc = "white" if (lv - vmin_) / span > 0.55 else "black"
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
                                           "Dye vs Guest (per Host)",
                                           "Host vs Guest (per Dye)"])
        self._heat_host_combo = QComboBox()
        self._heat_host_combo.setVisible(False)
        self._heat_host_lbl = QLabel("Host:")
        self._heat_host_lbl.setVisible(False)
        self._heat_dye_combo = QComboBox()
        self._heat_dye_combo.setVisible(False)
        self._heat_dye_lbl = QLabel("Dye:")
        self._heat_dye_lbl.setVisible(False)
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
            hf.addRow(self._heat_dye_lbl, self._heat_dye_combo)
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
        per_dye  = text == "Host vs Guest (per Dye)"
        self._heat_host_combo.setVisible(per_host)
        self._heat_host_lbl.setVisible(per_host)
        self._heat_dye_combo.setVisible(per_dye)
        self._heat_dye_lbl.setVisible(per_dye)

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
        if self._mode == "ki" and df is not None and "Dye" in df.columns:
            prev = self._heat_dye_combo.currentText()
            self._heat_dye_combo.blockSignals(True)
            self._heat_dye_combo.clear()
            self._heat_dye_combo.addItems(sorted(df["Dye"].dropna().unique()))
            if prev:
                idx = self._heat_dye_combo.findText(prev)
                if idx >= 0:
                    self._heat_dye_combo.setCurrentIndex(idx)
            self._heat_dye_combo.blockSignals(False)
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

        # Pooled (Compare Runs) input carries an "N" column — number of runs
        # behind each pooled point. Absent for normal single-run input, so
        # this never fires there and the chart is unchanged. A "Note" column
        # (set at n_runs==2 — wide/unreliable CI) travels alongside N so that
        # gap doesn't get silently dropped from this view: "⚠" is appended to
        # the n= annotation, same as the note is surfaced in the Compare Runs
        # table/export and plot titles.
        if "N" in df.columns:
            y_vals  = df["_y"].values
            y_tops  = y_vals + np.nan_to_num(yerr)
            y_bots  = y_vals - np.nan_to_num(yerr)
            pad     = max(np.nanmax(y_tops) - np.nanmin(y_bots), 1e-9) * 0.02
            notes = df["Note"].values if "Note" in df.columns else [""] * n
            for xi, (yt, nv, note) in enumerate(zip(y_tops, df["N"].values, notes)):
                if not np.isnan(nv):
                    lbl = f"n={int(nv)}" + (" ⚠" if note else "")
                    txt = ax.text(xi, yt + pad, lbl, ha="center", va="bottom",
                                  fontsize=max(tfs - 1, 6), fontfamily=ff)
                    if note:
                        txt.set_color("#b45309")
                        txt.set_fontweight("bold")

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
            row.text_combo = QComboBox()
            row.text_combo.addItems(["Auto", "Black text", "White text"])
            row.layout().addWidget(row.text_combo)
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
            elif cm == "Host vs Guest (per Dye)":
                dye = self._heat_dye_combo.currentText()
                if dye:
                    df = df[df["Dye"] == dye].copy()
                row_col = "Host";  col_col = "Guest"
            elif cm == "Host | Dye":
                df = df.copy()
                df["_hd"] = df["Host"] + " | " + df["Dye"]
                row_col = "Guest";  col_col = "_hd"
            else:
                row_col = "Guest";  col_col = cm

        pivot = df.groupby([row_col, col_col])[pv].mean().unstack(col_col)
        # Pooled (Compare Runs) input carries an "N" column — pivot it
        # alongside so cell annotations can show n. Absent for normal
        # single-run input, so this stays None and nothing changes there.
        n_pivot = (df.groupby([row_col, col_col])["N"].mean().unstack(col_col)
                   if "N" in df.columns else None)
        # "Note" (wide/unreliable CI at n_runs==2) travels alongside N —
        # .any() per cell since a cell can aggregate >1 combo. Same pattern
        # as the bar chart's "⚠" marker, so this view doesn't silently drop
        # the warning either.
        note_pivot = (df.assign(_has_note=df["Note"].astype(bool))
                        .groupby([row_col, col_col])["_has_note"].any().unstack(col_col)
                      if "Note" in df.columns else None)
        xlabel = col_col.replace("_hd", "Host | Dye")
        ylabel = row_col

        if portrait:
            pivot = pivot.T
            if n_pivot is not None:
                n_pivot = n_pivot.T
            if note_pivot is not None:
                note_pivot = note_pivot.T
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
            if self._heat_traffic_range_chk.isChecked():
                # Manual range: always honour the requested step count.
                t_bounds = np.linspace(vmin, vmax, n_steps + 1)
            else:
                t_bounds = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1.0)
                if len(t_bounds) - 1 > n_steps:
                    t_bounds = np.linspace(np.floor(vmin), np.ceil(vmax),
                                           n_steps + 1)
            n_colors = len(t_bounds) - 1
            if (self._heat_traffic_colors_chk.isChecked()
                    and self._heat_traffic_color_rows):
                _cols = []
                _text_overrides = []
                for _cr in self._heat_traffic_color_rows[:n_colors]:
                    _c = QColor(_cr.color)
                    _cols.append((_c.redF(), _c.greenF(), _c.blueF(), 1.0))
                    _text_overrides.append(
                        _cr.text_combo.currentText()
                        if hasattr(_cr, "text_combo") else "Auto")
                while len(_cols) < n_colors:
                    _cols.append((0.5, 0.5, 0.5, 1.0))
                    _text_overrides.append("Auto")
            else:
                _base_name = cmap[:-2] if cmap.endswith("_r") else cmap
                _base = matplotlib.colormaps.get_cmap(_base_name)
                _cols = [_base(i / max(n_colors - 1, 1)) for i in range(n_colors)]
                if cmap.endswith("_r"):
                    _cols = list(reversed(_cols))
                _text_overrides = ["Auto"] * n_colors
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
            n_data = n_pivot.values.astype(float) if n_pivot is not None else None
            note_data = note_pivot.values if note_pivot is not None else None
            for i in range(nr):
                for j in range(nc):
                    v = data[i, j]
                    if not np.isnan(v):
                        if traffic and t_bounds is not None:
                            band = int(np.clip(
                                np.searchsorted(t_bounds, v, side="right") - 1,
                                0, len(_cols) - 1))
                            txt_c = _traffic_text_color(_cols[band], _text_overrides[band])
                        else:
                            txt_c = "white" if (v - vmin) / span > 0.5 else "black"
                        txt = f"{v:{fmt}}"
                        if n_data is not None and not np.isnan(n_data[i, j]):
                            txt += f"\nn={int(n_data[i, j])}"
                            if note_data is not None and bool(note_data[i, j]):
                                txt += " ⚠"
                        ax.text(j, i, txt, ha="center", va="center",
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
        if self._mode == "ki":
            _col_mode = self._heat_col_combo.currentText()
            if _col_mode == "Dye vs Guest (per Host)":
                host = self._heat_host_combo.currentText()
                if host:
                    _title = f"{_pk} — {host}"
            elif _col_mode == "Host vs Guest (per Dye)":
                dye = self._heat_dye_combo.currentText()
                if dye:
                    _title = f"{_pk} — {dye}"
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
CV_FLAG_THRESHOLD_DEFAULT = 30.0   # cross-run CV% above this ⇒ "inconsistent across runs" triage flag


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

        self._min_n_grubbs_spin = QSpinBox()
        self._min_n_grubbs_spin.setRange(3, 20)
        self._min_n_grubbs_spin.setValue(3)
        self._min_n_grubbs_spin.setToolTip(
            "Minimum replicate count at one Host concentration before a\n"
            "within-replicate Grubbs test is run at all. Standard guidance\n"
            "treats Grubbs as unreliable below n≈6-7; default 3 (the\n"
            "minimum needed to compute a sample SD) is kept for continuity\n"
            "with prior runs. Curves where any concentration was tested at\n"
            "exactly 3 replicates are flagged via Grubbs_applied_at_n3 in\n"
            "the output regardless of this setting. Raise this to skip the\n"
            "test entirely at low replicate counts instead.")
        self._min_n_grubbs_spin.valueChanged.connect(self._on_grubbs_changed)

        self._grubbs_mult_chk = QCheckBox("Multiplicity correction")
        self._grubbs_mult_chk.setChecked(False)
        self._grubbs_mult_chk.setToolTip(
            "Grubbs' test is a single-outlier test — repeated iterative\n"
            "application at a fixed nominal alpha does not correct for\n"
            "repeated testing. When checked, applies a Bonferroni\n"
            "correction (alpha / iteration count) on each successive\n"
            "removal pass, making later passes progressively more\n"
            "conservative. Off by default — preserves existing behaviour\n"
            "exactly.")
        self._grubbs_mult_chk.stateChanged.connect(self._on_grubbs_changed)

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

        self._host_autofl_z_spin = QDoubleSpinBox()
        self._host_autofl_z_spin.setRange(0.5, 5.0)
        self._host_autofl_z_spin.setSingleStep(0.05)
        self._host_autofl_z_spin.setValue(pl_fda.HOST_AUTOFL_Z_DEFAULT)
        self._host_autofl_z_spin.setDecimals(2)
        self._host_autofl_z_spin.valueChanged.connect(self._on_r2_changed)
        self._host_autofl_z_spin.setToolTip(
            "FAIL a curve when host-only mean minus dye-blank mean exceeds\n"
            "z × the pooled SE (sqrt(host_SEM² + blank_SEM²)) — a textbook\n"
            "two-sample z-test, same statistical gate used for guest\n"
            "autofluorescence in the Ki pipeline, on the SAME plate/dye\n"
            "(and buffer, if used) as that curve.\n\n"
            "Default z=1.96 (two-tailed 95% CI). Raise it (e.g. 2.5–3) to\n"
            "be more permissive if curves are being failed for host\n"
            "autofluorescence that look fine otherwise.")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Kd range lo:",  self._kd_lo_spin)
        pf.addRow("Kd range hi:",  self._kd_hi_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Grubbs min n:", self._min_n_grubbs_spin)
        pf.addRow("",              self._grubbs_mult_chk)
        pf.addRow(self._cross_conc_chk, self._cross_conc_alpha_spin)
        pf.addRow("Model:",        self._model_combo)
        pf.addRow("Host autofl. gate (z):", self._host_autofl_z_spin)
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
            "min_n_for_grubbs":      self._min_n_grubbs_spin.value(),
            "grubbs_multiplicity_correction": self._grubbs_mult_chk.isChecked(),
            "use_cross_conc_grubbs": self._cross_conc_chk.isChecked(),
            "cross_conc_alpha":      self._cross_conc_alpha_spin.value(),
            "model_preference":      model_map[self._model_combo.currentText()],
            "host_autofl_z":         self._host_autofl_z_spin.value(),
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
        self._min_n_grubbs_spin.setValue(3)
        self._grubbs_mult_chk.setChecked(False)
        self._cross_conc_chk.setChecked(False)   # default OFF per pipeline v4
        self._cross_conc_alpha_spin.setValue(0.01)
        self._model_combo.setCurrentIndex(0)
        self._host_autofl_z_spin.setValue(pl_fda.HOST_AUTOFL_Z_DEFAULT)
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
                                 "min_n_for_grubbs":       cfg["min_n_for_grubbs"],
                                 "grubbs_multiplicity_correction": cfg["grubbs_multiplicity_correction"],
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
            kd_range_hi=cfg["kd_range_hi"],
            fi_df=self._state.fi_df,
            host_autofl_z=cfg["host_autofl_z"])
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

        self._min_n_grubbs_spin = QSpinBox()
        self._min_n_grubbs_spin.setRange(3, 20)
        self._min_n_grubbs_spin.setValue(3)
        self._min_n_grubbs_spin.setToolTip(
            "Minimum replicate count at one Guest concentration before a\n"
            "within-replicate Grubbs test is run at all. Standard guidance\n"
            "treats Grubbs as unreliable below n≈6-7; default 3 (the\n"
            "minimum needed to compute a sample SD) is kept for continuity\n"
            "with prior runs. Curves where any concentration was tested at\n"
            "exactly 3 replicates are flagged via Grubbs_applied_at_n3 in\n"
            "the output regardless of this setting. Raise this to skip the\n"
            "test entirely at low replicate counts instead.")
        self._min_n_grubbs_spin.valueChanged.connect(self._on_grubbs_changed)

        self._grubbs_mult_chk = QCheckBox("Multiplicity correction")
        self._grubbs_mult_chk.setChecked(False)
        self._grubbs_mult_chk.setToolTip(
            "Grubbs' test is a single-outlier test — repeated iterative\n"
            "application at a fixed nominal alpha does not correct for\n"
            "repeated testing. When checked, applies a Bonferroni\n"
            "correction (alpha / iteration count) on each successive\n"
            "removal pass, making later passes progressively more\n"
            "conservative. Off by default — preserves existing behaviour\n"
            "exactly.")
        self._grubbs_mult_chk.stateChanged.connect(self._on_grubbs_changed)

        self._ki_lo_spin = QDoubleSpinBox()
        self._ki_lo_spin.setRange(0.0, 1.0)
        self._ki_lo_spin.setSingleStep(0.05)
        self._ki_lo_spin.setValue(pl_ki.KI_RANGE_FACTOR_LO_DEFAULT)
        self._ki_lo_spin.setDecimals(2)
        self._ki_lo_spin.setToolTip(
            "FAIL if Ki < factor × min([Guest])\n(Ki below measured range — extrapolating left)")
        self._ki_lo_spin.valueChanged.connect(self._on_r2_changed)

        self._ki_hi_spin = QDoubleSpinBox()
        self._ki_hi_spin.setRange(1.0, 1000.0)
        self._ki_hi_spin.setSingleStep(1.0)
        self._ki_hi_spin.setValue(pl_ki.KI_RANGE_FACTOR_HI_DEFAULT)
        self._ki_hi_spin.setDecimals(1)
        self._ki_hi_spin.setToolTip(
            "FAIL if Ki > factor × max([Guest])\n(never reaches saturation — extrapolating right)")
        self._ki_hi_spin.valueChanged.connect(self._on_r2_changed)

        self._wang_spin = QDoubleSpinBox()
        self._wang_spin.setRange(0.01, 100.0)
        self._wang_spin.setSingleStep(0.1)
        self._wang_spin.setValue(pl_ki.WANG_GATE_DEFAULT)
        self._wang_spin.setDecimals(2)
        self._wang_spin.setToolTip(
            "Wang (exact, depletion-aware) model is tried when:\n"
            "  Ki_standard < gate × [Host]\n\n"
            "Wang accounts for depletion of free dye, host, and guest\n"
            "when a significant fraction is bound (Ki ≈ [Host]) — see\n"
            "Wang, Z.-X. FEBS Lett. 1995;360(2):111-114.\n\n"
            "Default 1.0 = only when Ki ≤ [Host] (true tight binding).\n"
            "Raise to e.g. 2–5 to be more permissive;\n"
            "lower to 0.1 to restrict to very tight binders only.")

        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["Auto (AICc)", "Standard", "Hill Slope", "Wang"])
        self._model_combo.setToolTip(
            "Auto (AICc): tries every applicable model (Standard, Hill Slope,\n"
            "Wang when Ki is within the Wang gate above) and keeps the best\n"
            "by AICc.\n\n"
            "Forcing a model (Standard/Hill Slope/Wang) fits only that\n"
            "model — Wang bypasses the gate above (still needs a Host\n"
            "concentration).")

        self._autofl_z_spin = QDoubleSpinBox()
        self._autofl_z_spin.setRange(0.5, 5.0)
        self._autofl_z_spin.setSingleStep(0.05)
        self._autofl_z_spin.setValue(pl_ki.GUEST_AUTOFL_Z_DEFAULT)
        self._autofl_z_spin.setDecimals(2)
        self._autofl_z_spin.valueChanged.connect(self._on_r2_changed)
        self._autofl_z_spin.setToolTip(
            "FAIL a curve when guest-only mean minus dye-blank mean exceeds\n"
            "z × the pooled SE (sqrt(guest_SEM² + blank_SEM²)) — a\n"
            "textbook two-sample z-test, on the SAME plate (and buffer,\n"
            "if used) as that curve.\n\n"
            "Default z=1.96 (two-tailed 95% CI). This is the ACTIVE gate\n"
            "(autofl_mode='stat'). Raise it (e.g. 2.5–3) to be more\n"
            "permissive if curves are being failed for autofluorescence\n"
            "that look fine otherwise (good R², well-determined Ki).")

        self._autofl_spin = QDoubleSpinBox()
        self._autofl_spin.setRange(0.1, 50.0)
        self._autofl_spin.setSingleStep(0.1)
        self._autofl_spin.setValue(pl_ki.GUEST_AUTOFL_FACTOR)
        self._autofl_spin.setDecimals(2)
        self._autofl_spin.valueChanged.connect(self._on_r2_changed)
        self._autofl_spin.setToolTip(
            "Fallback ratio-mode threshold (autofl_mode='ratio', not used\n"
            "unless that mode is selected in code): FAIL a curve if its\n"
            "guest-only wells' median fluorescence exceeds this factor ×\n"
            "the dye-blank mean, on the SAME plate (and buffer, if used).\n\n"
            "The statistical gate above (z-score) is the active default —\n"
            "this fixed-ratio test is kept only for continuity with prior\n"
            "runs that used it.")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Grubbs min n:", self._min_n_grubbs_spin)
        pf.addRow("",              self._grubbs_mult_chk)
        pf.addRow("Ki range lo:",  self._ki_lo_spin)
        pf.addRow("Ki range hi:",  self._ki_hi_spin)
        pf.addRow("Wang gate:",    self._wang_spin)
        pf.addRow("Model:",        self._model_combo)
        pf.addRow("Guest autofl. gate (z):", self._autofl_z_spin)
        pf.addRow("Guest autofl. gate (ratio, fallback):", self._autofl_spin)
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
            "min_n_for_grubbs":   self._min_n_grubbs_spin.value(),
            "grubbs_multiplicity_correction": self._grubbs_mult_chk.isChecked(),
            "ki_range_lo":        self._ki_lo_spin.value(),
            "ki_range_hi":        self._ki_hi_spin.value(),
            "wang_gate":          self._wang_spin.value(),
            "autofl_factor":      self._autofl_spin.value(),
            "autofl_z":           self._autofl_z_spin.value(),
            "model_preference":   {"Auto (AICc)": "auto", "Standard": "standard",
                                    "Hill Slope": "hillslope",
                                    "Wang": "wang"}[self._model_combo.currentText()],
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
        self._min_n_grubbs_spin.setValue(3)
        self._grubbs_mult_chk.setChecked(False)
        self._ki_lo_spin.setValue(pl_ki.KI_RANGE_FACTOR_LO_DEFAULT)
        self._ki_hi_spin.setValue(pl_ki.KI_RANGE_FACTOR_HI_DEFAULT)
        self._wang_spin.setValue(pl_ki.WANG_GATE_DEFAULT)
        self._autofl_spin.setValue(pl_ki.GUEST_AUTOFL_FACTOR)
        self._autofl_z_spin.setValue(pl_ki.GUEST_AUTOFL_Z_DEFAULT)
        self._model_combo.setCurrentIndex(0)
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
                               min_n_for_grubbs=cfg["min_n_for_grubbs"],
                               grubbs_multiplicity_correction=cfg["grubbs_multiplicity_correction"],
                               wang_gate=cfg["wang_gate"],
                               model_preference=cfg["model_preference"]))

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

    def _refresh_results(self):
        cfg = self._get_config()
        df  = pl_ki.apply_thresholds_ki(
            self._state.fit_results, self._state.plot_data, cfg["r2_threshold"],
            fi_df=self._state.fi_df, autofl_factor=cfg["autofl_factor"],
            autofl_z=cfg["autofl_z"],
            ki_range_lo=cfg["ki_range_lo"], ki_range_hi=cfg["ki_range_hi"])
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

        self._host_autofl_z_spin = QDoubleSpinBox()
        self._host_autofl_z_spin.setRange(0.5, 5.0)
        self._host_autofl_z_spin.setSingleStep(0.05)
        self._host_autofl_z_spin.setValue(pl_fda.HOST_AUTOFL_Z_DEFAULT)
        self._host_autofl_z_spin.setDecimals(2)
        self._host_autofl_z_spin.valueChanged.connect(self._on_r2_changed)
        self._host_autofl_z_spin.setToolTip(
            "FAIL a curve when host-only mean minus dye-blank mean exceeds\n"
            "z × the pooled SE (sqrt(host_SEM² + blank_SEM²)) — a textbook\n"
            "two-sample z-test, same statistical gate used for guest\n"
            "autofluorescence in the Ki pipeline. Default z=1.96 (two-\n"
            "tailed 95% CI).")

        pf.addRow("R² threshold:", self._r2_spin)
        pf.addRow("Kd range lo:",  self._kd_lo_spin)
        pf.addRow("Kd range hi:",  self._kd_hi_spin)
        pf.addRow("Grubbs α:",     self._grubbs_spin)
        pf.addRow("Model:",        self._model_combo)
        pf.addRow("Host autofl. gate (z):", self._host_autofl_z_spin)
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
            "host_autofl_z":  self._host_autofl_z_spin.value(),
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
        self._state.fi_df_fit      = result.get("fi_df_used")

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
        # fi_df_fit is the wavelength-restricted subset (only the wavelength
        # actually fit, per dye if per-dye wavelengths were used) — passing
        # the full multi-wavelength self._state.fi_df here would dilute the
        # host-autofluorescence gate across every scanned wavelength instead
        # of just the one the curve was fit at. Fall back to the full table
        # only if something upstream didn't populate it (defensive).
        df  = pl_fda.apply_thresholds(
            self._state.fit_results, self._state.plot_data,
            r2_threshold=cfg["r2_threshold"],
            kd_range_lo=cfg["kd_range_lo"],
            kd_range_hi=cfg["kd_range_hi"],
            fi_df=(self._state.fi_df_fit if self._state.fi_df_fit is not None
                   else self._state.fi_df),
            host_autofl_z=cfg["host_autofl_z"])
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
                 pooled_mode: bool = False, pool_fail_fallback: bool = True,
                 show_legend: bool = False,
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
                     pooled_mode=pooled_mode, pool_fail_fallback=pool_fail_fallback,
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
                   flip_norm=False, pooled_mode=False, pool_fail_fallback=True,
                   show_legend=True,
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
            pool_fail_fallback=pool_fail_fallback,
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
                _hc = combo[3] if len(combo) > 3 else -1.0
                _hc_str = f"  ({_hc:g} µM host)" if _hc != -1.0 else ""
                lbl = f"[{i+1}] {combo[0]} | {combo[1]} | {combo[2]}{_hc_str}"
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


class GuestToggleDialog(QDialog):
    """Tick-list to include/exclude individual guests (or dye concs, in Kd
    mode) from Compare Runs — separate from, and on top of, the existing
    single-select Host/Dye/Guest dropdown filters. Changes only take effect
    on OK; Cancel discards them."""

    def __init__(self, label: str, names: list, excluded: set, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Include/Exclude {label}")
        self.resize(300, 420)
        lay = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        all_btn  = QPushButton("Select All")
        none_btn = QPushButton("Deselect All")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._checks: dict = {}
        scroll_w = QWidget()
        scroll_lay = QVBoxLayout(scroll_w)
        scroll_lay.setAlignment(Qt.AlignTop)
        scroll_lay.setSpacing(2)
        for name in names:
            cb = QCheckBox(name)
            cb.setChecked(name not in excluded)
            self._checks[name] = cb
            scroll_lay.addWidget(cb)
        sa = QScrollArea()
        sa.setWidget(scroll_w)
        sa.setWidgetResizable(True)
        lay.addWidget(sa)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _set_all(self, checked: bool):
        for cb in self._checks.values():
            cb.setChecked(checked)

    def excluded_names(self) -> set:
        return {name for name, cb in self._checks.items() if not cb.isChecked()}


class CompareRunsWindow(QMainWindow):
    """Compare fitted Ki/Kd curves and values across independent runs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosphoMAX — Compare Runs")
        self.resize(1340, 860)
        self._run_items: list = []
        self._runs:      list = []   # {"item", "fit", "raw", "path"}
        self._mode:      str  = "ki"
        self._excluded_names: set = set()   # guests (or dye concs, Kd mode) ticked off
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
        self._status_flt = QComboBox()
        self._status_flt.addItems(["PASS only", "FAIL only", "All"])
        self._status_flt.setToolTip(
            "PASS only (default) — only PASS fit entries are shown.\n"
            "FAIL only — only FAIL fit entries are shown.\n"
            "All — both PASS and FAIL entries are shown.")
        self._cv_flag_spin = QDoubleSpinBox()
        self._cv_flag_spin.setRange(1.0, 200.0)
        self._cv_flag_spin.setSingleStep(1.0)
        self._cv_flag_spin.setValue(CV_FLAG_THRESHOLD_DEFAULT)
        self._cv_flag_spin.setDecimals(0)
        self._cv_flag_spin.setSuffix(" %")
        self._cv_flag_spin.setToolTip(
            "Cross-run CV% above this is flagged 'Inconsistent across runs\n"
            "— recommend orthogonal retest' in the Summary tab and XLSX\n"
            "exports, independent of any single run's PASS/FAIL Status.\n"
            "Only applies once a combo has ≥2 PASS observations across runs.\n\n"
            "Default 30% — for a triage screen, this is the headline\n"
            "reproducibility signal: a hit that's inconsistent run-to-run\n"
            "is a bad hit regardless of how correct any single run's model was.")
        self._guest_alias_edit = QLineEdit()
        self._guest_alias_edit.setPlaceholderText("e.g. spe=spm, admn=adm")
        self._guest_alias_edit.setToolTip(
            "Correct a guest mislabelled in one or more runs' own fit_results\n"
            "files — WITHOUT editing or re-analysing anything. Comma-separated\n"
            "old=new pairs, e.g. 'spe=spm' renames every 'spe' entry to 'spm'\n"
            "before combos are built, so runs that spelled the same guest\n"
            "differently pool together as one combo instead of showing up as\n"
            "two separate, never-pooled entries.\n\n"
            "Applies here only (Compare Runs) — the underlying fit_results\n"
            "files and any per-run exports are never touched.")
        self._guest_toggle_btn = QPushButton("Include/Exclude…")
        self._guest_toggle_btn.setToolTip(
            "Tick individual guests (or dye concs, in Kd mode) on/off — on\n"
            "top of, and independent from, the single-select dropdown above.\n"
            "Useful for dropping a handful of guests from the comparison\n"
            "without touching the 'Shared combos only'/Status filters.")
        self._guest_toggle_btn.clicked.connect(self._on_guest_toggle)
        flt_lay.addRow("Host:",        self._host_flt)
        flt_lay.addRow("Dye:",         self._dye_flt)
        flt_lay.addRow(self._guest_lbl, self._guest_flt)
        flt_lay.addRow("",             self._guest_toggle_btn)
        flt_lay.addRow(self._shared_chk)
        flt_lay.addRow("Status:",      self._status_flt)
        flt_lay.addRow("CV% flag threshold:", self._cv_flag_spin)
        flt_lay.addRow("Guest aliases:", self._guest_alias_edit)
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
        self._flip_chk.setChecked(False)
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

        self._pool_fail_chk = QCheckBox("Pool FAIL-only combos too")
        self._pool_fail_chk.setChecked(True)
        self._pool_fail_chk.setToolTip(
            "Pooled fit mode only. When a combo has zero PASS runs, pool\n"
            "across its FAIL runs instead of showing 'not enough data' —\n"
            "drawn in red and clearly marked as not a validated result.\n"
            "A combo that fails consistently across every run is itself\n"
            "informative (a plausible true negative). Never affects the\n"
            "reported/exported pooled Ki — plot only. Untick to restore\n"
            "the plain 'not enough data' placeholder.")

        p_lay.addWidget(self._normalise_chk)
        p_lay.addWidget(self._sci_notation_chk)
        p_lay.addWidget(self._dots_chk)
        p_lay.addWidget(self._resid_chk)
        p_lay.addWidget(self._flip_chk)
        p_lay.addWidget(self._legend_chk)
        p_lay.addWidget(self._ind_export_chk)
        p_lay.addLayout(avg_row)
        p_lay.addWidget(self._pool_fail_chk)
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
                    self._status_flt.currentIndexChanged,
                    self._cv_flag_spin.valueChanged,
                    self._dots_chk.stateChanged,
                    self._resid_chk.stateChanged,
                    self._flip_chk.stateChanged,
                    self._legend_chk.stateChanged,
                    self._pooled_rb.toggled,
                    self._pool_fail_chk.stateChanged,
                    self._guest_alias_edit.editingFinished):
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

        # Pooled-Ki summary plot — same SummaryTab widget/rendering used by
        # the Competitive Fitting tab, fed a pooled-across-runs dataframe.
        # Only meaningful in "ki" mode (pooling is Ki-specific); disabled
        # otherwise.
        self._summary_plot_tab = SummaryTab(mode="ki")
        self._summary_plot_idx = self._tabs.addTab(
            self._summary_plot_tab, "Pooled Ki Summary Plot")

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
            pool_fail_fallback=self._pool_fail_chk.isChecked(),
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

    def _status_keep(self, status) -> bool:
        sel = self._status_flt.currentText()
        if sel == "All":
            return True
        if sel == "FAIL only":
            return status == "FAIL"
        return status == "PASS"   # "PASS only" (default)

    def _guest_alias_map(self) -> dict:
        """Parse the 'Guest aliases' box into {old_title_cased: new_title_cased}.

        Format: comma-separated "old=new" pairs, e.g. "spe=spm, admn=adm".
        Lets a mislabelled guest spelling in one or more runs (a typo made at
        acquisition time, already baked into that run's own fit_results.xlsx)
        be corrected for pooling/comparison purposes only — without editing
        or re-analysing any file. Applied to every run's combo keys before
        anything else (filters, combo grid, pooling) sees them, so "spe" and
        "spm" entries from different runs merge into one combo instead of
        showing up as two never-pooled entries.
        """
        text = self._guest_alias_edit.text().strip()
        if not text:
            return {}
        out = {}
        for part in text.split(","):
            if "=" not in part:
                continue
            old, new = part.split("=", 1)
            old, new = old.strip(), new.strip()
            if old and new:
                out[old.title()] = new.title()
        return out

    def _apply_guest_alias(self, key: tuple, alias: dict) -> tuple:
        # Ki combo keys are (Host, Dye, Guest, HostConc); Kd combo keys have
        # no Guest field at all (Host, Dye, Dye_Conc) — alias is a no-op there.
        if not alias or self._mode != "ki":
            return key
        h, d, g, hc = key
        return (h, d, alias.get(g, g), hc)

    def _all_keys_with_counts(self) -> dict:
        alias = self._guest_alias_map()
        result: dict = {}
        for run in self._runs:
            for key, row in run["fit"].items():
                if not self._status_keep(row.get("_status")):
                    continue
                key = self._apply_guest_alias(key, alias)
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
        n_hidden = len(self._excluded_names & set(guests))
        self._guest_toggle_btn.setText(
            f"Include/Exclude… ({n_hidden} hidden)" if n_hidden else "Include/Exclude…")

    def _on_guest_toggle(self):
        kc    = getattr(self, "_cached_kc", None) or self._all_keys_with_counts()
        names = sorted({k[2] for k in kc.keys()})
        if not names:
            QMessageBox.information(self, "Nothing to toggle",
                                    "Add fit_results files first.")
            return
        label = self._guest_lbl.text().rstrip(":") or "Guest"
        dlg = GuestToggleDialog(label, names, self._excluded_names, self)
        if dlg.exec() == QDialog.Accepted:
            self._excluded_names = dlg.excluded_names()
            self._schedule_rebuild()

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
            and k[2] not in self._excluded_names
        ])

    def _active_runs(self) -> list:
        # Same alias applied to both fit and raw keys — _lookup_raw_combo
        # matches raw per-well data against a combo tuple, so the raw dict
        # must be remapped identically or a run's own scatter points/curve
        # reconstruction would stop resolving under the corrected name.
        alias = self._guest_alias_map()
        result = []
        for ri, run in zip(self._run_items, self._runs):
            fit = {self._apply_guest_alias(k, alias): v
                   for k, v in run["fit"].items()
                   if self._status_keep(v.get("_status"))}
            raw = {self._apply_guest_alias(k, alias): v
                   for k, v in run["raw"].items()}
            result.append({"label": ri.label, "color": ri.color,
                           "fit": fit, "raw": raw})
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
            self._tabs.setTabEnabled(self._summary_plot_idx, False)
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
            self._tabs.setTabEnabled(self._summary_plot_idx, False)
            self._status_lbl.setText(msg)
            return

        runs = self._active_runs()
        self._plots_tab.load(combos, runs, self._mode, **self._plot_kw())
        self._rebuild_summary(combos, runs)
        is_ki = self._mode == "ki"
        self._tabs.setTabEnabled(self._summary_plot_idx, is_ki)
        if is_ki:
            self._summary_plot_tab.load(_build_pooled_ki_summary_df(combos, runs))
        mode_str = "Ki" if self._mode == "ki" else "Kd"
        self._status_lbl.setText(
            f"{len(combos)} combo(s)  |  {len(runs)} run(s)  |  mode: {mode_str}")

    # ── summary table ─────────────────────────────────────────────────────────

    def _rebuild_summary(self, combos: list, runs: list):
        from PySide6.QtGui import QColor
        # Triage headline metric: cross-run CV% is the first thing to look
        # at — a hit that's inconsistent run-to-run is a bad hit regardless
        # of how correct any single run's model was. Placed first (with n)
        # so it's not buried among per-run columns, and flagged when it
        # exceeds the user-adjustable CV% flag threshold (Filters panel).
        cv_flag_threshold = self._cv_flag_spin.value()
        _cv_flag_color = QColor("#fff3cd")   # bootstrap-style "warning" yellow

        run_labels = [r["label"] for r in runs]
        is_ki      = self._mode == "ki"
        val_lbl    = "Ki (µM)" if is_ki else "Kd (µM)"
        _mkey      = "Best_Model" if is_ki else "Model"
        run_hdrs   = []
        for lbl in run_labels:
            run_hdrs += [f"{lbl}\n{val_lbl}", f"{lbl}\nfit SE", f"{lbl}\nR²adj"]
        headers = ["Combination", "CV%_linear_reference_only", "Log_CV%", "n", "Flags"] + run_hdrs + ["Mean", "SD", "Models"]
        _pooled_lbl = "Ki" if is_ki else "Kd"
        headers += [f"Pooled {_pooled_lbl}\n(µM)", "Pooled\n95% CI", "Pooled\nn runs", "Pooled\nnote"]
        self._summary_tbl.setRowCount(len(combos))
        self._summary_tbl.setColumnCount(len(headers))
        self._summary_tbl.setHorizontalHeaderLabels(headers)
        self._summary_tbl.setSortingEnabled(False)

        for ri, combo in enumerate(combos):
            if is_ki:
                h, d, g, hc = combo
                _hc_str = f"  ({hc:g} µM host)" if hc != -1.0 else ""
                combo_str = f"{h} | {d} | {g}{_hc_str}"
            else:
                h, d, dc = combo
                combo_str = f"{h} | {d} [{dc} µM]"
            self._summary_tbl.setItem(ri, 0, QTableWidgetItem(combo_str))

            vals: list = []          # PASS-only — feeds Mean/SD/CV%/n
            models_seen: list = []   # PASS-only — feeds the Models column
            flag_notes: list = []    # PASS-only — feeds the Flags column
            for ci, run in enumerate(runs):
                fr   = run["fit"].get(combo)
                base = 5 + ci * 3
                if fr is None:
                    for j in range(3):
                        self._summary_tbl.setItem(ri, base + j, QTableWidgetItem("—"))
                    continue
                try:
                    v  = float(fr.get("Ki_uM",    np.nan) if is_ki
                               else fr.get("Kd",    np.nan))
                    se = float(fr.get("Ki_err_uM", np.nan) if is_ki
                               else fr.get("Kd_SE", np.nan))
                    r2 = float(fr.get("R2_adj", np.nan))
                except Exception:
                    v = se = r2 = np.nan
                def _g(x, dp=3): return f"{x:.{dp}g}" if not np.isnan(x) else "—"
                # This run's own cell always shows its own value regardless of
                # status, so a FAIL run picked up by the Status filter is
                # still visible here for inspection.
                self._summary_tbl.setItem(ri, base,     QTableWidgetItem(_fmt_uM(v) if not np.isnan(v) else "—"))
                self._summary_tbl.setItem(ri, base + 1, QTableWidgetItem(_fmt_uM(se) if not np.isnan(se) else "—"))
                self._summary_tbl.setItem(ri, base + 2, QTableWidgetItem(_g(r2, 3)))
                if fr.get("_status") == "PASS":
                    models_seen.append(_model_disp(fr, _mkey))
                    if not np.isnan(v):
                        vals.append(v)
                    # Any-flags aggregate: wide-CI confidence, sign
                    # ambiguity, non-1 Hill slope, or an approximate
                    # (Cheng-Prusoff-shift) model basis on ANY contributing
                    # PASS run — a reviewer working only from this table
                    # should still see these without opening each run.
                    if fr.get("Confidence") and fr.get("Confidence") != "High":
                        flag_notes.append("Low confidence")
                    if fr.get("Sign_ambiguous"):
                        flag_notes.append("Sign ambiguous")
                    if fr.get(_mkey) == "HillSlope" and fr.get("Hill_flag"):
                        flag_notes.append("Hill slope flag")
                    if fr.get("Model_Basis") == "approximate_cp_shift":
                        flag_notes.append("Approx. model")

            unique_models   = sorted(set(models_seen))
            model_str       = ", ".join(unique_models) if unique_models else "—"
            unique_flags    = sorted(set(flag_notes))
            flags_str       = ", ".join(unique_flags) if unique_flags else "—"

            cv = np.nan
            if vals:
                mean = float(np.mean(vals))
                sd   = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
                cv   = sd / mean * 100 if (not np.isnan(sd) and mean) else np.nan

            # Reported pooled statistic: geometric mean +/- CI across
            # independent runs (_pooled_ki_simple / _pooled_kd_simple), NOT
            # a curve refit — see _pooled_ki_fit's docstring for why a
            # refit's SE is optimistic. Also drives the "inconsistent
            # across runs" flag: log_sem (log-scale dispersion) is used for
            # BOTH assay types instead of the linear CV%, since Kd and Ki
            # are multiplicative quantities. The highlighted/flagged column
            # (Log_CV%) is always the number that generates this flag — the
            # linear CV%_linear_reference_only column is display-only and
            # never highlighted, so the two columns can never disagree
            # about what a reader sees flagged.
            ps_stat = _pooled_ki_simple(combo, runs) if is_ki else _pooled_kd_simple(combo, runs)
            log_cv_pct = np.nan
            if ps_stat and ps_stat["n_runs"] >= 2 and not np.isnan(ps_stat["log_sem"]):
                log_cv_pct = (10.0 ** ps_stat["log_sem"] - 1.0) * 100.0

            # CV%/n go first (headline triage signal) — only meaningful once
            # >=2 runs contribute a PASS value for this combo.
            cv_item     = QTableWidgetItem(f"{cv:.1f}%" if not np.isnan(cv) else "—")
            log_cv_item = QTableWidgetItem(f"{log_cv_pct:.1f}%" if not np.isnan(log_cv_pct) else "—")
            n_item      = QTableWidgetItem(str(len(vals)))
            if not np.isnan(log_cv_pct) and log_cv_pct > cv_flag_threshold:
                log_cv_item.setBackground(_cv_flag_color)
                log_cv_item.setToolTip(
                    f"Log-scale dispersion={log_cv_pct:.1f}% > {cv_flag_threshold:.0f}% "
                    f"— inconsistent across runs; recommend orthogonal retest before "
                    f"follow-up, regardless of individual run Status. (CV%_linear_"
                    f"reference_only is display-only and no longer drives this flag — "
                    f"Kd/Ki are multiplicative quantities, so log-scale dispersion is the "
                    f"correct measure.)")
            self._summary_tbl.setItem(ri, 1, cv_item)
            self._summary_tbl.setItem(ri, 2, log_cv_item)
            self._summary_tbl.setItem(ri, 3, n_item)
            flags_item = QTableWidgetItem(flags_str)
            if unique_flags:
                flags_item.setBackground(_cv_flag_color)
                flags_item.setToolTip(", ".join(unique_flags) + " on at least one PASS run for this combo.")
            self._summary_tbl.setItem(ri, 4, flags_item)

            bs = 5 + len(runs) * 3
            if vals:
                self._summary_tbl.setItem(ri, bs,     QTableWidgetItem(f"{mean:.3g}"))
                self._summary_tbl.setItem(ri, bs + 1, QTableWidgetItem(f"{sd:.3g}" if not np.isnan(sd) else "—"))
            else:
                self._summary_tbl.setItem(ri, bs,     QTableWidgetItem("—"))
                self._summary_tbl.setItem(ri, bs + 1, QTableWidgetItem("—"))

            mdl_item = QTableWidgetItem(model_str)
            self._summary_tbl.setItem(ri, bs + 2, mdl_item)

            ps = bs + 3
            _val_key, _lo_key, _hi_key = ("ki", "ki_lo", "ki_hi") if is_ki else ("kd", "kd_lo", "kd_hi")
            if ps_stat:
                lo, hi = ps_stat[_lo_key], ps_stat[_hi_key]
                ci_str = "—" if np.isnan(lo) or np.isnan(hi) else f"{_fmt_uM(lo)}–{_fmt_uM(hi)}"
                self._summary_tbl.setItem(ri, ps,     QTableWidgetItem(_fmt_uM(ps_stat[_val_key])))
                self._summary_tbl.setItem(ri, ps + 1, QTableWidgetItem(ci_str))
                self._summary_tbl.setItem(ri, ps + 2, QTableWidgetItem(str(ps_stat["n_runs"])))
                note_item = QTableWidgetItem(ps_stat["note"] or "")
                if ps_stat["note"]:
                    note_item.setBackground(_cv_flag_color)
                self._summary_tbl.setItem(ri, ps + 3, note_item)
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
                    pool_fail_fallback=kw.get("pool_fail_fallback", True),
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
                    h, d, g, hc = combo
                    stem = f"{h}_{d}_{g}" + (f"_{hc:g}uM" if hc != -1.0 else "")
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

        runs = self._active_runs()
        cv_flag_threshold = self._cv_flag_spin.value()
        rows = [_build_compare_summary_row(combo, runs, self._mode, cv_flag_threshold)
                for combo in combos]

        try:
            df = _pd.DataFrame(rows)
            with _pd.ExcelWriter(path, engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="Summary", index=False)
                ws = w.sheets["Summary"]
                for ci, col in enumerate(df.columns):
                    max_len = max(len(str(col)),
                                  df[col].astype(str).str.len().max())
                    ws.set_column(ci, ci, min(max_len + 2, 30))
                if "Log_CV%" in df.columns and len(df) > 0:
                    # Conditional formatting must highlight the same number
                    # that drives the "Consistency" flag text (Log_CV%), not
                    # the linear CV%_linear_reference_only column — otherwise
                    # a reader can see a highlighted cell that disagrees with
                    # the Consistency text next to it.
                    _wb = w.book
                    _warn_fmt = _wb.add_format({"bg_color": "#fff3cd"})
                    _cv_col = list(df.columns).index("Log_CV%")
                    ws.conditional_format(
                        1, _cv_col, len(df), _cv_col,
                        {"type": "cell", "criteria": ">",
                         "value": cv_flag_threshold, "format": _warn_fmt})
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
        cv_flag_threshold = self._cv_flag_spin.value()
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
                    pool_fail_fallback=kw.get("pool_fail_fallback", True),
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
                    h, d, g, hc = combo
                    stem = f"{h}_{d}_{g}" + (f"_{hc:g}uM" if hc != -1.0 else "")
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
            rows_data = [_build_compare_summary_row(combo, runs, self._mode, cv_flag_threshold)
                         for combo in combos]

            df = _pd.DataFrame(rows_data)
            with _pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="Summary", index=False)
                ws = w.sheets["Summary"]
                for ci, col in enumerate(df.columns):
                    max_len = max(len(str(col)),
                                  df[col].astype(str).str.len().max())
                    ws.set_column(ci, ci, min(max_len + 2, 30))
                if "Log_CV%" in df.columns and len(df) > 0:
                    # Highlight the column that actually drives "Consistency"
                    # (Log_CV%), not CV%_linear_reference_only — see the
                    # twin block in _on_export_xlsx above for the rationale.
                    _wb = w.book
                    _warn_fmt = _wb.add_format({"bg_color": "#fff3cd"})
                    _cv_col = list(df.columns).index("Log_CV%")
                    ws.conditional_format(
                        1, _cv_col, len(df), _cv_col,
                        {"type": "cell", "criteria": ">",
                         "value": cv_flag_threshold, "format": _warn_fmt})
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
