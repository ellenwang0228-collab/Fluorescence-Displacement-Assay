#!/usr/bin/env python3
"""
Competitive binding pipeline (Ki) — multi-model, multi-chromatic.

Four models are attempted per (Host, Dye, Guest) curve; best selected by AICc:
  Standard  — classic logistic, Hill=1, Cheng-Prusoff correction
  HillSlope — free Hill slope
  Morrison  — quadratic tight-binding correction (gated on Ki vs host conc)
  Biphasic  — two-site competitive model
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import normaltest, t


ProgressCb = Optional[Callable[[str], None]]

PASS_R2_DEFAULT_KI       = 0.9
MORRISON_GATE_DEFAULT    = 1.0
MORRISON_KI_FRAC_DEFAULT = 0.05
PLOT_COLOR_DATA          = "#1e4572"
PLOT_COLOR_FIT           = "#6495ED"

MODEL_COLORS = {
    "Standard":  "dimgray",
    "HillSlope": "#8B4513",
    "Morrison":  "#007700",
    "Biphasic":  "#8B008B",
}


def _log(msg: str, cb: ProgressCb) -> None:
    if cb:
        cb(msg)
    else:
        print(msg)


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStateKi:
    merged_mapping: Optional[pd.DataFrame] = None
    hot_df:         Optional[pd.DataFrame] = None   # Kd lookup table
    fluorescence:   Optional[pd.DataFrame] = None
    merged:         Optional[pd.DataFrame] = None
    fi_df:          Optional[pd.DataFrame] = None
    qc_figures:     list = field(default_factory=list)   # list[Figure]
    fit_results:    list = field(default_factory=list)
    plot_data:      list = field(default_factory=list)
    df_results:     Optional[pd.DataFrame] = None


# ── Stage 0a: load mappings ───────────────────────────────────────────────────

def load_mappings_ki(dye_folder: str, host_folder: str, guest_folder: str,
                     progress_cb: ProgressCb = None) -> pd.DataFrame:
    def _load(folder):
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(".xlsx") and not f.startswith("~$"))
        return pd.concat([pd.read_excel(os.path.join(folder, f)) for f in files],
                         ignore_index=True), files

    dye_df,   dye_files   = _load(dye_folder)
    host_df,  host_files  = _load(host_folder)
    guest_df, guest_files = _load(guest_folder)

    _log(f"Dye maps:   {dye_files}", progress_cb)
    _log(f"Host maps:  {host_files}", progress_cb)
    _log(f"Guest maps: {guest_files}", progress_cb)

    def _sel(df, col_map):
        return df.loc[:, col_map.keys()].rename(columns=col_map)

    dye_df = _sel(dye_df, {
        "Compound ID": "Dye", "Destination Well": "Well",
        "Destination Concentration": "Dye_Concentration",
        "Destination Unit": "Dye_Unit", "Destination Plate Name": "Plate"})
    host_df = _sel(host_df, {
        "Compound ID": "Host", "Destination Well": "Well",
        "Destination Concentration": "Host_Concentration",
        "Destination Unit": "Host_Unit", "Destination Plate Name": "Plate"})
    guest_df = _sel(guest_df, {
        "Compound ID": "Guest", "Destination Well": "Well",
        "Destination Concentration": "Guest_Concentration",
        "Destination Unit": "Guest_Unit", "Destination Plate Name": "Plate"})

    merged = (dye_df
              .merge(host_df,  on=["Well", "Plate"], how="outer")
              .merge(guest_df, on=["Well", "Plate"], how="outer"))
    for col in ["Dye_Unit", "Host_Unit", "Guest_Unit"]:
        merged[col] = merged[col].str.replace(r"[^\w]", "", regex=True)

    _log(f"Hosts: {merged['Host'].nunique()} | Dyes: {merged['Dye'].nunique()} | "
         f"Guests: {merged['Guest'].nunique()}", progress_cb)
    return merged


# ── Stage 0b: load Kd table ───────────────────────────────────────────────────

def load_kd_table(kd_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(kd_folder)
                   if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    _log(f"Kd table files: {files}", progress_cb)
    df = pd.concat([pd.read_excel(os.path.join(kd_folder, f)) for f in files],
                   ignore_index=True)
    df = df.rename(columns={"Dye_Concentration": "HotNM", "Kd_uM": "HotKdNM"})
    df["HotNM"]   = pd.to_numeric(df["HotNM"],   errors="coerce")
    df["HotKdNM"] = pd.to_numeric(df["HotKdNM"], errors="coerce")
    df = df.dropna(subset=["Host", "Dye", "HotNM", "HotKdNM"])

    dups = df.duplicated(subset=["Host", "Dye"], keep=False)
    if dups.any():
        _log(f"WARNING: duplicate Host-Dye entries in Kd table — using first occurrence",
             progress_cb)
        df = df.drop_duplicates(subset=["Host", "Dye"], keep="first")

    _log(f"Kd table: {len(df)} Host-Dye entries", progress_cb)
    return df[["Host", "Dye", "HotNM", "HotKdNM"]]


# Combined stage 0 wrapper (both called together by the app)
def load_mappings_and_kd(dye_folder: str, host_folder: str, guest_folder: str,
                         kd_folder: str, progress_cb: ProgressCb = None) -> dict:
    merged_mapping = load_mappings_ki(dye_folder, host_folder, guest_folder, progress_cb)
    hot_df         = load_kd_table(kd_folder, progress_cb)
    return {"merged_mapping": merged_mapping, "hot_df": hot_df}


# ── Stage 1: load plates ──────────────────────────────────────────────────────

def _detect_chromatic_start_rows(df_raw: pd.DataFrame, data_offset: int = 3) -> list[int]:
    """Return data start rows for each chromatic block (any cell starting with 'chromatic')."""
    rows = []
    for i in range(len(df_raw)):
        if df_raw.iloc[i].dropna().astype(str).str.lower().str.startswith("chromatic").any():
            rows.append(i + data_offset)
    return rows


def _load_plate_ki(df_raw: pd.DataFrame, start_row: int,
                   plate_name: str, chromatic: int, chromatic_dye: str) -> pd.DataFrame:
    plate_df = df_raw.iloc[start_row:start_row + 16, 0:24].copy()
    if plate_df.shape != (16, 24):
        print(f"Warning: {plate_name} chromatic {chromatic} shape {plate_df.shape}")
    plate_df.index   = list(string.ascii_uppercase[:16])
    plate_df.columns = [str(i) for i in range(1, 25)]
    tidy = (plate_df.reset_index()
            .melt(id_vars="index", var_name="Column", value_name="Fluorescence")
            .rename(columns={"index": "Row"}))
    tidy["Well"]          = tidy["Row"] + tidy["Column"]
    tidy["Plate"]         = plate_name
    tidy["Chromatic"]     = chromatic
    tidy["Chromatic_Dye"] = chromatic_dye
    return tidy[["Well", "Fluorescence", "Plate", "Chromatic", "Chromatic_Dye"]].dropna(
        subset=["Fluorescence"])


def load_plates_ki(raw_folder: str, chromatic_to_dye: dict,
                   progress_cb: ProgressCb = None) -> pd.DataFrame:
    """
    chromatic_to_dye: {1: "DAPI", 2: "DASPI", 3: "H33"} — order maps to nth block found.
    """
    filepaths = sorted(
        os.path.join(raw_folder, f) for f in os.listdir(raw_folder)
        if f.lower().endswith(".xlsx") and not f.startswith("~$"))

    all_dfs = []
    for fp in filepaths:
        stem       = os.path.splitext(os.path.basename(fp))[0]
        tokens     = stem.split("_")
        plate_name = next((t for t in tokens if t and not (t.isdigit() and len(t) > 4)), stem)
        df_raw     = pd.read_excel(fp, engine="openpyxl", header=None, sheet_name=0)
        start_rows = _detect_chromatic_start_rows(df_raw)

        if len(start_rows) != len(chromatic_to_dye):
            _log(f"  WARNING: '{plate_name}' — found {len(start_rows)} chromatic block(s), "
                 f"expected {len(chromatic_to_dye)}", progress_cb)

        for (chrom_idx, dye_name), start_row in zip(chromatic_to_dye.items(), start_rows):
            df_chrom = _load_plate_ki(df_raw, start_row, plate_name, chrom_idx, dye_name)
            all_dfs.append(df_chrom)
            _log(f"  Loaded '{plate_name}' | Chromatic {chrom_idx} ({dye_name})", progress_cb)

    result = pd.concat(all_dfs, ignore_index=True)
    result["Fluorescence"] = pd.to_numeric(result["Fluorescence"], errors="coerce")
    _log(f"Loaded {result['Plate'].nunique()} plate(s), "
         f"{result['Chromatic'].nunique()} chromatic(s)", progress_cb)
    return result


# ── Stage 2: merge + blanks ───────────────────────────────────────────────────

def merge_ki(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
             blank_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    merged = (pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
              .sort_values(["Host", "Host_Concentration"]))

    # Keep only rows where the well's chromatic matches the dispensed dye;
    # background wells (no dye) are kept for all chromatics.
    merged = merged[
        (merged["Dye"] == merged["Chromatic_Dye"]) | merged["Dye"].isna()
    ].copy()
    _log(f"After chromatic filter: {len(merged)} rows", progress_cb)

    blank_files = sorted(f for f in os.listdir(blank_folder)
                         if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    blank_df = pd.concat(
        [pd.read_excel(os.path.join(blank_folder, f)) for f in blank_files],
        ignore_index=True)
    blank_df = blank_df[blank_df["Compound ID"].str.strip().str.lower() == "blank"]

    # LEFT join keeps all fluorescence rows and adds matching blank labels.
    # OUTER was wrong: blank_df rows with no fluorescence match created orphan
    # NaN rows that propagated through to Dye_Blank_Avg → FI-F0 = NaN.
    merged2 = pd.merge(merged, blank_df,
                       on=["Well", "Plate", "Dye", "Dye_Concentration"], how="left")
    _log(f"Merged with blanks: {len(merged2)} rows", progress_cb)
    return merged2


# ── Stage 3: background subtraction + QC figures ─────────────────────────────

def subtract_background_ki(merged_df: pd.DataFrame,
                            progress_cb: ProgressCb = None) -> dict:
    """Returns dict with 'fi_df' and 'qc_figures' (list of matplotlib Figures)."""
    df       = merged_df.copy()
    is_blank = df["Compound ID"].str.strip().str.lower() == "blank"

    bg_mean = (df[is_blank & df["Dye"].isna()]
               .groupby(["Plate", "Chromatic_Dye"])["Fluorescence"].mean())
    # Background_Avg (buffer-only wells) is stored for diagnostic output but does
    # not appear in the FI-F0 formula. Algebraically it cancels:
    #   FI-F0 = (F − Bg) − (Dye_blank − Bg) = F − Dye_blank_avg
    df["Background_Avg"] = (
        pd.MultiIndex.from_arrays([df["Plate"], df["Chromatic_Dye"]]).map(bg_mean))

    dye_blank_mean = (df[is_blank & df["Dye"].notna()]
                      .groupby(["Plate", "Dye"])["Fluorescence"].mean())
    df["Dye_Blank_Avg"] = (
        pd.MultiIndex.from_arrays([df["Plate"], df["Dye"]]).map(dye_blank_mean))

    df["FI-F0"] = df["Fluorescence"] - df["Dye_Blank_Avg"]
    _log("Background subtraction complete.", progress_cb)
    return {"fi_df": df}


def make_qc_figures_ki(df: pd.DataFrame, progress_cb: ProgressCb = None) -> list:
    """Generate QC figures for the Ki pipeline. Call from the main thread."""
    is_blank = df["Compound ID"].str.strip().str.lower() == "blank"
    return _make_qc_figures(df, is_blank, progress_cb)


def _make_qc_figures(df: pd.DataFrame, is_blank, progress_cb: ProgressCb) -> list:
    figures = []
    try:
        is_blank_lbl = df["Compound ID"].fillna("").str.strip().str.lower() == "blank"
        cond = pd.Series("other", index=df.index)
        cond[is_blank_lbl & df["Dye"].isna()]   = "Buffer blank"
        cond[is_blank_lbl & df["Dye"].notna()]  = "Dye blank"
        cond[df["Dye"].notna() & df["Host"].notna() &
             df["Guest"].isna() & ~is_blank_lbl] = "Host + Dye"
        cond[df["Dye"].notna() & df["Guest"].notna() &
             df["Host"].isna() & ~is_blank_lbl]  = "Dye + Guest"

        qc_df      = df.copy()
        qc_df["Condition"] = cond
        qc_df      = qc_df[qc_df["Condition"] != "other"]
        COND_ORDER = ["Buffer blank", "Dye blank", "Host + Dye", "Dye + Guest"]
        qc_dyes    = sorted(qc_df["Chromatic_Dye"].dropna().unique())
        qc_plates  = sorted(qc_df["Plate"].dropna().unique())
        palette    = dict(zip(qc_plates, sns.color_palette("tab10", len(qc_plates))))
        handles    = [mpatches.Patch(color=palette[p], label=p) for p in qc_plates]

        # QC plot 1: control conditions
        fig1 = Figure(figsize=(max(5, 4.5 * len(qc_dyes)), 5))
        axes = fig1.subplots(1, max(len(qc_dyes), 1), squeeze=False)
        for col_i, (ax, dye) in enumerate(zip(axes[0], qc_dyes)):
            sub = qc_df[qc_df["Chromatic_Dye"] == dye]
            sns.stripplot(data=sub, x="Condition", y="Fluorescence", order=COND_ORDER,
                          hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4, alpha=0.7)
            if ax.get_legend():
                ax.get_legend().remove()
            for xi, c in enumerate(COND_ORDER):
                vals = sub.loc[sub["Condition"] == c, "Fluorescence"].dropna()
                if len(vals):
                    ax.plot([xi - 0.3, xi + 0.3], [vals.mean(), vals.mean()],
                            lw=2.5, color="black", zorder=10)
            ax.set_title(dye, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Fluorescence (raw)" if col_i == 0 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        fig1.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                    loc="center left", fontsize=8)
        fig1.suptitle("QC — control well fluorescence", fontsize=12, fontweight="bold")
        fig1.tight_layout()
        figures.append(("QC: Control Wells", fig1))

        # Dye blank reference line for QC plots 2 & 3
        dye_blank_ref = (qc_df[qc_df["Condition"] == "Dye blank"]
                         .groupby("Chromatic_Dye")["Fluorescence"].mean())

        # QC plot 2: Host + Dye per host
        host_qc = qc_df[qc_df["Condition"] == "Host + Dye"]
        if not host_qc.empty:
            max_hosts = max(host_qc.groupby("Chromatic_Dye")["Host"].nunique().max(), 1)
            fig2 = Figure(figsize=(max(6, 0.65 * max_hosts * len(qc_dyes) + 2), 5))
            axes2 = fig2.subplots(1, max(len(qc_dyes), 1), squeeze=False)
            for col_i, (ax, dye) in enumerate(zip(axes2[0], qc_dyes)):
                sub = host_qc[host_qc["Chromatic_Dye"] == dye]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                order = (sub.groupby("Host")["Fluorescence"].median()
                         .sort_values().index.tolist())
                sns.stripplot(data=sub, x="Host", y="Fluorescence", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, h in enumerate(order):
                    vals = sub.loc[sub["Host"] == h, "Fluorescence"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.3, xi + 0.3], [vals.mean(), vals.mean()],
                                lw=2.5, color="black", zorder=10)
                if dye in dye_blank_ref.index:
                    ax.axhline(dye_blank_ref[dye], color="steelblue", lw=1.5,
                               ls="--", alpha=0.7, label="Dye blank")
                    ax.legend(fontsize=7)
                ax.set_title(dye, fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig2.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            fig2.suptitle("QC — Host + Dye per host", fontsize=12, fontweight="bold")
            fig2.tight_layout()
            figures.append(("QC: Host + Dye", fig2))

        # QC plot 3: Dye + Guest per guest
        guest_qc = qc_df[qc_df["Condition"] == "Dye + Guest"]
        if not guest_qc.empty:
            max_guests = max(guest_qc.groupby("Chromatic_Dye")["Guest"].nunique().max(), 1)
            fig3 = Figure(figsize=(max(6, 0.65 * max_guests * len(qc_dyes) + 2), 5))
            axes3 = fig3.subplots(1, max(len(qc_dyes), 1), squeeze=False)
            for col_i, (ax, dye) in enumerate(zip(axes3[0], qc_dyes)):
                sub = guest_qc[guest_qc["Chromatic_Dye"] == dye]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                order = (sub.groupby("Guest")["Fluorescence"].median()
                         .sort_values(ascending=False).index.tolist())
                sns.stripplot(data=sub, x="Guest", y="Fluorescence", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, g in enumerate(order):
                    vals = sub.loc[sub["Guest"] == g, "Fluorescence"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.3, xi + 0.3], [vals.mean(), vals.mean()],
                                lw=2.5, color="black", zorder=10)
                if dye in dye_blank_ref.index:
                    ax.axhline(dye_blank_ref[dye], color="steelblue", lw=1.5,
                               ls="--", alpha=0.7, label="Dye blank")
                    ax.legend(fontsize=7)
                ax.set_title(dye, fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig3.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            fig3.suptitle("QC — Dye + Guest per guest", fontsize=12, fontweight="bold")
            fig3.tight_layout()
            figures.append(("QC: Dye + Guest", fig3))

        _log(f"Generated {len(figures)} QC figure(s).", progress_cb)
    except Exception as e:
        _log(f"WARNING: QC figure generation failed: {e}", progress_cb)

    return figures


# ── Stage 4: multi-model fitting ──────────────────────────────────────────────

def _aicc(n: int, rss: float, k: int) -> float:
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2.0 * k + (2.0 * k * (k + 1)) / (n - k - 1)


def _grubbs_mask(data, alpha: float = 0.05) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    mask = np.ones(len(data), dtype=bool)
    while np.sum(mask) > 2:
        vals = data[mask]
        std  = np.std(vals, ddof=1)
        if std == 0:
            break
        abs_diffs  = np.abs(vals - np.mean(vals))
        idx_local  = int(np.argmax(abs_diffs))
        idx_global = np.where(mask)[0][idx_local]
        G      = abs_diffs[idx_local] / std
        N      = len(vals)
        t_crit = t.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = ((N - 1) / np.sqrt(N)) * np.sqrt(t_crit**2 / (N - 2 + t_crit**2))
        if G > G_crit:
            mask[idx_global] = False
        else:
            break
    return mask


def _comp_standard(HotNM, HotKdNM):
    def model(x, Top, Bottom, logKi):
        logEC50 = logKi + np.log10(1.0 + HotNM / HotKdNM)
        return Bottom + (Top - Bottom) / (1.0 + 10.0 ** (x - logEC50))
    return model


def _comp_hill(HotNM, HotKdNM):
    def model(x, Top, Bottom, logKi, HillSlope):
        logEC50 = logKi + np.log10(1.0 + HotNM / HotKdNM)
        return Bottom + (Top - Bottom) / (1.0 + 10.0 ** (HillSlope * (logEC50 - x)))
    return model


def _morrison(HotNM, HotKdNM, HostConc_uM):
    def model(x, Top, Bottom, logKi):
        Ki_app = 10.0 ** logKi * (1.0 + HotNM / HotKdNM)
        I_T    = 10.0 ** x
        E_T    = HostConc_uM
        A      = E_T + I_T + Ki_app
        disc   = np.maximum(A**2 - 4.0 * E_T * I_T, 0.0)
        frac   = np.clip((A - np.sqrt(disc)) / (2.0 * E_T), 0.0, 1.0)
        return Top - (Top - Bottom) * frac
    return model


def _biphasic(HotNM, HotKdNM):
    def model(x, Top, Bottom, logKi1, logKi2, Frac):
        Frac = np.clip(Frac, 0.0, 1.0)
        lEC1 = logKi1 + np.log10(1.0 + HotNM / HotKdNM)
        lEC2 = logKi2 + np.log10(1.0 + HotNM / HotKdNM)
        return Bottom + (Top - Bottom) * (
            Frac / (1.0 + 10.0 ** (x - lEC1)) +
            (1.0 - Frac) / (1.0 + 10.0 ** (x - lEC2)))
    return model


def _transition_span(factory, popt, x_min, x_max) -> float:
    Top, Bottom = float(popt[0]), float(popt[1])
    if abs(Top - Bottom) < 1e-6:
        return np.inf
    x_eval = np.linspace(x_min - 1.0, x_max + 1.0, 500)
    y_eval = factory(x_eval, *popt)
    def _cross(target):
        diffs = y_eval - target
        idx   = np.where(np.diff(np.sign(diffs)))[0]
        if not len(idx):
            return np.nan
        i  = idx[0]
        dy = y_eval[i + 1] - y_eval[i]
        return (x_eval[i] if abs(dy) < 1e-12
                else x_eval[i] + (target - y_eval[i]) * (x_eval[i + 1] - x_eval[i]) / dy)
    x10 = _cross(Top - 0.10 * (Top - Bottom))
    x90 = _cross(Top - 0.90 * (Top - Bottom))
    return np.inf if (np.isnan(x10) or np.isnan(x90)) else abs(x90 - x10)


def fit_curves_ki(fi_df: pd.DataFrame, hot_df: pd.DataFrame,
                  grubbs_alpha: float = 0.05,
                  morrison_gate: float = MORRISON_GATE_DEFAULT,
                  morrison_ki_frac: float = MORRISON_KI_FRAC_DEFAULT,
                  progress_cb: ProgressCb = None) -> tuple[list, list]:

    groups = (fi_df.dropna(subset=["Host", "Dye", "Guest", "Guest_Concentration", "FI-F0"])
              [["Host", "Dye", "Guest"]].drop_duplicates())
    total = len(groups)

    fit_results, plot_data = [], []

    for i, (_, row) in enumerate(groups.iterrows()):
        host, dye, guest = row["Host"], row["Dye"], row["Guest"]
        _log(f"  Fitting [{i+1}/{total}]: {host} | {dye} | {guest}", progress_cb)

        kd_row = hot_df.loc[(hot_df["Host"] == host) & (hot_df["Dye"] == dye)]
        if kd_row.empty:
            _log(f"    SKIP — no Kd entry for {host}-{dye}", progress_cb)
            continue
        HotNM   = float(kd_row["HotNM"].iloc[0])
        HotKdNM = float(kd_row["HotKdNM"].iloc[0])

        grp = fi_df.query("Host==@host and Dye==@dye and Guest==@guest")
        hc_vals     = grp["Host_Concentration"].dropna()
        HostConc_uM = float(hc_vals.iloc[0]) if not hc_vals.empty else np.nan
        if hc_vals.nunique() > 1:
            _log(f"    WARNING: multiple Host_Concentration values for {host}-{dye} "
                 f"{sorted(hc_vals.unique())} — Morrison uses first: {HostConc_uM} µM",
                 progress_cb)

        conc     = pd.to_numeric(grp["Guest_Concentration"], errors="coerce")
        mask_pos = conc > 0
        x_all    = np.log10(conc[mask_pos].values)
        y_all    = pd.to_numeric(grp.loc[mask_pos, "FI-F0"], errors="coerce").values
        ok       = ~(np.isnan(x_all) | np.isnan(y_all))
        x_all, y_all = x_all[ok], y_all[ok]

        if len(x_all) < 5 or len(np.unique(x_all)) < 3:
            _log(f"    SKIP — insufficient data points", progress_cb)
            continue

        keep = np.ones(len(y_all), dtype=bool)
        for xv in np.unique(x_all):
            idx = np.where(x_all == xv)[0]
            if len(idx) >= 3:
                keep[idx] = _grubbs_mask(y_all[idx], alpha=grubbs_alpha)

        x_cl  = x_all[keep]
        y_cl  = y_all[keep]
        x_out = x_all[~keep]
        y_out = y_all[~keep]

        if len(y_cl) < 5:
            _log(f"    SKIP — too few points after Grubbs", progress_cb)
            continue

        y_min = float(np.min(y_cl))
        y_max = float(np.max(y_cl))
        y_rng = max(abs(y_max - y_min), abs(y_max) + 1.0)
        x_med = float(np.median(x_cl))
        x_min_cl = float(np.min(x_cl))
        x_max_cl = float(np.max(x_cl))

        bounds_std = ([y_min - 2*y_rng, y_min - 2*y_rng, -14],
                      [y_max + 2*y_rng, y_max + 2*y_rng,  8])

        # Gate Morrison on Standard's Ki estimate
        std_Ki = np.nan
        if not np.isnan(HostConc_uM):
            try:
                popt_s, _ = curve_fit(_comp_standard(HotNM, HotKdNM),
                                      x_cl, y_cl, p0=[y_max, y_min, x_med],
                                      bounds=bounds_std, maxfev=20000)
                std_Ki = 10.0 ** float(popt_s[2])
            except Exception:
                pass

        use_morrison = (not np.isnan(HostConc_uM) and not np.isnan(std_Ki)
                        and std_Ki < morrison_gate * HostConc_uM)

        models = [
            {"name": "Standard",  "fn": _comp_standard(HotNM, HotKdNM), "k": 3,
             "p0": [y_max, y_min, x_med], "bounds": bounds_std},
            {"name": "HillSlope", "fn": _comp_hill(HotNM, HotKdNM),     "k": 4,
             "p0": [y_max, y_min, x_med, 1.0],
             "bounds": ([y_min-2*y_rng, y_min-2*y_rng, -14, 0.1],
                        [y_max+2*y_rng, y_max+2*y_rng,  8, 10.0])},
            {"name": "Biphasic",  "fn": _biphasic(HotNM, HotKdNM),      "k": 5,
             "p0": [y_max, y_min, x_med-1.0, x_med+1.0, 0.5],
             "bounds": ([y_min-2*y_rng, y_min-2*y_rng, -14, -14, 0.0],
                        [y_max+2*y_rng, y_max+2*y_rng,  8,   8, 1.0])},
        ]
        if use_morrison:
            for seed in [x_min_cl, x_med - 1.5, x_med - 0.5, x_med, x_med + 0.5]:
                models.append({"name": "Morrison",
                                "fn": _morrison(HotNM, HotKdNM, HostConc_uM), "k": 3,
                                "p0": [y_max, y_min, seed], "bounds": bounds_std})

        best = None
        for m in models:
            n_pts = len(y_cl)
            if n_pts <= m["k"]:
                continue
            try:
                popt, pcov = curve_fit(m["fn"], x_cl, y_cl, p0=m["p0"],
                                       bounds=m["bounds"], maxfev=20000)
            except (RuntimeError, ValueError):
                continue

            if m["name"] == "Morrison":
                Ki_fit = 10.0 ** float(popt[2])
                if Ki_fit < morrison_ki_frac * HostConc_uM:
                    continue
                if _transition_span(m["fn"], popt, x_min_cl, x_max_cl) < 1.0:
                    continue

            y_fit  = m["fn"](x_cl, *popt)
            rss    = float(np.sum((y_cl - y_fit)**2))
            ss_tot = float(np.sum((y_cl - np.mean(y_cl))**2))
            r2     = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
            r2_adj = (1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - m["k"] - 1)
                      if n_pts > m["k"] + 1 else r2)
            aic_val = _aicc(n_pts, rss, m["k"])

            if best is None or aic_val < best["aic"]:
                best = {"name": m["name"], "fn": m["fn"], "popt": popt, "pcov": pcov,
                        "r2_adj": r2_adj, "aic": aic_val, "k": m["k"]}

        if best is None:
            _log(f"    All models failed", progress_cb)
            continue

        logKi = float(best["popt"][2])
        try:
            kd_var    = float(best["pcov"][2, 2])
            logKi_err = float(np.sqrt(kd_var)) if kd_var >= 0 else np.nan
        except Exception:
            logKi_err = np.nan
        Ki     = 10.0 ** logKi
        Ki_err = Ki * np.log(10) * logKi_err if not np.isnan(logKi_err) else np.nan

        resid = y_cl - best["fn"](x_cl, *best["popt"])
        try:
            normal_p = normaltest(resid).pvalue if len(resid) >= 8 else np.nan
        except Exception:
            normal_p = np.nan

        extra = {}
        if best["name"] == "Biphasic":
            extra = {"logKi2":    float(best["popt"][3]),
                     "Ki2_uM":    10.0 ** float(best["popt"][3]),
                     "Frac_site1": float(best["popt"][4])}

        fit_results.append({
            "Host": host, "Dye": dye, "Guest": guest,
            "Best_Model":  best["name"],
            "logKi":       logKi,
            "logKi_err":   logKi_err,
            "Ki_uM":       Ki,
            "Ki_err_uM":   Ki_err,
            "R2_adj":      best["r2_adj"],
            "AICc":        best["aic"],
            "Normality_p": normal_p,
            "n_total":     len(y_all),
            "n_removed":   int(np.sum(~keep)),
            "HotNM_uM":    HotNM,
            "HotKdNM_uM":  HotKdNM,
            "HostConc_uM": HostConc_uM,
            "Top_fit":     float(best["popt"][0]),
            "Bottom_fit":  float(best["popt"][1]),
            **extra,
        })

        plot_data.append({
            "host": host, "dye": dye, "guest": guest,
            "x_cleaned":  x_cl,  "y_cleaned":  y_cl,
            "x_outliers": x_out, "y_outliers": y_out,
            "popt":       best["popt"],
            "factory":    best["fn"],
            "model_name": best["name"],
            "r2_adj":     best["r2_adj"],
            "logKi_err":  logKi_err,
            "HotNM":      HotNM, "HotKdNM": HotKdNM,
            "status":     None,
        })

    _log(f"Fitting complete: {len(fit_results)} curves.", progress_cb)
    return fit_results, plot_data


# ── apply thresholds ──────────────────────────────────────────────────────────

def apply_thresholds_ki(fit_results: list, plot_data: list,
                        r2_threshold: float = PASS_R2_DEFAULT_KI) -> pd.DataFrame:
    df = pd.DataFrame(fit_results).copy()
    if df.empty:
        return df

    fail_r2   = df["R2_adj"] < r2_threshold
    fail_ki   = df["Ki_uM"].isna()
    fail_sign = (df["Top_fit"] < 0) & (df["Bottom_fit"] < 0)

    fail_any = fail_r2 | fail_ki | fail_sign
    df["Status"]      = "PASS"
    df.loc[fail_any, "Status"] = "FAIL"
    df["Fail_reason"] = ""
    df.loc[fail_r2,   "Fail_reason"] += df.loc[fail_r2, "R2_adj"].apply(
        lambda v: f"R²_adj={v:.3f} < {r2_threshold}; ")
    df.loc[fail_ki,   "Fail_reason"] += "Ki is NaN; "
    df.loc[fail_sign, "Fail_reason"] += "Top and Bottom < 0; "
    df["Fail_reason"] = df["Fail_reason"].str.rstrip("; ")

    lookup = df.set_index(["Host", "Dye", "Guest"])["Status"].to_dict()
    for e in plot_data:
        e["status"] = lookup.get((e["host"], e["dye"], e["guest"]), "FAIL")

    return df


# ── plot rendering ────────────────────────────────────────────────────────────

def render_plot_ki(ax, entry: dict, show_status_color: bool = True,
                   color_fit: str = PLOT_COLOR_FIT,
                   color_data: str = PLOT_COLOR_DATA,
                   color_resid: str = None,
                   ax_resid=None):
    x_cl  = entry["x_cleaned"]
    y_cl  = entry["y_cleaned"]
    x_out = entry["x_outliers"]
    y_out = entry["y_outliers"]
    popt  = entry["popt"]
    fn    = entry["factory"]

    stats = (pd.DataFrame({"x": x_cl, "y": y_cl})
             .groupby("x")["y"].agg(["mean", "std"]))
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                fmt="o", color=color_data, ecolor=color_data,
                elinewidth=1, markersize=4, capsize=2)

    if len(x_out) > 0:
        ax.scatter(x_out, y_out, color="red", s=12, zorder=5)

    x_line = np.linspace(stats.index.min(), stats.index.max(), 300)
    ax.plot(x_line, fn(x_line, *popt), lw=2, color=color_fit)

    logKi     = float(popt[2])
    Ki        = 10.0 ** logKi
    logKi_err = entry.get("logKi_err", np.nan)
    Ki_err    = Ki * np.log(10) * logKi_err if not np.isnan(logKi_err) else np.nan
    model     = entry["model_name"]

    ax.text(0.98, 0.98,
            f"Ki = {Ki:.2f}{f' ± {Ki_err:.2f}' if not np.isnan(Ki_err) else ''} µM\n"
            f"R²adj = {entry['r2_adj']:.3f}\n[{model}]",
            transform=ax.transAxes, fontsize=6.5,
            va="top", ha="right",
            color=color_fit,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    status      = entry.get("status")
    title_color = "red" if (show_status_color and status == "FAIL") else "black"
    ax.set_title(f"{entry['host']} | {entry['dye']} | {entry['guest']}",
                 fontsize=7, color=title_color)
    ax.set_ylabel("FI – F₀", fontsize=7)
    ax.tick_params(labelsize=6)

    if ax_resid is not None:
        rc      = color_resid or color_data
        resid   = y_cl - fn(x_cl, *popt)
        stats_r = (pd.DataFrame({"x": x_cl, "r": resid})
                   .groupby("x")["r"].agg(["mean", "std"]))
        ax_resid.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax_resid.errorbar(stats_r.index, stats_r["mean"], yerr=stats_r["std"],
                         fmt="o", color=rc, ecolor=rc,
                         elinewidth=1, markersize=2.5, capsize=2, zorder=3)
        ax_resid.set_ylabel("Resid.", fontsize=5.5)
        ax_resid.tick_params(labelsize=5)
        ax_resid.set_xlabel("log[Guest] (µM)", fontsize=6)
    else:
        ax.set_xlabel("log[Guest] (µM)", fontsize=7)


# ── save outputs ──────────────────────────────────────────────────────────────

COLS_PER_PAGE = 4
ROWS_PER_PAGE = 3


def save_outputs_ki(state: PipelineStateKi, output_folder: str,
                    r2_threshold: float = PASS_R2_DEFAULT_KI,
                    progress_cb: ProgressCb = None,
                    plot_color_fit: str = PLOT_COLOR_FIT,
                    plot_color_data: str = PLOT_COLOR_DATA,
                    plot_color_resid: str = None,
                    show_residuals: bool = False,
                    export_individual: bool = False) -> list[str]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    for d in (reports_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    ts, saved = datetime.now().strftime("%y%m%d_%H%M%S"), []

    def _csv(df, label):
        p = os.path.join(reports_dir, f"{ts}_{label}.csv")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        _log(f"Saved: {p}", progress_cb)
        saved.append(p)

    if state.merged_mapping is not None: _csv(state.merged_mapping, "mapping")
    if state.hot_df         is not None: _csv(state.hot_df,         "kd_table")
    if state.fluorescence   is not None: _csv(state.fluorescence,   "raw_data")
    if state.merged         is not None: _csv(state.merged,         "merged_with_blanks")
    if state.fi_df          is not None: _csv(state.fi_df,          "blank_corrected")

    if state.df_results is not None:
        df_pass = state.df_results[state.df_results["Status"] == "PASS"]
        df_fail = state.df_results[state.df_results["Status"] == "FAIL"]
        xp = os.path.join(results_dir, f"{ts}_competitive_fit_results.xlsx")
        with pd.ExcelWriter(xp, engine="xlsxwriter") as w:
            df_pass.to_excel(w, sheet_name="PASS", index=False)
            df_fail.to_excel(w, sheet_name="FAIL", index=False)
        _log(f"Saved: {xp}", progress_cb)
        saved.append(xp)

        if state.fi_df is not None:
            lookup = state.df_results.set_index(["Host", "Dye", "Guest"])["Status"].to_dict()
            pb, fb = [], []
            src = state.fi_df.dropna(subset=["Host", "Dye", "Guest", "FI-F0"])
            for (host, dye, guest), grp in src.groupby(["Host", "Dye", "Guest"]):
                status = lookup.get((host, dye, guest), "FAIL")
                grp    = grp.sort_values("Guest_Concentration")
                concs  = grp["Guest_Concentration"].drop_duplicates().reset_index(drop=True)
                fi_bc  = grp.groupby("Guest_Concentration")["FI-F0"].apply(list)
                maxr   = fi_bc.apply(len).max()
                block  = pd.DataFrame({f"{host} | {dye} | {guest}": concs})
                for idx in range(maxr):
                    block[f"FI-F0_rep{idx+1}"] = (
                        fi_bc.apply(lambda v, _i=idx: v[_i] if _i < len(v) else pd.NA)
                        .reset_index(drop=True))
                (pb if status == "PASS" else fb).append(block)
            wp = os.path.join(results_dir, f"{ts}_competitive_data.xlsx")
            with pd.ExcelWriter(wp, engine="xlsxwriter") as w:
                (pd.concat(pb, axis=1) if pb else pd.DataFrame()).to_excel(
                    w, sheet_name="PASS", index=False)
                (pd.concat(fb, axis=1) if fb else pd.DataFrame()).to_excel(
                    w, sheet_name="FAIL", index=False)
            _log(f"Saved: {wp}", progress_cb)
            saved.append(wp)

    # QC PDFs
    for label, fig in state.qc_figures:
        p = os.path.join(reports_dir, f"{ts}_{label.replace(' ', '_').replace(':', '')}.pdf")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        _log(f"Saved: {p}", progress_cb)
        saved.append(p)

    # Binding curve PDF (1 host per page)
    if state.plot_data:
        pdf_path = os.path.join(plots_dir, f"{ts}_competitive_binding.pdf")
        hosts    = sorted(set(e["host"] for e in state.plot_data))
        with PdfPages(pdf_path) as pdf:
            for host in hosts:
                items = [e for e in state.plot_data if e["host"] == host]
                n     = len(items)
                cols  = min(COLS_PER_PAGE, n)
                rows  = int(np.ceil(n / cols))
                if show_residuals:
                    fig = plt.figure(figsize=(cols * 4, rows * 4.5))
                    gs  = GridSpec(rows * 2, cols, figure=fig,
                                   height_ratios=[3.5, 1] * rows,
                                   hspace=0.06, wspace=0.4)
                    for i, entry in enumerate(items):
                        ri, ci = divmod(i, cols)
                        ax   = fig.add_subplot(gs[ri * 2,     ci])
                        ax_r = fig.add_subplot(gs[ri * 2 + 1, ci], sharex=ax)
                        render_plot_ki(ax, entry,
                                       color_fit=plot_color_fit, color_data=plot_color_data,
                                       color_resid=plot_color_resid, ax_resid=ax_r)
                else:
                    fig, axes = plt.subplots(rows, cols,
                                             figsize=(cols * 4, rows * 3.5),
                                             squeeze=False)
                    for ax, entry in zip(axes.flatten(), items):
                        render_plot_ki(ax, entry,
                                       color_fit=plot_color_fit, color_data=plot_color_data,
                                       color_resid=plot_color_resid)
                    for ax in axes.flatten()[n:]:
                        fig.delaxes(ax)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                fig.suptitle(f"Host: {host}", fontsize=11, fontweight="bold")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        _log(f"Saved: {pdf_path}", progress_cb)
        saved.append(pdf_path)

        if export_individual:
            for status_dir in ("PASS", "FAIL"):
                os.makedirs(os.path.join(plots_dir, "individual", status_dir), exist_ok=True)
            for entry in state.plot_data:
                status   = entry.get("status", "FAIL")
                safe_nm  = re.sub(r"[^\w\-]", "_",
                                  f"{entry['host']}_{entry['dye']}_{entry['guest']}")
                ind_path = os.path.join(plots_dir, "individual", status,
                                        f"{safe_nm}.pdf")
                if show_residuals:
                    fig = plt.figure(figsize=(6, 5.5))
                    gs  = GridSpec(2, 1, figure=fig,
                                   height_ratios=[3.5, 1], hspace=0.06)
                    ax  = fig.add_subplot(gs[0])
                    ax_r = fig.add_subplot(gs[1], sharex=ax)
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax_r = None
                render_plot_ki(ax, entry,
                               color_fit=plot_color_fit, color_data=plot_color_data,
                               color_resid=plot_color_resid, ax_resid=ax_r)
                plt.tight_layout()
                fig.savefig(ind_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                saved.append(ind_path)
            _log(f"Saved {len(state.plot_data)} individual plot(s).", progress_cb)

    return saved


def preview_save_files_ki(state: PipelineStateKi, output_folder: str,
                          export_individual: bool = False) -> list[dict]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    ts, files   = datetime.now().strftime("%y%m%d_%H%M%S"), []

    def _f(folder, name, desc):
        files.append({"path": os.path.join(folder, f"{ts}_{name}"), "description": desc})

    if state.merged_mapping is not None: _f(reports_dir, "mapping.csv",              "Combined mapping (dye/host/guest)")
    if state.hot_df         is not None: _f(reports_dir, "kd_table.csv",             "Kd lookup table")
    if state.fluorescence   is not None: _f(reports_dir, "raw_data.csv",             "Raw fluorescence")
    if state.merged         is not None: _f(reports_dir, "merged_with_blanks.csv",   "Merged data + blank labels")
    if state.fi_df          is not None: _f(reports_dir, "blank_corrected.csv",      "Background-subtracted (FI-F0)")
    for label, _ in state.qc_figures:   _f(reports_dir, f"{label}.pdf",             f"QC plot: {label}")
    if state.df_results is not None:
        _f(results_dir, "competitive_fit_results.xlsx", "Ki fit results (PASS/FAIL sheets)")
        if state.fi_df is not None:
            _f(results_dir, "competitive_data.xlsx",    "Wide-format FI-F0 (PASS/FAIL sheets)")
    if state.plot_data:
        _f(plots_dir, "competitive_binding.pdf", "Competitive binding curves (1 host per page)")
        if export_individual:
            files.append({"path": os.path.join(plots_dir, "individual", "PASS", "…"),
                          "description": f"Individual PDFs — {len(state.plot_data)} plot(s) → individual/PASS|FAIL/"})
    return files
