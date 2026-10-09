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

import chromatic_db as cdb


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
    # Rename to unambiguous µM names.  Both columns are in µM; the ratio
    # DyeConc_uM / DyeKd_uM is dimensionless in the Cheng-Prusoff formula.
    df = df.rename(columns={"Dye_Concentration": "DyeConc_uM", "Kd_uM": "DyeKd_uM"})
    df["DyeConc_uM"] = pd.to_numeric(df["DyeConc_uM"], errors="coerce")
    df["DyeKd_uM"]   = pd.to_numeric(df["DyeKd_uM"],   errors="coerce")
    df = df.dropna(subset=["Host", "Dye", "DyeConc_uM", "DyeKd_uM"])

    dups = df.duplicated(subset=["Host", "Dye"], keep=False)
    if dups.any():
        _log(f"WARNING: duplicate Host-Dye entries in Kd table — using first occurrence",
             progress_cb)
        df = df.drop_duplicates(subset=["Host", "Dye"], keep="first")

    _log(f"Kd table: {len(df)} Host-Dye entries", progress_cb)
    return df[["Host", "Dye", "DyeConc_uM", "DyeKd_uM"]]


# Combined stage 0 wrapper (both called together by the app)
def load_mappings_and_kd(dye_folder: str, host_folder: str, guest_folder: str,
                         kd_folder: str, progress_cb: ProgressCb = None) -> dict:
    merged_mapping = load_mappings_ki(dye_folder, host_folder, guest_folder, progress_cb)
    hot_df         = load_kd_table(kd_folder, progress_cb)
    return {"merged_mapping": merged_mapping, "hot_df": hot_df}


# ── Stage 1: load plates ──────────────────────────────────────────────────────

def _detect_chromatic_start_rows(df_raw: pd.DataFrame) -> list[int]:
    """Return data start rows for each chromatic block by auto-detecting
    the first plate-data row after each 'Chromatic' header.

    A plate-data row is identified by either:
      (a) column 0 contains a single row-label letter A–P, OR
      (b) the row has ≥20 numeric values whose maximum exceeds 25.
          This threshold distinguishes fluorescence readings (always ≫25)
          from column-number header rows (max = 24) that some plate readers
          insert between the Chromatic header and the actual data.

    This avoids hardcoding a fixed offset, which breaks when different
    chromatic blocks (or different plate reader export versions) have different
    numbers of metadata rows before the data.
    """
    _row_labels = set(string.ascii_uppercase[:16])   # A–P
    rows = []
    for i in range(len(df_raw)):
        cell = str(df_raw.iloc[i, 0]).strip().lower()
        if cell.startswith("chromatic"):
            # Scan up to 10 rows ahead for the first data row
            for j in range(i + 1, min(i + 11, len(df_raw))):
                first_cell = str(df_raw.iloc[j, 0]).strip()
                # (a) row label in column 0
                if len(first_cell) == 1 and first_cell.upper() in _row_labels:
                    rows.append(j)
                    break
                # (b) enough large numeric values to be fluorescence data
                numeric = pd.to_numeric(df_raw.iloc[j], errors="coerce").dropna()
                if len(numeric) >= 20 and float(numeric.max()) > 25:
                    rows.append(j)
                    break
    return rows


def _load_plate_ki(df_raw: pd.DataFrame, start_row: int,
                   plate_name: str, chromatic: int, chromatic_dye: str) -> pd.DataFrame:
    # If column 0 of the first data row contains a row label (A–P), skip it
    # and read 24 data columns from column 1 onward — same logic as pipeline_fda.
    first_cell = str(df_raw.iloc[start_row, 0]).strip()
    col_start  = 1 if (len(first_cell) == 1 and first_cell.upper() in _ROW_LABELS_SET) else 0

    plate_df = df_raw.iloc[start_row:start_row + 16, col_start:col_start + 24].copy()
    if plate_df.shape != (16, 24):
        # Use print because progress_cb is not available here; this warning
        # will appear in the terminal but not the GUI log.
        print(f"WARNING: {plate_name} chromatic {chromatic} "
              f"unexpected shape {plate_df.shape} (expected 16×24) — "
              f"check data_offset and file format")
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


def _resolve_plate_mapping(chromatic_to_dye: dict, stem: str) -> dict:
    """Return the chromatic→dye mapping for the plate identified by *stem*.

    Accepts two formats:
      - Old / global:  {1: "DAPI", 2: "DASPI"}   — all int keys, applied to every plate.
      - Per-plate:     {"": {1: "DAPI"}, "PlateB": {1: "DASPI"}}
          Non-empty str key = filename substring; matched case-insensitively.
          Empty str key ""  = global fallback when no pattern matches.
    """
    if not chromatic_to_dye:
        return {}
    if all(isinstance(k, int) for k in chromatic_to_dye):
        return chromatic_to_dye
    stem_lower = stem.lower()
    for pattern, mapping in chromatic_to_dye.items():
        if pattern and pattern.lower() in stem_lower:
            return mapping
    return chromatic_to_dye.get("", {})


def load_plates_ki(raw_folder: str, chromatic_to_dye: dict,
                   chromatic_folder: str = None,
                   progress_cb: ProgressCb = None) -> pd.DataFrame:
    """
    chromatic_to_dye accepts two formats:
      - Global (old): {1: "DAPI", 2: "DASPI"} — same mapping for every plate.
      - Per-plate:    {"": {1: "DAPI"}, "PlateB": {1: "DASPI"}}
          "" key = global fallback; non-empty key = plate filename substring.
          Plates whose filename contains the substring use that mapping instead.

    chromatic_folder, if given, points at a Dye/Filter lookup table (see
    chromatic_db.py). Each raw file's own "Used filter settings and gain
    values" header is matched against this table to resolve chromatic
    index -> dye automatically, per file — so the result is the same
    whether a dye was run alone on a plate or alongside others on it.
    Entries in chromatic_to_dye take precedence over the auto-detected ones
    (manual override for files with no header or dyes not yet catalogued).
    """
    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) else None)

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

        auto_map   = cdb.resolve_chromatic_to_dye(df_raw, chromatic_db, plate_name, progress_cb)
        manual_map = _resolve_plate_mapping(chromatic_to_dye, stem)
        plate_map  = {**auto_map, **manual_map}   # manual entries override auto-detected ones

        if plate_map:
            if len(start_rows) != len(plate_map):
                _log(f"  WARNING: '{plate_name}' — found {len(start_rows)} chromatic block(s), "
                     f"expected {len(plate_map)}", progress_cb)
            for (chrom_idx, dye_name), start_row in zip(sorted(plate_map.items()), start_rows):
                df_chrom = _load_plate_ki(df_raw, start_row, plate_name, chrom_idx, dye_name)
                all_dfs.append(df_chrom)
                _log(f"  Loaded '{plate_name}' | Chromatic {chrom_idx} ({dye_name})", progress_cb)
        else:
            # No mapping — load the FIRST chromatic block only.
            # Loading all blocks would mix fluorescence channels: every well would
            # appear once per block, all with Chromatic_Dye=None, making background
            # subtraction wrong (dye-blank average would span multiple channels).
            # Multi-chromatic plates MUST specify a mapping.
            if len(start_rows) > 1:
                _log(f"  WARNING: '{plate_name}' has {len(start_rows)} chromatic blocks "
                     f"but no Chromatic→Dye mapping is set — loading block 1 only.\n"
                     f"  Enable 'Multi-chromatic plates' in the settings and add "
                     f"mappings to use additional blocks.", progress_cb)
            if start_rows:
                df_chrom = _load_plate_ki(df_raw, start_rows[0], plate_name, 1, None)
                all_dfs.append(df_chrom)
                _log(f"  Loaded '{plate_name}' | Chromatic 1 (unmapped)", progress_cb)

    if not all_dfs:
        raise ValueError(
            "No fluorescence data loaded — no chromatic blocks found in any plate file.")
    result = pd.concat(all_dfs, ignore_index=True)
    result["Fluorescence"] = pd.to_numeric(result["Fluorescence"], errors="coerce")
    _log(f"Loaded {result['Plate'].nunique()} plate(s), "
         f"{result['Chromatic'].nunique()} chromatic(s)", progress_cb)
    return result


# ── Stage 2: merge + blanks ───────────────────────────────────────────────────

def merge_ki(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
             blank_folder: str, chromatic_folder: str = None,
             progress_cb: ProgressCb = None) -> pd.DataFrame:
    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) else None)
    alias_map = cdb.build_alias_map(chromatic_db)

    merged = (pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
              .sort_values(["Host", "Host_Concentration"]))

    # Plate-name mismatch diagnostic: if most rows have NaN Host after the merge,
    # the plate names in the fluorescence files do not match those in the mapping.
    n_total   = len(merged)
    n_no_host = merged["Host"].isna().sum()
    if n_total > 0 and n_no_host / n_total > 0.5:
        fl_plates  = sorted(fluorescence_df["Plate"].dropna().unique())
        map_plates = sorted(merged_mapping["Plate"].dropna().unique()
                            if "Plate" in merged_mapping.columns else [])
        _log(f"WARNING: {n_no_host}/{n_total} rows have no Host after mapping merge.\n"
             f"  Fluorescence plate names : {fl_plates}\n"
             f"  Mapping plate names      : {map_plates}\n"
             f"  These must match exactly. Check filename stems vs 'Destination Plate Name' "
             f"in your Echo mapping files.", progress_cb)

    # Keep rows where chromatic matches the dispensed dye (aliases allowed —
    # e.g. dye_map says "H33" but the chromatic DB / Kd table say "H33258").
    # Rows with Chromatic_Dye=None (unmapped plates) bypass this filter entirely,
    # so single-dye plates work without a chromatic mapping.
    # Background wells (no dye dispensed) are kept for all chromatics.
    merged = merged[
        merged["Chromatic_Dye"].isna() | merged["Dye"].isna() |
        merged.apply(lambda r: cdb.dye_matches(r["Dye"], r["Chromatic_Dye"], alias_map), axis=1)
    ].copy()
    _log(f"After chromatic filter: {len(merged)} rows", progress_cb)

    blank_files = sorted(f for f in os.listdir(blank_folder)
                         if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    blank_df = pd.concat(
        [pd.read_excel(os.path.join(blank_folder, f)) for f in blank_files],
        ignore_index=True)
    blank_df = blank_df[blank_df["Compound ID"].fillna("").str.strip().str.lower() == "blank"]

    # LEFT join keeps all fluorescence rows and adds matching blank labels.
    # OUTER was wrong: blank_df rows with no fluorescence match created orphan
    # NaN rows that propagated through to Dye_Blank_Avg → FI-F0 = NaN.
    #
    # Join only on [Well, Plate, Dye] — NOT on Dye_Concentration.
    # Floating-point values parsed from two separate Excel files are often not
    # bit-identical (e.g. 0.5 vs 0.5000000000000001), causing the join to silently
    # drop all blank rows and leaving Dye_Blank_Avg = NaN for every well.
    merged2 = pd.merge(merged, blank_df,
                       on=["Well", "Plate", "Dye"], how="left",
                       suffixes=("", "_blank"))
    _log(f"Merged with blanks: {len(merged2)} rows "
         f"({merged2['Compound ID'].notna().sum()} blank-labelled)", progress_cb)
    return merged2


# ── Stage 3: background subtraction + QC figures ─────────────────────────────

def subtract_background_ki(merged_df: pd.DataFrame,
                            progress_cb: ProgressCb = None) -> dict:
    """Returns dict with 'fi_df' and 'qc_figures' (list of matplotlib Figures)."""
    df       = merged_df.copy()
    # fillna("") guards against numeric or NaN Compound ID values that would
    # crash .str accessor calls.
    is_blank = df["Compound ID"].fillna("").str.strip().str.lower() == "blank"

    bg_mean = (df[is_blank & df["Dye"].isna()]
               .groupby(["Plate", "Chromatic_Dye"])["Fluorescence"].mean())
    # Background_Avg (buffer-only wells) is stored for diagnostic output but does
    # not appear in the FI-F0 formula. Algebraically it cancels:
    #   FI-F0 = (F − Bg) − (Dye_blank − Bg) = F − Dye_blank_avg
    df["Background_Avg"] = (
        pd.MultiIndex.from_arrays([df["Plate"], df["Chromatic_Dye"]]).map(bg_mean))

    dye_blank_rows = df[is_blank & df["Dye"].notna()]
    if dye_blank_rows.empty:
        _log("ERROR: no dye-blank wells found — check that:\n"
             "  • blank_map files contain rows with Compound ID = 'blank'\n"
             "  • Dye_Concentration values in blank_map match those in dye_map (float precision)\n"
             "  • Plate names in blank_map match Plate names in the fluorescence data\n"
             "FI-F0 will be NaN for all wells; curve fitting will produce no results.",
             progress_cb)
    dye_blank_mean = dye_blank_rows.groupby(["Plate", "Dye"])["Fluorescence"].mean()
    df["Dye_Blank_Avg"] = (
        pd.MultiIndex.from_arrays([df["Plate"], df["Dye"]]).map(dye_blank_mean))

    df["FI-F0"] = df["Fluorescence"] - df["Dye_Blank_Avg"]
    n_valid = df["FI-F0"].notna().sum()
    n_total = len(df)
    _log(f"Background subtraction complete: {n_valid}/{n_total} rows have valid FI-F0.",
         progress_cb)
    if n_valid == 0:
        _log("ERROR: all FI-F0 values are NaN. Most likely causes:\n"
             "  1. Blank file merge failed (plate/dye name mismatch or float precision on Dye_Concentration)\n"
             "  2. Mapping merge failed (plate names in fluorescence files ≠ mapping files)\n"
             "  Check the warnings above.", progress_cb)
    return {"fi_df": df}


def make_qc_figures_ki(df: pd.DataFrame, progress_cb: ProgressCb = None) -> list:
    """Generate QC figures for the Ki pipeline. Call from the main thread."""
    is_blank = df["Compound ID"].fillna("").str.strip().str.lower() == "blank"
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
        # Guest-only: guest dispensed but no dye and no host — checks guest intrinsic fluorescence.
        cond[df["Dye"].isna() & df["Guest"].notna() &
             df["Host"].isna() & ~is_blank_lbl]  = "Guest only"
        # Competition wells (all three components present) were previously classified
        # "other" and excluded — added here so the actual assay signal is visible in QC.
        cond[df["Dye"].notna() & df["Host"].notna() &
             df["Guest"].notna() & ~is_blank_lbl] = "Competition"

        qc_df      = df.copy()
        qc_df["Condition"] = cond
        qc_df      = qc_df[qc_df["Condition"] != "other"]
        # For unmapped plates Chromatic_Dye is NaN — fall back to the dispensed Dye
        # so QC subplots are grouped by dye regardless of whether a mapping was given.
        qc_df["Chromatic_Dye"] = qc_df["Chromatic_Dye"].fillna(qc_df["Dye"])
        COND_ORDER = ["Buffer blank", "Dye blank", "Host + Dye", "Guest only",
                      "Dye + Guest", "Competition"]
        qc_dyes    = sorted(qc_df["Chromatic_Dye"].dropna().unique())
        # Buffer blank wells have no dispensed dye so Chromatic_Dye is still NaN
        # after the fillna above.  Duplicate them for every known dye so they appear
        # in each dye's subplot — matching behaviour when a chromatic mapping is set.
        unassigned_buf = ((qc_df["Condition"] == "Buffer blank") &
                          qc_df["Chromatic_Dye"].isna())
        if unassigned_buf.any() and qc_dyes:
            buf_rows = qc_df[unassigned_buf]
            qc_df = pd.concat(
                [qc_df[~unassigned_buf]] +
                [buf_rows.assign(**{"Chromatic_Dye": d}) for d in qc_dyes],
                ignore_index=True)
        qc_plates  = sorted(qc_df["Plate"].dropna().unique())
        palette    = dict(zip(qc_plates, sns.color_palette("tab10", len(qc_plates))))
        handles    = [mpatches.Patch(color=palette[p], label=p) for p in qc_plates]

        # QC plot 1: control conditions
        fig1 = Figure(figsize=(max(5, 4.5 * len(qc_dyes)), 5))
        axes = fig1.subplots(1, max(len(qc_dyes), 1), squeeze=False)
        for col_i, (ax, dye) in enumerate(zip(axes[0], qc_dyes)):
            sub = qc_df[qc_df["Chromatic_Dye"] == dye]
            order1 = [c for c in COND_ORDER if c in sub["Condition"].values]
            sns.stripplot(data=sub, x="Condition", y="Fluorescence", order=order1,
                          hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4, alpha=0.7)
            if ax.get_legend():
                ax.get_legend().remove()
            for xi, c in enumerate(order1):
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

        # QC plot 3: Dye + Guest per guest, with Guest-only overlay
        guest_qc      = qc_df[qc_df["Condition"] == "Dye + Guest"]
        guest_only_qc = qc_df[qc_df["Condition"] == "Guest only"]
        has_guest_only = not guest_only_qc.empty
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
                # Dye + Guest: filled circles, coloured by plate
                sns.stripplot(data=sub, x="Guest", y="Fluorescence", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.2, size=4)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, g in enumerate(order):
                    vals = sub.loc[sub["Guest"] == g, "Fluorescence"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.0, xi + 0.35], [vals.mean(), vals.mean()],
                                lw=2.5, color="black", zorder=10)
                # Guest only: diamond markers, shifted left, same plate colours
                sub_go = (guest_only_qc[guest_only_qc["Chromatic_Dye"] == dye]
                          if has_guest_only else pd.DataFrame())
                if not sub_go.empty:
                    rng = np.random.default_rng(0)
                    for xi, g in enumerate(order):
                        rows = sub_go[sub_go["Guest"] == g]
                        if rows.empty:
                            continue
                        xs = xi - 0.35 + rng.uniform(-0.1, 0.1, size=len(rows))
                        ax.scatter(xs, rows["Fluorescence"].values,
                                   c=[palette[p] for p in rows["Plate"]],
                                   marker="D", s=18, alpha=0.7, zorder=8,
                                   edgecolors="none")
                    # Per-guest mean bar for Guest only
                    for xi, g in enumerate(order):
                        vals_go = sub_go.loc[sub_go["Guest"] == g, "Fluorescence"].dropna()
                        if len(vals_go):
                            ax.plot([xi - 0.45, xi - 0.05],
                                    [vals_go.mean(), vals_go.mean()],
                                    lw=2.5, color="darkorange", zorder=10)
                # Legend elements
                legend_els = []
                legend_els.append(plt.Line2D([], [], color="black", lw=2.5,
                                             label="Dye+Guest (mean)"))
                if has_guest_only and not sub_go.empty:
                    legend_els.append(plt.Line2D([], [], color="darkorange", lw=2.5,
                                                 label="Guest only (mean)"))
                    legend_els.append(plt.Line2D([], [], marker="D", color="grey",
                                                 lw=0, markersize=5,
                                                 label="Guest only (pts)"))
                if dye in dye_blank_ref.index:
                    ax.axhline(dye_blank_ref[dye], color="steelblue", lw=1.5,
                               ls="--", alpha=0.7)
                    legend_els.append(plt.Line2D([], [], color="steelblue", lw=1.5,
                                                 ls="--", label="Dye blank"))
                ax.legend(handles=legend_els, fontsize=7, loc="upper right")
                ax.set_title(dye, fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig3.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            suffix = " (◆ = guest only)" if has_guest_only else ""
            fig3.suptitle(f"QC — Dye + Guest per guest{suffix}",
                          fontsize=12, fontweight="bold")
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


def _comp_standard(DyeConc_uM, DyeKd_uM):
    def model(x, Top, Bottom, logKi):
        logEC50 = logKi + np.log10(1.0 + DyeConc_uM / DyeKd_uM)
        return Bottom + (Top - Bottom) / (1.0 + 10.0 ** (x - logEC50))
    return model


def _comp_hill(DyeConc_uM, DyeKd_uM):
    def model(x, Top, Bottom, logKi, HillSlope):
        logEC50 = logKi + np.log10(1.0 + DyeConc_uM / DyeKd_uM)
        return Bottom + (Top - Bottom) / (1.0 + 10.0 ** (HillSlope * (logEC50 - x)))
    return model


def _morrison(DyeConc_uM, DyeKd_uM, HostConc_uM):
    def model(x, Top, Bottom, logKi):
        Ki_app = 10.0 ** logKi * (1.0 + DyeConc_uM / DyeKd_uM)
        I_T    = 10.0 ** x   # Guest concentration in µM (x = log10[Guest/µM])
        E_T    = HostConc_uM  # Host concentration in µM — same units as I_T ✓
        A      = E_T + I_T + Ki_app
        disc   = np.maximum(A**2 - 4.0 * E_T * I_T, 0.0)
        frac   = np.clip((A - np.sqrt(disc)) / (2.0 * E_T), 0.0, 1.0)
        return Top - (Top - Bottom) * frac
    return model


def _biphasic(DyeConc_uM, DyeKd_uM):
    def model(x, Top, Bottom, logKi1, logKi2, Frac):
        Frac = np.clip(Frac, 0.0, 1.0)
        lEC1 = logKi1 + np.log10(1.0 + DyeConc_uM / DyeKd_uM)
        lEC2 = logKi2 + np.log10(1.0 + DyeConc_uM / DyeKd_uM)
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
        DyeConc_uM = float(kd_row["DyeConc_uM"].iloc[0])
        DyeKd_uM   = float(kd_row["DyeKd_uM"].iloc[0])

        grp   = fi_df.query("Host==@host and Dye==@dye and Guest==@guest")
        plate = (grp["Plate"].dropna().iloc[0]
                 if "Plate" in grp.columns and not grp["Plate"].dropna().empty else "")
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
                popt_s, _ = curve_fit(_comp_standard(DyeConc_uM, DyeKd_uM),
                                      x_cl, y_cl, p0=[y_max, y_min, x_med],
                                      bounds=bounds_std, maxfev=20000)
                std_Ki = 10.0 ** float(popt_s[2])
            except Exception:
                pass

        use_morrison = (not np.isnan(HostConc_uM) and not np.isnan(std_Ki)
                        and std_Ki < morrison_gate * HostConc_uM)

        models = [
            {"name": "Standard",  "fn": _comp_standard(DyeConc_uM, DyeKd_uM), "k": 3,
             "p0": [y_max, y_min, x_med], "bounds": bounds_std},
            {"name": "HillSlope", "fn": _comp_hill(DyeConc_uM, DyeKd_uM),     "k": 4,
             "p0": [y_max, y_min, x_med, 1.0],
             "bounds": ([y_min-2*y_rng, y_min-2*y_rng, -14, 0.1],
                        [y_max+2*y_rng, y_max+2*y_rng,  8, 10.0])},
            {"name": "Biphasic",  "fn": _biphasic(DyeConc_uM, DyeKd_uM),      "k": 5,
             "p0": [y_max, y_min, x_med-1.0, x_med+1.0, 0.5],
             "bounds": ([y_min-2*y_rng, y_min-2*y_rng, -14, -14, 0.0],
                        [y_max+2*y_rng, y_max+2*y_rng,  8,   8, 1.0])},
        ]
        if use_morrison:
            for seed in [x_min_cl, x_med - 1.5, x_med - 0.5, x_med, x_med + 0.5]:
                models.append({"name": "Morrison",
                                "fn": _morrison(DyeConc_uM, DyeKd_uM, HostConc_uM), "k": 3,
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
        elif best["name"] == "HillSlope":
            extra = {"HillSlope": float(best["popt"][3])}

        fit_results.append({
            "Host": host, "Dye": dye, "Guest": guest, "Plate": plate,
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
            "DyeConc_uM":  DyeConc_uM,
            "DyeKd_uM":    DyeKd_uM,
            "HostConc_uM": HostConc_uM,
            "Top_fit":     float(best["popt"][0]),
            "Bottom_fit":  float(best["popt"][1]),
            **extra,
        })

        plot_data.append({
            "host": host, "dye": dye, "guest": guest, "plate": plate,
            "x_cleaned":  x_cl,  "y_cleaned":  y_cl,
            "x_outliers": x_out, "y_outliers": y_out,
            "popt":       best["popt"],
            "factory":    best["fn"],
            "model_name": best["name"],
            "r2_adj":     best["r2_adj"],
            "logKi_err":  logKi_err,
            "DyeConc_uM": DyeConc_uM, "DyeKd_uM": DyeKd_uM,
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
    # No sign check: both enhancing dyes (Top_fit > 0 → Bottom_fit ≈ 0) and
    # quenching dyes (Top_fit < 0 → Bottom_fit ≈ 0) are valid competition signals.
    # A wrong-direction fit produces poor R²_adj and is caught by fail_r2.

    fail_any = fail_r2 | fail_ki
    df["Status"]      = "PASS"
    df.loc[fail_any, "Status"] = "FAIL"
    df["Fail_reason"] = ""
    df.loc[fail_r2,   "Fail_reason"] += df.loc[fail_r2, "R2_adj"].apply(
        lambda v: f"R²_adj={v:.3f} < {r2_threshold}; ")
    df.loc[fail_ki,   "Fail_reason"] += "Ki is NaN; "
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
                   ax_resid=None,
                   title_fontsize: float = 8,
                   title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "bold",
                   title_fontstyle: str = "normal",
                   stats_in_title: bool = True):
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
    ki_str    = f"Ki = {Ki:.2f}{f' ± {Ki_err:.2f}' if not np.isnan(Ki_err) else ''} µM"

    status      = entry.get("status")
    title_color = "red" if (show_status_color and status == "FAIL") else "black"
    tkw = dict(fontsize=title_fontsize, fontfamily=title_fontfamily,
               fontweight=title_fontweight, fontstyle=title_fontstyle,
               color=title_color)
    if stats_in_title:
        ax.set_title(
            f"{entry['host']} | {entry['dye']} | {entry['guest']}\n"
            f"{ki_str}  |  R²adj = {entry['r2_adj']:.3f}\n"
            f"[{model}]",
            **tkw)
    else:
        ax.text(0.98, 0.98,
                f"{ki_str}\nR²adj = {entry['r2_adj']:.3f}\n[{model}]",
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="right", color=color_fit,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_title(f"{entry['host']} | {entry['dye']} | {entry['guest']}", **tkw)
    ax.set_ylabel("FI – F₀", fontsize=7)
    ax.tick_params(labelsize=6)

    if ax_resid is not None:
        ax.tick_params(labelbottom=False)
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


def _folder_date_prefix(folder_path: str) -> str:
    if not folder_path:
        return ""
    path = os.path.abspath(folder_path)
    for _ in range(5):
        name = os.path.basename(path)
        if not name:
            break
        parts = name.split("_")
        if parts and re.match(r"^\d{6}$", parts[0]):
            return parts[0] + "_"
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return ""


def save_outputs_ki(state: PipelineStateKi, output_folder: str,
                    r2_threshold: float = PASS_R2_DEFAULT_KI,
                    progress_cb: ProgressCb = None,
                    plot_color_fit: str = PLOT_COLOR_FIT,
                    plot_color_data: str = PLOT_COLOR_DATA,
                    plot_color_resid: str = None,
                    show_residuals: bool = False,
                    export_individual: bool = False,
                    title_fontsize: float = 8,
                    title_fontfamily: str = "sans-serif",
                    title_fontweight: str = "bold",
                    title_fontstyle: str = "normal",
                    layout_cfg: dict = None,
                    input_folder: str = "") -> list[str]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    for d in (reports_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    ts, saved = _folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S"), []

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

    if state.df_results is not None and state.df_results.empty:
        _log("No fit results to save (0 curves fitted) — skipping fit-results Excel/PDF.", progress_cb)
    elif state.df_results is not None:
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

    # QC PDF — all figures in one file, one page each
    if state.qc_figures:
        qc_pdf = os.path.join(reports_dir, f"{ts}_QC_plots.pdf")
        with PdfPages(qc_pdf) as _pdf:
            for _label, _fig in state.qc_figures:
                _pdf.savefig(_fig, bbox_inches="tight", dpi=150)
        _log(f"Saved: {qc_pdf}", progress_cb)
        saved.append(qc_pdf)

    # Binding curve PDF (1 plate per page)
    if state.plot_data:
        lc       = layout_cfg or {}
        pdf_path = os.path.join(plots_dir, f"{ts}_competitive_binding.pdf")
        plates   = sorted(set(e.get("plate", "") for e in state.plot_data))
        with PdfPages(pdf_path) as pdf:
            for plate in plates:
                items  = [e for e in state.plot_data if e.get("plate", "") == plate]
                n      = len(items)
                cols   = min(lc.get("pdf_cols", COLS_PER_PAGE), n)
                rows   = int(np.ceil(n / cols))
                col_w  = lc.get("fig_w", cols * 3.0) / cols
                fig_w  = col_w * cols
                tkw = dict(color_fit=plot_color_fit, color_data=plot_color_data,
                           color_resid=plot_color_resid,
                           title_fontsize=title_fontsize,
                           title_fontfamily=title_fontfamily,
                           title_fontweight=title_fontweight,
                           title_fontstyle=title_fontstyle)
                if show_residuals:
                    fig_h    = rows * max(2.5, col_w * 1.70)
                    fig      = plt.figure(figsize=(fig_w, fig_h))
                    outer_gs = GridSpec(rows, cols, figure=fig,
                                        hspace=lc.get("hspace", 0.40),
                                        wspace=lc.get("wspace", 0.40))
                    for i, entry in enumerate(items):
                        ri, ci   = divmod(i, cols)
                        inner_gs = outer_gs[ri, ci].subgridspec(
                            2, 1, height_ratios=[3.5, 1],
                            hspace=lc.get("resid_gap", 0.06))
                        ax   = fig.add_subplot(inner_gs[0])
                        ax_r = fig.add_subplot(inner_gs[1], sharex=ax)
                        render_plot_ki(ax, entry, ax_resid=ax_r, **tkw)
                else:
                    fig_h  = rows * max(2.0, col_w * 1.55)
                    fig    = plt.figure(figsize=(fig_w, fig_h))
                    gs     = GridSpec(rows, cols, figure=fig,
                                      hspace=lc.get("hspace", 0.40),
                                      wspace=lc.get("wspace", 0.40))
                    for i, entry in enumerate(items):
                        ri, ci = divmod(i, cols)
                        render_plot_ki(fig.add_subplot(gs[ri, ci]), entry, **tkw)
                    for i in range(n, rows * cols):
                        ri, ci = divmod(i, cols)
                        fig.add_subplot(gs[ri, ci]).set_visible(False)
                fig.subplots_adjust(
                    left=lc.get("left", 0.08),   right=lc.get("right", 0.97),
                    top=lc.get("top", 0.94),      bottom=lc.get("bottom", 0.08))
                fig.suptitle(f"Plate: {plate}" if plate else "Plate: (unknown)",
                             fontsize=11, fontweight="bold", y=0.998)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        _log(f"Saved: {pdf_path}", progress_cb)
        saved.append(pdf_path)

        if export_individual:
            ind_cols = min(lc.get("pdf_cols", COLS_PER_PAGE),
                          len(state.plot_data))
            col_w = lc.get("fig_w", ind_cols * 3.0) / ind_cols
            for status_dir in ("PASS", "FAIL"):
                os.makedirs(os.path.join(plots_dir, "individual", status_dir), exist_ok=True)
            for entry in state.plot_data:
                status   = entry.get("status", "FAIL")
                safe_nm  = re.sub(r"[^\w\-]", "_",
                                  f"{entry['host']}_{entry['dye']}_{entry['guest']}")
                ind_path = os.path.join(plots_dir, "individual", status,
                                        f"{safe_nm}.pdf")
                tkw_ind = dict(color_fit=plot_color_fit, color_data=plot_color_data,
                               color_resid=plot_color_resid,
                               title_fontsize=title_fontsize,
                               title_fontfamily=title_fontfamily,
                               title_fontweight=title_fontweight,
                               title_fontstyle=title_fontstyle)
                if show_residuals:
                    ind_fh = max(2.5, col_w * 1.70)
                    fig    = plt.figure(figsize=(col_w, ind_fh))
                    gs     = GridSpec(2, 1, figure=fig,
                                      height_ratios=[3.5, 1],
                                      hspace=lc.get("resid_gap", 0.06))
                    ax   = fig.add_subplot(gs[0])
                    ax_r = fig.add_subplot(gs[1], sharex=ax)
                else:
                    ind_fh = max(2.0, col_w * 1.55)
                    fig, ax = plt.subplots(figsize=(col_w, ind_fh))
                    ax_r = None
                render_plot_ki(ax, entry, ax_resid=ax_r, **tkw_ind)
                fig.subplots_adjust(
                    left=lc.get("left", 0.08),   right=lc.get("right", 0.97),
                    top=lc.get("top", 0.94),      bottom=lc.get("bottom", 0.08))
                fig.savefig(ind_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                saved.append(ind_path)
            _log(f"Saved {len(state.plot_data)} individual plot(s).", progress_cb)

    return saved


def preview_save_files_ki(state: PipelineStateKi, output_folder: str,
                          export_individual: bool = False,
                          input_folder: str = "") -> list[dict]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    ts, files   = _folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S"), []

    def _f(folder, name, desc):
        files.append({"path": os.path.join(folder, f"{ts}_{name}"), "description": desc})

    if state.merged_mapping is not None: _f(reports_dir, "mapping.csv",              "Combined mapping (dye/host/guest)")
    if state.hot_df         is not None: _f(reports_dir, "kd_table.csv",             "Kd lookup table")
    if state.fluorescence   is not None: _f(reports_dir, "raw_data.csv",             "Raw fluorescence")
    if state.merged         is not None: _f(reports_dir, "merged_with_blanks.csv",   "Merged data + blank labels")
    if state.fi_df          is not None: _f(reports_dir, "blank_corrected.csv",      "Background-subtracted (FI-F0)")
    if state.qc_figures:                 _f(reports_dir, "QC_plots.pdf",
                                              f"QC plots — {len(state.qc_figures)} page(s)")
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
