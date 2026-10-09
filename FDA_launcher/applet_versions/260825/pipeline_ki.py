#!/usr/bin/env python3
"""
Competitive binding pipeline (Ki) — multi-model, multi-chromatic.

Four models are attempted per (Host, Dye, Guest) curve; best selected by AICc:
  Standard  — classic logistic, Hill=1, Cheng-Prusoff correction
  HillSlope — free Hill slope
  Wang      — exact cubic solution for two-ligand (dye + guest) competitive
              binding to one site, accounting for depletion of dye, host,
              and guest (Wang, Z.-X. FEBS Lett. 1995;360(2):111-114).
              Gated on Ki vs host conc.

Ranking note (screening/triage use): within one Host/Dye pair, ranking
guests by Ki_uM is equivalent to ranking by raw EC50/IC50 (EC50_uM column)
— the Cheng-Prusoff shift is a constant added to every guest's curve in
that pair, regardless of whether it's theoretically exact for the fitted
Hill slope. Ki comparisons ACROSS different dyes/tracers within one screen
are only approximate, since the shift differs by dye (different
DyeConc_uM/DyeKd_uM) — rank within one Host/Dye pair first, and treat
cross-dye Ki comparisons as indicative only.
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
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import normaltest, t, norm

import chromatic_db as cdb
import layout_utils


ProgressCb = Optional[Callable[[str], None]]

PASS_R2_DEFAULT_KI       = 0.9
KI_RANGE_FACTOR_LO_DEFAULT = 0.1   # mirrors pipeline_fda.KD_RANGE_FACTOR_LO_DEFAULT
KI_RANGE_FACTOR_HI_DEFAULT = 10.0  # mirrors pipeline_fda.KD_RANGE_FACTOR_HI_DEFAULT
WANG_GATE_DEFAULT        = 1.0
WANG_GATE_MARGIN_BAND    = (0.5, 2.0)  # Wang_gate_margin (normalized to 1.0 at the gate
                                        # threshold) within this band ⇒ the choice to try
                                        # Wang was itself close to the boundary (see
                                        # fit_curves_ki docstring; mirrors
                                        # DEPLETION_GATE_MARGIN_BAND in pipeline_fda.py)
GUEST_AUTOFL_FACTOR      = 1.0
GUEST_AUTOFL_Z_DEFAULT       = 1.96  # two-tailed 95% CI z-score; separation threshold for the "stat" autofl mode
LOGKI_ERR_THRESHOLD_DEFAULT  = 0.217 # on the fitted log10 scale. Deliberately matches
                                      # pipeline_fda.KD_CV_THRESHOLD_DEFAULT's implied precision:
                                      # for small errors, SE(log10 K) ≈ (SE_K/K)/ln(10), so Kd's
                                      # 0.5 linear-CV threshold corresponds to a log10-SE of
                                      # ~0.217 — the same value used here. "Confidence: High"
                                      # now means the same precision in both pipelines.
NORMALITY_P_THRESHOLD_DEFAULT = 0.01 # deliberately lenient — diagnostic note, not a hard gate
RUNS_TEST_P_THRESHOLD_DEFAULT = 0.05 # diagnostic only — does not gate Status/Confidence
HILL_FLAG_LO_DEFAULT         = 0.5   # HillSlope outside [lo, hi] ⇒ Hill_flag=True (triage signal, not an error)
HILL_FLAG_HI_DEFAULT         = 2.0
DELTA_AICC_AMBIGUOUS         = 2.0   # Burnham & Anderson 2002: ΔAICc < 2 ⇒ no strong preference between models
                                      # (mirrors pipeline_fda.DELTA_AICC_AMBIGUOUS — keep both in sync if changed)
PLOT_COLOR_DATA          = "#1e4d72"
PLOT_COLOR_FIT           = "#5480ed"

MODEL_COLORS = {
    "Standard":  "dimgray",
    "HillSlope": "#8B4513",
    "Wang":      "#007700",
}


def _fmt_uM(v: float) -> str:
    """Format a Kd/Ki value in µM: fixed-point when compact, scientific when >= 5 digits."""
    if np.isnan(v):
        return "NaN"
    if abs(v) >= 10000:
        return f"{v:.4E}"
    return f"{v:.2f}"


def _log(msg: str, cb: ProgressCb) -> None:
    if cb:
        cb(msg)
    else:
        print(msg)


def _hc_key(v) -> float:
    """Stable, hashable key for a host concentration. Used to disambiguate
    Host/Dye/Guest combos that were run at multiple host concentrations in
    the same batch — NaN/missing values collapse to a fixed sentinel so
    dict lookups don't silently miss on NaN != NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1.0
    return round(v, 6) if not np.isnan(v) else -1.0


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
                     buffer_folder: str = None,
                     progress_cb: ProgressCb = None) -> pd.DataFrame:
    def _load(folder, label):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{label} folder not found: {folder}")
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(".xlsx") and not f.startswith("~$"))
        if not files:
            raise FileNotFoundError(f"No .xlsx files in {label} folder: {folder}")
        return pd.concat([pd.read_excel(os.path.join(folder, f)) for f in files],
                         ignore_index=True), files

    dye_df,   dye_files   = _load(dye_folder, "dye_map")
    host_df,  host_files  = _load(host_folder, "host_map")
    guest_df, guest_files = _load(guest_folder, "guest_map")

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

    if buffer_folder and os.path.isdir(buffer_folder):
        buf_files = sorted(f for f in os.listdir(buffer_folder)
                           if f.lower().endswith(".xlsx") and not f.startswith("~$"))
        if not buf_files:
            raise FileNotFoundError(f"No .xlsx files in buffer mapping folder: {buffer_folder}")
        _log(f"Loading buffer mapping: {buf_files}", progress_cb)
        buf_df = pd.concat(
            [pd.read_excel(os.path.join(buffer_folder, f)) for f in buf_files],
            ignore_index=True)
        buf_df = buf_df.loc[:, [
            "Compound ID", "Destination Well", "Destination Plate Name"
        ]].rename(columns={
            "Compound ID":            "Buffer",
            "Destination Well":       "Well",
            "Destination Plate Name": "Plate",
        }).drop_duplicates(subset=["Well", "Plate"])
        merged = pd.merge(merged, buf_df, on=["Well", "Plate"], how="left")
        merged["Buffer"] = merged["Buffer"].fillna("Unknown")
        _log(f"Buffers: {merged['Buffer'].nunique()}  ({sorted(merged['Buffer'].unique())})",
             progress_cb)

    _log(f"Hosts: {merged['Host'].nunique()} | Dyes: {merged['Dye'].nunique()} | "
         f"Guests: {merged['Guest'].nunique()}", progress_cb)
    return merged


# ── Stage 0b: load Kd table ───────────────────────────────────────────────────

def load_kd_table(kd_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    if not os.path.isdir(kd_folder):
        raise FileNotFoundError(f"Kd table folder not found: {kd_folder}")
    files = sorted(f for f in os.listdir(kd_folder)
                   if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files in Kd table folder: {kd_folder}")
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
                         kd_folder: str, buffer_folder: str = None,
                         progress_cb: ProgressCb = None) -> dict:
    merged_mapping = load_mappings_ki(dye_folder, host_folder, guest_folder,
                                      buffer_folder=buffer_folder,
                                      progress_cb=progress_cb)
    hot_df         = load_kd_table(kd_folder, progress_cb)
    return {"merged_mapping": merged_mapping, "hot_df": hot_df}


def load_exceptions(exceptions_folder: str, progress_cb: ProgressCb = None) -> set:
    """Wells to exclude because their dispense failed, from a folder of Echo
    'Exceptions' reports (.csv, columns 'Destination Plate Name' / 'Destination
    Well' — the same Echo transfer-report convention as dye_map/host_map/
    guest_map, but the failure log instead of the successful-transfer one).

    Every row in these files is a failed transfer, so no filtering on
    Transfer Status / Actual Volume is applied. Returns a set of
    (Plate, Well) tuples; empty if no folder/files are given.
    """
    if not exceptions_folder or not os.path.isdir(exceptions_folder):
        return set()
    files = sorted(f for f in os.listdir(exceptions_folder)
                   if f.lower().endswith(".csv") and not f.startswith("~$"))
    if not files:
        return set()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(os.path.join(exceptions_folder, f)))
        except Exception as exc:
            _log(f"WARNING: could not read exceptions file '{f}': {exc}", progress_cb)
    if not dfs:
        return set()
    df = pd.concat(dfs, ignore_index=True)
    _required = {"Destination Plate Name", "Destination Well"}
    missing = _required - set(df.columns)
    if missing:
        _log(f"WARNING: exceptions file(s) missing column(s) {sorted(missing)} — "
             f"skipping (no wells excluded)", progress_cb)
        return set()
    pairs = set(zip(df["Destination Plate Name"].astype(str).str.strip(),
                    df["Destination Well"].astype(str).str.strip()))
    _log(f"Exceptions: {len(pairs)} failed-dispense well(s) from {files}", progress_cb)
    return pairs


# ── Stage 1: load plates ──────────────────────────────────────────────────────

def _detect_chromatic_start_rows(df_raw: pd.DataFrame) -> list[int]:
    """Return data start rows for each chromatic block by auto-detecting
    the first plate-data row after each 'Chromatic' header.

    A plate-data row is identified by either:
      (a) column 0 contains a single row-label letter A–P, OR
      (b) the row has ≥3 numeric values whose maximum exceeds 25.
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
                if len(numeric) >= 3 and float(numeric.max()) > 25:
                    rows.append(j)
                    break
    return rows


def _load_plate_ki(df_raw: pd.DataFrame, start_row: int,
                   plate_name: str, chromatic: int, chromatic_dye: str) -> pd.DataFrame:
    # If column 0 of the first data row contains a row label (A–P), skip it
    # and read 24 data columns from column 1 onward — same logic as pipeline_fda.
    first_cell = str(df_raw.iloc[start_row, 0]).strip()
    col_start  = 1 if (len(first_cell) == 1 and first_cell.upper() in set(string.ascii_uppercase[:16])) else 0

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
    tidy["Fluorescence"]  = pd.to_numeric(tidy["Fluorescence"], errors="coerce")
    tidy["Well"]          = tidy["Row"] + tidy["Column"]
    tidy["Plate"]         = plate_name
    tidy["Chromatic"]     = chromatic_dye if chromatic_dye else chromatic
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
                   multi_chromatic: bool = False,
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

    multi_chromatic gates whether any of this targeting (auto-detected or
    manual) is used at all. Tick it only for plates that actually carry more
    than one dye. When off (default), every plate is treated as single-dye:
    the Chromatic DB is not consulted and only the first chromatic block is
    loaded (with a warning if more than one is found) — its dye then comes
    straight from the dye mapping in merge_ki.
    """
    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) and multi_chromatic
                    else None)

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

        if not multi_chromatic:
            auto_map, manual_map, plate_map = {}, {}, {}
            if len(start_rows) > 1:
                _log(f"  WARNING: '{plate_name}' has {len(start_rows)} chromatic blocks "
                     f"but 'Multi-chromatic plates' is unticked — loading block 1 only. "
                     f"Tick it to use the Chromatic DB to target each dye on this plate.",
                     progress_cb)
                start_rows = start_rows[:1]
        else:
            auto_map   = cdb.resolve_chromatic_to_dye(df_raw, chromatic_db, plate_name, progress_cb)
            manual_map = _resolve_plate_mapping(chromatic_to_dye, stem)
            plate_map  = {**auto_map, **manual_map}   # manual entries override auto-detected ones

        if plate_map:
            if len(start_rows) != len(plate_map):
                _log(f"  WARNING: '{plate_name}' — found {len(start_rows)} chromatic block(s), "
                     f"expected {len(plate_map)}", progress_cb)
            for (chrom_idx, dye_name), start_row in zip(sorted(plate_map.items()),
                                                       sorted(start_rows)):
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
             multi_chromatic: bool = False,
             exceptions_folder: str = None,
             progress_cb: ProgressCb = None) -> pd.DataFrame:
    """multi_chromatic gates whether a Chromatic_Dye/Dye mismatch drops rows
    (targeting real multi-dye plates via the Chromatic DB) or only produces a
    warning while the dye mapping is trusted as-is (single-dye plates, the
    default).

    exceptions_folder, if given, points at a folder of Echo 'Exceptions'
    reports (see load_exceptions) — matching wells are flagged in the
    'Dispense_Failed' column instead of dropped, so fit_curves_ki can
    exclude them from fitting while still plotting them as crosses."""
    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) else None)
    alias_map = cdb.build_alias_map(chromatic_db)

    merged = (pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
              .sort_values(["Host", "Host_Concentration"]))

    failed_wells = load_exceptions(exceptions_folder, progress_cb)
    merged["Dispense_Failed"] = (
        [(p, w) in failed_wells for p, w in zip(merged["Plate"].astype(str),
                                                merged["Well"].astype(str))]
        if failed_wells else False)

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

    # Compare each row's resolved Chromatic_Dye (only set when multi_chromatic
    # targeted this plate via the Chromatic DB / manual mapping — see
    # load_plates_ki) against the dispensed dye (aliases allowed — e.g.
    # dye_map says "H33" but the chromatic DB / Kd table say "H33258"). Rows
    # with Chromatic_Dye=None (unmapped/single-dye plates) bypass this check
    # entirely. Background wells (no dye dispensed) are kept for all chromatics.
    _both_present = merged["Chromatic_Dye"].notna() & merged["Dye"].notna()
    _match_vec = pd.Series(True, index=merged.index)
    if _both_present.any():
        _sub = merged.loc[_both_present]
        _match_vec.loc[_both_present] = [
            cdb.dye_matches(d, c, alias_map)
            for d, c in zip(_sub["Dye"], _sub["Chromatic_Dye"])
        ]
        _mismatched = _both_present & ~_match_vec
        if _mismatched.any():
            bad = merged.loc[_mismatched]
            for plate, dye, chrom_dye in sorted(set(zip(bad["Plate"], bad["Dye"], bad["Chromatic_Dye"]))):
                _log(f"  WARNING: plate '{plate}' — dye_map says dye '{dye}' but the "
                     f"chromatic block/filter resolves this channel to '{chrom_dye}'."
                     + (" Rows for this mismatch are dropped (multi-chromatic targeting is on)."
                        if multi_chromatic else
                        " Keeping the dye_map's value since 'Multi-chromatic plates' is "
                        "unticked — check the mapping if unexpected."),
                     progress_cb)

    # Dropping mismatched rows is only safe when targeting real multi-dye
    # plates via the Chromatic DB — otherwise (default) the dye mapping alone
    # decides each well's dye and nothing is filtered out.
    if multi_chromatic:
        merged = merged[~_both_present | _match_vec].copy()
    _log(f"After chromatic filter: {len(merged)} rows", progress_cb)

    if not os.path.isdir(blank_folder):
        raise FileNotFoundError(f"Blank mapping folder not found: {blank_folder}")
    blank_files = sorted(f for f in os.listdir(blank_folder)
                         if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    if not blank_files:
        raise FileNotFoundError(f"No .xlsx files in blank mapping folder: {blank_folder}")
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


def make_qc_figures_ki(df: pd.DataFrame, progress_cb: ProgressCb = None,
                       qc1_y_mode: str = "all",
                       qc1_y_min: float = None, qc1_y_max: float = None) -> list:
    """Generate QC figures for the Ki pipeline. Call from the main thread.

    qc1_y_mode controls the Y-axis range on QC1 (control wells):
      "all"    (default) — every point visible, plain auto-scaled axis.
      "tukey"  — robust IQR-fence range, so a couple of very fluorescent
                 guests don't stretch the axis and flatten everything else.
      "manual" — qc1_y_min/qc1_y_max (either can be given on its own; the
                 other side falls back to the Tukey bound).
    """
    return _make_qc_figures(df, progress_cb, qc1_y_mode, qc1_y_min, qc1_y_max)


def _make_qc_figures(df: pd.DataFrame, progress_cb: ProgressCb,
                     qc1_y_mode: str = "all",
                     qc1_y_min: float = None, qc1_y_max: float = None) -> list:
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
        # For unresolved rows, Chromatic_Dye is NaN — fall back to the row's
        # OWN dispensed Dye first (straight from the mapping, always correct
        # for that well, regardless of whether the Chromatic DB could resolve
        # it). Only rows with no dye of their own either (Guest-only /
        # Buffer-blank wells) fall further back to whichever Dye is used
        # elsewhere on the same PLATE — a safe guess only for single-dye
        # plates, so those wells still land in the right QC subplot when no
        # chromatic mapping is given.
        #
        # Falling back to the row's own Dye BEFORE the plate-wide guess
        # matters on genuinely multi-dye plates: if the Chromatic DB can't
        # resolve a channel (e.g. two dyes sharing a similar/ambiguous filter
        # entry), the old plate-wide "first dye seen" fallback silently
        # relabelled every unresolved row on that plate — including the
        # second dye's own Host+Dye/Dye blank/Dye+Guest wells — as whichever
        # dye happened to appear first, making the second dye's data vanish
        # from its own QC subplot (overwritten by the first dye's label).
        qc_df["Chromatic_Dye"] = qc_df["Chromatic_Dye"].fillna(qc_df["Dye"])
        _plate_dye_lookup = (qc_df.loc[qc_df["Dye"].notna(), ["Plate", "Dye"]]
                            .drop_duplicates(subset="Plate").set_index("Plate")["Dye"])
        qc_df["Chromatic_Dye"] = qc_df["Chromatic_Dye"].fillna(
            qc_df["Plate"].map(_plate_dye_lookup))
        COND_ORDER = ["Buffer blank", "Dye blank", "Host + Dye", "Guest only",
                      "Dye + Guest", "Competition"]
        qc_dyes    = sorted(qc_df["Chromatic_Dye"].dropna().unique())
        _log(f"  Plate -> Dye used for QC grouping: {_plate_dye_lookup.to_dict()}  |  "
             f"distinct dye subplot(s): {qc_dyes}", progress_cb)
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

        # QC-specific FI-F0 uses a DIFFERENT background reference per condition:
        #   - Buffer blank / Dye blank / Guest only: F0 = buffer blank average
        #     — these characterise a single component's (dye's or guest's) own
        #     intrinsic signal above true zero background.
        #   - Host + Dye / Dye + Guest / Competition: F0 = dye blank average
        #     — the free-dye baseline is the right zero-point once a second
        #     component (host and/or guest) is added, since the question is
        #     "how much has this changed relative to free dye" (matches the
        #     real Ki FI-F0 used for fitting in subtract_background_ki).
        # Grouped by (Plate, Chromatic_Dye), not Chromatic_Dye alone, so a
        # multi-plate dataset doesn't blend unrelated plates' background
        # levels into one reference.
        _DYE_REF_CONDS = {"Host + Dye", "Dye + Guest", "Competition"}
        buf_blank_ref = (qc_df[qc_df["Condition"] == "Buffer blank"]
                        .groupby(["Plate", "Chromatic_Dye"])["Fluorescence"].mean())
        dye_blank_ref = (qc_df[qc_df["Condition"] == "Dye blank"]
                        .groupby(["Plate", "Chromatic_Dye"])["Fluorescence"].mean())
        _plates_with_buf = sorted({p for p, _ in buf_blank_ref.index})
        _plates_with_dye = sorted({p for p, _ in dye_blank_ref.index})
        _plates_missing_buf = sorted(set(qc_plates) - set(_plates_with_buf))
        _log(f"  QC1 references: {len(buf_blank_ref)} buffer-blank (Plate,Dye) group(s) "
             f"covering plate(s) {_plates_with_buf}; "
             f"{len(dye_blank_ref)} dye-blank (Plate,Dye) group(s) covering plate(s) "
             f"{_plates_with_dye}. Condition counts: "
             f"{qc_df['Condition'].value_counts().to_dict()}",
             progress_cb)
        if _plates_missing_buf:
            _log(f"  NOTE: plate(s) {_plates_missing_buf} have NO buffer-blank wells — "
                 f"their Buffer blank/Dye blank/Guest only QC points will fall back to "
                 f"the dye-blank reference instead (smaller apparent signal than "
                 f"plates that do have a buffer blank).", progress_cb)

        _plate_dye   = pd.MultiIndex.from_arrays([qc_df["Plate"], qc_df["Chromatic_Dye"]])
        _use_dye_ref = qc_df["Condition"].isin(_DYE_REF_CONDS)
        qc_df["FI_F0_QC"] = np.where(
            _use_dye_ref,
            qc_df["Fluorescence"] - _plate_dye.map(dye_blank_ref),
            qc_df["Fluorescence"] - _plate_dye.map(buf_blank_ref))

        # Fall back to the dye-blank reference if a plate/dye has no buffer-blank
        # wells of its own (e.g. an assay design with only dye-blank controls) —
        # otherwise Buffer blank/Dye blank/Guest only would silently vanish from
        # the plot instead of just using a less-ideal reference.
        _missing = qc_df["FI_F0_QC"].isna() & ~_use_dye_ref
        if _missing.any():
            _log(f"  WARNING: {_missing.sum()} non-guest row(s) have no buffer-blank "
                 f"reference for their (Plate, Chromatic_Dye) — falling back to the "
                 f"dye-blank reference for these.", progress_cb)
            fallback = qc_df["Fluorescence"] - _plate_dye.map(dye_blank_ref)
            qc_df.loc[_missing, "FI_F0_QC"] = fallback[_missing]

        _still_missing = qc_df["FI_F0_QC"].isna()
        if _still_missing.any():
            _log(f"  WARNING: {_still_missing.sum()} row(s) still have no QC "
                 f"reference at all (no buffer-blank or dye-blank wells found for "
                 f"their Plate/Chromatic_Dye) — they will not appear in QC1/2/3.",
                 progress_cb)

        _med = (qc_df[qc_df["Condition"].isin(["Buffer blank", "Dye blank", "Host + Dye"])]
                .groupby(["Plate", "Condition"])["FI_F0_QC"].median())
        _log(f"  QC1 median FI_F0_QC per (Plate, Condition):\n{_med.to_string()}",
             progress_cb)

        # QC plot 1: control conditions, violin overlays only on the
        # guest-containing conditions (Guest only / Dye + Guest / Competition —
        # hundreds of points each, enough for a meaningful density shape).
        # The 18-well control conditions (Buffer blank / Dye blank / Host + Dye)
        # show points + median only, same as before.
        _VIOLIN_CONDS = ["Guest only", "Dye + Guest", "Competition"]
        fig1 = Figure(figsize=(max(5, 4.5 * len(qc_dyes)), 5))
        axes = fig1.subplots(1, max(len(qc_dyes), 1), squeeze=False)
        for col_i, (ax, dye) in enumerate(zip(axes[0], qc_dyes)):
            sub = qc_df[qc_df["Chromatic_Dye"] == dye]
            order1 = [c for c in COND_ORDER if c in sub["Condition"].values]

            # Y-axis range, per qc1_y_mode:
            #   "all"    — every point visible, plain auto-scaled axis (default).
            #   "tukey"  — Tukey's IQR fence, robust to a handful of extremely
            #     fluorescent guests among hundreds of normal points, which
            #     would otherwise stretch a plain min/max-scaled linear axis
            #     so much that every other condition looks flat. A fixed
            #     percentile (e.g. 2nd-98th) isn't reliable here: a couple of
            #     outlier guests with several concentration replicates each
            #     can exceed 2% of the total point count and still drag the
            #     cutoff up with them.
            #   "manual" — qc1_y_min/qc1_y_max, falling back to the Tukey
            #     bound on whichever side isn't given.
            _all_vals = sub["FI_F0_QC"].dropna()
            _lo = _hi = None
            if qc1_y_mode != "all" and not _all_vals.empty:
                _q1, _q3 = np.nanpercentile(_all_vals, [25, 75])
                _iqr = _q3 - _q1
                _auto_lo = max(_q1 - 1.5 * _iqr, float(_all_vals.min()))
                _auto_hi = min(_q3 + 1.5 * _iqr, float(_all_vals.max()))
                if qc1_y_mode == "manual":
                    _lo = qc1_y_min if qc1_y_min is not None else _auto_lo
                    _hi = qc1_y_max if qc1_y_max is not None else _auto_hi
                else:
                    _lo, _hi = _auto_lo, _auto_hi

            # Fit the violin's own KDE to points within the visible range only
            # (not the full outlier-inflated data) — otherwise its bandwidth
            # scales to the outliers' spread and the visible slice within the
            # tightened Y-limits ends up looking like a flat rectangle instead
            # of a tapered shape.
            violin_sub = sub[sub["Condition"].isin(_VIOLIN_CONDS)]
            if _lo is not None:
                violin_sub = violin_sub[violin_sub["FI_F0_QC"].between(_lo, _hi)]
            if not violin_sub.empty:
                n_coll = len(ax.collections)
                sns.violinplot(data=violin_sub, x="Condition", y="FI_F0_QC", order=order1,
                               ax=ax, inner=None, color="lightgrey", fill=True,
                               linewidth=0.8, saturation=1.0, density_norm="width",
                               bw_adjust=3, cut=2, zorder=1)
                for coll in ax.collections[n_coll:]:
                    coll.set_alpha(0.4)
            sns.stripplot(data=sub, x="Condition", y="FI_F0_QC", order=order1,
                          hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4,
                          alpha=0.7, zorder=5)
            if ax.get_legend():
                ax.get_legend().remove()
            for xi, c in enumerate(order1):
                vals = sub.loc[sub["Condition"] == c, "FI_F0_QC"].dropna()
                if len(vals):
                    ax.plot([xi - 0.3, xi + 0.3],
                            [vals.median(), vals.median()],
                            lw=2.5, color="black", zorder=10)
            # Outlier points are still plotted — just clipped at the edges —
            # and counted so it isn't misleading.
            if _lo is not None and _hi > _lo:
                _pad = 0.08 * (_hi - _lo)
                ax.set_ylim(_lo - _pad, _hi + _pad)
                _n_clipped = int(((_all_vals < _lo) | (_all_vals > _hi)).sum())
                if _n_clipped:
                    ax.text(0.99, 0.02, f"{_n_clipped} pt(s) outside range",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=6, color="grey", style="italic")
            ax.set_title(dye, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("FI − F₀" if col_i == 0 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        fig1.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                    loc="center left", fontsize=8)
        fig1.suptitle("QC — control well fluorescence (FI−F₀)",
                      fontsize=12, fontweight="bold")
        fig1.tight_layout()
        figures.append(("QC: Control Wells", fig1))

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
                order = (sub.groupby("Host")["FI_F0_QC"].median()
                         .sort_values().index.tolist())
                sns.stripplot(data=sub, x="Host", y="FI_F0_QC", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, h in enumerate(order):
                    vals = sub.loc[sub["Host"] == h, "FI_F0_QC"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.3, xi + 0.3],
                                [vals.median(), vals.median()],
                                lw=2.5, color="black", zorder=10)
                ax.axhline(0, color="steelblue", lw=1.5, ls="--", alpha=0.7,
                           label="Dye blank")
                ax.legend(fontsize=7)
                ax.set_title(dye, fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("FI − F₀" if col_i == 0 else "", fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig2.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            fig2.suptitle("QC — Host + Dye per host (FI−F₀)",
                          fontsize=12, fontweight="bold")
            fig2.tight_layout()
            figures.append(("QC: Host + Dye", fig2))

        # QC plot 3: Dye + Guest per guest, with Guest-only overlay
        #
        # Guest-only (diamonds) and Dye+Guest/Competition (circles) points use
        # DIFFERENT zero-references (buf_blank_ref vs dye_blank_ref — see the
        # FI_F0_QC computation above), but share one axis and one "Dye blank"
        # reference line drawn at 0, which is only a valid zero-point for the
        # circles. For diamonds, the true dye-blank-equivalent height is
        # offset by (dye_blank_ref - buf_blank_ref) for that (Plate,
        # Chromatic_Dye) — a plate/dye-specific gap that can be non-trivial
        # (confirmed on real data during the audit that introduced this fix:
        # ~4,500 fluorescence units in one case). Without a second reference
        # line at that offset, a guest can fail the actual PASS/FAIL
        # autofluorescence test (apply_thresholds_ki, which compares
        # Guest-only mean directly against Dye-blank mean, per Plate) while
        # looking unremarkable on this chart, or vice versa. This is a
        # visualisation-only fix — apply_thresholds_ki's statistics are
        # unchanged.
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
                order = (sub.groupby("Guest")["FI_F0_QC"].median()
                         .sort_values(ascending=False).index.tolist())
                # Dye + Guest: filled circles, coloured by plate
                sns.stripplot(data=sub, x="Guest", y="FI_F0_QC", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.2, size=4)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, g in enumerate(order):
                    vals = sub.loc[sub["Guest"] == g, "FI_F0_QC"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.0, xi + 0.35],
                                [vals.median(), vals.median()],
                                lw=2.5, color="black", zorder=10)
                # Guest only: diamond markers, shifted left, same plate colours
                sub_go = (guest_only_qc[guest_only_qc["Chromatic_Dye"] == dye]
                          if has_guest_only else pd.DataFrame())
                _go_offset = None   # dye-blank-equivalent height for diamonds (see below)
                if not sub_go.empty:
                    rng = np.random.default_rng(0)
                    for xi, g in enumerate(order):
                        rows = sub_go[sub_go["Guest"] == g]
                        if rows.empty:
                            continue
                        xs = xi - 0.35 + rng.uniform(-0.1, 0.1, size=len(rows))
                        ax.scatter(xs, rows["FI_F0_QC"].values,
                                   c=[palette[p] for p in rows["Plate"]],
                                   marker="D", s=18, alpha=0.7, zorder=8,
                                   edgecolors="none")
                    # Per-guest median bar for Guest only
                    for xi, g in enumerate(order):
                        vals_go = sub_go.loc[sub_go["Guest"] == g, "FI_F0_QC"].dropna()
                        if len(vals_go):
                            ax.plot([xi - 0.45, xi - 0.05],
                                    [vals_go.median(), vals_go.median()],
                                    lw=2.5, color="darkorange", zorder=10)
                    # Guest-only diamonds use buf_blank_ref as their zero-point
                    # (see FI_F0_QC above), but the actual PASS/FAIL
                    # autofluorescence test compares them against the
                    # dye-blank mean, per Plate — so the dye-blank-equivalent
                    # height for diamonds is offset by (dye_blank_ref -
                    # buf_blank_ref) for their (Plate, Chromatic_Dye), not 0.
                    _plates_go = sorted(sub_go["Plate"].dropna().unique())
                    _offsets = {p: dye_blank_ref[(p, dye)] - buf_blank_ref[(p, dye)]
                                for p in _plates_go
                                if (p, dye) in dye_blank_ref.index and (p, dye) in buf_blank_ref.index}
                    if _offsets:
                        _go_offset = float(np.mean(list(_offsets.values())))
                        if len(set(round(v, 3) for v in _offsets.values())) > 1:
                            _log(f"  NOTE: '{dye}' guest-only reference line is the mean "
                                 f"dye-blank/buffer-blank offset across plate(s) "
                                 f"{_plates_go} ({_offsets}) — these differ per plate, "
                                 f"so the single line shown is an average, not exact for "
                                 f"every plate's diamonds. See per-(Plate,Guest) values "
                                 f"in apply_thresholds_ki's Status/Fail_reason for the "
                                 f"real per-plate test.", progress_cb)
                    _log(f"  '{dye}': Guest-only and Dye+Guest/Competition points use "
                         f"DIFFERENT zero-references (buffer-blank vs dye-blank) and are "
                         f"NOT directly comparable by height without accounting for the "
                         f"gap between the two reference lines shown.", progress_cb)
                # Legend elements
                legend_els = []
                legend_els.append(plt.Line2D([], [], color="black", lw=2.5,
                                             label="Dye+Guest (median)"))
                if has_guest_only and not sub_go.empty:
                    legend_els.append(plt.Line2D([], [], color="darkorange", lw=2.5,
                                                 label="Guest only (median)"))
                    legend_els.append(plt.Line2D([], [], marker="D", color="grey",
                                                 lw=0, markersize=5,
                                                 label="Guest only (pts)"))
                ax.axhline(0, color="steelblue", lw=1.5, ls="--", alpha=0.7)
                legend_els.append(plt.Line2D([], [], color="steelblue", lw=1.5,
                                             ls="--", label="Dye blank"))
                if _go_offset is not None:
                    ax.axhline(_go_offset, color="firebrick", lw=1.5, ls="-.", alpha=0.7)
                    legend_els.append(plt.Line2D([], [], color="firebrick", lw=1.5,
                                                 ls="-.", label="Dye blank (guest-only reference)"))
                ax.legend(handles=legend_els, fontsize=7, loc="upper right")
                ax.set_title(dye, fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("FI − F₀" if col_i == 0 else "", fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig3.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            suffix = " (◆ = guest only)" if has_guest_only else ""
            fig3.suptitle(f"QC — Dye + Guest per guest (FI−F₀){suffix}",
                          fontsize=12, fontweight="bold")
            fig3.tight_layout()
            figures.append(("QC: Dye + Guest", fig3))

        _log(f"Generated {len(figures)} QC figure(s).", progress_cb)
    except Exception as e:
        _log(f"WARNING: QC figure generation failed: {e}", progress_cb)

    return figures


# ── Stage 4: multi-model fitting ──────────────────────────────────────────────

def _aicc(n: int, rss: float, k: int) -> float:
    """AICc for nonlinear regression: residual variance is itself an
    estimated parameter, so the parameter count used in the AIC/AICc
    formulas is K = k+1, not k (k = curve parameters only). GraphPad Curve
    Fitting Guide, "Model diagnostics": "k in the equations above is
    replaced by k+1" for nonlinear regression. Kept in sync with
    pipeline_fda.py's inline AICc computation — NOT inert here, since
    Standard/Wang (k=3) compete against HillSlope (k=4): the extra +1 in K
    changes the relative small-sample correction between models of
    different k, so this can flip Best_Model (verified against real
    replicate data during the audit that introduced this fix)."""
    K = k + 1
    if n <= K + 1:
        return np.inf
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2.0 * K + (2.0 * K * (K + 1)) / (n - K - 1)


def _runs_test_p(residuals: np.ndarray) -> float:
    """Wald-Wolfowitz runs test on residual signs vs. x-order: detects
    systematic (non-random) lack-of-fit — e.g. a hook effect or wrong
    functional form — that R2_adj/CI width alone can miss, and that
    (unlike scipy.stats.normaltest) remains computable below n=8.
    Diagnostic only; mirrors Normality_p's existing gating pattern —
    does NOT affect Status or Confidence."""
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    n1, n2 = (signs > 0).sum(), (signs < 0).sum()
    if n1 == 0 or n2 == 0:
        return np.nan
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    mean_runs = 2*n1*n2/(n1+n2) + 1
    var_runs = (2*n1*n2*(2*n1*n2-n1-n2)) / ((n1+n2)**2 * (n1+n2-1))
    if var_runs <= 0:
        return np.nan
    z = (runs - mean_runs) / np.sqrt(var_runs)
    return float(2 * (1 - norm.cdf(abs(z))))


def _grubbs_mask(data, alpha: float = 0.05,
                 multiplicity_correction: bool = False) -> np.ndarray:
    """Iteratively remove the single most extreme point per pass while it
    tests significant by Grubbs' test (two-sided), at a fixed nominal
    alpha per pass.

    Grubbs' test is, by construction, a single-outlier test — re-applying
    it at the same nominal alpha on every iterative pass (as this function
    always has) does not correct for repeated testing, and can inflate the
    true false-positive rate for outlier removal beyond the nominal alpha
    per curve. This is a known, common simplification, not a
    textbook-exact procedure.

    multiplicity_correction (default False, preserves existing behaviour
    exactly): when True, applies the simplest defensible correction —
    Bonferroni over the number of iterations already run on this group —
    by dividing alpha by the 1-based iteration count before computing
    t_crit each pass. This makes later passes progressively more
    conservative (harder to remove a further point) without changing the
    first-pass behaviour at all.

    At n=3 replicates per concentration (this lab's standard design),
    multiplicity_correction is a mathematical no-op: the `while
    np.sum(mask) > 2` stopping condition means at most one removal pass
    can ever run, so there is no second iteration for the
    Bonferroni-over-iterations correction to apply to. Confirmed
    empirically on a real batch: identical Ki, R2_adj, AICc, Best_Model,
    n_removed, and Status with the flag on vs. off, exact match, zero
    differing rows. This only becomes non-trivial if replicate count is
    ever raised to n>=4.
    """
    data = np.asarray(data, dtype=float)
    mask = np.ones(len(data), dtype=bool)
    iteration = 0
    while np.sum(mask) > 2:
        iteration += 1
        vals = data[mask]
        std  = np.std(vals, ddof=1)
        if std == 0:
            break
        abs_diffs  = np.abs(vals - np.mean(vals))
        idx_local  = int(np.argmax(abs_diffs))
        idx_global = np.where(mask)[0][idx_local]
        G      = abs_diffs[idx_local] / std
        N      = len(vals)
        eff_alpha = alpha / iteration if multiplicity_correction else alpha
        t_crit = t.ppf(1 - eff_alpha / (2 * N), N - 2)
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


# NOTE (Model_Basis = "approximate_cp_shift"): this reuses the Standard
# model's Cheng-Prusoff EC50 shift (logEC50 = logKi + log10(1+D/Kd)), which is
# only exactly derived for the Hill=1, single-site, non-depleting case. Ki
# from this model is an approximation whenever the fitted HillSlope departs
# from 1.
def _comp_hill(DyeConc_uM, DyeKd_uM):
    def model(x, Top, Bottom, logKi, HillSlope):
        logEC50 = logKi + np.log10(1.0 + DyeConc_uM / DyeKd_uM)
        return Bottom + (Top - Bottom) / (1.0 + 10.0 ** (HillSlope * (logEC50 - x)))
    return model


def _wang_free_receptor(A_tot, Ka, B_tot, Kb, P_tot):
    """Free receptor concentration — exact solution of the mass-action cubic
    for two ligands (A, B) competing for one site on receptor P, all three
    allowed to deplete. Derived from:
        P_tot = r + r*A_tot/(Ka+r) + r*B_tot/(Kb+r)
    which rearranges to  r^3 + a*r^2 + b*r + c = 0  with the coefficients
    below — algebraically identical to Wang, Z.-X. "An exact mathematical
    expression for describing competitive binding of two different ligands
    to a protein molecule." FEBS Lett. 1995;360(2):111-114 (solved here via
    the standard trigonometric depressed-cubic formula rather than Wang's
    own variable substitutions, but the same unique physical root 0<=r<=P_tot).
    """
    a = Ka + Kb + A_tot + B_tot - P_tot
    b = Kb * (A_tot - P_tot) + Ka * (B_tot - P_tot) + Ka * Kb
    c = -Ka * Kb * P_tot

    m = np.maximum(a**2 - 3.0 * b, 1e-30)          # a^2 - 3b, clamped > 0
    num = -2.0 * a**3 + 9.0 * a * b - 27.0 * c
    den = 2.0 * m**1.5
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    r = (2.0 * np.sqrt(m) * np.cos(theta / 3.0) - a) / 3.0
    return np.clip(r, 0.0, P_tot)


def _wang_cubic(DyeConc_uM, DyeKd_uM, HostConc_uM):
    """Wang (1995) exact competitive-binding model: dye (A) and guest (B)
    both compete for the host (P) site, and all three species are allowed
    to deplete. logKi is the guest's true (not apparent) dissociation
    constant. See _wang_free_receptor for the underlying math/reference."""
    A_tot = DyeConc_uM
    Ka    = DyeKd_uM
    P_tot = HostConc_uM

    def _bound_dye(B_tot, Kb):
        r = _wang_free_receptor(A_tot, Ka, B_tot, Kb, P_tot)
        return A_tot * r / (Ka + r)

    def model(x, Top, Bottom, logKi):
        Kb    = 10.0 ** logKi
        B_tot = 10.0 ** x   # Guest concentration in µM (x = log10[Guest/µM])
        RA0   = _bound_dye(0.0, Kb)          # bound dye with no competitor
        RA    = _bound_dye(B_tot, Kb)
        frac_displaced = np.clip(1.0 - RA / RA0, 0.0, 1.0)
        return Top - (Top - Bottom) * frac_displaced
    return model


def fit_curves_ki(fi_df: pd.DataFrame, hot_df: pd.DataFrame,
                  grubbs_alpha: float = 0.05,
                  min_n_for_grubbs: int = 3,
                  grubbs_multiplicity_correction: bool = False,
                  wang_gate: float = WANG_GATE_DEFAULT,
                  model_preference: str = "auto",
                  hill_flag_lo: float = HILL_FLAG_LO_DEFAULT,
                  hill_flag_hi: float = HILL_FLAG_HI_DEFAULT,
                  progress_cb: ProgressCb = None) -> tuple[list, list]:
    """model_preference: "auto" (default) tries every applicable model and
    picks the best by AICc. Or force a single model: "standard", "hillslope",
    "wang" — bypasses the Wang gate, since the user is explicitly asking
    for it.

    min_n_for_grubbs (default 3, unchanged behaviour): minimum replicate
    count at one concentration before a within-replicate Grubbs test is
    run at all. Standard guidance treats Grubbs as unreliable below n≈6-7;
    the default of 3 (the minimum needed to compute a sample SD at all) is
    kept for continuity with prior runs, but every curve where any
    concentration was tested at exactly 3 replicates is flagged via the
    output column Grubbs_applied_at_n3 so this can be audited/filtered
    later without re-deriving it. Raise this (e.g. to 6 or 7) to skip the
    test entirely at low replicate counts instead.

    grubbs_multiplicity_correction (default False, unchanged behaviour):
    see _grubbs_mask's docstring — applies a Bonferroni correction over
    the iterative removal passes already run on a given concentration's
    replicates, instead of testing every pass at the same nominal
    grubbs_alpha. At n=3 replicates (this lab's standard design) this
    flag is a mathematical no-op — see _grubbs_mask's docstring.

    wang_gate: Wang (exact, depletion-aware) is only added to the Auto
    candidate set when Standard's Ki estimate is < wang_gate x [Host] — i.e.
    only attempted in the tight-binding regime where the non-depleting
    Standard/Cheng-Prusoff assumption is likely to break down. An explicit
    "wang" request bypasses this gate (still needs a Host concentration).
    NOTE: this gate is circular — it decides whether to try the
    depletion-aware model using a Ki estimate (Standard's) that is itself
    potentially depletion-biased in exactly the regime the gate is trying to
    detect. Not corrected here (affects absolute Ki magnitude at the gate's
    margin, not which guests rank tightest within a screen — see module
    docstring); left as a documented limitation, not a code change.

    Wang_gate_margin (= std_Ki / (wang_gate x HostConc_uM), normalized so
    the gate threshold is 1.0) is persisted per curve so the gate decision
    can be reviewed post-hoc. When the winning Best_Model is "Wang" and
    this margin falls within WANG_GATE_MARGIN_BAND, a QC warning is logged
    (informational only — does not affect Status or Best_Model) noting
    that the choice to try Wang was itself close to the boundary.

    hill_flag_lo/hi: HillSlope fits with a fitted Hill slope outside
    [hill_flag_lo, hill_flag_hi] get Hill_flag=True in the output — a triage
    signal for possible multi-site binding, aggregation/nonspecific
    inhibition, or a fitting artifact (recommend orthogonal follow-up), not
    something to mathematically correct away."""
    _forced = model_preference if model_preference and model_preference != "auto" else None

    _has_buffer = "Buffer" in fi_df.columns
    _grp_cols   = (["Host", "Dye", "Guest", "Host_Concentration"]
                   + (["Buffer"] if _has_buffer else []))
    groups = (fi_df.dropna(subset=["Host", "Dye", "Guest", "Guest_Concentration", "FI-F0"])
              [_grp_cols].drop_duplicates())
    total = len(groups)

    fit_results, plot_data = [], []

    for i, (_, row) in enumerate(groups.iterrows()):
        host, dye, guest = row["Host"], row["Dye"], row["Guest"]
        host_conc   = row["Host_Concentration"]
        HostConc_uM = float(host_conc) if pd.notna(host_conc) else np.nan
        buffer = row["Buffer"] if _has_buffer else None
        # Host concentration is folded into the label/key everywhere below —
        # the same Host|Dye|Guest triple run at several host concentrations
        # in one batch must produce distinct curves, not one merged fit.
        _hc_str = f" ({_fmt_uM(HostConc_uM)} µM host)" if not np.isnan(HostConc_uM) else ""
        _lbl = f"{host} | {dye} | {guest}{_hc_str}" + (f" [{buffer}]" if buffer else "")
        _log(f"  Fitting [{i+1}/{total}]: {_lbl}", progress_cb)

        kd_row = hot_df.loc[(hot_df["Host"] == host) & (hot_df["Dye"] == dye)]
        if kd_row.empty:
            _log(f"    SKIP — no Kd entry for {host}-{dye}", progress_cb)
            continue
        DyeConc_uM = float(kd_row["DyeConc_uM"].iloc[0])
        DyeKd_uM   = float(kd_row["DyeKd_uM"].iloc[0])
        if DyeKd_uM <= 0 or np.isnan(DyeKd_uM):
            _log(f"    SKIP — invalid DyeKd_uM={DyeKd_uM} for {host}-{dye}", progress_cb)
            continue

        mask = ((fi_df["Host"] == host) & (fi_df["Dye"] == dye)
                 & (fi_df["Guest"] == guest))
        mask &= (fi_df["Host_Concentration"].isna() if np.isnan(HostConc_uM)
                  else fi_df["Host_Concentration"] == host_conc)
        if _has_buffer and "Buffer" in fi_df.columns:
            mask &= fi_df["Buffer"] == buffer
        grp   = fi_df[mask]
        plate = (grp["Plate"].dropna().iloc[0]
                 if "Plate" in grp.columns and not grp["Plate"].dropna().empty else "")

        conc     = pd.to_numeric(grp["Guest_Concentration"], errors="coerce")
        mask_pos = conc > 0

        # Failed-dispense wells (from an Echo exceptions file, see merge_ki)
        # are excluded from fitting entirely — plotted as crosses instead,
        # never fed to Grubbs or curve_fit.
        if "Dispense_Failed" in grp.columns:
            failed_col = grp["Dispense_Failed"].fillna(False).astype(bool)
        else:
            failed_col = pd.Series(False, index=grp.index)
        mask_failed = mask_pos & failed_col
        x_failed = np.log10(conc[mask_failed].values) if mask_failed.any() else np.array([])
        y_failed = (pd.to_numeric(grp.loc[mask_failed, "FI-F0"], errors="coerce").values
                    if mask_failed.any() else np.array([]))

        mask_use = mask_pos & ~failed_col
        x_all    = np.log10(conc[mask_use].values)
        y_all    = pd.to_numeric(grp.loc[mask_use, "FI-F0"], errors="coerce").values
        ok       = ~(np.isnan(x_all) | np.isnan(y_all))
        x_all, y_all = x_all[ok], y_all[ok]

        if len(x_all) < 5 or len(np.unique(x_all)) < 3:
            _log(f"    SKIP — insufficient data points", progress_cb)
            continue

        keep = np.ones(len(y_all), dtype=bool)
        n_grubbs_tests_run = 0
        grubbs_applied_at_n3 = False
        for xv in np.unique(x_all):
            idx = np.where(x_all == xv)[0]
            if len(idx) >= min_n_for_grubbs:
                keep[idx] = _grubbs_mask(y_all[idx], alpha=grubbs_alpha,
                                         multiplicity_correction=grubbs_multiplicity_correction)
                n_grubbs_tests_run += 1
                if len(idx) == 3:
                    grubbs_applied_at_n3 = True

        x_cl  = x_all[keep]
        y_cl  = y_all[keep]
        x_out = x_all[~keep]
        y_out = y_all[~keep]

        if len(y_cl) < 5:
            _log(f"    SKIP — too few points after Grubbs", progress_cb)
            continue

        y_min = float(np.min(y_cl))
        y_max = float(np.max(y_cl))
        y_rng = max(abs(y_max - y_min), 1.0)
        x_med = float(np.median(x_cl))
        x_min_cl = float(np.min(x_cl))
        x_max_cl = float(np.max(x_cl))

        bounds_std = ([y_min - 2*y_rng, y_min - 2*y_rng, -14],
                      [y_max + 2*y_rng, y_max + 2*y_rng,  8])

        # Gate Wang on Standard's Ki estimate (reused from main model loop)
        std_fn  = _comp_standard(DyeConc_uM, DyeKd_uM)
        std_Ki  = np.nan
        std_fit = None
        if not np.isnan(HostConc_uM):
            try:
                popt_s, pcov_s = curve_fit(std_fn, x_cl, y_cl,
                                           p0=[y_max, y_min, x_med],
                                           bounds=bounds_std, maxfev=20000)
                std_Ki  = 10.0 ** float(popt_s[2])
                std_fit = (popt_s, pcov_s)
            except Exception:
                pass

        use_wang = (not np.isnan(HostConc_uM) and not np.isnan(std_Ki)
                    and std_Ki < wang_gate * HostConc_uM)

        models = []
        if _forced in (None, "standard"):
            models.append(
                {"name": "Standard",  "fn": std_fn, "k": 3,
                 "p0": [y_max, y_min, x_med], "bounds": bounds_std,
                 "prefit": std_fit})
        if _forced in (None, "hillslope"):
            models.append(
                {"name": "HillSlope", "fn": _comp_hill(DyeConc_uM, DyeKd_uM),     "k": 4,
                 "p0": [y_max, y_min, x_med, 1.0],
                 "bounds": ([y_min-2*y_rng, y_min-2*y_rng, -14, 0.1],
                            [y_max+2*y_rng, y_max+2*y_rng,  8, 10.0])})
        # Auto mode gates Wang on the Standard Ki estimate; an explicit
        # "wang" request bypasses the gate (still needs a Host conc.).
        if _forced == "wang" and np.isnan(HostConc_uM):
            _log(f"    SKIP Wang — Host concentration unknown for {host}-{dye}",
                 progress_cb)
        elif _forced == "wang" or (_forced is None and use_wang):
            for seed in [x_min_cl, x_med - 1.5, x_med - 0.5, x_med, x_med + 0.5]:
                models.append({"name": "Wang",
                                "fn": _wang_cubic(DyeConc_uM, DyeKd_uM, HostConc_uM), "k": 3,
                                "p0": [y_max, y_min, seed], "bounds": bounds_std})

        if not models:
            _log(f"    SKIP — no applicable model for model_preference="
                 f"'{model_preference}' on {host}-{dye}", progress_cb)
            continue

        best = None
        best_by_name = {}
        for m in models:
            n_pts = len(y_cl)
            # Need at least k+3 points: k for parameters + 1 residual df
            # + 1 so R2_adj is a genuinely adjusted value + 1 more so
            # AICc's small-sample correction (K=k+1 inside _aicc) is
            # defined (mirrors pipeline_fda.fit_curves's "n <= k + 2:
            # continue" gate, kept in sync with the same AICc fix).
            if n_pts <= m["k"] + 2:
                continue
            prefit = m.get("prefit")
            if prefit is not None:
                popt, pcov = prefit
            else:
                try:
                    popt, pcov = curve_fit(m["fn"], x_cl, y_cl, p0=m["p0"],
                                           bounds=m["bounds"], maxfev=20000)
                except (RuntimeError, ValueError):
                    continue

            if m["name"] == "Wang":
                # Sanity rejections ported from fda_launcher._pooled_ki_fit
                # (~lines 652-667), which already applies them but was
                # never mirrored back here. A fit that converges onto the
                # logKi bound wall (-14 or 8, i.e. Ki ~1e-14 or ~1e8 uM) is
                # a numerical artifact of a degenerate near-vertical curve,
                # not a real near-zero/near-infinite Ki — reject it rather
                # than report a physically implausible value.
                _lo_bound, _hi_bound = m["bounds"][0][2], m["bounds"][1][2]
                if popt[2] <= _lo_bound + 0.5 or popt[2] >= _hi_bound - 0.5:
                    continue
                # Standard pharmacological QC: a logKi fitted many decades
                # outside the tested concentration range isn't actually
                # constrained by the data. Reject any logKi more than 3 log
                # units beyond [min, max] of the concentrations actually
                # tested (x_cl).
                if popt[2] < x_cl.min() - 3.0 or popt[2] > x_cl.max() + 3.0:
                    continue

            y_fit  = m["fn"](x_cl, *popt)
            rss    = float(np.sum((y_cl - y_fit)**2))
            ss_tot = float(np.sum((y_cl - np.mean(y_cl))**2))
            r2     = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
            # n_pts > m["k"] + 1 is guaranteed by the gate above, so this is
            # always a genuinely adjusted R2 (never silently falls back to r2).
            r2_adj = 1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - m["k"] - 1)
            aic_val = _aicc(n_pts, rss, m["k"])

            if m["name"] not in best_by_name or aic_val < best_by_name[m["name"]]["aic"]:
                best_by_name[m["name"]] = {"name": m["name"], "fn": m["fn"], "popt": popt,
                                            "pcov": pcov, "r2_adj": r2_adj, "aic": aic_val,
                                            "k": m["k"]}

            if best is None or aic_val < best["aic"]:
                best = {"name": m["name"], "fn": m["fn"], "popt": popt, "pcov": pcov,
                        "r2_adj": r2_adj, "aic": aic_val, "k": m["k"]}

        if best is None:
            _log(f"    All models failed", progress_cb)
            continue

        # Wang-gate boundary-proximity flag (additive, does not change
        # Status/Best_Model): the gate that decided whether to try Wang
        # used an estimate (std_Ki, from the simpler Standard model) that
        # is itself potentially depletion-biased in exactly the regime the
        # gate exists to detect (see docstring above). Persisting the
        # margin, and flagging when it's close to 1.0 (the gate
        # threshold), surfaces curves where that circularity is most
        # likely to matter, without re-running anything.
        wang_gate_margin = (std_Ki / (wang_gate * HostConc_uM)
                            if not np.isnan(std_Ki) and not np.isnan(HostConc_uM)
                            and wang_gate * HostConc_uM > 0 else np.nan)
        if best["name"] == "Wang" and not np.isnan(wang_gate_margin):
            _band_lo, _band_hi = WANG_GATE_MARGIN_BAND
            if _band_lo <= wang_gate_margin <= _band_hi:
                _log(f"    QC WARNING: Wang model won for {host}-{dye}-{guest} "
                     f"with Wang_gate_margin={wang_gate_margin:.3g} — within "
                     f"{_band_lo}-{_band_hi} of the gate threshold (1.0), so "
                     f"the decision to try this model was close to the "
                     f"boundary (informational only — does not affect "
                     f"Status).", progress_cb)

        # Model-selection ambiguity (mirrors pipeline_fda.fit_curves's
        # Sign_ambiguous): if the runner-up model's AICc is within
        # DELTA_AICC_AMBIGUOUS of the winner's, AICc has no strong
        # preference between them (Burnham & Anderson 2002) — e.g. Standard
        # vs. HillSlope picked apart by fitting noise, not real curve shape.
        # Compared per-model-name best (best_by_name), not per-candidate:
        # Wang fits 5 seeds, and comparing against the runner-up seed
        # (rather than the runner-up *model*) produced false ties whenever
        # Wang won, since multiple seeds often converge to the same optimum.
        sorted_aics = sorted(v["aic"] for v in best_by_name.values())
        second_best_aic = sorted_aics[1] if len(sorted_aics) >= 2 else np.nan
        delta_aicc_vs_runnerup = (float(second_best_aic - best["aic"])
                                   if not np.isnan(second_best_aic) else np.nan)
        model_ambiguous = (not np.isnan(delta_aicc_vs_runnerup)
                            and delta_aicc_vs_runnerup < DELTA_AICC_AMBIGUOUS)

        logKi = float(best["popt"][2])
        try:
            kd_var    = float(best["pcov"][2, 2])
            logKi_err = float(np.sqrt(kd_var)) if kd_var >= 0 else np.nan
        except Exception:
            logKi_err = np.nan
        Ki     = 10.0 ** logKi
        # Ki_err_uM is a symmetric linear approximation (delta method); for
        # the correct asymmetric confidence interval use
        # Ki_CI_low_uM/Ki_CI_high_uM.
        Ki_err = Ki * np.log(10) * logKi_err if not np.isnan(logKi_err) else np.nan

        # Ki_CI_low_uM/Ki_CI_high_uM: back-transform of the t-based CI on
        # logKi, mirroring pipeline_fda.fit_curves's Kd_95CI dof convention
        # (n - k).
        dof_ki = len(y_cl) - best["k"]
        t_crit_ki = t.ppf(0.975, dof_ki)
        if not np.isnan(logKi_err):
            ki_ci_low_uM  = 10.0 ** (logKi - t_crit_ki * logKi_err)
            ki_ci_high_uM = 10.0 ** (logKi + t_crit_ki * logKi_err)
        else:
            ki_ci_low_uM  = np.nan
            ki_ci_high_uM = np.nan

        resid = y_cl - best["fn"](x_cl, *best["popt"])
        try:
            normal_p = normaltest(resid).pvalue if len(resid) >= 8 else np.nan
        except Exception:
            normal_p = np.nan
        try:
            runs_test_p = _runs_test_p(resid[np.argsort(x_cl)])
        except Exception:
            runs_test_p = np.nan

        extra = {}
        if best["name"] == "HillSlope":
            hs = float(best["popt"][3])
            extra = {"HillSlope": hs,
                     "Hill_flag": bool(hs < hill_flag_lo or hs > hill_flag_hi)}

        # EC50/IC50 (item 7): the raw fitted midpoint, exposed alongside Ki
        # because it needs NO Cheng-Prusoff correction — within one Host/Dye
        # pair the shift log10(1+[Dye]/DyeKd) is a constant added to every
        # guest's logKi, so ranking guests by Ki_uM here is mathematically
        # equivalent to ranking by raw EC50/IC50. Recommended ranking metric
        # for triage within a single Host/Dye screen (see module docstring
        # for the caveat on comparing across different dyes/tracers).
        # Not meaningful for Wang, which fits the true (already-exact) Ki
        # directly with no Cheng-Prusoff shift involved.
        cp_shift = np.log10(1.0 + DyeConc_uM / DyeKd_uM)
        ec50_uM  = np.nan if best["name"] == "Wang" else 10.0 ** (logKi + cp_shift)

        # Guest concentration range actually used in the fit (post-Grubbs),
        # for the Ki range-sanity check in apply_thresholds_ki — mirrors
        # pipeline_fda.fit_curves's Host_Conc_min/Host_Conc_max. x_cl here
        # is already log10 of strictly-positive concentrations only (see
        # the mask_pos filter above), so no extra positive-only filtering
        # is needed before converting back to linear scale.
        guest_conc_min = float(10.0 ** np.min(x_cl))
        guest_conc_max = float(10.0 ** np.max(x_cl))

        _fit_row = {
            "Host": host, "Dye": dye, "Guest": guest, "Plate": plate,
            "Best_Model":  best["name"],
            "Model_Basis": ("exact" if best["name"] in ("Standard", "Wang")
                             else "approximate_cp_shift"),
            "logKi":       logKi,
            "logKi_err":   logKi_err,
            "Ki_uM":       Ki,
            "Ki_err_uM":   Ki_err,
            "Ki_CI_low_uM":  ki_ci_low_uM,
            "Ki_CI_high_uM": ki_ci_high_uM,
            "EC50_uM":     ec50_uM,
            "cp_shift":    cp_shift,
            "R2_adj":      best["r2_adj"],
            "AICc":        best["aic"],
            "Model_ambiguous":         model_ambiguous,
            "Delta_AICc_vs_runnerup":  delta_aicc_vs_runnerup,
            "Wang_gate_margin": wang_gate_margin,
            "Normality_p": normal_p,
            "Runs_test_p": runs_test_p,
            "n_total":     len(y_all),
            "n_removed":   int(np.sum(~keep)),
            "n_grubbs_tests_run": n_grubbs_tests_run,
            "Grubbs_applied_at_n3": grubbs_applied_at_n3,
            "n_dispense_failed": len(x_failed),
            "DyeConc_uM":  DyeConc_uM,
            "DyeKd_uM":    DyeKd_uM,
            "HostConc_uM": HostConc_uM,
            "Top_fit":     float(best["popt"][0]),
            "Bottom_fit":  float(best["popt"][1]),
            "Guest_Conc_min": guest_conc_min,
            "Guest_Conc_max": guest_conc_max,
            **extra,
        }
        if buffer is not None:
            _fit_row["Buffer"] = buffer
        fit_results.append(_fit_row)

        plot_data.append({
            "host": host, "dye": dye, "guest": guest, "plate": plate,
            "host_conc":  HostConc_uM,
            "buffer":     buffer,
            "x_cleaned":  x_cl,  "y_cleaned":  y_cl,
            "x_outliers": x_out, "y_outliers": y_out,
            "x_failed":   x_failed, "y_failed": y_failed,
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
                        r2_threshold: float = PASS_R2_DEFAULT_KI,
                        fi_df: pd.DataFrame = None,
                        autofl_factor: float = GUEST_AUTOFL_FACTOR,
                        autofl_mode: str = "stat",
                        autofl_z: float = GUEST_AUTOFL_Z_DEFAULT,
                        logki_err_threshold: float = LOGKI_ERR_THRESHOLD_DEFAULT,
                        normality_threshold: float = NORMALITY_P_THRESHOLD_DEFAULT,
                        runs_test_p_threshold: float = RUNS_TEST_P_THRESHOLD_DEFAULT,
                        ki_range_lo: float = KI_RANGE_FACTOR_LO_DEFAULT,
                        ki_range_hi: float = KI_RANGE_FACTOR_HI_DEFAULT) -> pd.DataFrame:
    """
    Adds Status, Fail_reason, and Confidence to a copy of fit_results
    DataFrame.

    Screening/triage use case: false hits and rank-order errors are costly
    (wasted follow-up), but imprecise absolute Ki magnitude is not — a real
    hit gets re-measured properly downstream anyway. So Status (PASS/FAIL)
    is driven only by R2_adj, Ki not NaN, Ki range sanity (fitted Ki must
    fall within [ki_range_lo, ki_range_hi] x the tested guest concentration
    range — mirrors pipeline_fda.apply_thresholds's Kd range check), and
    guest autofluorescence (a genuine false-hit source); a wide Ki
    confidence interval does NOT fail Status — it instead sets
    Confidence="Low — wide CI, recommend retest" (logKi_err >
    logki_err_threshold) alongside the existing Status, so a
    real-but-imprecise hit still surfaces for follow-up instead of being
    silently dropped.

    Model_ambiguous (set in fit_curves_ki, mirroring pipeline_fda's
    Sign_ambiguous) is surfaced the same additive way: a Fail_reason note
    when the winning model's AICc beats the runner-up by less than
    DELTA_AICC_AMBIGUOUS — does not affect Status or Confidence.

    autofl_mode: "stat" (default) flags guest autofluorescence via a
    textbook two-sample z-test against the dye-blank mean (same
    Plate/Dye/Buffer): guest_mean - blank_mean > z * sqrt(guest_sem**2 +
    blank_sem**2). "ratio" falls back to the older fixed-ratio
    median-vs-mean comparison (guest-only median > autofl_factor × blank
    mean) for continuity with prior runs.

    Normality_p (D'Agostino-Pearson test on residuals) is unreliable at the
    small n typical of these curves, so it is diagnostic only: curves with
    Normality_p < normality_threshold get a Fail_reason note but are NOT
    failed on it alone (does not affect Status).

    Runs_test_p (Wald-Wolfowitz runs test on residual signs vs. x-order,
    set in fit_curves_ki) is surfaced the same additive way: curves with
    Runs_test_p < runs_test_p_threshold get a Fail_reason note but are NOT
    failed on it alone (does not affect Status or Confidence).
    """
    df = pd.DataFrame(fit_results).copy()
    if df.empty:
        return df

    fail_r2   = df["R2_adj"] < r2_threshold
    fail_ki   = df["Ki_uM"].isna()
    fail_ki_lo = df["Ki_uM"] < ki_range_lo * df["Guest_Conc_min"]
    fail_ki_hi = df["Ki_uM"] > ki_range_hi * df["Guest_Conc_max"]

    # Confidence (Tier 2): additive, does NOT gate Status — a real hit with a
    # wide CI should still surface for follow-up, just flagged for retest.
    low_confidence = pd.Series(False, index=df.index)
    if "logKi_err" in df.columns and logki_err_threshold is not None:
        low_confidence = df["logKi_err"] > logki_err_threshold
    df["Confidence"] = "High"
    df.loc[low_confidence, "Confidence"] = "Low — wide CI, recommend retest"

    # Guest autofluorescence gate: flag curves whose guest fluoresces above
    # background IN THE SAME PLATE (and Buffer, if used).
    # Grouping must include Plate: guest-only wells carry no Dye tag (nothing
    # was dispensed there), so without Plate a guest tested across several
    # plates/dyes gets one median blended across unrelated fluorophores —
    # comparing a channel's own guest-only reading against a different
    # channel's blank level, which can both over- and under-flag curves.
    fail_autofl = pd.Series(False, index=df.index)
    autofl_not_assessed = pd.Series(False, index=df.index)
    if fi_df is not None and (autofl_factor is not None or autofl_z is not None):
        is_blank = fi_df["Compound ID"].fillna("").str.strip().str.lower() == "blank"
        go_mask  = (fi_df["Dye"].isna() & fi_df["Host"].isna() &
                    fi_df["Guest"].notna() & ~is_blank)
        db_mask  = is_blank & fi_df["Dye"].notna()
        _has_buf_fi = "Buffer" in fi_df.columns and "Buffer" in df.columns
        go_grp  = ["Plate", "Guest"] + (["Buffer"] if _has_buf_fi else [])
        db_grp  = ["Plate", "Dye"]   + (["Buffer"] if _has_buf_fi else [])

        if autofl_mode == "ratio":
            go_med  = fi_df[go_mask].groupby(go_grp)["Fluorescence"].median().to_dict()
            db_mean = fi_df[db_mask].groupby(db_grp)["Fluorescence"].mean().to_dict()
            for idx, row in df.iterrows():
                go_key = (row["Plate"], row["Guest"]) + ((row["Buffer"],) if _has_buf_fi else ())
                db_key = (row["Plate"], row["Dye"])   + ((row["Buffer"],) if _has_buf_fi else ())
                if go_key in go_med and db_key in db_mean:
                    if go_med[go_key] > autofl_factor * db_mean[db_key]:
                        fail_autofl.at[idx] = True
        else:
            # Statistical (default): two-sample Welch's t-test on the pooled
            # SE (sqrt(sem1^2+sem2^2)), not a non-overlapping-error-bar
            # heuristic. Uses mean vs. mean (consistent central tendency)
            # rather than a bare median-vs-mean ratio. sem is NaN only when
            # a group has a single well — that group's uncertainty must NOT
            # be silently zeroed out (doing so shrinks pooled_sem and makes
            # a singleton-well comparison the *easiest* to fail, the
            # opposite of conservative). Instead, skip the test for that
            # combo and leave a diagnostic note; when both groups have >=2
            # wells, use a Welch t critical value (dof = min(count)-1, the
            # conservative choice) instead of a fixed z, which converges to
            # the same z-based critical value at large replicate counts but
            # is more conservative at the small counts typical here.
            go_stats = (fi_df[go_mask].groupby(go_grp)["Fluorescence"]
                        .agg(["mean", "sem", "count"]))
            db_stats = (fi_df[db_mask].groupby(db_grp)["Fluorescence"]
                        .agg(["mean", "sem", "count"]))
            p_crit = float(norm.cdf(autofl_z))
            for idx, row in df.iterrows():
                go_key = (row["Plate"], row["Guest"]) + ((row["Buffer"],) if _has_buf_fi else ())
                db_key = (row["Plate"], row["Dye"])   + ((row["Buffer"],) if _has_buf_fi else ())
                if go_key in go_stats.index and db_key in db_stats.index:
                    go_mean, go_sem, go_n = go_stats.loc[go_key, ["mean", "sem", "count"]]
                    db_mean, db_sem, db_n = db_stats.loc[db_key, ["mean", "sem", "count"]]
                    if go_n < 2 or db_n < 2:
                        autofl_not_assessed.at[idx] = True
                        continue
                    dof = min(go_n, db_n) - 1
                    t_crit = float(t.ppf(p_crit, dof))
                    pooled_sem = np.sqrt(go_sem**2 + db_sem**2)
                    if (go_mean - db_mean) > t_crit * pooled_sem:
                        fail_autofl.at[idx] = True

    fail_normality = pd.Series(False, index=df.index)
    if "Normality_p" in df.columns and normality_threshold is not None:
        fail_normality = df["Normality_p"] < normality_threshold

    fail_runs_test = pd.Series(False, index=df.index)
    if "Runs_test_p" in df.columns and runs_test_p_threshold is not None:
        fail_runs_test = df["Runs_test_p"] < runs_test_p_threshold  # doesn't touch fail_any

    fail_any = fail_r2 | fail_ki | fail_ki_lo | fail_ki_hi | fail_autofl
    df["Status"]      = "PASS"
    df.loc[fail_any, "Status"] = "FAIL"
    df["Fail_reason"] = ""
    df.loc[fail_r2,   "Fail_reason"] += df.loc[fail_r2, "R2_adj"].apply(
        lambda v: f"R²_adj={v:.3f} < {r2_threshold}; ")
    df.loc[fail_ki,   "Fail_reason"] += "Ki is NaN; "
    df.loc[fail_ki_lo, "Fail_reason"] += df.loc[fail_ki_lo].apply(
        lambda r: f"Ki={r['Ki_uM']:.3g} < {ki_range_lo}×[Guest]_min={r['Guest_Conc_min']:.3g}; ", axis=1)
    df.loc[fail_ki_hi, "Fail_reason"] += df.loc[fail_ki_hi].apply(
        lambda r: f"Ki={r['Ki_uM']:.3g} > {ki_range_hi}×[Guest]_max={r['Guest_Conc_max']:.3g}; ", axis=1)
    df.loc[fail_autofl, "Fail_reason"] += "Guest autofluorescence; "
    df.loc[autofl_not_assessed, "Fail_reason"] += \
        "Guest autofluorescence not assessed — n=1 replicate well(s); "
    df.loc[fail_normality, "Fail_reason"] += df.loc[fail_normality, "Normality_p"].apply(
        lambda v: f"Residuals non-normal (p={v:.3f}, diagnostic only); ")
    df.loc[fail_runs_test, "Fail_reason"] += df.loc[fail_runs_test, "Runs_test_p"].apply(
        lambda v: f"Non-random residual pattern (runs test p={v:.4f}, diagnostic only); ")
    df.loc[low_confidence, "Fail_reason"] += df.loc[low_confidence, "logKi_err"].apply(
        lambda v: f"Low confidence (logKi_err={v:.3f} > {logki_err_threshold}); ")
    if "Model_ambiguous" in df.columns:
        _amb = df["Model_ambiguous"].fillna(False).astype(bool)
        df.loc[_amb, "Fail_reason"] += df.loc[_amb, "Delta_AICc_vs_runnerup"].apply(
            lambda v: f"Model selection ambiguous (ΔAICc={v:.2f} vs runner-up); ")
    df["Fail_reason"] = df["Fail_reason"].str.rstrip("; ")

    _has_buf   = "Buffer" in df.columns
    _key_parts = [df["Host"], df["Dye"], df["Guest"]]
    if _has_buf:
        _key_parts.append(df["Buffer"])
    _key_parts.append(df["HostConc_uM"].apply(_hc_key))
    lookup = dict(zip(zip(*_key_parts), df["Status"]))
    for e in plot_data:
        _key = (e["host"], e["dye"], e["guest"])
        if _has_buf:
            _key = _key + (e.get("buffer"),)
        _key = _key + (_hc_key(e.get("host_conc")),)
        e["status"] = lookup.get(_key, "FAIL")

    return df


# ── plot rendering ────────────────────────────────────────────────────────────

def render_plot_ki(ax, entry: dict, show_status_color: bool = True,
                   color_fit: str = PLOT_COLOR_FIT,
                   color_data: str = PLOT_COLOR_DATA,
                   color_resid: str = None,
                   ax_resid=None,
                   title_fontsize: float = 12,
                   title_fontfamily: str = "sans-serif",
                   title_fontweight: str = "normal",
                   title_fontstyle: str = "normal",
                   axis_fontsize: float = 14,
                   tick_fontsize: float = 12,
                   stats_in_title: bool = True,
                   normalise_y: bool = False,
                   sci_notation_y: bool = False,
                   lw_fit: float = 2.0,
                   ms_data: float = 4.0,
                   lw_errorbar: float = 1.5,
                   ms_resid: float = 2.5,
                   x_min: float = None,
                   x_max: float = None,
                   x_tick_step: float = None,
                   show_excluded: bool = True):
    x_cl     = entry["x_cleaned"]
    y_cl     = entry["y_cleaned"]
    x_out    = entry["x_outliers"]
    y_out    = entry["y_outliers"]
    x_failed = entry.get("x_failed", np.array([]))
    y_failed = entry.get("y_failed", np.array([]))
    popt  = entry["popt"]
    fn    = entry["factory"]

    y_plot_cl     = y_cl
    y_plot_out    = y_out
    y_plot_failed = y_failed
    if normalise_y:
        all_y   = np.concatenate([a for a in (y_cl, y_out, y_failed) if len(a)])
        y_min   = float(np.nanmin(all_y))
        y_max   = float(np.nanmax(all_y))
        y_range = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
        y_plot_cl     = (y_cl - y_min) / y_range
        y_plot_out    = (y_out - y_min) / y_range if len(y_out) else y_out
        y_plot_failed = (y_failed - y_min) / y_range if len(y_failed) else y_failed

    stats = (pd.DataFrame({"x": x_cl, "y": y_plot_cl})
             .groupby("x")["y"].agg(["mean", "std"]))
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                fmt="o", color=color_data, ecolor=color_data,
                elinewidth=lw_errorbar, markersize=ms_data, capsize=2)

    if len(x_out) > 0:
        ax.scatter(x_out, y_plot_out, color="red",
                   s=ms_data**2 * 0.75, zorder=5)

    if show_excluded and len(x_failed) > 0:
        ax.scatter(x_failed, y_plot_failed, color="black", marker="x",
                   s=ms_data**2 * 0.75, linewidths=1.5, zorder=6,
                   label="Dispense failed")

    x_line = np.linspace(stats.index.min(), stats.index.max(), 300)
    y_line = fn(x_line, *popt)
    if normalise_y:
        y_line = (y_line - y_min) / y_range
    ax.plot(x_line, y_line, lw=lw_fit, color=color_fit)

    logKi     = float(popt[2])
    Ki        = 10.0 ** logKi
    logKi_err = entry.get("logKi_err", np.nan)
    Ki_err    = Ki * np.log(10) * logKi_err if not np.isnan(logKi_err) else np.nan
    model     = entry["model_name"]
    ki_str    = f"Ki = {_fmt_uM(Ki)}{f' ± {_fmt_uM(Ki_err)}' if not np.isnan(Ki_err) else ''} µM"

    status      = entry.get("status")
    title_color = "red" if (show_status_color and status == "FAIL") else "black"
    tkw = dict(fontsize=title_fontsize, fontfamily=title_fontfamily,
               fontweight=title_fontweight, fontstyle=title_fontstyle,
               color=title_color)
    _buf_str = f"  [{entry['buffer']}]" if entry.get("buffer") else ""
    _hc      = entry.get("host_conc")
    _hc_str  = f"  {_fmt_uM(float(_hc))} µM" if _hc is not None and not np.isnan(float(_hc)) else ""
    _combo   = f"{entry['host']}{_hc_str} | {entry['dye']} | {entry['guest']}{_buf_str}"
    if stats_in_title:
        ax.set_title(
            f"{_combo}\n"
            f"{ki_str}  |  R²adj = {entry['r2_adj']:.3f}\n"
            f"[{model}]",
            **tkw)
    else:
        ax.text(0.98, 0.98,
                f"{ki_str}\nR²adj = {entry['r2_adj']:.3f}\n[{model}]",
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="right", color=color_fit,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set_title(_combo, **tkw)
    y_label = "Normalised Intensity" if normalise_y else "FI – F₀"
    ax.set_ylabel(y_label, fontsize=axis_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    ax.xaxis.set_major_locator(
        MultipleLocator(x_tick_step) if x_tick_step else MaxNLocator(nbins=6))
    if sci_notation_y and not normalise_y:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    if ax_resid is not None:
        ax.tick_params(labelbottom=False)
        rc      = color_resid or color_data
        if normalise_y:
            resid = y_plot_cl - (fn(x_cl, *popt) - y_min) / y_range
        else:
            resid = y_cl - fn(x_cl, *popt)
        stats_r = (pd.DataFrame({"x": x_cl, "r": resid})
                   .groupby("x")["r"].agg(["mean", "std"]))
        ax_resid.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax_resid.errorbar(stats_r.index, stats_r["mean"], yerr=stats_r["std"],
                         fmt="o", color=rc, ecolor=rc,
                         elinewidth=lw_errorbar * 0.7, markersize=ms_resid,
                         capsize=2, zorder=3)
        ax_resid.set_ylabel("Resid.", fontsize=axis_fontsize)
        ax_resid.tick_params(labelsize=tick_fontsize)
        ax_resid.set_xlabel("log[Guest] (µM)", fontsize=axis_fontsize)
    else:
        ax.set_xlabel("log[Guest] (µM)", fontsize=axis_fontsize)


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
                    title_fontsize: float = 12,
                    title_fontfamily: str = "sans-serif",
                    title_fontweight: str = "normal",
                    title_fontstyle: str = "normal",
                    axis_fontsize: float = 14,
                    tick_fontsize: float = 12,
                    layout_cfg: dict = None,
                    input_folder: str = "",
                    normalise_y: bool = False,
                    sci_notation_y: bool = False,
                    lw_fit: float = 2.0,
                    ms_data: float = 4.0,
                    lw_errorbar: float = 1.5,
                    ms_resid: float = 2.5,
                    x_min: float = None,
                    x_max: float = None,
                    x_tick_step: float = None,
                    show_excluded: bool = True) -> list[str]:
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
            _has_buf = ("Buffer" in state.df_results.columns
                        and "Buffer" in state.fi_df.columns)
            _res_key_parts = [state.df_results["Host"], state.df_results["Dye"],
                              state.df_results["Guest"]]
            if _has_buf:
                _res_key_parts.append(state.df_results["Buffer"])
            _res_key_parts.append(state.df_results["HostConc_uM"].apply(_hc_key))
            lookup = dict(zip(zip(*_res_key_parts), state.df_results["Status"]))

            _key_cols_s = (["Host", "Dye", "Guest"] + (["Buffer"] if _has_buf else [])
                           + ["Host_Concentration"])
            pb, fb = [], []
            src = state.fi_df.dropna(subset=["Host", "Dye", "Guest", "FI-F0"])
            for grp_key, grp in src.groupby(_key_cols_s, dropna=False):
                if _has_buf:
                    host, dye, guest, buf, host_conc = grp_key
                else:
                    host, dye, guest, host_conc = grp_key
                    buf = None
                lookup_key = (host, dye, guest) + ((buf,) if _has_buf else ()) + (_hc_key(host_conc),)
                status = lookup.get(lookup_key, "FAIL")
                grp    = grp.sort_values("Guest_Concentration")
                concs  = grp["Guest_Concentration"].drop_duplicates().reset_index(drop=True)
                fi_bc  = grp.groupby("Guest_Concentration")["FI-F0"].apply(list)
                maxr   = fi_bc.apply(len).max()
                _hc_s  = "NA" if pd.isna(host_conc) else f"{float(host_conc):.6g}"
                _hdr   = f"{host} | {dye} | {guest} | {_hc_s}"
                if buf:
                    _hdr += f" ({buf})"
                block  = pd.DataFrame({_hdr: concs})
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
                cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
                fl = layout_utils.figure_layout(lc, rows, cols, cl.row_h_in)
                tkw = dict(color_fit=plot_color_fit, color_data=plot_color_data,
                           color_resid=plot_color_resid,
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
                           x_min=x_min, x_max=x_max, x_tick_step=x_tick_step,
                           show_excluded=show_excluded)
                fig = plt.figure(figsize=(fl.fig_w, fl.fig_h))
                outer_gs = GridSpec(rows, cols, figure=fig, **fl.outer_kwargs())
                for i, entry in enumerate(items):
                    ri, ci = divmod(i, cols)
                    if show_residuals:
                        inner_gs = outer_gs[ri, ci].subgridspec(
                            cl.sub_rows, 1, height_ratios=cl.height_ratios,
                            hspace=cl.resid_gap)
                        ax   = fig.add_subplot(inner_gs[0])
                        ax_r = fig.add_subplot(inner_gs[1], sharex=ax)
                        render_plot_ki(ax, entry, ax_resid=ax_r, **tkw)
                    else:
                        render_plot_ki(fig.add_subplot(outer_gs[ri, ci]), entry, **tkw)
                for i in range(n, rows * cols):
                    ri, ci = divmod(i, cols)
                    fig.add_subplot(outer_gs[ri, ci]).set_visible(False)
                fig.suptitle(f"Plate: {plate}" if plate else "Plate: (unknown)",
                             fontsize=11, fontweight="bold", y=0.998)
                pdf.savefig(fig)
                plt.close(fig)
        _log(f"Saved: {pdf_path}", progress_cb)
        saved.append(pdf_path)

        if export_individual:
            _cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
            _fl = layout_utils.figure_layout(lc, 1, 1, _cl.row_h_in)
            for status_dir in ("PASS", "FAIL"):
                os.makedirs(os.path.join(plots_dir, "individual", status_dir), exist_ok=True)
            for entry in state.plot_data:
                status   = entry.get("status", "FAIL")
                _buf_sfx = f"_{entry['buffer']}" if entry.get("buffer") else ""
                _hc      = entry.get("host_conc")
                _hc_sfx  = (f"_{float(_hc):.6g}uM" if _hc is not None and not np.isnan(float(_hc))
                            else "")
                safe_nm  = re.sub(r"[^\w\-]", "_",
                                  f"{entry['host']}_{entry['dye']}_{entry['guest']}{_hc_sfx}{_buf_sfx}")
                ind_path = os.path.join(plots_dir, "individual", status,
                                        f"{safe_nm}.pdf")
                tkw_ind = dict(color_fit=plot_color_fit, color_data=plot_color_data,
                               color_resid=plot_color_resid,
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
                               x_min=x_min, x_max=x_max, x_tick_step=x_tick_step,
                               show_excluded=show_excluded)
                fig = plt.figure(figsize=(_fl.fig_w, _fl.fig_h))
                gs  = GridSpec(_cl.sub_rows, 1, figure=fig, height_ratios=_cl.height_ratios,
                               hspace=_cl.resid_gap, **_fl.single_kwargs())
                if show_residuals:
                    ax   = fig.add_subplot(gs[0])
                    ax_r = fig.add_subplot(gs[1], sharex=ax)
                else:
                    ax   = fig.add_subplot(gs[0])
                    ax_r = None
                render_plot_ki(ax, entry, ax_resid=ax_r, **tkw_ind)
                fig.savefig(ind_path, dpi=150)
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
