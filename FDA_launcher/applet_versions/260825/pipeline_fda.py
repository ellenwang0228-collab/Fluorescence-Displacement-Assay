#!/usr/bin/env python3
"""
Pipeline logic — v4 base (FDA.py).

Changes from pipeline.py:
  - PASS_R2_DEFAULT raised to 0.90
  - extract_plate_name: robust to varying filename formats
  - _load_plate_data_384: detects row-label column (A–P in col 0) and skips it
  - merge_blanks: LEFT join (merged ← blank_df) prevents orphan NaN rows
  - subtract_background: FI-F0 = Fluorescence − Dye_Blank_Avg (per Plate/Dye/Chromatic);
      host autofluorescence is NOT subtracted (host-only wells are kept for QC only);
      dropna=False in all groupby calls; missing-blank warnings;
      NaN FI-F0 diagnostic logged
  - fit_curves: sign-free one_site tried with positive and negative Bmax
      bounds separately; quadratic only when D_fixed ≥ 0.1×Kd_bind;
      stern_volmer removed; binding_mode classified post-hoc;
      fail_bmax check removed (bounds prevent it);
      within-rep Grubbs removals logged
  - Kd lo range check excludes zero concentrations
  - Column names: Model, Bmax, R2_adj, Binding_mode
  - Plot annotation shows model/binding_mode
  - Cross-conc Grubbs retained as toggleable option (default off per v4)
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import normaltest, t, norm
import seaborn as sns
import matplotlib.patches as mpatches

import chromatic_db as cdb
import layout_utils


ProgressCb = Optional[Callable[[str], None]]

PASS_R2_DEFAULT            = 0.90
KD_RANGE_FACTOR_LO_DEFAULT = 0.1
KD_RANGE_FACTOR_HI_DEFAULT = 10.0
PLOT_COLOR                 = "#5480ed"
PLOT_COLOR_DATA            = "#1e4d72"
KD_CV_THRESHOLD_DEFAULT      = 0.5   # Kd_SE/Kd worse than this ⇒ Confidence = Low (does not gate Status — screening use)
                                      # NOT directly comparable to pipeline_ki.LOGKI_ERR_THRESHOLD_DEFAULT — that's a
                                      # log10-scale SE, this is a linear relative SE. For small errors,
                                      # SE(log10 K) ≈ (SE_K/K)/ln(10), so this 0.5 linear-CV threshold corresponds to
                                      # a log10-SE of ~0.217 — i.e. TIGHTER (more stringent) than Ki's 0.3 log10-SE
                                      # threshold (which itself corresponds to a linear CV of ~0.69). "Confidence: High"
                                      # does not mean the same precision in both pipelines — see option (a) in the
                                      # statistical-fixes brief if these should be equalized (non-trivial: would
                                      # require refitting Kd's SE onto a log scale).
NORMALITY_P_THRESHOLD_DEFAULT = 0.01 # deliberately lenient — diagnostic note, not a hard gate
RUNS_TEST_P_THRESHOLD_DEFAULT = 0.05 # diagnostic only — does not gate Status/Confidence
HOST_AUTOFL_FACTOR_DEFAULT   = 1.0   # fallback ratio mode only; "stat" mode (default) uses HOST_AUTOFL_Z_DEFAULT
HOST_AUTOFL_Z_DEFAULT        = 1.96  # two-tailed 95% CI z-score; matches GUEST_AUTOFL_Z_DEFAULT in pipeline_ki.py
DEPLETION_GATE_DEFAULT       = 0.1   # only try the quadratic (depletion) model when D_fixed >= this x Kd_hyperbolic
DEPLETION_GATE_MARGIN_BAND   = (0.5, 2.0)  # Depletion_gate_margin within [band_lo, band_hi] x depletion_gate
                                            # ⇒ the choice to try the quadratic model was itself close to the
                                            # boundary (see fit_curves docstring; mirrors WANG_GATE_MARGIN_BAND
                                            # in pipeline_ki.py)
DELTA_AICC_AMBIGUOUS         = 2.0   # Burnham & Anderson 2002: ΔAICc < 2 ⇒ no strong preference between models


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


# ── Dataclass holding all intermediate state ──────────────────────────────────

@dataclass
class PipelineState:
    merged_mapping: Optional[pd.DataFrame] = None
    fluorescence:   Optional[pd.DataFrame] = None
    merged:         Optional[pd.DataFrame] = None
    fi_df:          Optional[pd.DataFrame] = None
    fit_results:    list = field(default_factory=list)
    plot_data:      list = field(default_factory=list)
    df_results:     Optional[pd.DataFrame] = None
    qc_figures:     list = field(default_factory=list)


# ── Stage 1: load mappings ────────────────────────────────────────────────────

def load_mappings(dye_folder: str, host_folder: str,
                  buffer_folder: str = None,
                  progress_cb: ProgressCb = None) -> pd.DataFrame:
    if not os.path.isdir(dye_folder):
        raise FileNotFoundError(f"Dye mapping folder not found: {dye_folder}")
    if not os.path.isdir(host_folder):
        raise FileNotFoundError(f"Host mapping folder not found: {host_folder}")

    dye_files  = sorted(f for f in os.listdir(dye_folder)
                        if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    host_files = sorted(f for f in os.listdir(host_folder)
                        if f.lower().endswith(".xlsx") and not f.startswith("~$"))

    if not dye_files:
        raise FileNotFoundError(f"No .xlsx files in dye mapping folder: {dye_folder}")
    if not host_files:
        raise FileNotFoundError(f"No .xlsx files in host mapping folder: {host_folder}")

    _log(f"Loading dye mapping:  {dye_files}", progress_cb)
    _log(f"Loading host mapping: {host_files}", progress_cb)

    dye_df = pd.concat(
        [pd.read_excel(os.path.join(dye_folder, f)) for f in dye_files],
        ignore_index=True)
    host_df = pd.concat(
        [pd.read_excel(os.path.join(host_folder, f)) for f in host_files],
        ignore_index=True)

    _required_cols = {"Compound ID", "Destination Well", "Destination Concentration",
                      "Destination Unit", "Destination Plate Name"}
    for label, df_check in [("dye_map", dye_df), ("host_map", host_df)]:
        missing = _required_cols - set(df_check.columns)
        if missing:
            raise ValueError(f"{label} file(s) missing required column(s): {sorted(missing)}")

    dye_df = dye_df.loc[:, [
        "Compound ID", "Destination Well", "Destination Concentration",
        "Destination Unit", "Destination Plate Name"
    ]].rename(columns={
        "Compound ID":               "Dye",
        "Destination Well":          "Well",
        "Destination Concentration": "Dye_Concentration",
        "Destination Unit":          "Dye_Unit",
        "Destination Plate Name":    "Plate",
    })

    host_df = host_df.loc[:, [
        "Compound ID", "Destination Well", "Destination Concentration",
        "Destination Unit", "Destination Plate Name"
    ]].rename(columns={
        "Compound ID":               "Host",
        "Destination Well":          "Well",
        "Destination Concentration": "Host_Concentration",
        "Destination Unit":          "Host_Unit",
        "Destination Plate Name":    "Plate",
    })

    merged = pd.merge(dye_df, host_df, on=["Well", "Plate"], how="outer")
    merged["Dye_Unit"]  = merged["Dye_Unit"].str.replace(r"[^\w]", "", regex=True)
    merged["Host_Unit"] = merged["Host_Unit"].str.replace(r"[^\w]", "", regex=True)

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

    _log(f"Hosts: {merged['Host'].nunique()}  |  Dyes: {merged['Dye'].nunique()}", progress_cb)
    return merged


def load_exceptions(exceptions_folder: str, progress_cb: ProgressCb = None) -> set:
    """Wells to exclude because their dispense failed, from a folder of Echo
    'Exceptions' reports (.csv, columns 'Destination Plate Name' / 'Destination
    Well' — the same Echo transfer-report convention as dye_map/host_map, but
    the failure log instead of the successful-transfer one).

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


# ── Stage 2: load raw plate data ──────────────────────────────────────────────

def _extract_plate_name(filepath: str) -> str:
    """
    Robust plate name extraction. Splits stem on '_', returns the first token
    that is not a pure-digit string longer than 4 characters (i.e. not a date).
    Falls back to the full stem if no token qualifies.
    """
    stem   = os.path.splitext(os.path.basename(filepath))[0]
    tokens = stem.split("_")
    for tok in tokens:
        if tok and not (tok.isdigit() and len(tok) > 4):
            return tok
    return stem


_ROW_LABELS = set(string.ascii_uppercase[:16])  # A–P


def _find_chromatic_blocks(df_raw: pd.DataFrame) -> list[tuple]:
    """Return list of (label, data_start_row) for each Chromatic: header.

    The data start row is auto-detected by scanning forward for the first
    plate-data row, rather than assuming a fixed offset — different chromatic
    blocks (or plate reader export versions) insert a different number of
    metadata/header rows between the "Chromatic:" label and the actual data,
    which silently shifted/truncated rows (e.g. dropping the top concentration)
    under the old fixed-offset (i + 3) approach. Mirrors
    pipeline_ki._detect_chromatic_start_rows.
    """
    blocks = []
    for i in range(len(df_raw)):
        cell = str(df_raw.iloc[i, 0]).strip()
        if cell.lower().startswith("chromatic:"):
            label = cell.split(":", 1)[1].strip()
            for j in range(i + 1, min(i + 11, len(df_raw))):
                first_cell = str(df_raw.iloc[j, 0]).strip()
                # (a) row label in column 0
                if len(first_cell) == 1 and first_cell.upper() in _ROW_LABELS:
                    blocks.append((label, j))
                    break
                # (b) enough large numeric values to be fluorescence data
                numeric = pd.to_numeric(df_raw.iloc[j], errors="coerce").dropna()
                if len(numeric) >= 3 and float(numeric.max()) > 25:
                    blocks.append((label, j))
                    break
    return blocks


def _resolve_block_dye(label: str, auto_map: dict) -> Optional[str]:
    """Resolve a single chromatic block's dye.

    - Numeric label (e.g. "1"): looked up against auto_map (built from the
      raw file's own filter-settings header + chromatic_db). None if the
      filter wasn't catalogued or the file has no header.
    - Non-numeric label: the user has renamed the chromatic block to the
      dye name directly in the raw export — keep that legacy convention as
      a manual per-file override.
    """
    try:
        return auto_map.get(int(label))
    except ValueError:
        return label


def _load_plate_data_384(filepath: str, plate_name: str,
                         chromatic_db: pd.DataFrame = None,
                         multi_chromatic: bool = False,
                         progress_cb: ProgressCb = None) -> pd.DataFrame:
    df_raw = pd.read_excel(filepath, engine="openpyxl", header=None, sheet_name=0)
    blocks = _find_chromatic_blocks(df_raw)

    if not blocks:
        return pd.DataFrame(columns=["Well", "Fluorescence", "Plate", "Chromatic", "Chromatic_Dye"])

    # The Chromatic DB is only consulted to target individual dyes when a plate
    # genuinely carries more than one — otherwise a filter that happens to
    # resolve (rightly or wrongly) would silently relabel a single-dye plate's
    # only channel instead of just trusting the dye mapping for it.
    auto_map = cdb.resolve_chromatic_to_dye(df_raw, chromatic_db, plate_name, progress_cb) \
        if multi_chromatic else {}

    if not multi_chromatic and len(blocks) > 1:
        _log(f"WARNING: '{plate_name}' has {len(blocks)} chromatic blocks but "
             f"'Multi-chromatic plates' is unticked — only the first block "
             f"('{blocks[0][0]}') is used; the rest are ignored. Tick "
             f"'Multi-chromatic plates' to use the Chromatic DB to target each "
             f"dye on this plate.", progress_cb)
        blocks = blocks[:1]

    row_labels = set(string.ascii_uppercase[:16])
    dfs = []
    for label, start_row in blocks:
        # Some plate readers put a row label (A–P) in column 0; detect and skip it.
        first_cell = str(df_raw.iloc[start_row, 0]).strip()
        col_start  = 1 if (len(first_cell) == 1 and first_cell.upper() in row_labels) else 0

        plate_df = df_raw.iloc[start_row:start_row + 16, col_start:col_start + 24].copy()
        nrows, ncols = plate_df.shape
        if nrows < 1 or ncols < 1:
            _log(f"Warning: {plate_name} chromatic '{label}' empty block ({nrows}x{ncols}) — skipped",
                 progress_cb)
            continue
        if (nrows, ncols) != (16, 24):
            _log(f"Warning: {plate_name} chromatic '{label}' shape {plate_df.shape} "
                 f"(expected 16×24)", progress_cb)
        plate_df.index   = list(string.ascii_uppercase[:nrows])
        plate_df.columns = [str(i) for i in range(1, ncols + 1)]
        tidy = (
            plate_df.reset_index()
            .melt(id_vars="index", var_name="Column", value_name="Fluorescence")
            .rename(columns={"index": "Row"})
        )
        tidy["Fluorescence"]  = pd.to_numeric(tidy["Fluorescence"], errors="coerce")
        tidy["Well"]          = tidy["Row"] + tidy["Column"]
        tidy["Plate"]         = plate_name
        resolved              = _resolve_block_dye(label, auto_map)
        tidy["Chromatic"]     = resolved if resolved else label
        tidy["Chromatic_Dye"] = resolved
        dfs.append(tidy[["Well", "Fluorescence", "Plate", "Chromatic", "Chromatic_Dye"]]
                  .dropna(subset=["Fluorescence"]))
    return pd.concat(dfs, ignore_index=True)


def load_plates(raw_folder: str, chromatic_folder: str = None,
                multi_chromatic: bool = False,
                progress_cb: ProgressCb = None) -> pd.DataFrame:
    """chromatic_folder, if given, points at a Dye/Filter lookup table (see
    chromatic_db.py) used to auto-resolve each raw file's chromatic blocks
    to dye names from its own filter-settings header — see
    chromatic_db.resolve_chromatic_to_dye for details.

    multi_chromatic gates whether that auto-resolution is used at all: tick
    it only for plates that actually carry more than one dye. When off
    (default), every plate is assumed single-dye — only its first chromatic
    block is loaded and its dye is taken straight from the dye mapping later
    in merge_blanks, without consulting the Chromatic DB."""
    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) and multi_chromatic
                    else None)

    if not os.path.isdir(raw_folder):
        raise FileNotFoundError(f"Raw data folder not found: {raw_folder}")

    filepaths = sorted(
        os.path.join(raw_folder, f)
        for f in os.listdir(raw_folder)
        if f.lower().endswith(".xlsx") and not f.startswith("~$")
    )

    if not filepaths:
        raise FileNotFoundError(f"No .xlsx files in raw data folder: {raw_folder}")

    all_dfs = []
    for fp in filepaths:
        plate_name = _extract_plate_name(fp)
        df_plate   = _load_plate_data_384(fp, plate_name, chromatic_db, multi_chromatic, progress_cb)
        all_dfs.append(df_plate)
        chroms = df_plate["Chromatic"].unique().tolist()
        _log(f"  Loaded '{plate_name}': {len(chroms)} chromatic(s) → {chroms}", progress_cb)

    result = pd.concat(all_dfs, ignore_index=True)
    result["Fluorescence"] = pd.to_numeric(result["Fluorescence"], errors="coerce")
    _log(f"Loaded {result['Plate'].nunique()} plate(s), "
         f"{result['Chromatic'].nunique()} unique chromatic(s)", progress_cb)
    return result


# ── Stage 3: merge with mapping + blanks ─────────────────────────────────────

def merge_blanks(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
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
    'Dispense_Failed' column instead of dropped, so fit_curves can exclude
    them from fitting while still plotting them as crosses."""
    merged = pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
    merged = merged.sort_values(["Host", "Host_Concentration"])

    failed_wells = load_exceptions(exceptions_folder, progress_cb)
    merged["Dispense_Failed"] = (
        [(p, w) in failed_wells for p, w in zip(merged["Plate"].astype(str),
                                                merged["Well"].astype(str))]
        if failed_wells else False)

    chrom_counts       = fluorescence_df.groupby("Plate")["Chromatic"].nunique()
    multi_chrom_plates = set(chrom_counts[chrom_counts > 1].index)

    chromatic_db = (cdb.load_chromatic_db(chromatic_folder, progress_cb)
                    if chromatic_folder and os.path.isdir(chromatic_folder) else None)
    alias_map = cdb.build_alias_map(chromatic_db)

    if multi_chrom_plates:
        _log(f"Multi-chromatic plates: {sorted(multi_chrom_plates)}", progress_cb)

    # Compare each row's resolved Chromatic_Dye — set either by the Chromatic DB
    # (only populated when multi_chromatic is on, see _load_plate_data_384) or by
    # a manually renamed block label (works either way) — against the mapping's
    # Dye. Aliases allowed (e.g. dye_map says "H33" but the chromatic DB / Kd
    # table say "H33258").
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
                        "unticked — check the mapping or the raw block label if unexpected."),
                     progress_cb)

    if not multi_chromatic:
        # Trust the mapping for the Chromatic label too, not just for which
        # rows are kept — otherwise a mismatched raw-file block label (or a
        # DB mis-resolution) would still show up as the plot/grouping label
        # even though the mapping's dye is what's actually used.
        #
        # Relabel by PLATE, not by row: every well on a plate shares the same
        # physical channel in single-dye mode, but host-only/buffer-blank
        # wells have Dye=NaN (no dye was dispensed there) so a row-by-row
        # "Dye.notna()" mask would only relabel Dye-blank/Host+Dye rows,
        # leaving host-only/buffer-blank rows on the old raw block label —
        # splitting one plate's wells across two different Chromatic groups
        # in QC and background-subtraction grouping.
        plate_dye = (merged.loc[merged["Dye"].notna(), ["Plate", "Dye"]]
                     .drop_duplicates(subset="Plate").set_index("Plate")["Dye"])
        _resolved = merged["Plate"].map(plate_dye)
        _has_resolved = _resolved.notna()
        merged.loc[_has_resolved, "Chromatic"]     = _resolved[_has_resolved]
        merged.loc[_has_resolved, "Chromatic_Dye"] = _resolved[_has_resolved]

    # Dropping mismatched rows is only safe when targeting real multi-dye
    # plates via the Chromatic DB — otherwise (default) the mapping alone
    # decides each well's dye and nothing is filtered out.
    if multi_chromatic and multi_chrom_plates:
        # Rows with Chromatic_Dye=None (unresolved channel — ambiguous filter)
        # are kept ONLY if no resolved channel on the same plate already covers
        # that well's dye; otherwise the resolved channel is preferred.
        _resolved_dyes_per_plate: dict[str, set] = {}
        for plate in multi_chrom_plates:
            _resolved_dyes_per_plate[plate] = set(
                fluorescence_df.loc[
                    (fluorescence_df["Plate"] == plate) & fluorescence_df["Chromatic_Dye"].notna(),
                    "Chromatic_Dye"
                ].unique()
            )

        _unresolved = merged["Chromatic_Dye"].isna() & merged["Dye"].notna()
        for plate in multi_chrom_plates:
            resolved = _resolved_dyes_per_plate[plate]
            plate_unresolved = _unresolved & (merged["Plate"] == plate)
            if plate_unresolved.any():
                _match_vec.loc[plate_unresolved] = [
                    not any(cdb.dye_matches(d, r, alias_map) for r in resolved)
                    for d in merged.loc[plate_unresolved, "Dye"]
                ]

        merged = merged[~(_both_present | _unresolved) | _match_vec].copy()

        for plate in multi_chrom_plates:
            plate_dyes   = merged_mapping.loc[merged_mapping["Plate"] == plate, "Dye"].dropna().unique()
            plate_chroms = (fluorescence_df.loc[fluorescence_df["Plate"] == plate, "Chromatic_Dye"]
                            .dropna().unique())
            for dye in plate_dyes:
                if not any(cdb.dye_matches(dye, c, alias_map) for c in plate_chroms):
                    _log(f"  WARNING: plate {plate}: dye '{dye}' matched no chromatic channel "
                         f"(resolved channels: {list(plate_chroms)}) — check the chromatic DB "
                         f"(including Aliases) or rename the chromatic block to the dye name.",
                         progress_cb)

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

    # LEFT join keeps all fluorescence rows and adds blank labels where matched.
    # Join only on [Well, Plate, Dye] — NOT on Dye_Concentration.
    # Merging on a float key is unreliable: 0.5 parsed from two separate Excel files
    # may differ in the last ULP and silently produce 0 matches, leaving every well
    # without a Dye_Blank_Avg and making all FI-F0 = NaN.
    merged2 = pd.merge(merged, blank_df,
                       on=["Well", "Plate", "Dye"],
                       how="left",
                       suffixes=("", "_blank"))

    n_blank_matched = merged2["Compound ID"].notna().sum()
    _log(f"Merged with blanks: {len(merged2)} rows "
         f"({n_blank_matched} blank-labelled)", progress_cb)
    if n_blank_matched == 0:
        _log("WARNING: no blank wells matched after merge. Check that:\n"
             "  • blank_map files contain rows with Compound ID = 'blank'\n"
             "  • Plate and Well names match the fluorescence plate files\n"
             "  • Dye names in blank_map match those in dye_map exactly\n"
             "FI-F0 will be NaN for all wells.", progress_cb)

    # Plate-name mismatch diagnostic
    n_no_host = merged2["Host"].isna().sum()
    if n_no_host / max(len(merged2), 1) > 0.5:
        fl_plates  = sorted(fluorescence_df["Plate"].dropna().unique())
        map_plates = sorted(merged_mapping["Plate"].dropna().unique()
                            if "Plate" in merged_mapping.columns else [])
        _log(f"WARNING: {n_no_host}/{len(merged2)} rows have no Host after merge.\n"
             f"  Fluorescence plate names: {fl_plates}\n"
             f"  Mapping plate names:      {map_plates}\n"
             f"  These must match exactly.", progress_cb)

    # Backfill Chromatic / Chromatic_Dye from the mapping's Dye column for
    # rows left unresolved (ambiguous filter).  Use the canonical dye name
    # (via alias map) so the Chromatic label matches resolved channels that
    # used the DB's canonical spelling.
    def _canonical(dye):
        return alias_map.get(str(dye).strip().lower(), dye) if pd.notna(dye) else dye

    _unres = merged2["Chromatic_Dye"].isna()
    _unres_with_dye = _unres & merged2["Dye"].notna()
    if _unres_with_dye.any():
        canon = merged2.loc[_unres_with_dye, "Dye"].map(_canonical)
        merged2.loc[_unres_with_dye, "Chromatic"]     = canon
        merged2.loc[_unres_with_dye, "Chromatic_Dye"] = canon

    # Unmapped wells (Dye=None) on unresolved channels — e.g. buffer blanks.
    # Assign them to whichever dye(s) on the plate have no resolved channel.
    # When multiple dyes share the ambiguous filter, duplicate the row so
    # each dye gets its own buffer-blank entry (same fluorescence reading).
    # Use alias-aware comparison so 'H333' in the mapping is recognised as
    # covered by resolved channel 'H33342'.
    _unres_no_dye = _unres & merged2["Dye"].isna()
    if _unres_no_dye.any():
        _unres_dyes_per_plate: dict[str, list] = {}
        plates_to_check = multi_chrom_plates if multi_chrom_plates else set(merged2["Plate"].unique())
        for plate in plates_to_check:
            all_plate_dyes = set(
                merged_mapping.loc[merged_mapping["Plate"] == plate, "Dye"]
                .dropna().unique())
            resolved = set(
                fluorescence_df.loc[
                    (fluorescence_df["Plate"] == plate)
                    & fluorescence_df["Chromatic_Dye"].notna(),
                    "Chromatic_Dye"].unique())
            unres_canonical = sorted({
                _canonical(d) for d in all_plate_dyes
                if not any(cdb.dye_matches(d, r, alias_map) for r in resolved)
            })
            _unres_dyes_per_plate[plate] = unres_canonical

        to_drop = []
        new_rows = []
        for idx in merged2.index[_unres_no_dye]:
            plate = merged2.at[idx, "Plate"]
            candidates = _unres_dyes_per_plate.get(plate, [])
            if len(candidates) >= 1:
                merged2.at[idx, "Chromatic"]     = candidates[0]
                merged2.at[idx, "Chromatic_Dye"] = candidates[0]
                for extra_dye in candidates[1:]:
                    row = merged2.loc[idx].copy()
                    row["Chromatic"]     = extra_dye
                    row["Chromatic_Dye"] = extra_dye
                    new_rows.append(row)
            else:
                to_drop.append(idx)
        if new_rows:
            merged2 = pd.concat([merged2, pd.DataFrame(new_rows)], ignore_index=True)
        if to_drop:
            merged2 = merged2.drop(to_drop).copy()

    return merged2


# ── Stage 4: background subtraction ──────────────────────────────────────────

def subtract_background(merged_df: pd.DataFrame,
                        progress_cb: ProgressCb = None) -> pd.DataFrame:
    """FI-F0 = Fluorescence − Dye_Blank_Avg (per Plate/Dye/Chromatic)."""
    df       = merged_df.copy()
    is_blank = df["Compound ID"].fillna("").str.strip().str.lower() == "blank"

    # Dye blank average per (Plate, Dye, Chromatic).
    # dropna=False keeps rows whose group key contains NaN in a NaN group
    # rather than silently dropping them.
    df["Dye_Blank_Avg"] = (
        df
        .where(is_blank & df["Dye"].notna())
        .groupby(["Plate", "Dye", "Chromatic"], dropna=False)["Fluorescence"]
        .transform("mean")
    )
    df["Dye_Blank_Avg"] = (
        df.groupby(["Plate", "Dye", "Chromatic"], dropna=False)["Dye_Blank_Avg"]
        .transform("first")
    )

    missing = (
        df[df["Dye"].notna()]
        .groupby(["Plate", "Dye", "Chromatic"], dropna=False)["Dye_Blank_Avg"]
        .first()
        .pipe(lambda s: s[s.isna()])
    )
    if not missing.empty:
        _log("WARNING: no dye blank wells found for:", progress_cb)
        for idx in missing.index:
            _log(f"  {idx}", progress_cb)

    df["FI-F0"] = df["Fluorescence"] - df["Dye_Blank_Avg"]

    nan_rows = df[df["Dye"].notna() & df["FI-F0"].isna()][
        ["Plate", "Dye", "Chromatic", "Host_Concentration"]].drop_duplicates()
    if not nan_rows.empty:
        _log(f"NOTE: {len(nan_rows)} row(s) with NaN FI-F0 will be dropped before fitting.",
             progress_cb)

    _log("Background subtraction complete (FI-F0 = F – Dye_Blank).", progress_cb)
    return df


# ── Stage 4b: QC figures ─────────────────────────────────────────────────────

def make_qc_figures_fda(df: pd.DataFrame, progress_cb: ProgressCb = None) -> list:
    """
    QC figures for the Kd pipeline (modelled after Ki QC):
    1. Control well fluorescence per chromatic (Buffer blank / Dye blank / Host-only / Assay)
    2. Assay wells (Host + Dye) per host per chromatic, with dye-blank reference line
    3. Host-only autofluorescence per host per chromatic, with dye-blank reference line
    """
    figures = []
    try:
        chromaticss  = sorted(df["Chromatic"].dropna().unique())
        # Title subplots by resolved dye name where available, falling back to
        # the raw chromatic label for unresolved channels (no DB match / no header).
        chrom_label  = (df.dropna(subset=["Chromatic", "Chromatic_Dye"])
                        .drop_duplicates("Chromatic")
                        .set_index("Chromatic")["Chromatic_Dye"]
                        if "Chromatic_Dye" in df.columns else pd.Series(dtype=object))
        chrom_title  = lambda chrom: chrom_label.get(chrom, f"Chromatic {chrom}")
        plates       = sorted(df["Plate"].dropna().unique())
        palette      = dict(zip(plates, sns.color_palette("tab10", len(plates))))
        handles      = [mpatches.Patch(color=palette[p], label=p) for p in plates]
        is_blank_lbl = df["Compound ID"].fillna("").str.strip().str.lower() == "blank"

        # ── label each row with its condition ──────────────────────────────────
        cond = pd.Series("other", index=df.index)
        cond[is_blank_lbl & df["Dye"].isna()]                              = "Buffer blank"
        cond[is_blank_lbl & df["Dye"].notna()]                             = "Dye blank"
        cond[df["Host"].notna() & df["Dye"].isna()  & ~is_blank_lbl]      = "Host-only"
        cond[df["Host"].notna() & df["Dye"].notna() & ~is_blank_lbl]      = "Host + Dye"

        qc_df = df.copy()
        qc_df["Condition"] = cond
        qc_df = qc_df[qc_df["Condition"] != "other"]

        COND_ORDER = ["Buffer blank", "Dye blank", "Host-only", "Host + Dye"]

        # ── QC 1: control wells per chromatic ──────────────────────────────────
        fig1 = Figure(figsize=(max(5, 4.5 * len(chromaticss)), 5))
        axes = fig1.subplots(1, max(len(chromaticss), 1), squeeze=False)
        for col_i, (ax, chrom) in enumerate(zip(axes[0], chromaticss)):
            sub = qc_df[qc_df["Chromatic"] == chrom]
            if sub.empty:
                ax.set_visible(False)
                continue
            order = [c for c in COND_ORDER if c in sub["Condition"].values]
            sns.stripplot(data=sub, x="Condition", y="Fluorescence", order=order,
                          hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4, alpha=0.7)
            if ax.get_legend():
                ax.get_legend().remove()
            for xi, c in enumerate(order):
                vals = sub.loc[sub["Condition"] == c, "Fluorescence"].dropna()
                if len(vals):
                    ax.plot([xi - 0.3, xi + 0.3],
                            [vals.median(), vals.median()],
                            lw=2.5, color="black", zorder=10)
            ax.set_title(chrom_title(chrom), fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        fig1.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                    loc="center left", fontsize=8)
        fig1.suptitle("QC — control well fluorescence",
                      fontsize=12, fontweight="bold")
        fig1.tight_layout()
        figures.append(("QC: Control Wells", fig1))

        # Dye-blank reference per (Plate, Chromatic) — used in QC 2 & 3 below.
        # Scoped per plate (not just per chromatic) so a multi-plate dataset
        # doesn't blend unrelated plates' background levels into one line
        # (same reasoning as the Ki QC dye-blank reference).
        dye_blank_ref = (qc_df[qc_df["Condition"] == "Dye blank"]
                         .groupby(["Plate", "Chromatic"])["Fluorescence"].mean())

        # ── QC 2: Host + Dye assay wells per host per chromatic ────────────────
        host_dye = qc_df[qc_df["Condition"] == "Host + Dye"]
        if not host_dye.empty:
            max_hosts = max(host_dye.groupby("Chromatic")["Host"].nunique().max(), 1)
            fig2 = Figure(figsize=(max(6, 0.65 * max_hosts * len(chromaticss) + 2), 5))
            axes2 = fig2.subplots(1, max(len(chromaticss), 1), squeeze=False)
            for col_i, (ax, chrom) in enumerate(zip(axes2[0], chromaticss)):
                sub = host_dye[host_dye["Chromatic"] == chrom]
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
                        ax.plot([xi - 0.3, xi + 0.3],
                                [vals.median(), vals.median()],
                                lw=2.5, color="black", zorder=10)
                _dye_label_used = False
                for p in sorted(sub["Plate"].dropna().unique()):
                    if (p, chrom) in dye_blank_ref.index:
                        ax.axhline(dye_blank_ref[(p, chrom)], color=palette[p], lw=1.5,
                                   ls="--", alpha=0.7,
                                   label=None if _dye_label_used else "Dye blank")
                        _dye_label_used = True
                if _dye_label_used:
                    ax.legend(fontsize=7)
                ax.set_title(chrom_title(chrom), fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=8)
            fig2.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            fig2.suptitle("QC — Host + Dye per host",
                          fontsize=12, fontweight="bold")
            fig2.tight_layout()
            figures.append(("QC: Host + Dye", fig2))

        # ── QC 3: Host-only autofluorescence per host per chromatic ────────────
        host_only = qc_df[qc_df["Condition"] == "Host-only"]
        if not host_only.empty:
            max_hosts = max(host_only.groupby("Chromatic")["Host"].nunique().max(), 1)
            fig3 = Figure(figsize=(max(6, 0.65 * max_hosts * len(chromaticss) + 2), 5))
            axes3 = fig3.subplots(1, max(len(chromaticss), 1), squeeze=False)
            for col_i, (ax, chrom) in enumerate(zip(axes3[0], chromaticss)):
                sub = host_only[host_only["Chromatic"] == chrom]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                order = (sub.groupby("Host")["Fluorescence"].median()
                         .sort_values().index.tolist())
                sns.stripplot(data=sub, x="Host", y="Fluorescence", order=order,
                              hue="Plate", palette=palette, ax=ax, jitter=0.25, size=4, alpha=0.7)
                if ax.get_legend():
                    ax.get_legend().remove()
                for xi, h in enumerate(order):
                    vals = sub.loc[sub["Host"] == h, "Fluorescence"].dropna()
                    if len(vals):
                        ax.plot([xi - 0.3, xi + 0.3],
                                [vals.median(), vals.median()],
                                lw=2.5, color="black", zorder=10)
                _dye_label_used = False
                for p in sorted(sub["Plate"].dropna().unique()):
                    if (p, chrom) in dye_blank_ref.index:
                        ax.axhline(dye_blank_ref[(p, chrom)], color=palette[p], lw=1.5,
                                   ls="--", alpha=0.7,
                                   label=None if _dye_label_used else "Dye blank")
                        _dye_label_used = True
                if _dye_label_used:
                    ax.legend(fontsize=7)
                ax.set_title(chrom_title(chrom), fontsize=11, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Fluorescence" if col_i == 0 else "", fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=8)
            fig3.legend(handles=handles, title="Plate", bbox_to_anchor=(1.01, 0.5),
                        loc="center left", fontsize=8)
            fig3.suptitle("QC — Host autofluorescence (host-only wells)",
                          fontsize=12, fontweight="bold")
            fig3.tight_layout()
            figures.append(("QC: Host Autofluorescence", fig3))

        _log(f"Generated {len(figures)} QC figure(s).", progress_cb)
    except Exception as e:
        _log(f"WARNING: QC figure generation failed: {e}", progress_cb)
    return figures


# ── Stage 5: curve fitting ────────────────────────────────────────────────────

def _one_site(x, Bmax, Kd):
    """Sign-free hyperbolic. Bmax > 0 → binding; Bmax < 0 → quenching."""
    x = np.asarray(x, dtype=float)
    return Bmax * x / (Kd + x)


def _quadratic_binding(x, Fmax, Kd, D_fixed):
    D_fixed = max(D_fixed, 1e-12)
    arg = np.clip((x + D_fixed + Kd)**2 - 4 * x * D_fixed, 0, None)
    return Fmax * (x + D_fixed + Kd - np.sqrt(arg)) / (2 * D_fixed)


def _binding_mode(bmax: float) -> str:
    return "binding" if bmax >= 0 else "quenching"


def _eval_model(model_name: str, x, popt, D_fixed: float):
    if model_name == "quadratic":
        return _quadratic_binding(x, *popt, D_fixed)
    return _one_site(x, *popt)


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
    first-pass behaviour at all. Kept in sync with pipeline_ki.py's
    _grubbs_mask.

    At n=3 replicates per concentration (this lab's standard design),
    multiplicity_correction is a mathematical no-op: the `while
    np.sum(mask) >= 3` stopping condition means at most one removal pass
    can ever run, so there is no second iteration for the
    Bonferroni-over-iterations correction to apply to. Confirmed
    empirically on a real batch: identical Kd, R2_adj, AICc, Model,
    n_removed, and Status with the flag on vs. off, exact match, zero
    differing rows. This only becomes non-trivial if replicate count is
    ever raised to n>=4.
    """
    data = np.asarray(data, dtype=float)
    mask = np.ones(len(data), dtype=bool)
    iteration = 0
    while np.sum(mask) >= 3:
        iteration += 1
        vals = data[mask]
        std  = np.std(vals, ddof=1)
        if std == 0:
            break
        mean       = np.mean(vals)
        abs_diffs  = np.abs(vals - mean)
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


def fit_curves(fi_df: pd.DataFrame,
               grubbs_alpha: float = 0.05,
               min_n_for_grubbs: int = 3,
               grubbs_multiplicity_correction: bool = False,
               use_cross_conc_grubbs: bool = False,
               cross_conc_alpha: float = 0.01,
               model_preference: str = "auto",
               depletion_gate: float = DEPLETION_GATE_DEFAULT,
               progress_cb: ProgressCb = None) -> tuple[list, list]:
    """
    Returns (fit_results, plot_data).
    model_preference: "auto" | "one_site" | "quadratic" | "stern_volmer"
      "stern_volmer" selects the negative-Bmax one_site (quenching) model only.

    min_n_for_grubbs (default 3, unchanged behaviour): minimum replicate
    count at one Host concentration before a within-replicate Grubbs test
    is run at all. Standard guidance treats Grubbs as unreliable below
    n≈6-7; the default of 3 (the minimum needed to compute a sample SD at
    all) is kept for continuity with prior runs, but every curve where any
    concentration was tested at exactly 3 replicates is flagged via the
    output column Grubbs_applied_at_n3 so this can be audited/filtered
    later without re-deriving it. Raise this (e.g. to 6 or 7) to skip the
    test entirely at low replicate counts instead.

    grubbs_multiplicity_correction (default False, unchanged behaviour):
    see _grubbs_mask's docstring — applies a Bonferroni correction over
    the iterative removal passes already run on a given concentration's
    replicates (and on the cross-concentration pass, if enabled), instead
    of testing every pass at the same nominal alpha. At n=3 replicates
    (this lab's standard design) this flag is a mathematical no-op — see
    _grubbs_mask's docstring.

    depletion_gate: the quadratic (depletion-correction) model is only added
    to the Auto candidate set when D_fixed >= depletion_gate x the
    hyperbolic (one_site) Kd estimate for that curve — i.e. only attempted
    in the regime where the non-depleting one_site assumption is likely to
    break down (mirrors wang_gate in pipeline_ki.py). An explicit
    "quadratic" request bypasses this gate and logs a QC warning if used
    outside the expected regime.
    NOTE: this gate is circular — it decides whether to try the
    depletion-aware quadratic model using a Kd estimate (Kd_bind_prefit,
    from the plain non-depleting one-site fit) that is itself potentially
    depletion-biased in exactly the regime the gate is trying to detect
    (identical structural issue to wang_gate in pipeline_ki.py — see that
    docstring). Not corrected here (affects absolute Kd magnitude at the
    gate's margin, not which curves rank tightest within a screen); left as
    a documented limitation, not a code change.

    Depletion_gate_margin (= D_fixed / Kd_bind_prefit or Kd_quench_prefit,
    matching whichever prefit the winning model's sign corresponds to) is
    persisted per curve so the gate decision can be reviewed post-hoc. When
    the winning Model is "quadratic" and this margin falls within
    DEPLETION_GATE_MARGIN_BAND x depletion_gate, a QC warning is logged
    (informational only — does not affect Status or Model) noting that the
    choice to try the depletion-aware model was itself close to the
    boundary.
    """
    if use_cross_conc_grubbs:
        _log("WARNING: Cross-concentration Grubbs is ENABLED — this can "
             "silently remove genuine hook-effect/saturation points that "
             "are real, not outliers. n_cross_conc_removed is persisted per "
             "curve in the results table for audit.", progress_cb)

    _has_buffer = "Buffer" in fi_df.columns
    _grp_cols   = ["Plate", "Host", "Dye"] + (["Buffer"] if _has_buffer else [])
    groups = (
        fi_df.dropna(subset=["Host", "Dye", "Host_Concentration", "FI-F0"])
        [_grp_cols]
        .drop_duplicates()
    )

    fit_results = []
    plot_data   = []
    total       = len(groups)

    for i, (_, row) in enumerate(groups.iterrows()):
        plate, host, dye = row["Plate"], row["Host"], row["Dye"]
        buffer = row["Buffer"] if _has_buffer else None
        _lbl = f"{plate} {host}-{dye}" + (f" [{buffer}]" if buffer else "")
        _log(f"  Fitting [{i+1}/{total}]: {_lbl}", progress_cb)

        _q = "Plate==@plate and Host==@host and Dye==@dye"
        if _has_buffer:
            _q += " and Buffer==@buffer"
        _plate_dye = fi_df.query(_q)
        D_vals = _plate_dye["Dye_Concentration"].dropna().unique()

        for D_fixed in D_vals:
            D_fixed = float(D_fixed)
            _tol = 1e-9 * max(abs(D_fixed), 1)
            _dc = pd.to_numeric(_plate_dye["Dye_Concentration"], errors="coerce")
            df_sub = _plate_dye[(_dc - D_fixed).abs() < _tol
            ].dropna(subset=["Host_Concentration", "FI-F0"])

            # Failed-dispense wells (from an Echo exceptions file, see
            # merge_blanks) are excluded from fitting entirely — plotted as
            # crosses instead, never fed to Grubbs or curve_fit.
            if "Dispense_Failed" in df_sub.columns:
                _failed = df_sub["Dispense_Failed"].fillna(False).astype(bool).values
            else:
                _failed = np.zeros(len(df_sub), dtype=bool)
            x_failed = df_sub["Host_Concentration"].astype(float).values[_failed]
            y_failed = df_sub["FI-F0"].astype(float).values[_failed]

            x_raw = df_sub["Host_Concentration"].astype(float).values[~_failed]
            y_raw = df_sub["FI-F0"].astype(float).values[~_failed]

            # Within-replicate Grubbs
            keep_mask = np.ones(len(y_raw), dtype=bool)
            n_grubbs_tests_run = 0
            grubbs_applied_at_n3 = False
            for conc in np.unique(x_raw):
                idx = np.where(x_raw == conc)[0]
                if len(idx) >= min_n_for_grubbs:
                    local_mask = _grubbs_mask(y_raw[idx], alpha=grubbs_alpha,
                                              multiplicity_correction=grubbs_multiplicity_correction)
                    keep_mask[idx] = local_mask
                    n_grubbs_tests_run += 1
                    if len(idx) == 3:
                        grubbs_applied_at_n3 = True
                    if not local_mask.all():
                        dropped = y_raw[idx][~local_mask]
                        _log(f"    Within-rep Grubbs: dropped {len(dropped)} value(s) "
                             f"at [Host]={conc} µM ({dropped})", progress_cb)

            x_cl  = x_raw[keep_mask]
            y_cl  = y_raw[keep_mask]
            x_out = x_raw[~keep_mask]
            y_out = y_raw[~keep_mask]

            # Cross-concentration Grubbs (optional, default off per v4)
            n_cross_conc_removed = 0
            if use_cross_conc_grubbs:
                unique_concs = np.unique(x_cl)
                if len(unique_concs) >= 4:
                    conc_means = np.array([y_cl[x_cl == c].mean() for c in unique_concs])
                    curve_ok   = _grubbs_mask(conc_means, alpha=cross_conc_alpha,
                                              multiplicity_correction=grubbs_multiplicity_correction)
                    bad_concs  = set(unique_concs[~curve_ok])
                    if bad_concs:
                        cross_keep = np.array([c not in bad_concs for c in x_cl])
                        n_cross_conc_removed = int((~cross_keep).sum())
                        x_out = np.concatenate([x_out, x_cl[~cross_keep]])
                        y_out = np.concatenate([y_out, y_cl[~cross_keep]])
                        x_cl  = x_cl[cross_keep]
                        y_cl  = y_cl[cross_keep]
                        _log(f"    Cross-conc Grubbs removed concentrations: {bad_concs}",
                             progress_cb)
                        # Warn if the removed point is at a curve extreme — real hook
                        # effects and saturation drops look like outliers to Grubbs.
                        if (max(bad_concs) >= max(unique_concs) or
                                min(bad_concs) <= min(unique_concs)):
                            _log(f"    WARNING: removed point is at the curve boundary "
                                 f"— this may be a real hook effect. "
                                 f"Disable Cross-conc Grubbs if unexpected.",
                                 progress_cb)

            n = len(y_cl)
            if n < 3:
                _log(f"    Skipped (< 3 points after outlier removal)", progress_cb)
                continue

            try:
                y_max       = float(np.nanmax(y_cl))
                y_min       = float(np.nanmin(y_cl))
                Bmax_bind   = y_max if y_max != 0 else 1e-6
                Bmax_quench = y_min if y_min != 0 else -1e-6
                x_med       = float(np.nanmedian(x_cl))
                Kd_bind     = max(float(x_cl[np.argmin(np.abs(y_cl - 0.5 * Bmax_bind))]),   1e-12)
                Kd_quench   = max(float(x_cl[np.argmin(np.abs(y_cl - 0.5 * Bmax_quench))]), 1e-12)
                D_cap       = D_fixed

                bind_cfg = dict(
                    name="one_site", func=_one_site,
                    p0=[abs(Bmax_bind),   Kd_bind],
                    lo=[0,        1e-12], hi=[np.inf, np.inf])
                quench_cfg = dict(
                    name="one_site", func=_one_site,
                    p0=[-abs(Bmax_quench), Kd_quench],
                    lo=[-np.inf, 1e-12],  hi=[0, np.inf])
                quad_bind_cfg = dict(
                    name="quadratic",
                    func=lambda x, Fmax, Kd, _D=D_cap: _quadratic_binding(x, Fmax, Kd, _D),
                    p0=[abs(Bmax_bind), Kd_bind],
                    lo=[0, 1e-12], hi=[np.inf, np.inf])
                # Quenching quadratic: same tight-binding correction but Fmax < 0.
                # Required when the FDA signal is quenching (fluorescence decreases
                # on binding) AND D ≈ Kd so the linear approximation breaks down.
                quad_quench_cfg = dict(
                    name="quadratic",
                    func=lambda x, Fmax, Kd, _D=D_cap: _quadratic_binding(x, Fmax, Kd, _D),
                    p0=[-abs(Bmax_quench), Kd_quench],
                    lo=[-np.inf, 1e-12], hi=[0, np.inf])

                # Prefit the sign-free hyperbolic (one_site) models — used both
                # as Auto-mode candidates (reused, not refit) and to gate the
                # quadratic (depletion) models on D_fixed vs Kd_hyperbolic.
                try:
                    bind_cfg["prefit"] = curve_fit(
                        bind_cfg["func"], x_cl, y_cl, p0=bind_cfg["p0"],
                        bounds=(bind_cfg["lo"], bind_cfg["hi"]), maxfev=10000)
                except (RuntimeError, ValueError):
                    bind_cfg["prefit"] = None
                try:
                    quench_cfg["prefit"] = curve_fit(
                        quench_cfg["func"], x_cl, y_cl, p0=quench_cfg["p0"],
                        bounds=(quench_cfg["lo"], quench_cfg["hi"]), maxfev=10000)
                except (RuntimeError, ValueError):
                    quench_cfg["prefit"] = None

                Kd_bind_prefit   = (float(bind_cfg["prefit"][0][1])
                                     if bind_cfg["prefit"] is not None else Kd_bind)
                Kd_quench_prefit = (float(quench_cfg["prefit"][0][1])
                                     if quench_cfg["prefit"] is not None else Kd_quench)
                use_quad_bind   = D_fixed >= depletion_gate * Kd_bind_prefit
                use_quad_quench = D_fixed >= depletion_gate * Kd_quench_prefit

                if model_preference == "quadratic":
                    model_configs = [quad_bind_cfg, quad_quench_cfg]
                    if not (use_quad_bind or use_quad_quench):
                        _log(f"    QC WARNING: quadratic model forced outside its expected "
                             f"depletion regime (D_fixed={D_fixed} < {depletion_gate}×Kd_hyperbolic) "
                             f"for {plate} {host}-{dye} — Kd may be biased by noise, not "
                             f"depletion.", progress_cb)
                elif model_preference == "stern_volmer":
                    model_configs = [quench_cfg]
                else:
                    # Auto and one_site: hyperbolic models always offered; the
                    # depletion-correction (quadratic) models are gated on
                    # whether D_fixed is comparable to the hyperbolic Kd.
                    model_configs = [bind_cfg, quench_cfg]
                    if use_quad_bind:
                        model_configs.append(quad_bind_cfg)
                    if use_quad_quench:
                        model_configs.append(quad_quench_cfg)

                best_fit = None
                best_by_sign = {"binding": None, "quenching": None}
                for mcfg in model_configs:
                    k = len(mcfg["p0"])
                    if n <= k:
                        continue
                    prefit = mcfg.get("prefit")
                    if prefit is not None:
                        popt, pcov = prefit
                    else:
                        try:
                            popt, pcov = curve_fit(
                                mcfg["func"], x_cl, y_cl,
                                p0=mcfg["p0"], bounds=(mcfg["lo"], mcfg["hi"]), maxfev=10000)
                        except (RuntimeError, ValueError):
                            try:
                                popt, pcov = curve_fit(
                                    mcfg["func"], x_cl, y_cl,
                                    p0=[mcfg["p0"][0], x_med],
                                    bounds=(mcfg["lo"], mcfg["hi"]), maxfev=10000)
                            except (RuntimeError, ValueError):
                                continue

                    # Need at least k+3 points: k for parameters + 1 residual df
                    # + 1 so r²_adj is defined + 1 more so AICc's small-sample
                    # correction (2K(K+1)/(n-K-1), K=k+1 below) is defined.
                    if n <= k + 2:
                        continue

                    y_pred    = mcfg["func"](x_cl, *popt)
                    residuals = y_cl - y_pred
                    ss_res    = float(np.sum(residuals**2))
                    if ss_res <= 0:
                        continue
                    ss_tot = float(np.sum((y_cl - np.mean(y_cl))**2))
                    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                    # AICc for nonlinear regression: residual variance is
                    # itself an estimated parameter, so the parameter count
                    # used in the AIC/AICc formulas is K = k+1, not k (k =
                    # curve parameters only). GraphPad Curve Fitting Guide,
                    # "Model diagnostics": "k in the equations above is
                    # replaced by k+1" for nonlinear regression. Inert here —
                    # every candidate model in this pipeline has the same
                    # k=2, so the correction is a constant offset that
                    # cancels in every AICc comparison (Model/Binding_mode
                    # selection does not change) — but the reported absolute
                    # AICc value does change, and pipeline_ki.py's _aicc
                    # (kept in sync with this) is NOT inert, since Standard/
                    # Wang (k=3) compete against HillSlope (k=4) there.
                    K    = k + 1
                    aicc = n * np.log(ss_res / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)

                    cand = dict(name=mcfg["name"], popt=popt, pcov=pcov,
                                r2_adj=r2_adj, residuals=residuals, aicc=aicc, k=k)
                    sign = _binding_mode(float(popt[0]))
                    if best_by_sign[sign] is None or aicc < best_by_sign[sign]["aicc"]:
                        best_by_sign[sign] = cand
                    if best_fit is None or aicc < best_fit["aicc"]:
                        best_fit = cand

                if best_fit is None:
                    _log(f"    All models failed for D={D_fixed}", progress_cb)
                    continue

                # Sign ambiguity (auto mode only): was the winning binding/
                # quenching direction decided by a wide AICc margin, or could
                # fitting noise have picked either sign? ΔAICc < 2 is the
                # conventional "no strong preference" threshold (Burnham &
                # Anderson, Model Selection and Multimodel Inference, 2002).
                _winning_sign  = _binding_mode(float(best_fit["popt"][0]))
                _opposite_sign = "quenching" if _winning_sign == "binding" else "binding"
                _opp_fit = best_by_sign.get(_opposite_sign)
                if _opp_fit is not None:
                    delta_aicc_vs_opposite_sign = float(_opp_fit["aicc"] - best_fit["aicc"])
                    sign_ambiguous = delta_aicc_vs_opposite_sign < DELTA_AICC_AMBIGUOUS
                else:
                    delta_aicc_vs_opposite_sign = np.nan
                    sign_ambiguous = False

                # Depletion-gate boundary-proximity flag (additive, does not
                # change Status/Model): the gate that decided whether to try
                # the depletion-aware quadratic model used an estimate
                # (Kd_bind_prefit/Kd_quench_prefit) from the simpler,
                # non-depleting model — itself potentially biased in exactly
                # the regime the gate exists to detect (see docstring above).
                # Persisting the margin, and flagging when it's close to the
                # gate threshold, surfaces curves where that circularity is
                # most likely to matter, without re-running anything.
                depletion_gate_margin = (D_fixed / Kd_bind_prefit if _winning_sign == "binding"
                                          else D_fixed / Kd_quench_prefit)
                _band_lo, _band_hi = DEPLETION_GATE_MARGIN_BAND
                _margin_in_band = (depletion_gate * _band_lo <= depletion_gate_margin
                                    <= depletion_gate * _band_hi)
                if best_fit["name"] == "quadratic" and _margin_in_band:
                    _log(f"    QC WARNING: quadratic model won for {plate} {host}-{dye} "
                         f"(D={D_fixed}) with Depletion_gate_margin={depletion_gate_margin:.3g} "
                         f"— within {_band_lo}-{_band_hi}x the depletion_gate threshold "
                         f"({depletion_gate}) itself, so the decision to try this model "
                         f"was close to the boundary (informational only — does not "
                         f"affect Status).", progress_cb)
                elif best_fit["name"] == "one_site" and _margin_in_band:
                    # Same circularity, other side of the gate: one_site won, but
                    # the decision to SKIP quadratic (never tried, so it never got
                    # a chance to compete) was itself close to the gate threshold.
                    _quad_tried = use_quad_bind if _winning_sign == "binding" else use_quad_quench
                    if not _quad_tried:
                        _log(f"    QC WARNING: one_site model won for {plate} {host}-{dye} "
                             f"(D={D_fixed}) with Depletion_gate_margin={depletion_gate_margin:.3g} "
                             f"— within {_band_lo}-{_band_hi}x the depletion_gate threshold "
                             f"({depletion_gate}) itself; quadratic was NOT tried here, but "
                             f"the decision to skip it was close to the boundary "
                             f"(informational only — does not affect Status).", progress_cb)

                popt, pcov = best_fit["popt"], best_fit["pcov"]
                dof        = n - best_fit["k"]
                kd_var     = float(pcov[1, 1])
                # Guard against inf covariance (non-convergence / ill-conditioned fit)
                kd_se      = (float(np.sqrt(kd_var))
                              if kd_var >= 0 and np.isfinite(kd_var) else np.nan)
                t_crit_val = t.ppf(0.975, dof)
                kd_ci_low  = float(popt[1]) - t_crit_val * kd_se if not np.isnan(kd_se) else np.nan
                kd_ci_high = float(popt[1]) + t_crit_val * kd_se if not np.isnan(kd_se) else np.nan

                try:
                    _, p_normal = (normaltest(best_fit["residuals"])
                                   if len(best_fit["residuals"]) >= 8
                                   else (None, np.nan))
                except Exception:
                    p_normal = np.nan
                if p_normal is None:
                    p_normal = np.nan
                try:
                    runs_test_p = _runs_test_p(best_fit["residuals"][np.argsort(x_cl)])
                except Exception:
                    runs_test_p = np.nan

                # Positive concentrations only for the lo range check
                x_pos     = x_cl[x_cl > 0]
                conc_min  = float(np.min(x_pos)) if len(x_pos) > 0 else float(np.nanmin(x_cl))
                conc_max  = float(np.nanmax(x_cl))

                plot_data.append({
                    "plate":        plate,
                    "host":         host,
                    "dye":          dye,
                    "buffer":       buffer,
                    "x_cleaned":    x_cl,
                    "y_cleaned":    y_cl,
                    "x_outliers":   x_out,
                    "y_outliers":   y_out,
                    "x_failed":     x_failed,
                    "y_failed":     y_failed,
                    "model_name":   best_fit["name"],
                    "binding_mode": _binding_mode(float(popt[0])),
                    "popt":         popt,
                    "D_fixed":      D_fixed,
                    "r2_adj":       best_fit["r2_adj"],
                    "kd_se":        kd_se,
                    "sign_ambiguous": sign_ambiguous,
                    "status":       None,
                })

                _fit_row = {
                    "Plate":             plate,
                    "Host":              host,
                    "Dye":               dye,
                    "Dye_Concentration": D_fixed,
                    "Model":             best_fit["name"],
                    "Binding_mode":      _binding_mode(float(popt[0])),
                    "Bmax":              float(popt[0]),
                    "Kd":                float(popt[1]),
                    "Kd_SE":             kd_se,
                    "Kd_95CI_low":       kd_ci_low,
                    "Kd_95CI_high":      kd_ci_high,
                    "R2_adj":            best_fit["r2_adj"],
                    "AICc":              best_fit["aicc"],
                    "Sign_ambiguous":    sign_ambiguous,
                    "Delta_AICc_vs_opposite_sign": delta_aicc_vs_opposite_sign,
                    "Depletion_gate_margin": depletion_gate_margin,
                    "n_points":          n,
                    "n_outliers":        len(x_out),
                    "n_dispense_failed": len(x_failed),
                    "n_grubbs_tests_run":    n_grubbs_tests_run,
                    "Grubbs_applied_at_n3":  grubbs_applied_at_n3,
                    "n_cross_conc_removed":  n_cross_conc_removed,
                    "Normality_p":       p_normal,
                    "Runs_test_p":       runs_test_p,
                    "Host_Conc_min":     conc_min,
                    "Host_Conc_max":     conc_max,
                }
                if buffer is not None:
                    _fit_row["Buffer"] = buffer
                fit_results.append(_fit_row)

            except Exception as exc:
                _log(f"    Error fitting D={D_fixed}: {exc}", progress_cb)
                continue

    _log(f"Fitting complete: {len(fit_results)} curves.", progress_cb)
    return fit_results, plot_data


# ── Stage 6: apply thresholds (cheap, no refit) ───────────────────────────────

def apply_thresholds(fit_results: list, plot_data: list,
                     r2_threshold: float = PASS_R2_DEFAULT,
                     kd_range_lo: float = KD_RANGE_FACTOR_LO_DEFAULT,
                     kd_range_hi: float = KD_RANGE_FACTOR_HI_DEFAULT,
                     kd_cv_threshold: float = KD_CV_THRESHOLD_DEFAULT,
                     normality_threshold: float = NORMALITY_P_THRESHOLD_DEFAULT,
                     runs_test_p_threshold: float = RUNS_TEST_P_THRESHOLD_DEFAULT,
                     fi_df: pd.DataFrame = None,
                     host_autofl_factor: float = HOST_AUTOFL_FACTOR_DEFAULT,
                     host_autofl_mode: str = "stat",
                     host_autofl_z: float = HOST_AUTOFL_Z_DEFAULT) -> pd.DataFrame:
    """
    Adds Status, Fail_reason, and Confidence to a copy of fit_results
    DataFrame. Also updates the 'status' field in each plot_data dict
    in-place.

    Screening/triage use case: false hits and rank-order errors are costly
    (wasted follow-up), but imprecise absolute Kd magnitude is not — a real
    hit gets re-measured properly downstream anyway. So Status (PASS/FAIL)
    is driven only by R2_adj, Kd range sanity, and autofluorescence (a
    genuine false-hit source); a wide Kd confidence interval does NOT fail
    Status — it instead sets Confidence="Low — wide CI, recommend retest"
    alongside the existing Status, so a real-but-imprecise hit still
    surfaces for follow-up instead of being silently dropped.

    host_autofl_mode: "stat" (default) flags host autofluorescence via a
    textbook two-sample z-test against the dye-blank mean (same
    Plate/Dye/Buffer): host_mean - blank_mean > z * sqrt(host_sem**2 +
    blank_sem**2) — mirrors the guest-autofluorescence gate in
    pipeline_ki.apply_thresholds_ki. "ratio" falls back to the older
    fixed-ratio mean comparison (host-only mean > host_autofl_factor ×
    blank mean).

    Normality_p (D'Agostino-Pearson test on residuals) is diagnostic only
    and does not affect Status: curves with Normality_p < normality_threshold
    get a Fail_reason note but are NOT failed on it alone — unreliable at
    the small n typical of these curves.

    Runs_test_p (Wald-Wolfowitz runs test on residual signs vs. x-order,
    set in fit_curves) is surfaced the same additive way: curves with
    Runs_test_p < runs_test_p_threshold get a Fail_reason note but are NOT
    failed on it alone (does not affect Status or Confidence).
    """
    df = pd.DataFrame(fit_results).copy()
    if df.empty:
        return df

    fail_r2    = df["R2_adj"] < r2_threshold
    fail_kd    = df["Kd"].isna()
    fail_kd_lo = df["Kd"] < kd_range_lo * df["Host_Conc_min"]
    fail_kd_hi = df["Kd"] > kd_range_hi * df["Host_Conc_max"]

    # Confidence (Tier 2): additive, does NOT gate Status — a real hit with a
    # wide CI should still surface for follow-up, just flagged for retest.
    low_kd_cv = pd.Series(False, index=df.index)
    if "Kd_SE" in df.columns and kd_cv_threshold is not None:
        low_kd_cv = (df["Kd_SE"] / df["Kd"]) > kd_cv_threshold
    low_kd_ci = pd.Series(False, index=df.index)
    if "Kd_95CI_low" in df.columns:
        low_kd_ci = df["Kd_95CI_low"] <= 0
    low_confidence = low_kd_cv | low_kd_ci
    df["Confidence"] = "High"
    df.loc[low_confidence, "Confidence"] = "Low — wide CI, recommend retest"

    fail_normality = pd.Series(False, index=df.index)
    if "Normality_p" in df.columns and normality_threshold is not None:
        fail_normality = df["Normality_p"] < normality_threshold

    fail_runs_test = pd.Series(False, index=df.index)
    if "Runs_test_p" in df.columns and runs_test_p_threshold is not None:
        fail_runs_test = df["Runs_test_p"] < runs_test_p_threshold  # doesn't touch fail_any

    # Host autofluorescence gate: flag curves whose host-only wells fluoresce
    # above background for the same Plate/Dye (and Buffer, if used).
    # Host-only wells carry no Dye tag, so the dye is resolved via the
    # Chromatic channel used on that Plate.
    fail_host_autofl = pd.Series(False, index=df.index)
    host_autofl_not_assessed = pd.Series(False, index=df.index)
    if fi_df is not None and (host_autofl_factor is not None or host_autofl_z is not None):
        is_blank = fi_df["Compound ID"].fillna("").str.strip().str.lower() == "blank"
        ho_mask  = fi_df["Host"].notna() & fi_df["Dye"].isna() & ~is_blank
        db_mask  = is_blank & fi_df["Dye"].notna()
        _has_buf_fi = "Buffer" in fi_df.columns and "Buffer" in df.columns
        dye_lookup = (fi_df[fi_df["Dye"].notna()]
                      .groupby(["Plate", "Chromatic"])["Dye"].first().to_dict())
        ho = fi_df[ho_mask].copy()
        ho["Dye"] = [dye_lookup.get((p, c)) for p, c in zip(ho["Plate"], ho["Chromatic"])]
        ho = ho.dropna(subset=["Dye"])
        ho_grp  = ["Plate", "Host", "Dye"] + (["Buffer"] if _has_buf_fi else [])
        db_grp  = ["Plate", "Dye"]         + (["Buffer"] if _has_buf_fi else [])

        if host_autofl_mode == "ratio":
            ho_mean = ho.groupby(ho_grp)["Fluorescence"].mean().to_dict()
            db_mean = fi_df[db_mask].groupby(db_grp)["Fluorescence"].mean().to_dict()
            for idx, row in df.iterrows():
                ho_key = (row["Plate"], row["Host"], row["Dye"]) + ((row["Buffer"],) if _has_buf_fi else ())
                db_key = (row["Plate"], row["Dye"])               + ((row["Buffer"],) if _has_buf_fi else ())
                if ho_key in ho_mean and db_key in db_mean:
                    if ho_mean[ho_key] > host_autofl_factor * db_mean[db_key]:
                        fail_host_autofl.at[idx] = True
        else:
            # Statistical (default): two-sample Welch's t-test on the pooled
            # SE (sqrt(sem1^2+sem2^2)), mirroring the guest-autofluorescence
            # gate in pipeline_ki.py. sem is NaN only when a group has a
            # single well — that group's uncertainty must NOT be silently
            # zeroed out (doing so shrinks pooled_sem and makes a singleton-
            # well comparison the *easiest* to fail, the opposite of
            # conservative). Instead, skip the test for that combo and leave
            # a diagnostic note; when both groups have >=2 wells, use a
            # Welch t critical value (dof = min(count)-1, the conservative
            # choice) instead of a fixed z, which converges to the same
            # z-based critical value at large replicate counts but is more
            # conservative at the small counts typical here.
            ho_stats = ho.groupby(ho_grp)["Fluorescence"].agg(["mean", "sem", "count"])
            db_stats = fi_df[db_mask].groupby(db_grp)["Fluorescence"].agg(["mean", "sem", "count"])
            p_crit = float(norm.cdf(host_autofl_z))
            for idx, row in df.iterrows():
                ho_key = (row["Plate"], row["Host"], row["Dye"]) + ((row["Buffer"],) if _has_buf_fi else ())
                db_key = (row["Plate"], row["Dye"])               + ((row["Buffer"],) if _has_buf_fi else ())
                if ho_key in ho_stats.index and db_key in db_stats.index:
                    ho_mean_v, ho_sem, ho_n = ho_stats.loc[ho_key, ["mean", "sem", "count"]]
                    db_mean_v, db_sem, db_n = db_stats.loc[db_key, ["mean", "sem", "count"]]
                    if ho_n < 2 or db_n < 2:
                        host_autofl_not_assessed.at[idx] = True
                        continue
                    dof = min(ho_n, db_n) - 1
                    t_crit = float(t.ppf(p_crit, dof))
                    pooled_sem = np.sqrt(ho_sem**2 + db_sem**2)
                    if (ho_mean_v - db_mean_v) > t_crit * pooled_sem:
                        fail_host_autofl.at[idx] = True

    fail_any = fail_r2 | fail_kd | fail_kd_lo | fail_kd_hi | fail_host_autofl
    df["Status"]      = "PASS"
    df.loc[fail_any, "Status"] = "FAIL"

    df["Fail_reason"] = ""
    df.loc[fail_r2,    "Fail_reason"] += df.loc[fail_r2, "R2_adj"].apply(
        lambda v: f"R²_adj={v:.3f} < {r2_threshold}; ")
    df.loc[fail_kd,    "Fail_reason"] += "Kd is NaN; "
    df.loc[fail_kd_lo, "Fail_reason"] += df.loc[fail_kd_lo].apply(
        lambda r: f"Kd={r['Kd']:.3g} < {kd_range_lo}×[Host]_min={r['Host_Conc_min']:.3g}; ", axis=1)
    df.loc[fail_kd_hi, "Fail_reason"] += df.loc[fail_kd_hi].apply(
        lambda r: f"Kd={r['Kd']:.3g} > {kd_range_hi}×[Host]_max={r['Host_Conc_max']:.3g}; ", axis=1)
    df.loc[fail_host_autofl, "Fail_reason"] += "Host autofluorescence; "
    df.loc[host_autofl_not_assessed, "Fail_reason"] += \
        "Host autofluorescence not assessed — n=1 replicate well(s); "
    if "Sign_ambiguous" in df.columns:
        _amb = df["Sign_ambiguous"].fillna(False).astype(bool)
        df.loc[_amb, "Fail_reason"] += df.loc[_amb, "Delta_AICc_vs_opposite_sign"].apply(
            lambda v: f"Binding/quenching sign ambiguous (ΔAICc={v:.2f} vs opposite sign); ")
    df.loc[fail_normality, "Fail_reason"] += df.loc[fail_normality, "Normality_p"].apply(
        lambda v: f"Residuals non-normal (p={v:.3f}, diagnostic only); ")
    df.loc[fail_runs_test, "Fail_reason"] += df.loc[fail_runs_test, "Runs_test_p"].apply(
        lambda v: f"Non-random residual pattern (runs test p={v:.4f}, diagnostic only); ")
    df.loc[low_confidence, "Fail_reason"] += df.loc[low_confidence].apply(
        lambda r: f"Low confidence (Kd_SE/Kd={r['Kd_SE']/r['Kd']:.2f}, 95% CI low={r['Kd_95CI_low']:.3g}); "
        if "Kd_SE" in r and "Kd_95CI_low" in r else "Low confidence; ", axis=1)
    df["Fail_reason"] = df["Fail_reason"].str.rstrip("; ")

    _has_buf  = "Buffer" in df.columns
    key_cols  = ["Plate", "Host", "Dye", "Dye_Concentration"] + (["Buffer"] if _has_buf else [])
    status_lookup = df.set_index(key_cols)["Status"].to_dict()
    for entry in plot_data:
        _key = (entry.get("plate", ""), entry["host"], entry["dye"], entry["D_fixed"])
        if _has_buf:
            _key = _key + (entry.get("buffer"),)
        entry["status"] = status_lookup.get(_key, "FAIL")

    return df


# ── Plot rendering ────────────────────────────────────────────────────────────

def render_plot(ax, entry: dict, show_status_color: bool = True,
                color_data: str = PLOT_COLOR_DATA, color_fit: str = PLOT_COLOR,
                color_resid: str = None,
                ax_resid=None,
                title_fontsize: float = 12,
                title_fontfamily: str = "sans-serif",
                title_fontweight: str = "normal",
                title_fontstyle: str = "normal",
                axis_fontsize: float = 14,
                tick_fontsize: float = 12,
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

    stats = (
        pd.DataFrame({"x": x_cl, "y": y_plot_cl})
        .groupby("x")["y"]
        .agg(["mean", "std"])
    )
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                fmt="o", color=color_data, ecolor=color_data,
                elinewidth=lw_errorbar, markersize=ms_data, capsize=2, zorder=3)

    if len(x_out) > 0:
        ax.scatter(x_out, y_plot_out, color="red", marker="o",
                   s=ms_data**2 * 1.5, zorder=5)

    if show_excluded and len(x_failed) > 0:
        ax.scatter(x_failed, y_plot_failed, color="black", marker="x",
                   s=ms_data**2 * 1.5, linewidths=1.5, zorder=6,
                   label="Dispense failed")

    x_line = np.linspace(x_cl.min(), x_cl.max(), 300)
    y_line = _eval_model(entry["model_name"], x_line, popt, entry["D_fixed"])
    if normalise_y:
        y_line = (y_line - y_min) / y_range
    ax.plot(x_line, y_line, color=color_fit, linewidth=lw_fit, zorder=2)

    kd_se    = entry["kd_se"]
    se_str   = f" ± {_fmt_uM(kd_se)}" if not np.isnan(kd_se) else ""
    mode_str = entry.get("binding_mode", "")
    if entry.get("sign_ambiguous"):
        mode_str += " (sign ambiguous)"

    status      = entry.get("status")
    title_color = "red" if (show_status_color and status == "FAIL") else "black"
    _buf_str    = f"  [{entry['buffer']}]" if entry.get("buffer") else ""
    ax.set_title(
        f"{entry['host']} – {entry['dye']}  (D = {entry['D_fixed']} µM){_buf_str}\n"
        f"Kd = {_fmt_uM(float(popt[1]))}{se_str} µM  |  "
        f"R²adj = {entry['r2_adj']:.3f}\n"
        f"[{entry['model_name']} / {mode_str}]",
        fontsize=title_fontsize, fontweight=title_fontweight,
        fontfamily=title_fontfamily, fontstyle=title_fontstyle,
        color=title_color)
    y_label = "Normalised Intensity" if normalise_y else "FI – F₀"
    ax.set_ylabel(y_label, fontsize=axis_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    ax.xaxis.set_major_locator(
        MultipleLocator(x_tick_step) if x_tick_step else MaxNLocator(nbins=6))
    if sci_notation_y and not normalise_y:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    resid_axis_fs = axis_fontsize
    resid_tick_fs = tick_fontsize
    if ax_resid is not None:
        ax.tick_params(labelbottom=False)
        rc      = color_resid or color_data
        y_pred  = _eval_model(entry["model_name"], x_cl, popt, entry["D_fixed"])
        if normalise_y:
            resid = y_plot_cl - (y_pred - y_min) / y_range
        else:
            resid = y_cl - y_pred
        stats_r = (pd.DataFrame({"x": x_cl, "r": resid})
                   .groupby("x")["r"].agg(["mean", "std"]))
        ax_resid.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
        ax_resid.errorbar(stats_r.index, stats_r["mean"], yerr=stats_r["std"],
                         fmt="o", color=rc, ecolor=rc,
                         elinewidth=lw_errorbar * 0.7, markersize=ms_resid,
                         capsize=2, zorder=3)
        ax_resid.set_ylabel("Resid.", fontsize=resid_axis_fs)
        ax_resid.tick_params(labelsize=resid_tick_fs)
        ax_resid.set_xlabel("[Host] (µM)", fontsize=resid_axis_fs)
    else:
        ax.set_xlabel("[Host] (µM)", fontsize=axis_fontsize)


# ── Stage 7: save outputs ─────────────────────────────────────────────────────

COLS_PER_PAGE  = 4
ROWS_PER_PAGE  = 3


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


def save_outputs(state: PipelineState, output_folder: str,
                 r2_threshold: float = PASS_R2_DEFAULT,
                 progress_cb: ProgressCb = None,
                 plot_color_data: str = PLOT_COLOR_DATA,
                 plot_color_fit: str = PLOT_COLOR,
                 plot_color_resid: str = None,
                 show_residuals: bool = False,
                 export_individual: bool = False,
                 layout_cfg: dict = None,
                 title_fontsize: float = 12,
                 title_fontfamily: str = "sans-serif",
                 title_fontweight: str = "normal",
                 title_fontstyle: str = "normal",
                 axis_fontsize: float = 14,
                 tick_fontsize: float = 12,
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
    _tkw = dict(title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
                title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
                axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
                normalise_y=normalise_y, sci_notation_y=sci_notation_y,
                lw_fit=lw_fit, ms_data=ms_data,
                lw_errorbar=lw_errorbar, ms_resid=ms_resid,
                x_min=x_min, x_max=x_max, x_tick_step=x_tick_step,
                show_excluded=show_excluded)
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    for d in (reports_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    ts    = _folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S")
    saved = []

    def _save_csv(df, folder, label):
        path = os.path.join(folder, f"{ts}_{label}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        _log(f"Saved: {path}", progress_cb)
        saved.append(path)

    if state.merged_mapping is not None:
        _save_csv(state.merged_mapping, reports_dir, "mapping")
    if state.fluorescence is not None:
        _save_csv(state.fluorescence, reports_dir, "raw_data")
    if state.merged is not None:
        _save_csv(state.merged, reports_dir, "merged_with_blanks")
    if state.fi_df is not None:
        _save_csv(state.fi_df, reports_dir, "blank_corrected")

    if state.df_results is not None and state.df_results.empty:
        _log("No fit results to save (0 curves fitted) — skipping fit-results Excel/PDF.", progress_cb)
    elif state.df_results is not None:
        df_pass = state.df_results[state.df_results["Status"] == "PASS"]
        df_fail = state.df_results[state.df_results["Status"] == "FAIL"]

        results_path = os.path.join(results_dir, f"{ts}_binding_fit_results.xlsx")
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            df_pass.to_excel(writer, sheet_name="PASS", index=False)
            df_fail.to_excel(writer, sheet_name="FAIL", index=False)
        _log(f"Saved: {results_path}", progress_cb)
        saved.append(results_path)

        if state.fi_df is not None:
            _has_buf = ("Buffer" in state.df_results.columns
                        and "Buffer" in state.fi_df.columns)
            _sk = ["Plate", "Host", "Dye", "Dye_Concentration"] + (["Buffer"] if _has_buf else [])
            status_lookup = (
                state.df_results
                .set_index(_sk)["Status"]
                .to_dict()
            )
            pass_blocks, fail_blocks = [], []
            src = state.fi_df.dropna(subset=["Host", "Dye", "FI-F0"])
            for grp_key, grp in src.groupby(_sk):
                if _has_buf:
                    plate, host, dye, D_fixed, buf = grp_key
                else:
                    plate, host, dye, D_fixed = grp_key
                    buf = None
                grp    = grp.sort_values("Host_Concentration")
                status = status_lookup.get(grp_key, "FAIL")
                concs  = grp["Host_Concentration"].drop_duplicates().reset_index(drop=True)
                fi_by_conc = grp.groupby("Host_Concentration")["FI-F0"].apply(list)
                max_reps   = fi_by_conc.apply(len).max()
                _hdr   = f"{host} | {dye} [{D_fixed}] µM"
                if buf:
                    _hdr += f" ({buf})"
                block  = pd.DataFrame({_hdr: concs})
                for i in range(max_reps):
                    block[f"FI-F0_{i+1}"] = (
                        fi_by_conc.apply(lambda v, _i=i: v[_i] if _i < len(v) else pd.NA)
                        .reset_index(drop=True)
                    )
                (pass_blocks if status == "PASS" else fail_blocks).append(block)

            wide_path = os.path.join(results_dir, f"{ts}_data.xlsx")
            with pd.ExcelWriter(wide_path, engine="xlsxwriter") as writer:
                (pd.concat(pass_blocks, axis=1) if pass_blocks else pd.DataFrame()).to_excel(
                    writer, sheet_name="PASS", index=False)
                (pd.concat(fail_blocks, axis=1) if fail_blocks else pd.DataFrame()).to_excel(
                    writer, sheet_name="FAIL", index=False)
            _log(f"Saved: {wide_path}", progress_cb)
            saved.append(wide_path)

    # QC PDFs
    for label, fig in state.qc_figures:
        p = os.path.join(reports_dir,
                         f"{ts}_{label.replace(' ', '_').replace(':', '')}.pdf")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        _log(f"Saved: {p}", progress_cb)
        saved.append(p)

    if state.plot_data:
        lc       = layout_cfg or {}
        pdf_path = os.path.join(plots_dir, f"{ts}_binding_results.pdf")
        plates   = sorted(set(e["plate"] for e in state.plot_data))
        with PdfPages(pdf_path) as pdf:
            for plate in plates:
                page_items = [e for e in state.plot_data if e["plate"] == plate]
                n     = len(page_items)
                cols  = min(lc.get("pdf_cols", COLS_PER_PAGE), n)
                rows  = int(np.ceil(n / cols))
                cl = layout_utils.cell_layout(lc, show_residuals=show_residuals, show_dots=False)
                fl = layout_utils.figure_layout(lc, rows, cols, cl.row_h_in)
                fig   = plt.figure(figsize=(fl.fig_w, fl.fig_h))
                outer = GridSpec(rows, cols, figure=fig, **fl.outer_kwargs())
                for i, entry in enumerate(page_items):
                    ri, ci = divmod(i, cols)
                    if show_residuals:
                        inner = outer[ri, ci].subgridspec(
                            cl.sub_rows, 1, height_ratios=cl.height_ratios,
                            hspace=cl.resid_gap)
                        ax   = fig.add_subplot(inner[0])
                        ax_r = fig.add_subplot(inner[1], sharex=ax)
                        render_plot(ax, entry,
                                    color_data=plot_color_data, color_fit=plot_color_fit,
                                    color_resid=plot_color_resid, ax_resid=ax_r, **_tkw)
                    else:
                        render_plot(fig.add_subplot(outer[ri, ci]), entry,
                                    color_data=plot_color_data, color_fit=plot_color_fit,
                                    color_resid=plot_color_resid, **_tkw)
                for i in range(n, rows * cols):
                    ri, ci = divmod(i, cols)
                    fig.add_subplot(outer[ri, ci]).set_visible(False)
                fig.suptitle(f"Plate: {plate}", fontsize=11, fontweight="bold", y=0.998)
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
                safe_nm  = re.sub(r"[^\w\-]", "_",
                                  f"{entry['host']}_{entry['dye']}_{entry['D_fixed']}uM{_buf_sfx}")
                ind_path = os.path.join(plots_dir, "individual", status,
                                        f"{safe_nm}.pdf")
                fig = plt.figure(figsize=(_fl.fig_w, _fl.fig_h))
                gs  = GridSpec(_cl.sub_rows, 1, figure=fig, height_ratios=_cl.height_ratios,
                               hspace=_cl.resid_gap, **_fl.single_kwargs())
                if show_residuals:
                    ax   = fig.add_subplot(gs[0])
                    ax_r = fig.add_subplot(gs[1], sharex=ax)
                else:
                    ax   = fig.add_subplot(gs[0])
                    ax_r = None
                render_plot(ax, entry,
                            color_data=plot_color_data, color_fit=plot_color_fit,
                            color_resid=plot_color_resid, ax_resid=ax_r, **_tkw)
                fig.savefig(ind_path, dpi=150)
                plt.close(fig)
                saved.append(ind_path)
            _log(f"Saved {len(state.plot_data)} individual plot(s).", progress_cb)

    return saved


# ── Preview: list files that save_outputs would write ─────────────────────────

def preview_save_files(state: PipelineState, output_folder: str,
                       export_individual: bool = False,
                       input_folder: str = "") -> list[dict]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    ts          = _folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S")
    files       = []

    def _f(folder, name, desc):
        files.append({"path": os.path.join(folder, f"{ts}_{name}"), "description": desc})

    if state.merged_mapping is not None:
        _f(reports_dir, "mapping.csv",              "Merged dye/host mapping")
    if state.fluorescence is not None:
        _f(reports_dir, "raw_data.csv",             "Raw fluorescence data")
    if state.merged is not None:
        _f(reports_dir, "merged_with_blanks.csv",   "Merged data (with blank labels)")
    if state.fi_df is not None:
        _f(reports_dir, "blank_corrected.csv",      "Background-subtracted (FI-F0)")
    for label, _ in state.qc_figures:
        _f(reports_dir, f"{label.replace(' ', '_').replace(':', '')}.pdf",
           f"QC plot: {label}")
    if state.df_results is not None:
        _f(results_dir, "binding_fit_results.xlsx", "Fit results (PASS/FAIL sheets)")
        if state.fi_df is not None:
            _f(results_dir, "data.xlsx",            "Wide-format FI-F0 (PASS/FAIL sheets)")
    if state.plot_data:
        _f(plots_dir, "binding_results.pdf", "Binding curve plots (1 plate per page)")
        if export_individual:
            files.append({"path": os.path.join(plots_dir, "individual", "PASS", "…"),
                          "description": f"Individual PDFs — {len(state.plot_data)} plot(s) → individual/PASS|FAIL/"})

    return files
