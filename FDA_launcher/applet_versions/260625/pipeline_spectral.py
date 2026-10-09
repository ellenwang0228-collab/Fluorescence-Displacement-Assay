#!/usr/bin/env python3
"""
Spectral scan pipeline — excitation/emission spectra + single-wavelength
binding fit, reusing the Direct Binding (Kd) pipeline wherever the data
shape lines up.

Raw files are plate-reader spectral-scan exports (.csv, latin1-encoded):
  - A header block declares one or more scan procedures (excitation scan:
    Ex wavelength varies, Em fixed; emission scan: Em varies, Ex fixed) and
    the number of wavelength steps in each.
  - Each wavelength step is recorded as its own "Chromatic: N" 16x24 plate
    block (only the wells actually used in that run are filled in).

Host/Dye mapping and dye-blank handling reuse the same Echo-export
dye_map/host_map/blank_map convention as Direct Binding — a spectral scan is
the same host-dye titration, just read out across many wavelengths instead
of one filter pair. Once a single wavelength is selected, the resulting
table has exactly the shape pipeline_fda.fit_curves expects (Host, Dye,
Dye_Concentration, Host_Concentration, FI-F0, Plate), so the Kd fit itself,
its plot rendering and its Excel/PDF export are delegated to pipeline_fda
rather than re-implemented here.
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
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import pipeline_fda as pl_fda

ProgressCb = Optional[Callable[[str], None]]

_ROW_LABELS = list(string.ascii_uppercase[:16])  # A–P

_RANGE_RE = re.compile(
    r'^\s*(\d+):\s*(\d+)-(\d+)/(\d+)-(\d+)\s*-->\s*(\d+)-(\d+)/(\d+)-(\d+)')
_STEPS_RE = re.compile(r'^\s*(\d+):\s*No\.\s*of scan values:\s*(\d+)')
_ID1_RE   = re.compile(r'ID1:\s*(\S+)')
_CHROM_RE = re.compile(r'Chromatic:\s*(\d+)')


def _log(msg: str, cb: ProgressCb) -> None:
    if cb:
        cb(msg)
    else:
        print(msg)


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStateSpectral:
    merged_mapping: Optional[pd.DataFrame] = None
    fluorescence:   Optional[pd.DataFrame] = None   # long: Well/Plate/Chromatic/ScanType/Wavelength/Fluorescence
    merged:         Optional[pd.DataFrame] = None
    fi_df:          Optional[pd.DataFrame] = None
    spectra_figures: list = field(default_factory=list)   # list[(label, Figure)]
    fit_results:    list = field(default_factory=list)
    plot_data:      list = field(default_factory=list)
    df_results:     Optional[pd.DataFrame] = None
    overlay_figures: list = field(default_factory=list)   # list[(label, Figure)]
    fit_wavelength: Optional[float] = None
    fit_scan_type:  Optional[str]   = None


# ── Stage 1: load raw spectral scan files ─────────────────────────────────────

def _parse_scan_procedures(lines: list[str]) -> dict:
    """Parse the 'Used filter settings and gain values' header block into
    {procedure_idx: {"scan_type": "ex"|"em", "start": float, "end": float, "steps": int}}.

    Example:
      1: 320-10/460-16 --> 400-10/460-16   (Ex varies 320->400, Em fixed at 460 -> excitation scan)
      1: No. of scan values: 81
      2: 360-16/400-10 --> 360-16/550-10   (Ex fixed at 360, Em varies 400->550 -> emission scan)
      2: No. of scan values: 31
    """
    procedures: dict[int, dict] = {}
    for line in lines:
        m = _RANGE_RE.match(line)
        if m:
            idx = int(m.group(1))
            ex1, _bw1, em1, _bw2, ex2, _bw3, em2, _bw4 = (int(g) for g in m.groups()[1:])
            if ex1 != ex2 and em1 == em2:
                procedures.setdefault(idx, {}).update(
                    scan_type="ex", start=float(ex1), end=float(ex2))
            elif em1 != em2 and ex1 == ex2:
                procedures.setdefault(idx, {}).update(
                    scan_type="em", start=float(em1), end=float(em2))
            continue
        m2 = _STEPS_RE.match(line)
        if m2:
            idx = int(m2.group(1))
            procedures.setdefault(idx, {})["steps"] = int(m2.group(2))
    return procedures


def _build_chromatic_map(procedures: dict) -> dict:
    """{chromatic_index (1-based): {"scan_type": .., "wavelength": ..}}, in the
    same order the 'Chromatic: N' blocks appear (procedure 1's steps first,
    then procedure 2's, etc.)."""
    chrom_map = {}
    offset = 0
    for idx in sorted(procedures):
        p = procedures[idx]
        if "scan_type" not in p or "steps" not in p:
            continue
        wavelengths = np.linspace(p["start"], p["end"], p["steps"])
        for i, wl in enumerate(wavelengths):
            chrom_map[offset + i + 1] = {"scan_type": p["scan_type"], "wavelength": float(wl)}
        offset += p["steps"]
    return chrom_map


def _parse_plate_name(lines: list[str], filepath: str) -> str:
    """Extract the plate name that lines up with 'Destination Plate Name' in
    the Echo mapping files.

    Priority:
      1. Filename starts with 'Plate N' (matches the Echo convention directly).
      2. The instrument's ID1 header field (first underscore-separated token).
      3. Filename's first non-date token.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    plate_m = re.match(r'^(Plate[\s_]*[A-Za-z0-9]+)', stem, re.IGNORECASE)
    if plate_m:
        return plate_m.group(1)
    for line in lines:
        m = _ID1_RE.search(line)
        if m and m.group(1):
            return m.group(1).split("_")[0]
    tokens = stem.split("_")
    for tok in tokens:
        if tok and not (tok.isdigit() and len(tok) > 4):
            return tok
    return stem


def _parse_data_blocks(lines: list[str], n_chromatics: int) -> np.ndarray:
    data          = np.full((n_chromatics, 16, 24), np.nan)
    current_chrom = None
    in_data_block = False
    row_idx       = 0
    for line in lines:
        line = line.strip()
        m = _CHROM_RE.match(line)
        if m:
            current_chrom = int(m.group(1)) - 1
            row_idx       = 0
            in_data_block = False
            continue
        if line.startswith("Time"):
            in_data_block = True
            continue
        if (in_data_block and current_chrom is not None
                and 0 <= current_chrom < n_chromatics and row_idx < 16):
            parts = line.split(",")
            if len(parts) >= 48:
                for c in range(24):
                    val_str = parts[c * 2].strip()
                    if val_str != "-":
                        try:
                            data[current_chrom, row_idx, c] = float(val_str)
                        except ValueError:
                            pass
            row_idx += 1
    return data


def _load_one_file(filepath: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    with open(filepath, "r", encoding="latin1") as f:
        lines = f.readlines()

    procedures = _parse_scan_procedures(lines)
    chrom_map  = _build_chromatic_map(procedures)
    if not chrom_map:
        _log(f"  WARNING: '{os.path.basename(filepath)}' — could not parse scan "
             f"procedures from header; skipping.", progress_cb)
        return pd.DataFrame(columns=["Well", "Fluorescence", "Plate", "Chromatic",
                                     "ScanType", "Wavelength"])

    plate_name   = _parse_plate_name(lines, filepath)
    n_chromatics = max(chrom_map)
    data         = _parse_data_blocks(lines, n_chromatics)

    rows = []
    for chrom_idx in range(n_chromatics):
        info = chrom_map.get(chrom_idx + 1)
        if info is None:
            continue
        block = data[chrom_idx]
        for r in range(16):
            for c in range(24):
                val = block[r, c]
                if not np.isnan(val):
                    rows.append({
                        "Well":         f"{_ROW_LABELS[r]}{c + 1}",
                        "Fluorescence": val,
                        "Plate":        plate_name,
                        "Chromatic":    chrom_idx + 1,
                        "ScanType":     info["scan_type"],
                        "Wavelength":   info["wavelength"],
                    })
    df   = pd.DataFrame(rows)
    n_ex = sum(1 for v in chrom_map.values() if v["scan_type"] == "ex")
    n_em = sum(1 for v in chrom_map.values() if v["scan_type"] == "em")
    _log(f"  Loaded '{plate_name}' ({os.path.basename(filepath)}): "
         f"{n_ex} excitation step(s), {n_em} emission step(s), "
         f"{len(df)} well-readings", progress_cb)
    return df


def load_plates_spectral(raw_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    filepaths = sorted(
        os.path.join(raw_folder, f) for f in os.listdir(raw_folder)
        if f.lower().endswith(".csv") and not f.startswith("~$"))
    if not filepaths:
        raise ValueError(f"No .csv spectral scan files found in {raw_folder}")
    dfs    = [_load_one_file(fp, progress_cb) for fp in filepaths]
    result = pd.concat(dfs, ignore_index=True)
    result["Fluorescence"] = pd.to_numeric(result["Fluorescence"], errors="coerce")
    _log(f"Loaded {result['Plate'].nunique()} plate(s), "
         f"{result['Chromatic'].nunique()} wavelength step(s) total", progress_cb)
    return result


# ── Stage 2: merge with mapping + blanks ──────────────────────────────────────

def _align_plate_names(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
                       progress_cb: ProgressCb = None) -> pd.DataFrame:
    """Auto-align spectral plate names to mapping plate names when they
    don't match (e.g. 'P1' from ID1 vs 'Plate 1' from Echo export)."""
    fl_plates  = sorted(fluorescence_df["Plate"].dropna().unique())
    map_plates = sorted(merged_mapping["Plate"].dropna().unique()
                        if "Plate" in merged_mapping.columns else [])
    if not fl_plates or not map_plates:
        return fluorescence_df
    if any(p in map_plates for p in fl_plates):
        return fluorescence_df

    fluorescence_df = fluorescence_df.copy()
    if len(map_plates) == 1:
        _log(f"Auto-aligned plate names {fl_plates} → '{map_plates[0]}'", progress_cb)
        fluorescence_df["Plate"] = map_plates[0]
    elif len(fl_plates) == len(map_plates):
        rename = dict(zip(fl_plates, map_plates))
        fluorescence_df["Plate"] = fluorescence_df["Plate"].map(rename)
        _log(f"Auto-aligned plate names by position: {rename}", progress_cb)
    else:
        _log(f"WARNING: plate name mismatch — spectral: {fl_plates}, "
             f"mapping: {map_plates}. Merge may produce NaN.", progress_cb)
    return fluorescence_df


def merge_blanks_spectral(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
                          blank_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    """Same Well/Plate/Dye join as pipeline_fda.merge_blanks, without the
    multi-chromatic dye-filter step — every 'Chromatic' here is a wavelength
    step of one continuous scan of a single dye, not a separate filter
    channel, so no chromatic->dye resolution is needed."""
    fluorescence_df = _align_plate_names(fluorescence_df, merged_mapping, progress_cb)
    merged = pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
    merged = merged.sort_values(["Host", "Host_Concentration"])

    if not os.path.isdir(blank_folder):
        raise FileNotFoundError(f"Blank mapping folder not found: {blank_folder}")
    blank_files = sorted(f for f in os.listdir(blank_folder)
                         if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    if not blank_files:
        raise FileNotFoundError(f"No .xlsx files in blank mapping folder: {blank_folder}")
    blank_df = pd.concat(
        [pd.read_excel(os.path.join(blank_folder, f)) for f in blank_files],
        ignore_index=True)
    if "Compound ID" not in blank_df.columns:
        raise ValueError("Blank mapping file(s) missing required column 'Compound ID'")
    blank_df = blank_df[blank_df["Compound ID"].fillna("").str.strip().str.lower() == "blank"]

    merged2 = pd.merge(merged, blank_df,
                       on=["Well", "Plate", "Dye"], how="left",
                       suffixes=("", "_blank"))

    n_blank_matched = merged2["Compound ID"].notna().sum()
    _log(f"Merged with blanks: {len(merged2)} rows "
         f"({n_blank_matched} blank-labelled)", progress_cb)
    if n_blank_matched == 0:
        _log("WARNING: no blank wells matched after merge. Check that:\n"
             "  • blank_map files contain rows with Compound ID = 'blank'\n"
             "  • Plate and Well names match the spectral scan plate files\n"
             "  • Dye names in blank_map match those in dye_map exactly\n"
             "FI-F0 will be NaN for all wells.", progress_cb)

    n_no_host = merged2["Host"].isna().sum()
    if n_no_host / max(len(merged2), 1) > 0.5:
        fl_plates  = sorted(fluorescence_df["Plate"].dropna().unique())
        map_plates = sorted(merged_mapping["Plate"].dropna().unique()
                            if "Plate" in merged_mapping.columns else [])
        _log(f"WARNING: {n_no_host}/{len(merged2)} rows have no Host after merge.\n"
             f"  Spectral scan plate names: {fl_plates}\n"
             f"  Mapping plate names:       {map_plates}\n"
             f"  These must match exactly.", progress_cb)

    return merged2


# Background subtraction is identical to pipeline_fda's: FI-F0 = Fluorescence
# − Dye_Blank_Avg, grouped by (Plate, Dye, Chromatic). "Chromatic" being a
# wavelength step rather than a filter channel makes no difference to the
# grouping logic, so it is reused as-is rather than duplicated.
subtract_background_spectral = pl_fda.subtract_background


# ── Spectra plotting ───────────────────────────────────────────────────────────

def render_spectra_figure(fi_df: pd.DataFrame, host: Optional[str], dye: str,
                          palette: str = "viridis",
                          color_overrides: Optional[dict] = None,
                          display_mode: str = "side_by_side",
                          selected_concs: Optional[list] = None,
                          fig_w: float = 11.0,
                          fig_h: float = 5.0,
                          title_fontsize: float = 12,
                          axis_fontsize: float = 14,
                          tick_fontsize: float = 12) -> Figure:
    """Excitation + emission spectra for one Host/Dye pair, colour-coded by
    Host concentration, with the dye-blank (no host) overlaid as a dashed
    reference curve.

    palette:        seaborn/matplotlib colour palette name
    color_overrides: {concentration_float: "#hex"} per-conc overrides
    display_mode:   "side_by_side" | "combined" | "separate"
    selected_concs: if not None, only show these host concentrations
    """
    sub       = fi_df[fi_df["Dye"] == dye].copy()
    host_sub  = sub[sub["Host"] == host] if host else sub[sub["Host"].notna()]
    is_blank  = sub["Compound ID"].fillna("").str.strip().str.lower() == "blank"
    blank_sub = sub[is_blank & sub["Host"].isna()]

    concs  = sorted(host_sub["Host_Concentration"].dropna().unique())
    if selected_concs is not None:
        concs = [c for c in concs if c in selected_concs]

    pal = sns.color_palette(palette, max(len(concs), 1))
    overrides = color_overrides or {}
    colors = [overrides[c] if c in overrides else pal[i]
              for i, c in enumerate(concs)]

    def _plot_on_ax(ax, scan_type, show_ylabel=True, label_prefix="",
                    add_legend_labels=True):
        scan_data  = host_sub[host_sub["ScanType"] == scan_type]
        blank_data = blank_sub[blank_sub["ScanType"] == scan_type]
        scan_label = "Ex" if scan_type == "ex" else "Em"
        for conc, color in zip(concs, colors):
            curve = (scan_data[scan_data["Host_Concentration"] == conc]
                     .groupby("Wavelength")["Fluorescence"].mean().sort_index())
            if not curve.empty:
                lbl = (f"{label_prefix}{conc:g} µM" if add_legend_labels
                       else "_nolegend_")
                ax.plot(curve.index, curve.values, color=color, label=lbl)
        if not blank_data.empty:
            blank_curve = blank_data.groupby("Wavelength")["Fluorescence"].mean().sort_index()
            lbl = (f"{label_prefix}0 µM (Dye Blank)" if add_legend_labels
                   else "_nolegend_")
            ax.plot(blank_curve.index, blank_curve.values, "k--", linewidth=2,
                    label=lbl)
        xlabel = "Excitation" if scan_type == "ex" else "Emission"
        ax.set_xlabel(f"{xlabel} Wavelength (nm)", fontsize=axis_fontsize)
        if show_ylabel:
            ax.set_ylabel("Fluorescence Intensity", fontsize=axis_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

    sep_h = fig_h * 1.8  # taller for stacked layout

    if display_mode == "combined":
        fig = Figure(figsize=(fig_w, fig_h))
        ax  = fig.subplots(1, 1)
        ex_wls = sorted(host_sub.loc[host_sub["ScanType"] == "ex", "Wavelength"].dropna().unique())
        em_wls = sorted(host_sub.loc[host_sub["ScanType"] == "em", "Wavelength"].dropna().unique())
        _plot_on_ax(ax, "ex", label_prefix="Ex: ")
        _plot_on_ax(ax, "em", show_ylabel=False, label_prefix="Em: ")
        if ex_wls and em_wls:
            gap_lo = max(ex_wls)
            gap_hi = min(em_wls)
            if gap_hi > gap_lo + 1:
                ax.axvline(x=(gap_lo + gap_hi) / 2, color="grey", linestyle=":",
                           linewidth=1, alpha=0.7)
        ax.set_xlabel("Wavelength (nm)", fontsize=axis_fontsize)
        ax.set_ylabel("Fluorescence Intensity", fontsize=axis_fontsize)
        ax.legend(title="Host Conc.", bbox_to_anchor=(1.05, 1),
                  loc="upper left", fontsize=tick_fontsize)
    elif display_mode == "separate":
        fig = Figure(figsize=(fig_w, sep_h))
        axs = fig.subplots(2, 1)
        for scan_type, ax in zip(["ex", "em"], axs):
            _plot_on_ax(ax, scan_type)
            label = "Excitation" if scan_type == "ex" else "Emission"
            ax.set_title(f"{label} Spectrum", fontsize=title_fontsize)
        axs[1].legend(title="Host Conc.", bbox_to_anchor=(1.05, 1),
                      loc="upper left", fontsize=tick_fontsize)
    else:
        fig = Figure(figsize=(fig_w, fig_h))
        axs = fig.subplots(1, 2)
        for scan_type, ax in zip(["ex", "em"], axs):
            _plot_on_ax(ax, scan_type)
            label = "Excitation" if scan_type == "ex" else "Emission"
            ax.set_title(f"{label} Spectrum", fontsize=title_fontsize)
        axs[1].legend(title="Host Conc.", bbox_to_anchor=(1.05, 1),
                      loc="upper left", fontsize=tick_fontsize)

    fig.suptitle(f"{dye}" + (f" + {host}" if host else ""), fontsize=title_fontsize + 1)
    fig.tight_layout()
    return fig


def _host_color_with_alpha(base_hex: str, alpha: float) -> tuple:
    """Return an RGBA tuple from a hex colour and an alpha value."""
    from matplotlib.colors import to_rgba
    r, g, b, _a = to_rgba(base_hex)
    return (r, g, b, alpha)


_DEFAULT_HOST_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def render_spectra_figure_by_dye(fi_df: pd.DataFrame, dye: str,
                                 palette: str = "viridis",
                                 color_overrides: Optional[dict] = None,
                                 host_color_overrides: Optional[dict] = None,
                                 conc_opacity_overrides: Optional[dict] = None,
                                 display_mode: str = "side_by_side",
                                 selected_concs: Optional[list] = None,
                                 fig_w: float = 11.0,
                                 fig_h: float = 5.0,
                                 title_fontsize: float = 12,
                                 axis_fontsize: float = 14,
                                 tick_fontsize: float = 12) -> Figure:
    """All hosts overlaid on one figure for a single dye, colour-coded by
    host with opacity variation for concentration."""
    import matplotlib.lines as mlines

    sub = fi_df[fi_df["Dye"] == dye].copy()
    hosts = sorted(sub["Host"].dropna().unique())
    is_blank = sub["Compound ID"].fillna("").str.strip().str.lower() == "blank"
    blank_sub = sub[is_blank & sub["Host"].isna()]

    concs = sorted(sub.loc[sub["Host"].notna(), "Host_Concentration"].dropna().unique())
    if selected_concs is not None:
        concs = [c for c in concs if c in selected_concs]

    h_overrides = host_color_overrides or {}
    host_colors = {}
    for i, h in enumerate(hosts):
        if h in h_overrides:
            host_colors[h] = h_overrides[h]
        else:
            host_colors[h] = _DEFAULT_HOST_COLORS[i % len(_DEFAULT_HOST_COLORS)]

    o_overrides = conc_opacity_overrides or {}
    n_concs = len(concs)
    conc_alphas = {}
    for i, c in enumerate(concs):
        if c in o_overrides:
            conc_alphas[c] = o_overrides[c]
        elif n_concs <= 1:
            conc_alphas[c] = 1.0
        else:
            conc_alphas[c] = 0.3 + 0.7 * i / (n_concs - 1)

    def _plot_on_ax(ax, scan_type, show_ylabel=True):
        for host in hosts:
            host_data = sub[(sub["Host"] == host) & (sub["ScanType"] == scan_type)]
            base = host_colors[host]
            for conc in concs:
                curve = (host_data[host_data["Host_Concentration"] == conc]
                         .groupby("Wavelength")["Fluorescence"].mean().sort_index())
                if not curve.empty:
                    ax.plot(curve.index, curve.values,
                            color=_host_color_with_alpha(base, conc_alphas[conc]),
                            label="_nolegend_")
        if not blank_sub.empty:
            blank_data = blank_sub[blank_sub["ScanType"] == scan_type]
            if not blank_data.empty:
                blank_curve = blank_data.groupby("Wavelength")["Fluorescence"].mean().sort_index()
                ax.plot(blank_curve.index, blank_curve.values, "k--",
                        linewidth=2, label="_nolegend_")
        xlabel = "Excitation" if scan_type == "ex" else "Emission"
        ax.set_xlabel(f"{xlabel} Wavelength (nm)", fontsize=axis_fontsize)
        if show_ylabel:
            ax.set_ylabel("Fluorescence Intensity", fontsize=axis_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

    sep_h = fig_h * 1.8

    if display_mode == "combined":
        fig = Figure(figsize=(fig_w, fig_h))
        ax = fig.subplots(1, 1)
        _plot_on_ax(ax, "ex")
        _plot_on_ax(ax, "em", show_ylabel=False)
        ax.set_xlabel("Wavelength (nm)", fontsize=axis_fontsize)
        ax.set_ylabel("Fluorescence Intensity", fontsize=axis_fontsize)
    elif display_mode == "separate":
        fig = Figure(figsize=(fig_w, sep_h))
        axs = fig.subplots(2, 1)
        for st, ax in zip(["ex", "em"], axs):
            _plot_on_ax(ax, st)
            ax.set_title(f"{'Excitation' if st == 'ex' else 'Emission'} Spectrum",
                         fontsize=title_fontsize)
    else:
        fig = Figure(figsize=(fig_w, fig_h))
        axs = fig.subplots(1, 2)
        for st, ax in zip(["ex", "em"], axs):
            _plot_on_ax(ax, st)
            ax.set_title(f"{'Excitation' if st == 'ex' else 'Emission'} Spectrum",
                         fontsize=title_fontsize)

    host_handles = [mlines.Line2D([], [], color=host_colors[h], linewidth=2,
                                   label=h) for h in hosts]
    conc_handles = [mlines.Line2D([], [], color="gray",
                                   alpha=conc_alphas[c],
                                   linewidth=2, label=f"{c:g} µM") for c in concs]
    if not blank_sub.empty:
        conc_handles.append(mlines.Line2D([], [], color="black", linestyle="--",
                                           linewidth=2, label="0 µM (Dye Blank)"))

    target_ax = fig.axes[-1] if fig.axes else fig.add_subplot(111)
    leg1 = target_ax.legend(handles=host_handles, title="Host",
                             bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=tick_fontsize * 0.7)
    target_ax.add_artist(leg1)
    target_ax.legend(handles=conc_handles, title="Host Conc.",
                      bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=tick_fontsize * 0.7)

    fig.suptitle(f"{dye} — All Hosts", fontsize=title_fontsize + 1)
    fig.tight_layout()
    return fig


def make_spectra_figures(fi_df: pd.DataFrame, progress_cb: ProgressCb = None,
                         palette: str = "viridis",
                         color_overrides: Optional[dict] = None,
                         host_color_overrides: Optional[dict] = None,
                         conc_opacity_overrides: Optional[dict] = None,
                         display_mode: str = "side_by_side",
                         selected_concs: Optional[list] = None,
                         grouping: str = "per_host",
                         fig_w: float = 11.0,
                         fig_h: float = 5.0,
                         title_fontsize: float = 12,
                         axis_fontsize: float = 14,
                         tick_fontsize: float = 12) -> list:
    """Spectra figures.

    grouping:
        "per_host"  — one figure per (Host, Dye) pair  [default]
        "by_dye"    — one figure per Dye, all hosts overlaid
    """
    figures = []
    _fkw = dict(title_fontsize=title_fontsize, axis_fontsize=axis_fontsize,
                tick_fontsize=tick_fontsize)
    if grouping == "by_dye":
        dyes = sorted(fi_df["Dye"].dropna().unique())
        for dye in dyes:
            fig = render_spectra_figure_by_dye(
                fi_df, dye, palette=palette,
                color_overrides=color_overrides,
                host_color_overrides=host_color_overrides,
                conc_opacity_overrides=conc_opacity_overrides,
                display_mode=display_mode,
                selected_concs=selected_concs,
                fig_w=fig_w, fig_h=fig_h, **_fkw)
            figures.append((f"{dye} — All Hosts", fig))
    else:
        pairs = (fi_df.dropna(subset=["Host", "Dye"])[["Host", "Dye"]]
                 .drop_duplicates().sort_values(["Dye", "Host"]))
        for _, row in pairs.iterrows():
            fig = render_spectra_figure(fi_df, row["Host"], row["Dye"],
                                        palette=palette,
                                        color_overrides=color_overrides,
                                        display_mode=display_mode,
                                        selected_concs=selected_concs,
                                        fig_w=fig_w, fig_h=fig_h, **_fkw)
            figures.append((f"{row['Dye']} + {row['Host']}", fig))
    _log(f"Generated {len(figures)} spectra figure(s) (grouping={grouping}).", progress_cb)
    return figures


# ── Binding fit at a single wavelength ────────────────────────────────────────

def select_wavelength_subset(fi_df: pd.DataFrame, scan_type: str, wavelength: float,
                             progress_cb: ProgressCb = None) -> tuple[pd.DataFrame, float]:
    """Pick the scanned wavelength nearest *wavelength* within *scan_type*
    ('ex' or 'em') and return (rows at that wavelength, nearest wavelength)."""
    sub = fi_df[fi_df["ScanType"] == scan_type]
    if sub.empty:
        raise ValueError(f"No '{scan_type}' scan data found.")
    wavelengths = sub["Wavelength"].dropna().unique()
    if len(wavelengths) == 0:
        raise ValueError(f"No valid wavelengths found for '{scan_type}' scan data.")
    nearest     = float(min(wavelengths, key=lambda w: abs(w - wavelength)))
    out         = sub[np.isclose(sub["Wavelength"], nearest)].copy()
    _log(f"Selected {scan_type} wavelength {nearest:.1f} nm "
         f"(target {wavelength:g} nm), {len(out)} rows", progress_cb)
    return out, nearest


def fit_binding_at_wavelength(fi_df: pd.DataFrame, scan_type: str, wavelength: float,
                              grubbs_alpha: float = 0.05,
                              use_cross_conc_grubbs: bool = False,
                              cross_conc_alpha: float = 0.01,
                              model_preference: str = "auto",
                              progress_cb: ProgressCb = None) -> dict:
    """Binding fit from spectral data: select the single wavelength step
    closest to *wavelength* within *scan_type*, then run the same curve
    fitting used by Direct Binding (FI-F0 vs [Host])."""
    subset, nearest = select_wavelength_subset(fi_df, scan_type, wavelength, progress_cb)
    fit_results, plot_data = pl_fda.fit_curves(
        subset, grubbs_alpha=grubbs_alpha,
        use_cross_conc_grubbs=use_cross_conc_grubbs,
        cross_conc_alpha=cross_conc_alpha,
        model_preference=model_preference,
        progress_cb=progress_cb)
    return {"fit_results": fit_results, "plot_data": plot_data, "nearest_wavelength": nearest}


def fit_binding_per_dye(fi_df: pd.DataFrame, scan_type: str,
                        wavelength_map: dict,
                        grubbs_alpha: float = 0.05,
                        use_cross_conc_grubbs: bool = False,
                        cross_conc_alpha: float = 0.01,
                        model_preference: str = "auto",
                        progress_cb: ProgressCb = None) -> dict:
    """Like fit_binding_at_wavelength but with a per-dye wavelength.

    wavelength_map: {dye_name: target_wavelength_float}
    Returns the same dict shape, with nearest_wavelength as a dict per dye.
    """
    all_fit_results = []
    all_plot_data   = []
    nearest_map     = {}
    for dye_name, wl in wavelength_map.items():
        dye_sub = fi_df[fi_df["Dye"] == dye_name]
        if dye_sub.empty:
            _log(f"  Skipping dye '{dye_name}' — no data.", progress_cb)
            continue
        _log(f"  Fitting dye '{dye_name}' at target λ={wl:g} nm …", progress_cb)
        subset, nearest = select_wavelength_subset(dye_sub, scan_type, wl, progress_cb)
        nearest_map[dye_name] = nearest
        fit_results, plot_data = pl_fda.fit_curves(
            subset, grubbs_alpha=grubbs_alpha,
            use_cross_conc_grubbs=use_cross_conc_grubbs,
            cross_conc_alpha=cross_conc_alpha,
            model_preference=model_preference,
            progress_cb=progress_cb)
        all_fit_results.extend(fit_results)
        all_plot_data.extend(plot_data)
    return {"fit_results": all_fit_results, "plot_data": all_plot_data,
            "nearest_wavelength": nearest_map}


# ── Save outputs ───────────────────────────────────────────────────────────────

def save_outputs_spectral(state: PipelineStateSpectral, output_folder: str,
                          r2_threshold: float = pl_fda.PASS_R2_DEFAULT,
                          progress_cb: ProgressCb = None,
                          plot_color_data: str = pl_fda.PLOT_COLOR_DATA,
                          plot_color_fit: str = pl_fda.PLOT_COLOR,
                          plot_color_resid: str = None,
                          show_residuals: bool = False,
                          export_individual: bool = False,
                          export_individual_spectra: bool = False,
                          layout_cfg: dict = None,
                          title_fontsize: float = 12,
                          title_fontfamily: str = "sans-serif",
                          title_fontweight: str = "normal",
                          title_fontstyle: str = "normal",
                          axis_fontsize: float = 14,
                          tick_fontsize: float = 12,
                          input_folder: str = "",
                          normalise_y: bool = False,
                          sci_notation_y: bool = False) -> list[str]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    for d in (reports_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    ts, saved = pl_fda._folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S"), []

    def _csv(df, label):
        p = os.path.join(reports_dir, f"{ts}_spectral_{label}.csv")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        _log(f"Saved: {p}", progress_cb)
        saved.append(p)

    if state.merged_mapping is not None: _csv(state.merged_mapping, "mapping")
    if state.fluorescence   is not None: _csv(state.fluorescence,   "raw_data")
    if state.merged         is not None: _csv(state.merged,         "merged_with_blanks")
    if state.fi_df          is not None: _csv(state.fi_df,          "blank_corrected")

    if state.spectra_figures:
        pdf_path = os.path.join(plots_dir, f"{ts}_spectra.pdf")
        with PdfPages(pdf_path) as pdf:
            for _label, fig in state.spectra_figures:
                pdf.savefig(fig, bbox_inches="tight")
        _log(f"Saved: {pdf_path}", progress_cb)
        saved.append(pdf_path)

        if export_individual_spectra:
            ind_dir = os.path.join(plots_dir, "spectra_individual")
            os.makedirs(ind_dir, exist_ok=True)
            for label, fig in state.spectra_figures:
                safe = re.sub(r'[^\w\-. ]+', '_', label).strip('_')
                ind_path = os.path.join(ind_dir, f"{ts}_{safe}.pdf")
                fig.savefig(ind_path, bbox_inches="tight")
                saved.append(ind_path)
            _log(f"Saved {len(state.spectra_figures)} individual spectra PDF(s) "
                 f"to {ind_dir}", progress_cb)

    if state.overlay_figures:
        overlay_path = os.path.join(plots_dir, f"{ts}_overlay_by_dye.pdf")
        with PdfPages(overlay_path) as pdf:
            for _label, fig in state.overlay_figures:
                pdf.savefig(fig, bbox_inches="tight")
        _log(f"Saved: {overlay_path}", progress_cb)
        saved.append(overlay_path)

    if state.df_results is not None:
        fda_state = pl_fda.PipelineState(
            fit_results=state.fit_results, plot_data=state.plot_data,
            df_results=state.df_results)
        saved.extend(pl_fda.save_outputs(
            fda_state, output_folder, r2_threshold, progress_cb,
            plot_color_data=plot_color_data, plot_color_fit=plot_color_fit,
            plot_color_resid=plot_color_resid, show_residuals=show_residuals,
            export_individual=export_individual,
            layout_cfg=layout_cfg,
            title_fontsize=title_fontsize, title_fontfamily=title_fontfamily,
            title_fontweight=title_fontweight, title_fontstyle=title_fontstyle,
            axis_fontsize=axis_fontsize, tick_fontsize=tick_fontsize,
            input_folder=input_folder,
            normalise_y=normalise_y, sci_notation_y=sci_notation_y))

    return saved


def preview_save_files_spectral(state: PipelineStateSpectral, output_folder: str,
                                 export_individual: bool = False,
                                 export_individual_spectra: bool = False,
                                 input_folder: str = "") -> list[dict]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    ts, files   = pl_fda._folder_date_prefix(input_folder) + datetime.now().strftime("%y%m%d_%H%M%S"), []

    def _f(folder, name, desc):
        files.append({"path": os.path.join(folder, f"{ts}_spectral_{name}"), "description": desc})

    if state.merged_mapping is not None: _f(reports_dir, "mapping.csv",            "Combined mapping (dye/host)")
    if state.fluorescence   is not None: _f(reports_dir, "raw_data.csv",           "Raw spectral fluorescence")
    if state.merged         is not None: _f(reports_dir, "merged_with_blanks.csv", "Merged data + blank labels")
    if state.fi_df          is not None: _f(reports_dir, "blank_corrected.csv",    "Background-subtracted (FI-F0)")
    if state.spectra_figures:
        files.append({"path": os.path.join(plots_dir, f"{ts}_spectra.pdf"),
                      "description": f"Excitation/Emission spectra — {len(state.spectra_figures)} page(s)"})
        if export_individual_spectra:
            ind_dir = os.path.join(plots_dir, "spectra_individual")
            files.append({"path": ind_dir,
                          "description": f"Individual spectra PDFs — {len(state.spectra_figures)} file(s)"})
    if state.overlay_figures:
        files.append({"path": os.path.join(plots_dir, f"{ts}_overlay_by_dye.pdf"),
                      "description": f"Overlay fits by dye — {len(state.overlay_figures)} page(s)"})
    if state.df_results is not None:
        fda_state = pl_fda.PipelineState(fit_results=state.fit_results,
                                         plot_data=state.plot_data, df_results=state.df_results)
        files.extend(pl_fda.preview_save_files(fda_state, output_folder, export_individual,
                                               input_folder=input_folder))
    return files
