#!/usr/bin/env python3
"""
Pipeline logic for the multi-chromatic FAILPASS binding assay (v3 base).
No top-level side effects — all execution is triggered by calling functions.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.stats import normaltest, t


ProgressCb = Optional[Callable[[str], None]]

PASS_R2_DEFAULT        = 0.6
KD_RANGE_FACTOR_LO_DEFAULT = 0.1   # Kd < factor × min([Host]) → FAIL (extrapolating left)
KD_RANGE_FACTOR_HI_DEFAULT = 10.0  # Kd > factor × max([Host]) → FAIL (never reaches saturation)
PLOT_COLOR             = "#6495ED"


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
    merged:         Optional[pd.DataFrame] = None   # after blank merge + multi-chrom filter
    fi_df:          Optional[pd.DataFrame] = None   # after background subtraction
    fit_results:    list = field(default_factory=list)   # list[dict]
    plot_data:      list = field(default_factory=list)   # list[dict]
    df_results:     Optional[pd.DataFrame] = None   # after apply_thresholds


# ── Stage 1: load mappings ────────────────────────────────────────────────────

def load_mappings(dye_folder: str, host_folder: str,
                  progress_cb: ProgressCb = None) -> pd.DataFrame:
    dye_files  = sorted(f for f in os.listdir(dye_folder)
                        if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    host_files = sorted(f for f in os.listdir(host_folder)
                        if f.lower().endswith(".xlsx") and not f.startswith("~$"))

    _log(f"Loading dye mapping:  {dye_files}", progress_cb)
    _log(f"Loading host mapping: {host_files}", progress_cb)

    dye_df = pd.concat(
        [pd.read_excel(os.path.join(dye_folder, f)) for f in dye_files],
        ignore_index=True)
    host_df = pd.concat(
        [pd.read_excel(os.path.join(host_folder, f)) for f in host_files],
        ignore_index=True)

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

    _log(f"Hosts: {merged['Host'].nunique()}  |  Dyes: {merged['Dye'].nunique()}", progress_cb)
    return merged


# ── Stage 2: load raw plate data ──────────────────────────────────────────────

def _find_chromatic_blocks(df_raw: pd.DataFrame) -> list[tuple]:
    """Return list of (label, data_start_row) for each Chromatic: header."""
    blocks = []
    for i in range(len(df_raw)):
        cell = str(df_raw.iloc[i, 0]).strip()
        if cell.lower().startswith("chromatic:"):
            label = cell.split(":", 1)[1].strip()
            blocks.append((label, i + 3))
    return blocks


def _load_plate_data_384(filepath: str, plate_name: str) -> pd.DataFrame:
    df_raw = pd.read_excel(filepath, engine="openpyxl", header=None, sheet_name=0)
    blocks = _find_chromatic_blocks(df_raw)

    if not blocks:
        return pd.DataFrame(columns=["Well", "Fluorescence", "Plate", "Chromatic"])

    dfs = []
    for label, start_row in blocks:
        plate_df = df_raw.iloc[start_row:start_row + 16, 0:24].copy()
        plate_df.index   = list(string.ascii_uppercase[:16])
        plate_df.columns = [str(i) for i in range(1, 25)]
        tidy = (
            plate_df.reset_index()
            .melt(id_vars="index", var_name="Column", value_name="Fluorescence")
            .rename(columns={"index": "Row"})
        )
        tidy["Well"]      = tidy["Row"] + tidy["Column"]
        tidy["Plate"]     = plate_name
        tidy["Chromatic"] = label
        dfs.append(tidy[["Well", "Fluorescence", "Plate", "Chromatic"]].dropna(subset=["Fluorescence"]))
    return pd.concat(dfs, ignore_index=True)


def load_plates(raw_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    filepaths = sorted(
        os.path.join(raw_folder, f)
        for f in os.listdir(raw_folder)
        if f.lower().endswith(".xlsx") and not f.startswith("~$")
    )

    all_dfs = []
    for fp in filepaths:
        plate_name = os.path.basename(fp).split("_")[0]
        df_plate   = _load_plate_data_384(fp, plate_name)
        all_dfs.append(df_plate)
        chroms = df_plate["Chromatic"].unique().tolist()
        _log(f"  Loaded {plate_name}: {len(chroms)} chromatic(s) → {chroms}", progress_cb)

    result = pd.concat(all_dfs, ignore_index=True)
    result["Fluorescence"] = pd.to_numeric(result["Fluorescence"], errors="coerce")
    _log(f"Loaded {result['Plate'].nunique()} plate(s), "
         f"{result['Chromatic'].nunique()} unique chromatic(s)", progress_cb)
    return result


# ── Stage 3: merge with mapping + blanks ─────────────────────────────────────

def merge_blanks(fluorescence_df: pd.DataFrame, merged_mapping: pd.DataFrame,
                 blank_folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    merged = pd.merge(fluorescence_df, merged_mapping, on=["Well", "Plate"], how="left")
    merged = merged.sort_values(["Host", "Host_Concentration"])

    # For multi-chromatic plates: keep only the chromatic whose name matches the dye
    chrom_counts        = fluorescence_df.groupby("Plate")["Chromatic"].nunique()
    multi_chrom_plates  = set(chrom_counts[chrom_counts > 1].index)

    if multi_chrom_plates:
        _log(f"Multi-chromatic plates: {sorted(multi_chrom_plates)}", progress_cb)
        is_multi  = merged["Plate"].isin(multi_chrom_plates)
        has_dye   = merged["Dye"].notna()

        def _chrom_matches_dye(row):
            dye   = str(row["Dye"]).lower()
            chrom = str(row["Chromatic"]).lower()
            return dye in chrom or chrom in dye

        chrom_ok = merged.apply(_chrom_matches_dye, axis=1)
        merged   = merged[~is_multi | ~has_dye | chrom_ok].copy()

        # Warn if any dye matched no chromatic
        for plate in multi_chrom_plates:
            plate_dyes  = merged_mapping.loc[merged_mapping["Plate"] == plate, "Dye"].dropna().unique()
            plate_chroms = fluorescence_df.loc[fluorescence_df["Plate"] == plate, "Chromatic"].unique()
            for dye in plate_dyes:
                if not any(dye.lower() in c.lower() or c.lower() in dye.lower()
                           for c in plate_chroms):
                    _log(f"  WARNING: plate {plate}: dye '{dye}' matched no chromatic "
                         f"{list(plate_chroms)}", progress_cb)

    blank_files = sorted(f for f in os.listdir(blank_folder)
                         if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    blank_df = pd.concat(
        [pd.read_excel(os.path.join(blank_folder, f)) for f in blank_files],
        ignore_index=True)
    blank_df = blank_df[blank_df["Compound ID"].str.strip().str.lower() == "blank"]

    merged2 = pd.merge(blank_df, merged,
                       on=["Well", "Plate", "Dye", "Dye_Concentration"],
                       how="outer")
    _log(f"Merged with blanks: {len(merged2)} rows", progress_cb)
    return merged2


# ── Stage 4: background subtraction ──────────────────────────────────────────

def subtract_background(merged_df: pd.DataFrame,
                        progress_cb: ProgressCb = None) -> pd.DataFrame:
    df       = merged_df.copy()
    is_blank = df["Compound ID"].str.strip().str.lower() == "blank"

    df["Dye_Blank_Avg"] = (
        df
        .where(is_blank & df["Dye"].notna())
        .groupby(["Plate", "Dye", "Chromatic"])["Fluorescence"]
        .transform("mean")
    )
    df["Dye_Blank_Avg"] = (
        df.groupby(["Plate", "Dye", "Chromatic"])["Dye_Blank_Avg"]
        .transform("first")
    )

    df["FI-F0"] = df["Fluorescence"] - df["Dye_Blank_Avg"]
    _log("Background subtraction complete (FI-F0 = F – Dye_Blank_Avg).", progress_cb)
    return df


# ── Stage 5: curve fitting ────────────────────────────────────────────────────

def _one_site_specific(x, Bmax, Kd):
    x = np.asarray(x, dtype=float)
    return Bmax * x / (Kd + x)


def _quadratic_binding(x, Fmax, Kd, D_fixed):
    arg = np.clip((x + D_fixed + Kd)**2 - 4 * x * D_fixed, 0, None)
    return Fmax * (x + D_fixed + Kd - np.sqrt(arg)) / (2 * D_fixed)


def _stern_volmer(x, Bottom, Kd):
    """Quenching model: signal decreases on binding (Bottom < 0)."""
    x = np.asarray(x, dtype=float)
    return Bottom * x / (Kd + x)


def _eval_model(model_name: str, x, popt, D_fixed: float):
    if model_name == "one_site":
        return _one_site_specific(x, *popt)
    if model_name == "quadratic":
        return _quadratic_binding(x, *popt, D_fixed)
    if model_name == "stern_volmer":
        return _stern_volmer(x, *popt)
    return _one_site_specific(x, *popt)


def _grubbs_mask(data, alpha: float = 0.05) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    mask = np.ones(len(data), dtype=bool)
    while np.sum(mask) >= 3:
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
        t_crit = t.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = ((N - 1) / np.sqrt(N)) * np.sqrt(t_crit**2 / (N - 2 + t_crit**2))
        if G > G_crit:
            mask[idx_global] = False
        else:
            break
    return mask


def fit_curves(fi_df: pd.DataFrame,
               grubbs_alpha: float = 0.05,
               use_cross_conc_grubbs: bool = True,
               cross_conc_alpha: float = 0.01,
               model_preference: str = "auto",
               progress_cb: ProgressCb = None) -> tuple[list, list]:
    """
    Returns (fit_results, plot_data).
    fit_results: list[dict]
    plot_data:   list[dict]
    """
    groups = (
        fi_df.dropna(subset=["Host", "Dye", "Host_Concentration", "FI-F0"])
        [["Plate", "Host", "Dye"]]
        .drop_duplicates()
    )

    fit_results = []
    plot_data   = []
    total       = len(groups)

    for i, (_, row) in enumerate(groups.iterrows()):
        plate, host, dye = row["Plate"], row["Host"], row["Dye"]
        _log(f"  Fitting [{i+1}/{total}]: {plate} {host}-{dye}", progress_cb)

        D_vals = (fi_df.query("Plate==@plate and Host==@host and Dye==@dye")
                  ["Dye_Concentration"].dropna().unique())

        for D_fixed in D_vals:
            D_fixed = float(D_fixed)
            df_sub  = fi_df.query(
                "Plate==@plate and Host==@host and Dye==@dye "
                "and Dye_Concentration==@D_fixed"
            ).dropna(subset=["Host_Concentration", "FI-F0"])

            x_raw = df_sub["Host_Concentration"].astype(float).values
            y_raw = df_sub["FI-F0"].astype(float).values

            # Within-replicate Grubbs (α = grubbs_alpha, requires n ≥ 3)
            keep_mask = np.ones(len(y_raw), dtype=bool)
            for conc in np.unique(x_raw):
                idx = np.where(x_raw == conc)[0]
                if len(idx) >= 3:
                    keep_mask[idx] = _grubbs_mask(y_raw[idx], alpha=grubbs_alpha)

            x_cl  = x_raw[keep_mask]
            y_cl  = y_raw[keep_mask]
            x_out = x_raw[~keep_mask]
            y_out = y_raw[~keep_mask]

            # Cross-concentration Grubbs on per-concentration means
            # Detects hook-effect drops where all replicates at a concentration
            # fall together — systematic artifact, not replicate noise.
            if use_cross_conc_grubbs:
                unique_concs = np.unique(x_cl)
                if len(unique_concs) >= 4:
                    conc_means = np.array([y_cl[x_cl == c].mean() for c in unique_concs])
                    curve_ok   = _grubbs_mask(conc_means, alpha=cross_conc_alpha)
                    bad_concs  = set(unique_concs[~curve_ok])
                    if bad_concs:
                        cross_keep = np.array([c not in bad_concs for c in x_cl])
                        x_out = np.concatenate([x_out, x_cl[~cross_keep]])
                        y_out = np.concatenate([y_out, y_cl[~cross_keep]])
                        x_cl  = x_cl[cross_keep]
                        y_cl  = y_cl[cross_keep]

            n = len(y_cl)
            if n < 3:
                _log(f"    Skipped (< 3 points after outlier removal)", progress_cb)
                continue

            try:
                Fmax_init   = max(float(np.nanmax(y_cl)), 1e-6)
                Bmin_init   = min(float(np.nanmin(y_cl)), -1e-6)
                Kd_init     = max(float(x_cl[np.argmin(np.abs(y_cl - 0.5 * Fmax_init))]), 1e-12)
                x_med       = float(np.nanmedian(x_cl))
                half_quench = 0.5 * Bmin_init
                sv_kd_init  = max(float(x_cl[np.argmin(np.abs(y_cl - half_quench))]), 1e-12)
                D_cap       = D_fixed

                sv_cfg = dict(
                    name="stern_volmer", func=_stern_volmer,
                    p0=[Bmin_init, sv_kd_init], lo=[-np.inf, 1e-12], hi=[0.0, np.inf])
                pos_cfg = [
                    dict(name="one_site", func=_one_site_specific,
                         p0=[Fmax_init, Kd_init], lo=[0, 1e-12], hi=[np.inf, np.inf]),
                    dict(name="quadratic",
                         func=lambda x, Fmax, Kd, _D=D_cap: _quadratic_binding(x, Fmax, Kd, _D),
                         p0=[Fmax_init, Kd_init], lo=[0, 1e-12], hi=[np.inf, np.inf]),
                ]

                if model_preference == "quadratic":
                    model_configs = [pos_cfg[1]]
                elif model_preference == "one_site":
                    model_configs = [pos_cfg[0]]
                elif model_preference == "stern_volmer":
                    model_configs = [sv_cfg]
                else:
                    # Auto: only offer models that can fit the sign of the data.
                    # Prevents one_site/quadratic (Bmax ≥ 0) from "winning" on
                    # quenching data by converging to a degenerate flat line at 0.
                    data_max = float(np.nanmax(y_cl))
                    data_min = float(np.nanmin(y_cl))
                    if data_max < 0:
                        # All values negative → quenching only
                        model_configs = [sv_cfg]
                    elif data_min >= 0:
                        # All values positive → enhancement only
                        model_configs = pos_cfg
                    else:
                        # Mixed sign → try all
                        model_configs = pos_cfg + [sv_cfg]

                best_fit = None
                for mcfg in model_configs:
                    k = len(mcfg["p0"])
                    if n <= k:
                        continue
                    try:
                        popt, pcov = curve_fit(
                            mcfg["func"], x_cl, y_cl,
                            p0=mcfg["p0"], bounds=(mcfg["lo"], mcfg["hi"]), maxfev=10000)
                    except (RuntimeError, ValueError):
                        try:
                            fb = ([Bmin_init, x_med] if mcfg["name"] == "stern_volmer"
                                  else [Fmax_init, x_med])
                            popt, pcov = curve_fit(
                                mcfg["func"], x_cl, y_cl,
                                p0=fb, bounds=(mcfg["lo"], mcfg["hi"]), maxfev=10000)
                        except (RuntimeError, ValueError):
                            continue

                    y_pred    = mcfg["func"](x_cl, *popt)
                    residuals = y_cl - y_pred
                    ss_res    = float(np.sum(residuals**2))
                    if ss_res <= 0:
                        continue
                    ss_tot = float(np.sum((y_cl - np.mean(y_cl))**2))
                    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                    aicc   = (n * np.log(ss_res / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
                              if n - k - 1 > 0
                              else n * np.log(ss_res / n) + 2 * k)

                    if best_fit is None or aicc < best_fit["aicc"]:
                        best_fit = dict(name=mcfg["name"], popt=popt, pcov=pcov,
                                        r2_adj=r2_adj, residuals=residuals, aicc=aicc, k=k)

                if best_fit is None:
                    _log(f"    All models failed for D={D_fixed}", progress_cb)
                    continue

                popt, pcov = best_fit["popt"], best_fit["pcov"]
                dof        = n - best_fit["k"]
                kd_var     = float(pcov[1, 1])
                kd_se      = float(np.sqrt(kd_var)) if kd_var >= 0 else np.nan
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

                plot_data.append({
                    "plate":      plate,
                    "host":       host,
                    "dye":        dye,
                    "x_cleaned":  x_cl,
                    "y_cleaned":  y_cl,
                    "x_outliers": x_out,
                    "y_outliers": y_out,
                    "model_name": best_fit["name"],
                    "popt":       popt,
                    "D_fixed":    D_fixed,
                    "r2_adj":     best_fit["r2_adj"],
                    "kd_se":      kd_se,
                    "status":     None,   # filled in by apply_thresholds
                })

                fit_results.append({
                    "Plate":             plate,
                    "Host":              host,
                    "Dye":               dye,
                    "Dye_Concentration": D_fixed,
                    "Chosen_Model":      best_fit["name"],
                    "Bmax_or_Bottom":    float(popt[0]),
                    "Kd":                float(popt[1]),
                    "Kd_SE":             kd_se,
                    "Kd_95CI_low":       kd_ci_low,
                    "Kd_95CI_high":      kd_ci_high,
                    "R2_adjusted":       best_fit["r2_adj"],
                    "AICc":              best_fit["aicc"],
                    "n_points":          n,
                    "n_outliers":        len(x_out),
                    "Normality_p":       p_normal,
                    "Host_Conc_min":     float(np.nanmin(x_cl)),
                    "Host_Conc_max":     float(np.nanmax(x_cl)),
                })

            except Exception as exc:
                _log(f"    Error fitting D={D_fixed}: {exc}", progress_cb)
                continue

    _log(f"Fitting complete: {len(fit_results)} curves.", progress_cb)
    return fit_results, plot_data


# ── Stage 6: apply thresholds (cheap, no refit) ───────────────────────────────

def apply_thresholds(fit_results: list, plot_data: list,
                     r2_threshold: float = PASS_R2_DEFAULT,
                     kd_range_lo: float = KD_RANGE_FACTOR_LO_DEFAULT,
                     kd_range_hi: float = KD_RANGE_FACTOR_HI_DEFAULT) -> pd.DataFrame:
    """
    Adds Status and Fail_reason to a copy of fit_results DataFrame.
    Also updates the 'status' field in each plot_data dict in-place.
    """
    df = pd.DataFrame(fit_results).copy()
    if df.empty:
        return df

    fail_r2   = df["R2_adjusted"] < r2_threshold
    fail_bmax = (df["Chosen_Model"] != "stern_volmer") & (df["Bmax_or_Bottom"] < 0)
    fail_kd   = df["Kd"].isna()
    fail_kd_lo = df["Kd"] < kd_range_lo * df["Host_Conc_min"]
    fail_kd_hi = df["Kd"] > kd_range_hi * df["Host_Conc_max"]

    fail_any = fail_r2 | fail_bmax | fail_kd | fail_kd_lo | fail_kd_hi
    df["Status"]      = "PASS"
    df.loc[fail_any, "Status"] = "FAIL"

    df["Fail_reason"] = ""
    df.loc[fail_r2,    "Fail_reason"] += df.loc[fail_r2,   "R2_adjusted"].apply(
        lambda v: f"R²_adj={v:.3f} < {r2_threshold}; ")
    df.loc[fail_bmax,  "Fail_reason"] += "Bmax < 0; "
    df.loc[fail_kd,    "Fail_reason"] += "Kd is NaN; "
    df.loc[fail_kd_lo, "Fail_reason"] += df.loc[fail_kd_lo].apply(
        lambda r: f"Kd={r['Kd']:.3g} < {kd_range_lo}×[Host]_min={r['Host_Conc_min']:.3g}; ", axis=1)
    df.loc[fail_kd_hi, "Fail_reason"] += df.loc[fail_kd_hi].apply(
        lambda r: f"Kd={r['Kd']:.3g} > {kd_range_hi}×[Host]_max={r['Host_Conc_max']:.3g}; ", axis=1)
    df["Fail_reason"] = df["Fail_reason"].str.rstrip("; ")

    # Propagate status back into plot_data dicts
    status_lookup = df.set_index(["Host", "Dye", "Dye_Concentration"])["Status"].to_dict()
    for entry in plot_data:
        entry["status"] = status_lookup.get(
            (entry["host"], entry["dye"], entry["D_fixed"]), "FAIL")

    return df


# ── Plot rendering (shared by GUI and save_outputs) ───────────────────────────

def render_plot(ax, entry: dict, show_status_color: bool = True):
    x_cl  = entry["x_cleaned"]
    y_cl  = entry["y_cleaned"]
    x_out = entry["x_outliers"]
    y_out = entry["y_outliers"]
    popt  = entry["popt"]

    stats = (
        pd.DataFrame({"x": x_cl, "y": y_cl})
        .groupby("x")["y"]
        .agg(["mean", "std"])
    )
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                fmt="o", color=PLOT_COLOR, ecolor=PLOT_COLOR,
                elinewidth=1.5, markersize=4, capsize=2, zorder=3)

    if len(x_out) > 0:
        ax.scatter(x_out, y_out, color="red", marker="o", s=30, zorder=5)

    x_line = np.linspace(x_cl.min(), x_cl.max(), 300)
    y_line = _eval_model(entry["model_name"], x_line, popt, entry["D_fixed"])
    ax.plot(x_line, y_line, color=PLOT_COLOR, linewidth=1.8, zorder=2)

    kd_se  = entry["kd_se"]
    se_str = f" ± {kd_se:.2f}" if not np.isnan(kd_se) else ""
    ax.text(0.96, 0.05,
            f"Kd = {float(popt[1]):.2f}{se_str} µM\n"
            f"R²adj = {entry['r2_adj']:.3f}\n"
            f"[{entry['model_name']}]",
            transform=ax.transAxes, fontsize=6.5, color=PLOT_COLOR,
            va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

    status       = entry.get("status")
    title_color  = ("red" if (show_status_color and status == "FAIL") else "black")
    ax.set_title(f"{entry['host']} – {entry['dye']}\n(D = {entry['D_fixed']} µM)",
                 fontsize=8, color=title_color)
    ax.set_xlabel("[Host] (µM)", fontsize=7)
    ax.set_ylabel("FI – F₀",    fontsize=7)
    ax.tick_params(labelsize=6)


# ── Stage 7: save outputs ─────────────────────────────────────────────────────

COLS_PER_PAGE  = 4
ROWS_PER_PAGE  = 3
PLOTS_PER_PAGE = COLS_PER_PAGE * ROWS_PER_PAGE


def save_outputs(state: PipelineState, output_folder: str,
                 r2_threshold: float = PASS_R2_DEFAULT,
                 progress_cb: ProgressCb = None) -> list[str]:
    """Save all outputs. Returns list of written file paths."""
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    for d in (reports_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    ts    = datetime.now().strftime("%y%m%d_%H%M%S")
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

    if state.df_results is not None:
        df_pass = state.df_results[state.df_results["Status"] == "PASS"]
        df_fail = state.df_results[state.df_results["Status"] == "FAIL"]

        results_path = os.path.join(results_dir, f"{ts}_binding_fit_results.xlsx")
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            df_pass.to_excel(writer, sheet_name="PASS", index=False)
            df_fail.to_excel(writer, sheet_name="FAIL", index=False)
        _log(f"Saved: {results_path}", progress_cb)
        saved.append(results_path)

        # Wide-format FI-F0 export
        if state.fi_df is not None:
            status_lookup = (
                state.df_results
                .set_index(["Host", "Dye", "Dye_Concentration"])["Status"]
                .to_dict()
            )
            pass_blocks, fail_blocks = [], []
            src = state.fi_df.dropna(subset=["Host", "Dye", "FI-F0"])
            for (host, dye, D_fixed), grp in src.groupby(["Host", "Dye", "Dye_Concentration"]):
                grp    = grp.sort_values("Host_Concentration")
                status = status_lookup.get((host, dye, D_fixed), "FAIL")
                concs  = grp["Host_Concentration"].drop_duplicates().reset_index(drop=True)
                fi_by_conc = grp.groupby("Host_Concentration")["FI-F0"].apply(list)
                max_reps   = fi_by_conc.apply(len).max()
                block      = pd.DataFrame({f"{host} | {dye} [{D_fixed}] µM": concs})
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

    if state.plot_data:
        pdf_path = os.path.join(plots_dir, f"{ts}_binding_results.pdf")
        plates   = sorted(set(e["plate"] for e in state.plot_data))
        with PdfPages(pdf_path) as pdf:
            for plate in plates:
                page_items = [e for e in state.plot_data if e["plate"] == plate]
                n     = len(page_items)
                cols  = min(COLS_PER_PAGE, n)
                rows  = int(np.ceil(n / cols))
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 4, rows * 3.2),
                                         squeeze=False)
                axes_flat = axes.flatten()
                for ax, entry in zip(axes_flat, page_items):
                    render_plot(ax, entry)
                for ax in axes_flat[n:]:
                    fig.delaxes(ax)
                fig.suptitle(f"Plate: {plate}", fontsize=11, fontweight="bold")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        _log(f"Saved: {pdf_path}", progress_cb)
        saved.append(pdf_path)

    return saved


# ── Preview: list files that save_outputs would write ─────────────────────────

def preview_save_files(state: PipelineState, output_folder: str) -> list[dict]:
    reports_dir = os.path.join(output_folder, "output_reports")
    results_dir = os.path.join(output_folder, "results")
    plots_dir   = os.path.join(results_dir, "plots")
    ts          = datetime.now().strftime("%y%m%d_%H%M%S")
    files       = []

    def _f(folder, name, desc):
        files.append({"path": os.path.join(folder, f"{ts}_{name}"), "description": desc})

    if state.merged_mapping is not None:
        _f(reports_dir, "mapping.csv",            "Merged dye/host mapping")
    if state.fluorescence is not None:
        _f(reports_dir, "raw_data.csv",           "Raw fluorescence data")
    if state.merged is not None:
        _f(reports_dir, "merged_with_blanks.csv", "Merged data (with blank labels)")
    if state.fi_df is not None:
        _f(reports_dir, "blank_corrected.csv",    "Background-subtracted (FI-F0)")
    if state.df_results is not None:
        _f(results_dir, "binding_fit_results.xlsx", "Fit results (PASS/FAIL sheets)")
        if state.fi_df is not None:
            _f(results_dir, "data.xlsx",          "Wide-format FI-F0 (PASS/FAIL sheets)")
    if state.plot_data:
        _f(plots_dir,   "binding_results.pdf",    "All binding curve plots (FAIL titles in red)")

    return files
