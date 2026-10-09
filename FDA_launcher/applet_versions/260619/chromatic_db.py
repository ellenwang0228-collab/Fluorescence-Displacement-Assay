#!/usr/bin/env python3
"""
Chromatic-channel -> dye resolution, shared by pipeline_fda.py and pipeline_ki.py.

Plate readers record the physical filter (Ex-bandwidth/Em-bandwidth) used for
each acquisition channel in a header block at the top of every raw export:

    No. of Channels / Multichromatics: 3
    Used filter settings and gain values:
      1: 360-20/460-30                                         auto
      2: 355-20/455-30                                         auto
      3: 369-15/505-20                                         auto
      4: -
      5: -

The 1-based index here is the same index used by the "Chromatic: N" data
blocks later in the same sheet (confirmed against real exports from both
pipelines). Looking up each filter string in a small Dye -> Filter database
lets every raw file resolve its own channel -> dye mapping from data the
instrument already wrote, instead of guessing chromatic order per run — so
the result is identical whether one dye was run per plate (one channel per
file) or several dyes were run on the same plate (several channels per
file).
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

import pandas as pd

ProgressCb = Optional[Callable[[str], None]]

_FILTER_RE       = re.compile(r"(\d+-\d+/\d+-\d+)")
_FILTER_CTR_RE   = re.compile(r"^(\d+)-\d+/(\d+)-\d+$")


def _filter_centers(filt: str) -> Optional[tuple]:
    """Return (exc_center_nm, em_center_nm) from a normalised filter string, or None."""
    m = _FILTER_CTR_RE.match(filt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _log(msg: str, cb: ProgressCb) -> None:
    if cb:
        cb(msg)
    else:
        print(msg)


def normalize_filter(raw) -> Optional[str]:
    """Extract the bare 'exc-bw/em-bw' token, stripping gain/auto suffixes."""
    if not isinstance(raw, str):
        return None
    m = _FILTER_RE.search(raw)
    return m.group(1) if m else None


def load_chromatic_db(folder: str, progress_cb: ProgressCb = None) -> pd.DataFrame:
    """Load the Dye -> Filter lookup table from a folder of .xlsx files
    (same convention as the Kd_comp table). Expected columns: Dye, Filter,
    and an optional Aliases column (semicolon-separated alternate spellings
    of the same physical dye, e.g. Dye=H33258, Aliases=H33) — see
    build_alias_map.
    """
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    _log(f"Chromatic DB files: {files}", progress_cb)
    if not files:
        return pd.DataFrame(columns=["Dye", "Filter", "Aliases"])

    df = pd.concat([pd.read_excel(os.path.join(folder, f)) for f in files],
                   ignore_index=True)
    if "Aliases" not in df.columns:
        df["Aliases"] = None
    df = df.loc[:, ["Dye", "Filter", "Aliases"]].copy()
    df["Dye"]    = df["Dye"].astype(str).str.strip()
    df["Filter"] = df["Filter"].apply(normalize_filter)
    df = df.dropna(subset=["Dye", "Filter"])
    df = df[df["Dye"] != ""]

    dups = df.duplicated(subset=["Filter"], keep=False)
    if dups.any():
        _log("NOTE: chromatic DB has filter(s) shared by more than one Dye row "
             "— auto-detection picks one arbitrarily for these. If they're the "
             "same physical dye under different spellings, use the Aliases "
             f"column on a single row instead: {sorted(df.loc[dups, 'Filter'].unique())}",
             progress_cb)

    _log(f"Chromatic DB: {len(df)} dye-filter entries", progress_cb)
    return df


def build_alias_map(chromatic_db: Optional[pd.DataFrame]) -> dict[str, str]:
    """Map every known alias (case-insensitive, stripped) to its canonical
    Dye name, using the optional 'Aliases' column in the chromatic DB.

    Lets the same physical dye be recognised under different spellings
    across experiments (e.g. dye_map says "H33" but the chromatic DB's
    canonical name, and the Kd table, say "H33258") without renaming
    anything in the experiment's own files.
    """
    alias_map: dict[str, str] = {}
    if chromatic_db is None or chromatic_db.empty:
        return alias_map
    for _, row in chromatic_db.iterrows():
        canonical = row["Dye"]
        alias_map[canonical.strip().lower()] = canonical
        aliases = row.get("Aliases")
        if isinstance(aliases, str):
            for alias in aliases.split(";"):
                alias = alias.strip()
                if alias:
                    alias_map[alias.lower()] = canonical
    return alias_map


def dye_matches(dye, chromatic_dye, alias_map: dict) -> bool:
    """True if dye and chromatic_dye are the same physical dye, resolving
    both through alias_map first (see build_alias_map). NaN on either side
    counts as a match (background wells / unresolved channels bypass this
    check elsewhere)."""
    if pd.isna(dye) or pd.isna(chromatic_dye):
        return True
    a = alias_map.get(str(dye).strip().lower(), dye)
    b = alias_map.get(str(chromatic_dye).strip().lower(), chromatic_dye)
    return a == b


def parse_filter_settings(df_raw: pd.DataFrame) -> dict[int, str]:
    """Parse the 'Used filter settings and gain values:' header block.

    Returns {channel_index: normalized_filter_string}, skipping channels the
    instrument marks unused ('-').
    """
    settings: dict[int, str] = {}
    in_block = False
    for i in range(len(df_raw)):
        cell = str(df_raw.iloc[i, 0]).strip()
        if cell.lower().startswith("used filter settings"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"(\d+):\s*(.+)", cell)
            if not m:
                break
            idx  = int(m.group(1))
            filt = normalize_filter(m.group(2))
            if filt:
                settings[idx] = filt
    return settings


def resolve_chromatic_to_dye(df_raw: pd.DataFrame, chromatic_db: Optional[pd.DataFrame],
                             plate_name: str = "",
                             progress_cb: ProgressCb = None) -> dict[int, str]:
    """Build {chromatic_index: dye_name} for one raw file from its own
    filter-settings header, using chromatic_db as the Filter -> Dye lookup.

    Returns {} if the file has no filter-settings header (older export
    format) or the database is empty — callers should fall back to their
    existing behaviour (manual mapping / legacy label matching) in that case.
    """
    if chromatic_db is None or chromatic_db.empty:
        return {}
    settings = parse_filter_settings(df_raw)
    if not settings:
        return {}

    # Build exact filter → dye lookup.  Multiple rows for the same dye (different
    # filter strings) are all preserved — later rows overwrite earlier ones only
    # when the FILTER string itself is duplicated (already warned in load_chromatic_db).
    filter_to_dye = dict(zip(chromatic_db["Filter"], chromatic_db["Dye"]))

    # Build centre-wavelength fallback: (exc_centre, em_centre) → dye.
    # Used when the plate reader records a slightly different bandwidth than what
    # is in the DB (e.g. "360-20/460-30" vs "360-20/460-35").
    centre_to_dye: dict[tuple, str] = {}
    for filt, dye in filter_to_dye.items():
        ctrs = _filter_centers(filt)
        if ctrs and ctrs not in centre_to_dye:
            centre_to_dye[ctrs] = dye

    result, unmatched = {}, []
    for idx, filt in settings.items():
        dye = filter_to_dye.get(filt)
        if dye is None:
            ctrs = _filter_centers(filt)
            if ctrs:
                dye = centre_to_dye.get(ctrs)
                if dye:
                    _log(f"  Note: '{plate_name}' channel {idx} ({filt}) matched "
                         f"'{dye}' via centre wavelengths (bandwidth differs from DB).",
                         progress_cb)
        if dye:
            result[idx] = dye
        else:
            unmatched.append(f"{idx}: {filt}")

    if unmatched:
        _log(f"  WARNING: '{plate_name}' — no chromatic DB match for filter(s) "
             f"{unmatched}; these channels were not auto-assigned a dye.",
             progress_cb)
    return result
