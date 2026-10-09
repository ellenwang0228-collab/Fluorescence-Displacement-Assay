#!/usr/bin/env python3
"""
PhosphoMAX — session save / load.

A "session" is a single self-contained ``.phosmax`` file holding everything
needed to reopen an analysis exactly as it was left: every widget setting, the
pipeline dataframes, the fit results and the per-curve plot arrays.  Reopening
needs nothing but the file itself — not the original Raw/ folder, not the same
machine, not the same day.

──────────────────────────────────────────────────────────────────────────────
DESIGN RULES  (these are the reason the format survives code changes)
──────────────────────────────────────────────────────────────────────────────
1. NO PICKLE, EVER.  The file contains only JSON and raw numeric buffers.
   Nothing in it names a class, a module or a function in this codebase, so
   renaming a widget, moving a class or editing a pipeline can never make an
   old session unreadable.

2. EVERY READ IS BEST-EFFORT.  Loading never raises on a mismatch.  Anything
   that cannot be restored is recorded in a ``LoadReport`` and shown to the
   user afterwards; the rest still loads.

3. SETTINGS ARE KEYED BY NAME, NOT POSITION.  Combo boxes are stored by their
   visible TEXT, not their index, so reordering or inserting items in a combo
   does not silently change a restored setting.  Spin boxes are clamped into
   the widget's current range if the range was tightened.

4. THE SETTINGS SNAPSHOT IS SELF-MAINTAINING.  Widgets are discovered by
   walking the window's attributes, so a widget added to the GUI later is
   saved and restored automatically with no change to this module.

──────────────────────────────────────────────────────────────────────────────
FILE LAYOUT  (.phosmax is an ordinary zip — you can unzip it to inspect)
──────────────────────────────────────────────────────────────────────────────
    manifest.json    format + app version, mode, timestamp, library versions
    settings.json    {dotted.widget.key: {"t": <type tag>, "v": <value>}}
    extras.json      free-form per-mode dict (file lists, exclusions, …)
    state.json       pipeline state: dataframes + fit_results + plot_data,
                     with every numpy array replaced by {"__arr__": "<key>"}
    arrays.npz       all numpy arrays, compressed

FORMAT_VERSION is bumped only for changes that a reader must understand.
Readers accept any version <= their own and warn (but still load) above it.
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
MAGIC = "phosmax-session"
SESSION_EXT = ".phosmax"

# Bump when the launcher itself changes in a way worth recording in the file.
# Purely informational — never used to gate a load.
APP_VERSION = "2026.08"


# ══════════════════════════════════════════════════════════════════════════════
#  Load report
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadReport:
    """Everything that did not restore cleanly. Never fatal, always shown."""
    source: str = ""
    saved_app_version: str = ""
    saved_format_version: int = 0
    saved_at: str = ""
    unknown_settings: list = field(default_factory=list)   # in file, not in GUI
    type_changed: list = field(default_factory=list)       # widget type differs
    value_rejected: list = field(default_factory=list)     # combo text is gone
    value_clamped: list = field(default_factory=list)      # out of spin range
    missing_state: list = field(default_factory=list)      # absent from file
    errors: list = field(default_factory=list)             # decode failures
    notes: list = field(default_factory=list)              # informational

    @property
    def clean(self) -> bool:
        return not (self.unknown_settings or self.type_changed
                    or self.value_rejected or self.value_clamped
                    or self.errors)

    def to_text(self) -> str:
        L = []
        L.append(f"Session:  {os.path.basename(self.source)}")
        L.append(f"Saved:    {self.saved_at or 'unknown'}"
                 f"   (app {self.saved_app_version or '?'},"
                 f" format v{self.saved_format_version})")
        if self.clean and not self.missing_state and not self.notes:
            L.append("")
            L.append("Restored completely — no differences found.")
            return "\n".join(L)

        def _block(title, items, limit=25):
            if not items:
                return
            L.append("")
            L.append(f"{title}  ({len(items)})")
            for it in items[:limit]:
                L.append(f"   • {it}")
            if len(items) > limit:
                L.append(f"   … and {len(items) - limit} more")

        _block("Settings in the file that no longer exist in this version "
               "(ignored)", self.unknown_settings)
        _block("Settings whose control changed type (left at default)",
               self.type_changed)
        _block("Saved choices no longer offered (left at default)",
               self.value_rejected)
        _block("Values outside the current allowed range (clamped)",
               self.value_clamped)
        _block("Results not present in the file", self.missing_state)
        _block("Decode problems", self.errors)
        _block("Notes", self.notes)
        L.append("")
        L.append("Everything not listed above was restored.")
        return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════════
#  Value codec — JSON-safe encoding of numpy / pandas / python objects
# ══════════════════════════════════════════════════════════════════════════════
#
# Encoded forms (all dicts with a single reserved "__tag__" key):
#   {"__arr__":  "k7"}                     numpy array, payload in arrays.npz
#   {"__df__":   {...}}                    pandas DataFrame (columnar)
#   {"__tuple__": [...]}                   tuple
#   {"__set__":  [...]}                    set
#   {"__float__": "nan"|"inf"|"-inf"}      non-finite float
#   {"__ts__":   "<iso>"}                  pandas/py Timestamp
#   {"__dropped__": "<why>"}               unserialisable (e.g. a closure)
#   {"__keydict__": [[k, v], ...]}         dict with non-string keys
#
# A decoder that meets an unknown "__…__" tag returns None and logs it, so a
# NEWER file opened by an OLDER build degrades gracefully instead of crashing.

_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


class _ArrayStore:
    """Collects numpy arrays during encoding, hands them back for the npz."""

    def __init__(self):
        self._arrays: dict = {}
        self._n = 0

    def put(self, arr: np.ndarray) -> str:
        key = f"a{self._n}"
        self._n += 1
        self._arrays[key] = arr
        return key

    def get(self, key: str):
        return self._arrays.get(key)

    def load(self, arrays: dict):
        self._arrays = arrays

    @property
    def arrays(self) -> dict:
        return self._arrays


def _encode(obj, store: _ArrayStore, report: LoadReport = None, _depth: int = 0):
    """Recursively turn *obj* into something json.dump can write."""
    if _depth > 60:
        return {"__dropped__": "max recursion depth"}

    # ── scalars ───────────────────────────────────────────────────────────────
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__float__": "nan"}
        if math.isinf(obj):
            return {"__float__": "inf" if obj > 0 else "-inf"}
        return obj

    # ── numpy ─────────────────────────────────────────────────────────────────
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # Object arrays are not npz-safe; store element-wise as JSON.
            return {"__objarr__": [_encode(v, store, report, _depth + 1)
                                   for v in obj.ravel().tolist()],
                    "shape": list(obj.shape)}
        return {"__arr__": store.put(obj)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _encode(float(obj), store, report, _depth + 1)
    if isinstance(obj, np.generic):
        return _encode(obj.item(), store, report, _depth + 1)

    # ── pandas ────────────────────────────────────────────────────────────────
    if isinstance(obj, pd.DataFrame):
        return {"__df__": _encode_df(obj, store, report, _depth)}
    if isinstance(obj, pd.Series):
        return {"__df__": _encode_df(obj.to_frame(), store, report, _depth),
                "series_name": str(obj.name) if obj.name is not None else None}
    if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime):
        return {"__ts__": obj.isoformat()}
    if obj is pd.NaT:
        return None

    # ── containers ────────────────────────────────────────────────────────────
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, store, report, _depth + 1) for v in obj]}
    if isinstance(obj, (set, frozenset)):
        return {"__set__": [_encode(v, store, report, _depth + 1)
                            for v in sorted(obj, key=repr)]}
    if isinstance(obj, list):
        return [_encode(v, store, report, _depth + 1) for v in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: _encode(v, store, report, _depth + 1) for k, v in obj.items()}
        return {"__keydict__": [[_encode(k, store, report, _depth + 1),
                                 _encode(v, store, report, _depth + 1)]
                                for k, v in obj.items()]}

    # ── anything else ─────────────────────────────────────────────────────────
    # Callables (the Ki model factory) land here.  They are deliberately NOT
    # stored — a pickled closure would tie the file to this exact source file,
    # which is the one thing this format must never do.  They are rebuilt on
    # load from the model name and its parameters, which ARE stored.
    if callable(obj):
        return {"__dropped__": "callable (rebuilt on load)"}
    return {"__dropped__": f"unsupported type {type(obj).__name__}"}


def _decode(obj, store: _ArrayStore, report: LoadReport, _depth: int = 0):
    if _depth > 60:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_decode(v, store, report, _depth + 1) for v in obj]
    if not isinstance(obj, dict):
        return obj

    tags = [k for k in obj if k.startswith("__") and k.endswith("__")]
    if not tags:
        return {k: _decode(v, store, report, _depth + 1) for k, v in obj.items()}

    tag = tags[0]
    try:
        if tag == "__arr__":
            arr = store.get(obj[tag])
            if arr is None:
                report.errors.append(f"array '{obj[tag]}' missing from archive")
                return np.array([])
            return arr
        if tag == "__objarr__":
            vals = [_decode(v, store, report, _depth + 1) for v in obj[tag]]
            arr = np.empty(len(vals), dtype=object)
            for i, v in enumerate(vals):
                arr[i] = v
            shape = obj.get("shape")
            return arr.reshape(shape) if shape else arr
        if tag == "__df__":
            df = _decode_df(obj[tag], store, report, _depth)
            if obj.get("series_name") is not None or df.shape[1] == 1 and "series_name" in obj:
                return df.iloc[:, 0].rename(obj.get("series_name"))
            return df
        if tag == "__tuple__":
            return tuple(_decode(v, store, report, _depth + 1) for v in obj[tag])
        if tag == "__set__":
            return set(_decode(v, store, report, _depth + 1) for v in obj[tag])
        if tag == "__float__":
            return _NON_FINITE.get(obj[tag], math.nan)
        if tag == "__ts__":
            return pd.Timestamp(obj[tag])
        if tag == "__keydict__":
            return {_decode(k, store, report, _depth + 1):
                    _decode(v, store, report, _depth + 1) for k, v in obj[tag]}
        if tag == "__dropped__":
            return None
    except Exception as exc:                                   # never fatal
        report.errors.append(f"{tag}: {type(exc).__name__}: {exc}")
        return None

    # Unknown tag → this file was written by a NEWER build.  Degrade quietly.
    report.errors.append(f"unrecognised tag {tag} (file is newer than this build)")
    return None


# ── DataFrame codec ───────────────────────────────────────────────────────────
#
# Numeric / bool / datetime columns go into the npz as typed arrays (exact,
# compact, fast).  Object columns go to JSON element-wise.  Column dtype is
# recorded so a round trip preserves it.

def _encode_df(df: pd.DataFrame, store: _ArrayStore, report, _depth: int) -> dict:
    cols = []
    for name in df.columns:
        s = df[name]
        entry = {"name": str(name), "dtype": str(s.dtype)}
        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            entry["json"] = [_encode(v, store, report, _depth + 1)
                             for v in s.tolist()]
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry["arr"] = store.put(s.values.astype("datetime64[ns]").view("int64"))
            entry["kind"] = "datetime"
        else:
            try:
                entry["arr"] = store.put(np.asarray(s.values))
            except Exception:
                entry["json"] = [_encode(v, store, report, _depth + 1)
                                 for v in s.tolist()]
        cols.append(entry)

    idx = df.index
    index = {"name": [str(n) if n is not None else None
                      for n in (idx.names if idx.nlevels > 1 else [idx.name])]}
    if isinstance(idx, pd.RangeIndex):
        index["kind"] = "range"
        index["start"], index["stop"], index["step"] = idx.start, idx.stop, idx.step
    else:
        index["kind"] = "values"
        index["values"] = [_encode(v, store, report, _depth + 1)
                           for v in idx.tolist()]
        index["nlevels"] = idx.nlevels
    return {"columns": cols, "index": index, "nrows": int(len(df))}


def _decode_df(blob: dict, store: _ArrayStore, report: LoadReport, _depth: int) -> pd.DataFrame:
    data = {}
    order = []
    for entry in blob.get("columns", []):
        name = entry.get("name")
        order.append(name)
        try:
            if "arr" in entry:
                arr = store.get(entry["arr"])
                if arr is None:
                    report.errors.append(f"column '{name}': array missing")
                    arr = np.full(blob.get("nrows", 0), np.nan)
                if entry.get("kind") == "datetime":
                    arr = arr.astype("int64").view("datetime64[ns]")
                data[name] = arr
            else:
                data[name] = [_decode(v, store, report, _depth + 1)
                              for v in entry.get("json", [])]
        except Exception as exc:
            report.errors.append(f"column '{name}': {type(exc).__name__}: {exc}")
            data[name] = [None] * blob.get("nrows", 0)

    df = pd.DataFrame(data, columns=order) if order else pd.DataFrame()

    # Restore declared dtypes where still possible; a failure is cosmetic.
    for entry in blob.get("columns", []):
        want = entry.get("dtype")
        name = entry.get("name")
        if not want or name not in df.columns:
            continue
        if str(df[name].dtype) == want:
            continue
        try:
            df[name] = df[name].astype(want)
        except Exception:
            pass

    idx = blob.get("index", {})
    try:
        if idx.get("kind") == "range":
            df.index = pd.RangeIndex(idx["start"], idx["stop"], idx["step"])
        elif idx.get("kind") == "values":
            vals = [_decode(v, store, report, _depth + 1) for v in idx.get("values", [])]
            if len(vals) == len(df):
                if idx.get("nlevels", 1) > 1:
                    df.index = pd.MultiIndex.from_tuples(
                        [tuple(v) if isinstance(v, (list, tuple)) else (v,) for v in vals])
                else:
                    df.index = pd.Index(vals)
        names = idx.get("name") or [None]
        if len(names) == df.index.nlevels:
            df.index.names = names
    except Exception as exc:
        report.errors.append(f"index: {type(exc).__name__}: {exc}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Archive read / write
# ══════════════════════════════════════════════════════════════════════════════

def write_session(path: str, *, mode: str, settings: dict,
                  state: dict, extras: dict = None,
                  attachments: list = None,
                  app_version: str = APP_VERSION) -> str:
    """Write a .phosmax file.  Written to a temp name and renamed on success,
    so an interrupted save can never leave a half-written file in its place.

    ``attachments`` is a list of source file paths (the Summary-Plots and
    Compare-Runs modes read .xlsx files rather than running a pipeline).  A
    copy of each is tucked inside the session so it still opens after the
    originals have been moved, renamed or left on another machine.
    """
    if not path.lower().endswith(SESSION_EXT):
        path += SESSION_EXT

    store = LoadReport()      # reused purely as an error sink during encode
    arrays = _ArrayStore()
    state_blob = _encode(state or {}, arrays, store)
    extras_blob = _encode(extras or {}, arrays, store)
    settings_blob = _encode(settings or {}, arrays, store)

    # Copy in the source workbooks, de-duplicated, skipping anything unreadable.
    att_index, att_blobs, seen = [], {}, set()
    for i, src in enumerate(attachments or []):
        src = str(src)
        if src in seen:
            continue
        seen.add(src)
        try:
            with open(src, "rb") as fh:
                data = fh.read()
        except Exception:
            att_index.append({"path": src, "stored": None})
            continue
        member = f"sources/{i:03d}_{os.path.basename(src)}"
        att_blobs[member] = data
        att_index.append({"path": src, "stored": member,
                          "size": len(data)})

    manifest = {
        "magic":          MAGIC,
        "format_version": FORMAT_VERSION,
        "app_version":    app_version,
        "mode":           mode,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
        "libs": {"pandas": pd.__version__, "numpy": np.__version__},
        "attachments":    att_index,
    }

    npz_buf = io.BytesIO()
    if arrays.arrays:
        np.savez_compressed(npz_buf, **arrays.arrays)

    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("settings.json", json.dumps(settings_blob))
        z.writestr("extras.json",   json.dumps(extras_blob))
        z.writestr("state.json",    json.dumps(state_blob))
        if arrays.arrays:
            z.writestr("arrays.npz", npz_buf.getvalue())
        for member, data in att_blobs.items():
            z.writestr(member, data)
    os.replace(tmp, path)
    return path


def extract_attachments(session_path: str, dest_dir: str,
                        report: LoadReport = None) -> dict:
    """Unpack the stored source workbooks into *dest_dir*.

    Returns {original path: usable path}.  A file still sitting where it was
    saved is used in place; anything missing falls back to the copy inside
    the session, so a session opened on a different machine still works.
    """
    out = {}
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(session_path, "r") as z:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
            for entry in manifest.get("attachments") or []:
                orig, member = entry.get("path"), entry.get("stored")
                if orig and os.path.isfile(orig):
                    out[orig] = orig
                    continue
                if not member or member not in z.namelist():
                    if report is not None:
                        report.errors.append(f"source file unavailable: {orig}")
                    continue
                dest = os.path.join(dest_dir, os.path.basename(member))
                with open(dest, "wb") as fh:
                    fh.write(z.read(member))
                out[orig] = dest
                if report is not None:
                    report.notes.append(
                        f"{os.path.basename(orig)} was not at its original "
                        f"location — used the copy stored in the session")
    except Exception as exc:
        if report is not None:
            report.errors.append(f"attachments: {type(exc).__name__}: {exc}")
    return out


@dataclass
class SessionBundle:
    mode: str = ""
    settings: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    report: LoadReport = field(default_factory=LoadReport)


def read_session(path: str) -> SessionBundle:
    """Read a .phosmax file.  Raises only if the file is not a readable
    archive; every lesser problem lands in ``bundle.report``."""
    report = LoadReport(source=path)

    with zipfile.ZipFile(path, "r") as z:
        names = set(z.namelist())

        def _json(name, default):
            if name not in names:
                report.missing_state.append(name)
                return default
            try:
                return json.loads(z.read(name).decode("utf-8"))
            except Exception as exc:
                report.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                return default

        manifest = _json("manifest.json", {})
        if manifest.get("magic") not in (MAGIC, None):
            report.notes.append("File does not carry the PhosphoMAX marker — "
                                "reading it anyway.")
        fv = int(manifest.get("format_version") or 0)
        report.saved_format_version = fv
        report.saved_app_version = str(manifest.get("app_version") or "")
        report.saved_at = str(manifest.get("saved_at") or "")
        if fv > FORMAT_VERSION:
            report.notes.append(
                f"This session was written by a newer version of the app "
                f"(format v{fv} vs v{FORMAT_VERSION}). Anything this build "
                f"does not understand is listed below.")

        arrays = _ArrayStore()
        if "arrays.npz" in names:
            try:
                with np.load(io.BytesIO(z.read("arrays.npz")),
                             allow_pickle=False) as npz:
                    arrays.load({k: npz[k] for k in npz.files})
            except Exception as exc:
                report.errors.append(f"arrays.npz: {type(exc).__name__}: {exc}")

        settings = _decode(_json("settings.json", {}), arrays, report)
        state    = _decode(_json("state.json", {}),    arrays, report)
        extras   = _decode(_json("extras.json", {}),   arrays, report)

    return SessionBundle(mode=str(manifest.get("mode") or ""),
                         settings=settings or {}, state=state or {},
                         extras=extras or {}, manifest=manifest, report=report)


def peek_mode(path: str) -> str:
    """Read just the mode from a session file — used by the Mode Picker so it
    can open the right window before doing any heavy decoding."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            return str(json.loads(z.read("manifest.json").decode("utf-8")).get("mode") or "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  Widget snapshot / restore
# ══════════════════════════════════════════════════════════════════════════════
#
# The window's widgets are found by walking its attributes, so every control
# is captured without this module holding a hand-written list that would drift
# out of date the moment a new spin box is added to the GUI.
#
# A key is the dotted attribute path from the window, e.g.
#       "_r2_spin"                       (a spin box on the window)
#       "_summary_tab._bar_col"          (a colour row inside a tab)
# If an attribute in that path disappears the setting is simply reported and
# skipped; if a NEW one appears it keeps its default.

try:                                                              # noqa: E402
    from PySide6.QtWidgets import (
        QWidget, QDoubleSpinBox, QSpinBox, QCheckBox, QRadioButton,
        QComboBox, QLineEdit, QSlider, QPlainTextEdit, QGroupBox,
    )
    _HAVE_QT = True
except ImportError:      # allows a session file to be inspected headlessly
    _HAVE_QT = False

    class _NoQt:         # placeholders so the module still imports
        pass

    QWidget = QDoubleSpinBox = QSpinBox = QCheckBox = QRadioButton = _NoQt
    QComboBox = QLineEdit = QSlider = QPlainTextEdit = QGroupBox = _NoQt

# Attributes never worth saving: transient runtime handles, log panes, and
# anything holding live data rather than a user choice.  Everything else is
# filtered by type — a table model or a canvas is neither a settable control
# nor an application widget, so it is skipped without needing to be named.
_SKIP_ATTRS = {
    "_worker", "_state", "_log", "_status_lbl", "_refresh_timer",
    "_current_fig", "_current_figs", "_df", "_runs", "_run_items",
    "parent", "_parent",
}


def _is_own_widget(obj) -> bool:
    """True for widgets defined by this application, false for stock Qt ones.

    Recursion is limited to the app's own composite widgets. Walking into Qt's
    containers instead would traverse the entire widget tree and produce
    thousands of meaningless keys. Testing the module name (rather than a
    hard-coded list of class names) means a widget class added later — or the
    launcher being split across several modules — needs no change here.
    """
    if not isinstance(obj, QWidget):
        return False
    mod = type(obj).__module__ or ""
    return not mod.startswith("PySide6") and not mod.startswith("shiboken")


# ── leaf handlers ─────────────────────────────────────────────────────────────
# Each is (tag, matches(obj), get(obj), set(obj, value, key, report)).
# Order matters: the first match wins, so QRadioButton is tested before the
# generic QCheckBox branch it does not inherit from, and duck-typed custom
# widgets are tested before the Qt classes they are built from.

def _set_spin(w, v, key, report):
    try:
        v = float(v)
    except (TypeError, ValueError):
        report.value_rejected.append(f"{key}: {v!r} is not a number")
        return
    lo, hi = w.minimum(), w.maximum()
    if v < lo or v > hi:
        report.value_clamped.append(f"{key}: {v:g} → {min(max(v, lo), hi):g} "
                                    f"(allowed {lo:g}…{hi:g})")
        v = min(max(v, lo), hi)
    w.setValue(v)


def _set_int_spin(w, v, key, report):
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        report.value_rejected.append(f"{key}: {v!r} is not a whole number")
        return
    lo, hi = w.minimum(), w.maximum()
    if v < lo or v > hi:
        report.value_clamped.append(f"{key}: {v} → {min(max(v, lo), hi)} "
                                    f"(allowed {lo}…{hi})")
        v = min(max(v, lo), hi)
    w.setValue(v)


def _set_combo(w, v, key, report):
    """Match by visible text, case-insensitively, so reordering the items in a
    combo — or changing their capitalisation — does not silently pick a
    different option than the one that was saved."""
    text = "" if v is None else str(v)
    items = [w.itemText(i) for i in range(w.count())]
    if text in items:
        w.setCurrentIndex(items.index(text))
        return
    lowered = [s.strip().lower() for s in items]
    if text.strip().lower() in lowered:
        w.setCurrentIndex(lowered.index(text.strip().lower()))
        return
    if w.isEditable():
        w.setCurrentText(text)
        return
    report.value_rejected.append(
        f"{key}: saved choice {text!r} is no longer offered "
        f"(kept {w.currentText()!r})")


def _set_checkable(w, v, key, report):
    w.setChecked(bool(v))


def _set_text(w, v, key, report):
    w.setText("" if v is None else str(v))


def _set_plain(w, v, key, report):
    w.setPlainText("" if v is None else str(v))


def _set_color(w, v, key, report):
    if isinstance(v, str) and v:
        w.set_color(v)
    else:
        report.value_rejected.append(f"{key}: {v!r} is not a colour")


def _set_path(w, v, key, report):
    w.edit.setText("" if v is None else str(v))
    if v and not os.path.isdir(str(v)):
        report.notes.append(f"{key}: folder no longer exists — {v}")


def _set_cfg(w, v, key, report):
    if isinstance(v, dict):
        w.set_session_cfg(v)
    else:
        report.value_rejected.append(f"{key}: expected a settings block")


_LEAF_HANDLERS = [
    # custom application widgets, identified by their API not their class name
    ("color", lambda o: hasattr(o, "color") and hasattr(o, "set_color"),
     lambda o: o.color, _set_color),
    ("path",  lambda o: hasattr(o, "edit") and hasattr(o, "path"),
     lambda o: o.path, _set_path),
    ("cfg",   lambda o: hasattr(o, "get_session_cfg") and hasattr(o, "set_session_cfg"),
     lambda o: o.get_session_cfg(), _set_cfg),
    # plain Qt controls
    ("float", lambda o: isinstance(o, QDoubleSpinBox), lambda o: float(o.value()), _set_spin),
    ("int",   lambda o: isinstance(o, (QSpinBox, QSlider)), lambda o: int(o.value()), _set_int_spin),
    ("bool",  lambda o: isinstance(o, (QCheckBox, QRadioButton)), lambda o: bool(o.isChecked()), _set_checkable),
    ("combo", lambda o: isinstance(o, QComboBox), lambda o: o.currentText(), _set_combo),
    ("text",  lambda o: isinstance(o, QLineEdit), lambda o: o.text(), _set_text),
    ("plain", lambda o: isinstance(o, QPlainTextEdit), lambda o: o.toPlainText(), _set_plain),
    ("groupbox", lambda o: isinstance(o, QGroupBox) and o.isCheckable(),
     lambda o: bool(o.isChecked()), _set_checkable),
]


def _leaf_for(obj):
    for tag, matches, getter, setter in _LEAF_HANDLERS:
        try:
            if matches(obj):
                return tag, getter, setter
        except Exception:
            continue
    return None


def _handler_by_tag(tag):
    for t, _m, getter, setter in _LEAF_HANDLERS:
        if t == tag:
            return getter, setter
    return None, None


def _skip(name: str) -> bool:
    return name in _SKIP_ATTRS


def collect_settings(window, _obj=None, _prefix="", _out=None, _seen=None,
                     _depth=0) -> dict:
    """Snapshot every user-settable control reachable from *window*.

    Returns {dotted_key: {"t": tag, "v": value}}.  Dynamic groups — lists of
    (label, widget) pairs built after data loads, such as the spectral
    per-concentration colour rows — are stored keyed by their label so they
    survive a different set of concentrations next time.
    """
    obj = window if _obj is None else _obj
    out = {} if _out is None else _out
    seen = set() if _seen is None else _seen

    if _depth > 6 or id(obj) in seen:
        return out
    seen.add(id(obj))

    for name, value in list(vars(obj).items()):
        if _skip(name):
            continue
        key = f"{_prefix}{name}"

        leaf = _leaf_for(value)
        if leaf:
            tag, getter, _setter = leaf
            try:
                out[key] = {"t": tag, "v": getter(value)}
            except Exception:
                pass
            continue

        # dynamic group: list/tuple of (label, widget) pairs
        if isinstance(value, (list, tuple)) and value:
            pairs = []
            ok = True
            for item in value:
                if (isinstance(item, (list, tuple)) and len(item) == 2
                        and _leaf_for(item[1])):
                    tag, getter, _s = _leaf_for(item[1])
                    try:
                        pairs.append([str(item[0]), tag, getter(item[1])])
                    except Exception:
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok and pairs:
                out[key] = {"t": "dyn", "v": pairs}
            continue

        if _is_own_widget(value):
            collect_settings(window, value, key + ".", out, seen, _depth + 1)

    return out


def _resolve(window, key: str):
    """Walk a dotted key to its widget. Returns (widget, None) or (None, why)."""
    obj = window
    parts = key.split(".")
    for i, part in enumerate(parts):
        if not hasattr(obj, part):
            return None, f"no '{part}' in {'.'.join(parts[:i]) or 'window'}"
        obj = getattr(obj, part)
    return obj, None


def apply_settings(window, settings: dict, report: LoadReport,
                   dynamic: bool = False, block_signals: bool = True) -> None:
    """Push saved settings back onto the window.

    Call once with dynamic=False right after the window is built, then again
    with dynamic=True after the results have been restored — the dynamic
    groups (per-concentration colours, per-host colours …) only exist once
    there is data to build them from.

    Signals are blocked while writing so that setting forty controls does not
    trigger forty plot redraws; the caller refreshes once at the end.
    """
    if not isinstance(settings, dict):
        return

    for key, blob in sorted(settings.items()):
        if not isinstance(blob, dict) or "t" not in blob:
            report.unknown_settings.append(f"{key} (unreadable entry)")
            continue
        tag, value = blob["t"], blob.get("v")

        if (tag == "dyn") != dynamic:
            continue

        widget, why = _resolve(window, key)
        if widget is None:
            if not dynamic:
                report.unknown_settings.append(f"{key} — {why}")
            continue

        try:
            if tag == "dyn":
                _apply_dynamic(widget, value, key, report, block_signals)
                continue

            live = _leaf_for(widget)
            if live is None:
                report.type_changed.append(f"{key} — control is no longer settable")
                continue
            live_tag, _getter, setter = live
            if live_tag != tag:
                report.type_changed.append(
                    f"{key} — was saved as {tag}, is now {live_tag}")
                continue

            blocked = block_signals and hasattr(widget, "blockSignals")
            if blocked:
                widget.blockSignals(True)
            try:
                setter(widget, value, key, report)
            finally:
                if blocked:
                    widget.blockSignals(False)
        except Exception as exc:
            report.errors.append(f"{key}: {type(exc).__name__}: {exc}")


def _apply_dynamic(group, pairs, key, report, block_signals):
    """Restore a list of (label, widget) pairs by matching on the label.
    Labels present then but not now are ignored; labels present now but not
    then keep their freshly-built default."""
    if not isinstance(group, (list, tuple)) or not isinstance(pairs, list):
        return
    saved = {}
    for entry in pairs:
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            saved[str(entry[0])] = (entry[1], entry[2])

    matched = 0
    for item in group:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        label, widget = str(item[0]), item[1]
        if label not in saved:
            continue
        tag, value = saved[label]
        live = _leaf_for(widget)
        if live is None or live[0] != tag:
            continue
        blocked = block_signals and hasattr(widget, "blockSignals")
        if blocked:
            widget.blockSignals(True)
        try:
            live[2](widget, value, f"{key}[{label}]", report)
            matched += 1
        finally:
            if blocked:
                widget.blockSignals(False)

    if saved and matched < len(saved):
        report.notes.append(
            f"{key}: {len(saved) - matched} of {len(saved)} saved entries no "
            f"longer apply to this data")


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline state
# ══════════════════════════════════════════════════════════════════════════════
#
# The PipelineState dataclasses differ per mode and will keep changing, so
# nothing here names their fields.  Every public field is walked generically;
# fields that vanish are ignored on load and fields that appear keep their
# dataclass default.
#
# Matplotlib figures are NOT stored.  They are large, they pickle badly across
# matplotlib versions, and they are cheap to regenerate from fi_df — which IS
# stored.  The window regenerates them on load.

_FIGURE_FIELDS = ("qc_figures", "spectra_figures", "overlay_figures")


def dump_pipeline_state(state) -> dict:
    """Field-by-field snapshot of a PipelineState / PipelineStateKi /
    PipelineStateSpectral, skipping matplotlib figures."""
    if state is None:
        return {}
    out = {}
    for name, value in vars(state).items():
        if name.startswith("_") or name in _FIGURE_FIELDS:
            continue
        out[name] = value
    return out


def restore_pipeline_state(state, blob: dict, report: LoadReport):
    """Write decoded fields back onto a freshly constructed *state* object.
    Unknown field names are reported, not applied."""
    if not isinstance(blob, dict):
        return state
    known = set(vars(state).keys())
    for name, value in blob.items():
        if name in _FIGURE_FIELDS:
            continue
        if name not in known:
            report.unknown_settings.append(f"results.{name} (no longer used)")
            continue
        try:
            setattr(state, name, value)
        except Exception as exc:
            report.errors.append(f"results.{name}: {type(exc).__name__}: {exc}")
    for name in sorted(known - set(blob) - set(_FIGURE_FIELDS)):
        report.missing_state.append(f"results.{name}")
    return state


def rebuild_ki_factories(plot_data, pl_ki, report: LoadReport) -> None:
    """The competitive plot entries carry a fitted model as a live closure
    (``entry["factory"]``).  A closure cannot be written to a portable file,
    so it is dropped on save and rebuilt here from the model name and the
    concentrations, which are stored.  Rebuilt in place."""
    if not plot_data:
        return
    builders = {
        "Standard":  lambda e: pl_ki._comp_standard(e.get("DyeConc_uM"),
                                                    e.get("DyeKd_uM")),
        "HillSlope": lambda e: pl_ki._comp_hill(e.get("DyeConc_uM"),
                                                e.get("DyeKd_uM")),
        "Wang":      lambda e: pl_ki._wang_cubic(e.get("DyeConc_uM"),
                                                 e.get("DyeKd_uM"),
                                                 e.get("host_conc")),
    }
    unknown = set()
    for entry in plot_data:
        if not isinstance(entry, dict) or callable(entry.get("factory")):
            continue
        name = str(entry.get("model_name") or "")
        build = builders.get(name)
        if build is None:
            unknown.add(name or "<unnamed>")
            continue
        try:
            entry["factory"] = build(entry)
        except Exception as exc:
            report.errors.append(
                f"rebuilding {name} model: {type(exc).__name__}: {exc}")
    if unknown:
        report.errors.append(
            "curves using unknown model(s) " + ", ".join(sorted(unknown)) +
            " could not be redrawn — re-run the fit stage to restore them")


def default_session_name(mode: str, input_folder: str = "") -> str:
    """Suggested filename for the save dialog."""
    stem = os.path.basename(os.path.normpath(input_folder)) if input_folder else ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = [p for p in (stem, mode, stamp) if p]
    return "_".join(parts) + SESSION_EXT

