#!/usr/bin/env python3
"""Assemble PhosphoMAX.app.

The bundle is a thin wrapper: it holds an icon, a plist and a shell stub, and
points at fda_launcher.py where it already lives. Nothing is copied in, so
editing the analysis code takes effect on the next launch with no rebuild.

    python3 build_app.py <FDA_launcher folder>
"""

import os
import plistlib
import shutil
import stat
import sys

import make_icon

BUNDLE_ID = "uk.ac.crick.phosphomax"
VERSION = "1.0"

# The stub is deliberately POSIX sh: it must run before we know anything about
# the machine. It resolves the bundle's own location, then hands over to the
# Python launcher, which does the real work of finding an interpreter.
STUB = r"""#!/bin/sh
# PhosphoMAX.app launcher stub — see launch_phosphomax.py for the logic.
set -u

# Contents/MacOS/PhosphoMAX -> the folder holding the .app
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=$(cd "$HERE/../.." && pwd)
ROOT=$(dirname "$BUNDLE")

export PHOSPHOMAX_TARGET="$ROOT/fda_launcher.py"
LAUNCHER="$ROOT/launch_phosphomax.py"

if [ ! -f "$LAUNCHER" ]; then
    osascript -e 'display dialog "PhosphoMAX cannot find launch_phosphomax.py.

The app must stay in the same folder as the analysis code. If you want it \
somewhere else, make an alias instead of moving it." with title "PhosphoMAX" \
buttons {"OK"} default button "OK" with icon stop' >/dev/null 2>&1
    exit 1
fi

# /usr/bin/python3 exists on every modern macOS. It only has to be good
# enough to run the launcher, which then finds the Python that has PySide6.
for PY in /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
    if [ -x "$PY" ]; then
        exec "$PY" "$LAUNCHER"
    fi
done

osascript -e 'display dialog "PhosphoMAX could not find any Python 3 to \
start with. Install the Xcode command line tools with:

    xcode-select --install" with title "PhosphoMAX" buttons {"OK"} \
default button "OK" with icon stop' >/dev/null 2>&1
exit 1
"""


def build(root):
    root = os.path.abspath(root)
    app = os.path.join(root, "PhosphoMAX.app")
    contents = os.path.join(app, "Contents")
    macos = os.path.join(contents, "MacOS")
    resources = os.path.join(contents, "Resources")

    if os.path.isdir(app):
        shutil.rmtree(app)
    os.makedirs(macos)
    os.makedirs(resources)

    # ── icon ─────────────────────────────────────────────────────────────
    icns = os.path.join(resources, "PhosphoMAX.icns")
    make_icon.build_icns(icns)

    # ── Info.plist ───────────────────────────────────────────────────────
    info = {
        "CFBundleName":                 "PhosphoMAX",
        "CFBundleDisplayName":          "PhosphoMAX",
        "CFBundleExecutable":           "PhosphoMAX",
        "CFBundleIdentifier":           BUNDLE_ID,
        "CFBundleIconFile":             "PhosphoMAX",
        "CFBundleVersion":              VERSION,
        "CFBundleShortVersionString":   VERSION,
        "CFBundlePackageType":          "APPL",
        "CFBundleSignature":            "????",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion":       "10.15",
        # Retina: without this the window renders at 1x and looks soft.
        "NSHighResolutionCapable":      True,
        # It is a real windowed app, so it belongs in the Dock and can be
        # brought to the front — not a background agent.
        "LSBackgroundOnly":             False,
        "LSUIElement":                  False,
        # Reading data from Dropbox/Desktop/Documents triggers these prompts
        # on recent macOS; the strings explain why it is asking.
        "NSDesktopFolderUsageDescription":
            "PhosphoMAX needs access to read your plate reader data files.",
        "NSDocumentsFolderUsageDescription":
            "PhosphoMAX needs access to read your plate reader data files.",
        "NSDownloadsFolderUsageDescription":
            "PhosphoMAX needs access to read your plate reader data files.",
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName":        "PhosphoMAX Session",
            "CFBundleTypeExtensions":  ["phosmax"],
            "CFBundleTypeRole":        "Editor",
            "LSHandlerRank":           "Owner",
            "CFBundleTypeIconFile":    "PhosphoMAX",
        }],
    }
    with open(os.path.join(contents, "Info.plist"), "wb") as fh:
        plistlib.dump(info, fh)

    # ── executable stub ──────────────────────────────────────────────────
    exe = os.path.join(macos, "PhosphoMAX")
    with open(exe, "w", encoding="utf-8") as fh:
        fh.write(STUB)
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP |
             stat.S_IXOTH)

    # PkgInfo is legacy but harmless, and some tools still look for it.
    with open(os.path.join(contents, "PkgInfo"), "w") as fh:
        fh.write("APPL????")

    return app


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    path = build(target)
    print(f"built: {path}")
    for dirpath, _dirs, files in os.walk(path):
        rel = os.path.relpath(dirpath, os.path.dirname(path))
        for f in sorted(files):
            full = os.path.join(dirpath, f)
            mode = "x" if os.access(full, os.X_OK) else "-"
            print(f"  [{mode}] {os.path.join(rel, f)}  "
                  f"({os.path.getsize(full):,} bytes)")
