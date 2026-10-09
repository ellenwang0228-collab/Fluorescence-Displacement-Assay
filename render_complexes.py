#!/usr/bin/env python3
"""
render_complexes.py — PyMOL image renderer for host-guest complexes.

THREE WAYS TO USE IT
--------------------

1. Run it IN the complex directory (simplest — just cd there first):

       cd /path/to/reformed/MyHost/MyGuest
       python3 /path/to/render_complexes.py
       # → writes MyHost_MyGuest.png in the current directory

2. Point it at one complex directory:

       python3 render_complexes.py --dir reformed/MyHost/MyGuest

3. Batch — scan a whole reformed/ tree and render everything:

       python3 render_complexes.py --batch reformed/
       # → writes renders/<host>_<guest>.png next to the reformed/ folder

PYMOL PATH
----------
The script tries common locations automatically. If yours is elsewhere:

       python3 render_complexes.py --pymol /Applications/PyMOL.app/Contents/bin/pymol

IMAGE SETTINGS (your spec)
--------------------------
  Sticks  |  stick_radius 0.25
  C grey60  H white  N blue  O vivid-red  metals as spheres (scale 0.6)
  ray 2000×2000  |  dpi 500  |  ray_trace_mode 1  |  bg white
  light_count 8  |  ambient 0  |  direct 0.1  |  reflect 1.5
  shadow decay_factor 2  |  decay_range 0.2
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  PyMOL scripts
# ─────────────────────────────────────────────────────────────────────────────

# Shared colour/lighting block — used by every template below
_BODY = """\
hide everything
remove (elem H)
show sticks, all
set stick_radius, 0.25

# Show all bonds as uniform single sticks — suppresses double/aromatic bond
# display from mol2 bond-order data, and makes PDB and mol2 look identical
set valence, 0
set half_bonds, 0
bg white

# Element colouring
color grey60, symbol C
color white, symbol H
color blue, symbol N
set_color oxygen, [1.0, 0.1, 0.1]
color oxygen, symbol O
set_color brightorange, [1.0, 0.7, 0.2]

# Guest carbons: warm lightorange so the pharmacophore reads against grey host
color lightorange, (guest and elem C)

# Metal centres: spheres, scale 0.6
select metals, elem Fe+Zn+Cu+Mn+Mg+Ca
show spheres, metals
set sphere_scale, 0.6, metals
color orange, elem Zn
color orange, elem Fe
color darksalmon, elem Cu
color lightblue, elem Mg
color lightblue, elem Ca
color violet, elem Mn

# Lighting (your settings)
set light_count, 8
set spec_count, 2
set shininess, 10
set specular, 0.25
set ambient, 0
set direct, 0.1
set reflect, 1.5
set depth_cue, 1

# Ray trace
set ray_trace_mode, 1
set ray_shadow_decay_factor, 2
set ray_shadow_decay_range, 0.2

zoom guest, 8
ray 2000, 2000
png {output_png}, dpi=500
quit
"""

# Two objects: host PDBQT + guest mol2 — best option (separate colours)
TMPL_TWO = """\
load {host_pdbqt}, host
load {guest_mol}, guest
""" + _BODY

# One object fallback (no host PDBQT found nearby)
# Creates a dummy empty 'guest' selection so (guest and elem C) doesn't error
TMPL_ONE = """\
load {mol2}, host
select guest, none
""" + _BODY.replace("zoom guest, 8", "zoom host, 8")

# CREST best conformer (xyz — PyMOL infers bonds from geometry)
TMPL_CREST = """\
load {xyz}, host
select guest, none
""" + _BODY.replace("zoom guest, 8", "zoom host, 8")


# ─────────────────────────────────────────────────────────────────────────────
#  PyMOL binary discovery
# ─────────────────────────────────────────────────────────────────────────────

_PYMOL_CANDIDATES = [
    "pymol",
    "/Applications/PyMOL.app/Contents/bin/pymol",            # macOS official
    "/usr/local/bin/pymol",                                   # brew / manual
    "/opt/homebrew/bin/pymol",                                # brew Apple-silicon
    os.path.expanduser("~/opt/miniconda3/envs/pymol/bin/pymol"),
    os.path.expanduser("~/miniconda3/envs/pymol/bin/pymol"),
    os.path.expanduser("~/anaconda3/envs/pymol/bin/pymol"),
]


def find_pymol(override: str = "") -> str:
    if override:
        if not (Path(override).is_file() or
                subprocess.run(["which", override], capture_output=True).returncode == 0):
            sys.exit(f"ERROR: PyMOL binary not found at '{override}'")
        return override
    for c in _PYMOL_CANDIDATES:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0 or b"PyMOL" in r.stdout + r.stderr:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    sys.exit(
        "ERROR: PyMOL not found. Either:\n"
        "  • Pass --pymol /path/to/pymol\n"
        "  • macOS: /Applications/PyMOL.app/Contents/bin/pymol\n"
        "  • conda: conda activate pymol  then re-run\n"
        "  • cluster: ml PyMOL  then re-run"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Detect complex files in a directory
# ─────────────────────────────────────────────────────────────────────────────

def detect_complex(directory: Path):
    """
    Given a directory, return a dict of available complex files found in it.
    Returns None if no recognisable complex files are present.
    Accepts both mol2 and pdb — the render logic picks the best available.
    """
    d = directory.resolve()
    files = {}
    for name in ("complex_full.mol2", "complex_full.pdb",
                 "complex_raw.pdb", "complex.pdb",
                 "docked_top.pdbqt", "guest_H.mol2",
                 "crest_best.xyz", "complex.xyz"):
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            files[name] = p
    has_complex = any(k in files for k in (
        "complex_full.mol2", "complex_full.pdb",
        "complex_raw.pdb", "complex.pdb",
        "crest_best.xyz", "complex.xyz",
    ))
    return files if has_complex else None


def find_host_pdbqt_near(reform_dir: Path, host_name: str):
    """
    Walk up the directory tree from a reform dir to find pdbqt/<host>.pdbqt.
    Handles both:
      cluster layout: reformed/<host>/<guest>/  →  ../../pdbqt/<host>.pdbqt
      local layout:   wherever the user put it
    """
    # Try up to 4 levels up
    d = reform_dir.resolve()
    for _ in range(4):
        d = d.parent
        candidate = d / "pdbqt" / f"{host_name}.pdbqt"
        if candidate.is_file():
            return candidate
    return None


def infer_names(reform_dir: Path):
    """
    Try to infer host and guest names from directory structure.
    reformed/<host>/<guest>/  →  host_name, guest_name
    Otherwise use directory name as guest_name, parent as host_name.
    """
    d = reform_dir.resolve()
    guest_name = d.name
    host_name  = d.parent.name
    # If parent looks like 'reformed' or similar, collapse to single name
    if host_name.lower() in ("reformed", "complexes", "results", ""):
        host_name = guest_name
        guest_name = ""
    return host_name, guest_name


# ─────────────────────────────────────────────────────────────────────────────
#  Render one complex directory
# ─────────────────────────────────────────────────────────────────────────────

def render_dir(reform_dir: Path, output_png: Path, source: str,
               pymol_bin: str, force: bool, verbose: bool) -> bool:
    if output_png.exists() and not force:
        print(f"  [SKIP]  {output_png.name}  (already exists — pass --force to re-render)")
        return True

    files = detect_complex(reform_dir)
    if not files:
        print(f"  [SKIP]  {reform_dir}  (no complex_full.mol2 or crest_best.xyz found)")
        return False

    host_name, guest_name = infer_names(reform_dir)

    # ── Choose template and inputs ────────────────────────────────────────────
    if source == "crest_best":
        xyz = files.get("crest_best.xyz") or files.get("complex.xyz")
        if not xyz:
            print(f"  [WARN]  No crest_best.xyz found in {reform_dir} — skipping")
            return False
        pml = TMPL_CREST.format(xyz=xyz, output_png=output_png)

    else:
        # Best available complex file: mol2 preferred (has bond info),
        # pdb variants as fallback. set valence,0 + set half_bonds,0 in the
        # PyMOL script normalises display to single sticks regardless.
        complex_file = (
            files.get("complex_full.mol2") or
            files.get("complex_full.pdb")  or
            files.get("complex_raw.pdb")   or
            files.get("complex.pdb")
        )
        if not complex_file:
            print(f"  [SKIP]  {reform_dir} — no complex mol2/pdb found")
            return False

        host_pdbqt = find_host_pdbqt_near(reform_dir, host_name)
        guest_mol  = (files.get("guest_H.mol2") or
                      files.get("docked_top.pdbqt") or
                      complex_file)

        if host_pdbqt:
            pml = TMPL_TWO.format(
                host_pdbqt=host_pdbqt,
                guest_mol=guest_mol,
                output_png=output_png,
            )
        else:
            print(f"  [INFO]  No host PDBQT found near {reform_dir} — "
                  f"rendering as single object (all C will be grey60)")
            pml = TMPL_ONE.format(mol2=complex_file, output_png=output_png)

    # ── Write temp .pml and invoke PyMOL ─────────────────────────────────────
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pml", delete=False, dir=str(output_png.parent)) as tf:
        tf.write(pml)
        pml_path = tf.name

    try:
        result = subprocess.run(
            [pymol_bin, "-cq", pml_path],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        os.unlink(pml_path)
        sys.exit(f"ERROR: PyMOL binary not found: '{pymol_bin}'")
    except subprocess.TimeoutExpired:
        print(f"  [FAIL]  Timed out (>600 s) for {reform_dir.name}")
        os.unlink(pml_path)
        return False
    finally:
        try:
            os.unlink(pml_path)
        except OSError:
            pass

    if not output_png.exists():
        print(f"  [FAIL]  {reform_dir.name}")
        if verbose:
            for label, text in [("stdout", result.stdout), ("stderr", result.stderr)]:
                if text.strip():
                    print(f"    {label}: {text.strip()[-600:]}")
        return False

    kb = output_png.stat().st_size // 1024
    print(f"  [OK]    {output_png.name}  ({kb} KB)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def scan_for_complexes(root: Path):
    """
    Recursively find all directories under root that contain a complex file.
    Returns list of (host_name, guest_name, path) tuples.
    """
    seen = set()
    targets = []
    for pattern in ("complex_full.mol2", "complex_full.pdb",
                    "complex_raw.pdb", "complex.pdb"):
        for d in sorted(root.rglob(pattern)):
            guest_dir = d.parent
            if guest_dir in seen:
                continue
            seen.add(guest_dir)
            host_name, guest_name = infer_names(guest_dir)
            targets.append((host_name, guest_name, guest_dir))
    return sorted(targets)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Render host-guest complex images with PyMOL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  cd reformed/  &&  python3 render_complexes.py\n"
            "      → scans CWD for all complex subdirectories, renders all\n\n"
            "  python3 render_complexes.py --dir reformed/\n"
            "      → same, explicit path\n\n"
            "  python3 render_complexes.py --source crest_best\n"
            "      → render lowest-energy CREST conformers instead"
        ),
    )
    ap.add_argument(
        "--dir", metavar="DIR", default=None,
        help="Root directory to scan (default: CWD). Can be the reformed/ folder "
             "(scans all complex subdirectories) or a single complex directory "
             "(contains complex_full.mol2 directly).",
    )
    ap.add_argument(
        "--output-dir", metavar="OUTPUT_DIR", default=None,
        help="Where to write PNGs (default: renders/ alongside the scanned directory). "
             "In single-complex mode the PNG is written into the complex directory itself.",
    )
    ap.add_argument(
        "--source", choices=["docked", "crest_best"], default="docked",
        help="docked=complex_full.mol2 pre-CREST pose (default); "
             "crest_best=lowest-energy CREST conformer xyz",
    )
    ap.add_argument(
        "--pymol", default="",
        help="Path to PyMOL binary (auto-detected from common locations if not given)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-render even if PNG already exists",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print PyMOL stdout/stderr on failure",
    )
    args = ap.parse_args()

    pymol_bin = find_pymol(args.pymol)
    root = Path(args.dir).resolve() if args.dir else Path.cwd()

    print(f"render_complexes.py")
    print(f"  Scanning : {root}")
    print(f"  PyMOL    : {pymol_bin}")
    print(f"  Source   : {args.source}")

    # ── Auto-detect: are we sitting inside a single complex dir or a root? ────
    # A "single complex dir" contains complex_full.mol2 directly.
    # A "root" contains subdirectories that in turn contain complex_full.mol2.
    _single_markers = ("complex_full.mol2", "complex_full.pdb",
                        "complex_raw.pdb", "complex.pdb")
    if any((root / m).exists() for m in _single_markers):
        # Single complex directory mode
        host_name, guest_name = infer_names(root)
        stem = f"{host_name}_{guest_name}" if guest_name else host_name
        if args.output_dir:
            output_png = Path(args.output_dir).resolve() / f"{stem}.png"
        else:
            output_png = root / f"{stem}.png"

        print(f"  Mode     : single complex ({host_name} + {guest_name})")
        print(f"  Output   : {output_png}")
        print()
        ok = render_dir(root, output_png, args.source,
                        pymol_bin, args.force, args.verbose)
        sys.exit(0 if ok else 1)

    # ── Scan root for all complex subdirectories ──────────────────────────────
    targets = scan_for_complexes(root)

    if not targets:
        sys.exit(
            f"No complex_full.mol2 files found under {root}\n\n"
            f"Make sure you are either:\n"
            f"  • Inside the reformed/ directory (cd reformed/ then run this script), or\n"
            f"  • Passing --dir /path/to/reformed"
        )

    out_root = Path(args.output_dir).resolve() if args.output_dir \
               else root.parent / "renders"

    print(f"  Mode     : batch ({len(targets)} complex(es))")
    print(f"  Output   : {out_root}")
    print()

    ok = fail = 0
    for host_name, guest_name, reform_dir in targets:
        stem = f"{host_name}_{guest_name}" if guest_name else host_name
        png  = out_root / f"{stem}.png"
        print(f"  {host_name} + {guest_name}")
        if render_dir(reform_dir, png, args.source,
                      pymol_bin, args.force, args.verbose):
            ok += 1
        else:
            fail += 1

    print(f"\nDone.  OK={ok}  Failed={fail}")
    if fail:
        print("Re-run with --verbose to see PyMOL output for failed renders.")
        sys.exit(1)


if __name__ == "__main__":
    main()
