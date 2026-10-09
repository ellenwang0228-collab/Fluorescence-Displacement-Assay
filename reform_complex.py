#!/usr/bin/env python3

import sys
import os
import subprocess
import argparse
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build complex.xyz directly from host + docked guest PDBQTs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--docked-pdbqt", required=True,
                   help="Vina docked guest PDBQT (top MODEL 1 will be used)")
    p.add_argument("--host-pdbqt", required=True,
                   help="Host receptor PDBQT (same coordinate frame as docking; "
                        "typically pdbqt/<host>.pdbqt, written with -xr)")
    p.add_argument("--curated-host-mol2", default=None,
                   help="OPTIONAL: experimentally curated host mol2 with the correct "
                        "protonation state (curated/<host>.mol2). If provided, this "
                        "file is used AS-IS for the host portion of the complex — "
                        "obabel will NOT add any further H to the host. The docked "
                        "guest still gets full -h treatment (Vina strips guest H). "
                        "This means you only need to curate the protonation ONCE per "
                        "host, and all complexes for that host will use it "
                        "automatically. If not provided, the original pipeline runs: "
                        "join both PDBQTs and add all H to the whole complex.")
    p.add_argument("--host-ref", default=None,
                   help="OPTIONAL: host reference mol2/xyz (hosts/<host>.*) -- "
                        "used only for the expected-H-count QC check")
    p.add_argument("--guest-ref", default=None,
                   help="OPTIONAL: guest reference mol2/xyz (guests/<guest>.*) -- "
                        "used only for the expected-H-count QC check")
    p.add_argument("--out-dir", required=True,
                   help="Output directory")
    p.add_argument("--host-name", required=True,
                   help="Host name (for logging)")
    p.add_argument("--guest-name", required=True,
                   help="Guest name (for logging)")
    p.add_argument("--clash-warn", type=float, default=2.0,
                   help="Å threshold below which host-guest heavy-atom pairs are "
                        "flagged as clashes (default 2.0)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  PDBQT HELPERS  (plain text -- no obabel round-trip, preserves everything)
# ─────────────────────────────────────────────────────────────────────────────

def extract_docking_score(pdbqt_path: str):
    """Pull the top-pose binding affinity from a Vina docked PDBQT's REMARK lines."""
    with open(pdbqt_path, errors="ignore") as fh:
        for line in fh:
            if "VINA RESULT" in line:
                try:
                    return float(line.split()[3])
                except (IndexError, ValueError):
                    pass
            elif "minimizedAffinity" in line:
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError):
                    pass
            if line.startswith("ENDMDL"):
                break
    return None


def extract_model1(pdbqt_path: str, out_path: str):
    """
    Extract MODEL 1 from a (possibly multi-model) PDBQT, writing every line
    verbatim -- atom-type labels, ROOT/BRANCH/TORSDOF, etc. are preserved
    exactly as Vina wrote them.

    If the file has no MODEL/ENDMDL records at all (e.g. a rigid receptor
    written with -xr), the whole file is treated as "model 1".
    """
    lines = open(pdbqt_path, errors="ignore").readlines()

    out = []
    in_model = True
    seen_model = False

    for line in lines:
        if line.startswith("MODEL"):
            seen_model = True
            parts = line.split()
            in_model = (len(parts) < 2) or (parts[1] == "1")
            continue
        if line.startswith("ENDMDL"):
            if seen_model and in_model:
                break
            in_model = False
            continue
        if in_model:
            out.append(line)

    if not any(l.startswith(("ATOM", "HETATM")) for l in out):
        raise ValueError(f"No ATOM/HETATM records found in MODEL 1 of {pdbqt_path}")

    with open(out_path, "w") as fh:
        fh.writelines(out)


def count_pdbqt_atoms(pdbqt_path: str) -> int:
    """Count ATOM/HETATM records in a (single-model) PDBQT."""
    n = 0
    with open(pdbqt_path, errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


def count_pdbqt_H(pdbqt_path: str) -> int:
    """Count hydrogen ATOM/HETATM records in a PDBQT file.
    Matches atom names starting with H (columns 13-16, e.g. HD, HN, HO, H1)."""
    n = 0
    with open(pdbqt_path, errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name.upper().startswith("H"):
                    n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
#  REFERENCE STRUCTURE ELEMENT COUNTING  (for the optional H-count QC check)
# ─────────────────────────────────────────────────────────────────────────────

def _format_element(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "?"
    return raw.upper() if len(raw) == 1 else raw[0].upper() + raw[1:].lower()


def count_elements_mol2(path: str) -> Counter:
    counts = Counter()
    in_atom = False
    for line in open(path, errors="ignore"):
        s = line.strip()
        if s.startswith("@<TRIPOS>"):
            in_atom = (s == "@<TRIPOS>ATOM")
            continue
        if in_atom:
            parts = s.split()
            if len(parts) >= 6:
                counts[_format_element(parts[5].split(".")[0])] += 1
    return counts


def count_elements_xyz(path: str) -> Counter:
    counts = Counter()
    lines = open(path, errors="ignore").readlines()
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        n = max(len(lines) - 2, 0)
    for line in lines[2:2 + n]:
        parts = line.split()
        if parts:
            counts[_format_element(parts[0])] += 1
    return counts


def count_elements_ref(path: str) -> Counter:
    ext = Path(path).suffix.lower()
    if ext == ".mol2":
        return count_elements_mol2(path)
    if ext == ".xyz":
        return count_elements_xyz(path)
    raise ValueError(f"Unsupported reference file type '{ext}' for {path} "
                      "(expected .mol2 or .xyz)")


def format_formula(counts: Counter) -> str:
    def order_key(e):
        if e == "C":
            return (0, e)
        if e == "H":
            return (1, e)
        return (2, e)
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}"
                    for e in sorted(counts, key=order_key))


# ─────────────────────────────────────────────────────────────────────────────
#  MOL2 PARSING  (for the rehydrogenated complex)
# ─────────────────────────────────────────────────────────────────────────────

def parse_mol2_full(path: str):
    """
    Parse @<TRIPOS>ATOM and @<TRIPOS>BOND from complex_full.mol2.

    Returns:
      atoms : list of {element, x, y, z}, in file order
      bonds : list of (atom1_id, atom2_id), 1-indexed as written in the file
    """
    atoms, bonds = [], []
    section = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if s.startswith("@<TRIPOS>"):
            section = s[9:]
            continue
        if section == "ATOM":
            parts = s.split()
            if len(parts) >= 6:
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                atoms.append({"element": _format_element(parts[5].split(".")[0]),
                               "x": x, "y": y, "z": z})
        elif section == "BOND":
            parts = s.split()
            if len(parts) >= 3:
                try:
                    bonds.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    if not atoms:
        raise ValueError(f"No atoms parsed from {path} -- is it a valid mol2 file?")
    return atoms, bonds


# ─────────────────────────────────────────────────────────────────────────────
#  CLASH DETECTION  (pure Python -- complex sizes here are tiny)
# ─────────────────────────────────────────────────────────────────────────────

def detect_clashes(host_heavy: list, guest_heavy: list, threshold: float):
    if not host_heavy or not guest_heavy:
        return float("inf"), 0
    min_dist = float("inf")
    n_clash = 0
    for h in host_heavy:
        for g in guest_heavy:
            d = ((h["x"] - g["x"]) ** 2 +
                 (h["y"] - g["y"]) ** 2 +
                 (h["z"] - g["z"]) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
            if d < threshold:
                n_clash += 1
    return min_dist, n_clash


# ─────────────────────────────────────────────────────────────────────────────
#  OBABEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_obabel(args: list, log_path: Path, step_name: str):
    """
    Run obabel with the given argument list, appending the exact command and
    its full stdout/stderr to log_path. Raises RuntimeError with a helpful
    message (including the tail of stderr) on any failure.
    """
    cmd = ["obabel"] + [str(a) for a in args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise RuntimeError(f"{step_name}: 'obabel' not found in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{step_name}: obabel timed out after 300s.\n"
                            f"  command: {' '.join(cmd)}")

    with open(log_path, "a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        if result.stdout:
            fh.write(result.stdout)
        if result.stderr:
            fh.write(result.stderr)
        fh.write("\n")

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name}: obabel exited with code {result.returncode} "
            f"-- see {log_path}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr (tail): {result.stderr.strip()[-500:]}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "obabel_commands.log"
    if log_path.exists():
        log_path.unlink()

    sep = "─" * 60

    def step(n, msg):
        print(f"\n[reform] Step {n} — {msg}")
        print(f"         {sep}")

    print(f"\n[reform] {'═'*60}")
    print(f"[reform]  {args.host_name}  +  {args.guest_name}")
    print(f"[reform] {'═'*60}")
    print(f"[reform]  Docked guest PDBQT : {args.docked_pdbqt}")
    print(f"[reform]  Host PDBQT         : {args.host_pdbqt}")
    if args.curated_host_mol2:
        print(f"[reform]  Curated host mol2  : {args.curated_host_mol2}  ← HOST H PRESERVED")
    print(f"[reform]  Output dir         : {out_dir}")
    print(f"[reform]  obabel commands log: {log_path}")

    # ── Step 1: top-1 docked pose ─────────────────────────────────────────────
    step(1, "Extracting top-1 docked pose (MODEL 1) from guest PDBQT")
    docking_score = extract_docking_score(args.docked_pdbqt)
    docked_top = out_dir / "docked_top.pdbqt"
    extract_model1(args.docked_pdbqt, str(docked_top))
    n_guest_atoms = count_pdbqt_atoms(str(docked_top))
    print(f"         Docking score (top pose)          : {docking_score} kcal/mol")
    print(f"         Guest atoms in PDBQT (united-atom): {n_guest_atoms}")
    print(f"         -> {docked_top}")

    if args.curated_host_mol2:
        # ── CURATED HOST PATH ─────────────────────────────────────────────────
        # User has provided a mol2 with the experimentally correct protonation.
        # Host H is used AS-IS; only the guest gets -h treatment.
        # This is set-once-per-host: every guest complex for this host will use
        # the same curated mol2, so no per-complex editing is needed.

        curated = Path(args.curated_host_mol2)
        if not curated.exists():
            raise FileNotFoundError(
                f"Curated host mol2 not found: {curated}\n"
                f"  Create it by opening {args.host_pdbqt} (or the original host mol2)\n"
                f"  in PyMOL, removing the unwanted H atoms, and saving as mol2\n"
                f"  to curated/{args.host_name}.mol2 in your project directory."
            )

        step(2, f"Using curated host mol2 (H preserved exactly): {curated.name}")
        host_atoms_raw, _ = parse_mol2_full(str(curated))
        n_host_atoms = len(host_atoms_raw)
        n_host_H = sum(1 for a in host_atoms_raw if a["element"] == "H")
        print(f"         Host atoms (curated mol2): {n_host_atoms}  (H: {n_host_H})")
        print(f"         obabel will NOT add any further H to the host.")

        step(3, "Converting docked guest PDBQT → mol2 + adding H (Vina strips guest H)")
        guest_mol2 = out_dir / "guest_H.mol2"
        run_obabel([str(docked_top), "-O", str(guest_mol2),
                    "-h", "--partialcharge", "gasteiger"],
                   log_path, "Step 3 (guest → mol2 + -h)")
        guest_atoms_raw, _ = parse_mol2_full(str(guest_mol2))
        n_guest_with_H = len(guest_atoms_raw)
        n_guest_H = sum(1 for a in guest_atoms_raw if a["element"] == "H")
        print(f"         Guest atoms (with H): {n_guest_with_H}  (H: {n_guest_H})")
        print(f"         -> {guest_mol2}")

        step(4, "Joining curated host mol2 + hydrogenated guest mol2")
        complex_full = out_dir / "complex_full.mol2"
        # --join with two mol2 inputs: no -h (host H already correct, guest H already added)
        run_obabel([str(curated), str(guest_mol2), "-O", str(complex_full), "--join"],
                   log_path, "Step 4 (join curated host + guest_H.mol2)")
        atoms, bonds = parse_mol2_full(str(complex_full))
        formula = Counter(a["element"] for a in atoms)
        n_total = len(atoms)
        n_h_actual = formula.get("H", 0)
        print(f"         -> {complex_full}")
        print(f"         Total atoms : {n_total}  ({n_host_atoms} host + {n_guest_with_H} guest)")
        print(f"         Formula     : {format_formula(formula)}")
        print(f"         Host H in complex: {n_host_H} (from curated mol2 — NOT re-perceived)")

    else:
        # ── STANDARD PATH (no curated host mol2) ─────────────────────────────
        # Join both PDBQTs, then add all H to the whole complex.
        # The pipeline will pause after reform for you to manually edit
        # complex.xyz if needed.

        step(2, "Joining host + docked guest into one structure (no coordinate transform)")
        if not Path(args.host_pdbqt).exists():
            raise FileNotFoundError(f"Host PDBQT not found: {args.host_pdbqt}")
        n_host_pdbqt_atoms = count_pdbqt_atoms(args.host_pdbqt)
        n_host_atoms = n_host_pdbqt_atoms
        print(f"         Host atoms in PDBQT: {n_host_pdbqt_atoms}")
        complex_raw = out_dir / "complex_raw.pdb"
        run_obabel([args.host_pdbqt, str(docked_top), "-O", str(complex_raw), "--join"],
                   log_path, "Step 2 (join)")
        print(f"         -> {complex_raw}")
        print(f"         (host and guest already co-registered from Vina —")
        print(f"          pure concatenation, no coordinate transform)")

        step(3, "Perceiving bonds + adding ALL hydrogens (obabel -h, Gasteiger charges)")
        print(f"         TIP: provide --curated-host-mol2 curated/<host>.mol2 to fix")
        print(f"         host protonation once for all guests, instead of editing")
        print(f"         each complex.xyz individually after this step.")
        complex_full = out_dir / "complex_full.mol2"
        run_obabel([str(complex_raw), "-O", str(complex_full), "-h",
                    "--partialcharge", "gasteiger"],
                   log_path, "Step 3 (add hydrogens)")
        atoms, bonds = parse_mol2_full(str(complex_full))
        formula = Counter(a["element"] for a in atoms)
        n_total = len(atoms)
        n_h_actual = formula.get("H", 0)
        print(f"         -> {complex_full}")
        print(f"         Total atoms : {n_total}")
        print(f"         Formula     : {format_formula(formula)}")

    # ── Step 5: complex.xyz for CREST ─────────────────────────────────────────
    step(5, "Writing complex.xyz (CREST input)")
    complex_xyz = out_dir / "complex.xyz"
    run_obabel([str(complex_full), "-O", str(complex_xyz)], log_path, "Step 5 (to xyz)")
    print(f"         -> {complex_xyz}")
    if args.curated_host_mol2:
        print(f"         Host protonation reflects your curated mol2 — no further")
        print(f"         editing of complex.xyz should be needed for the host.")
    else:
        print(f"         If you need to remove specific polar H, edit complex.xyz")
        print(f"         (delete H lines + update atom count on line 1).")
        print(f"         The pipeline will pause for you after all reform steps complete.")

    # ── Step 5: QC checks ──────────────────────────────────────────────────────
    step(5, "Quality checks")

    qc_lines = []  # collected for reform_summary.txt

    # 5a. Expected vs actual H-count / heavy-atom-count (if references given)
    if args.host_ref and args.guest_ref:
        try:
            host_ref_counts = count_elements_ref(args.host_ref)
            guest_ref_counts = count_elements_ref(args.guest_ref)
            expected = host_ref_counts + guest_ref_counts
            exp_h = expected.get("H", 0)
            exp_heavy = sum(c for e, c in expected.items() if e != "H")
            act_heavy = n_total - n_h_actual

            print(f"         Expected (host_ref + guest_ref): "
                  f"{format_formula(expected)}  ({exp_heavy} heavy + {exp_h} H)")
            print(f"         Actual   (complex_full.mol2)   : "
                  f"{format_formula(formula)}  ({act_heavy} heavy + {n_h_actual} H)")

            if act_heavy != exp_heavy:
                msg = (f"*** HEAVY-ATOM COUNT MISMATCH: expected {exp_heavy}, "
                       f"got {act_heavy} ***")
                print(f"\n         {msg}")
                print("         This should not happen from -h alone (it only adds H).")
                print("         Check host.pdbqt / docked PDBQT for unexpected atoms,")
                print("         or a cross-fragment bond merging atoms unexpectedly (see 5b).")
                qc_lines.append(msg)
            elif n_h_actual != exp_h:
                msg = (f"NOTE: H-count differs from curated references "
                       f"(expected {exp_h}, got {n_h_actual}).")
                print(f"\n         {msg}")
                print("         obabel's geometry-based valence/protonation assignment can")
                print("         differ from your curated references for groups like amines,")
                print("         carboxylic acids, etc. Worth a visual check on this complex.")
                qc_lines.append(msg)
            else:
                msg = "Formula matches curated references exactly."
                print(f"         \u2713 {msg}")
                qc_lines.append(msg)
        except (ValueError, OSError) as exc:
            msg = f"Could not compare against references: {exc}"
            print(f"         WARNING: {msg}")
            qc_lines.append(f"WARNING: {msg}")
    else:
        msg = "No --host-ref/--guest-ref given -- skipping expected-formula check."
        print(f"         {msg}")
        qc_lines.append(msg)

    # 5b. Cross-fragment bonds (host atom bonded directly to a guest atom)
    boundary = n_host_atoms + n_guest_atoms
    cross_bonds = [
        (a, b) for a, b in bonds
        if a <= boundary and b <= boundary
        and (a <= n_host_atoms) != (b <= n_host_atoms)
    ]
    if cross_bonds:
        msg = (f"*** {len(cross_bonds)} bond(s) perceived ACROSS the host/guest "
               f"boundary: {cross_bonds} ***")
        print(f"\n         {msg}")
        print("         obabel read the host and guest as ONE covalently-bonded")
        print("         molecule at these atom pairs -- usually means a host and")
        print("         guest atom are close enough to look bonded by geometry.")
        print(f"         Inspect {complex_full.name} around these atom indices")
        print("         (1-indexed, host atoms first, then guest atoms) before CREST.")
        qc_lines.append(msg)
    else:
        msg = "No bonds perceived across the host/guest boundary."
        print(f"         \u2713 {msg}")
        qc_lines.append(msg)

    # 5c. Host-guest heavy-atom clash distances
    host_heavy = [a for a in atoms[:n_host_atoms] if a["element"] != "H"]
    guest_heavy = [a for a in atoms[n_host_atoms:boundary] if a["element"] != "H"]
    min_dist, n_clashes = detect_clashes(host_heavy, guest_heavy, args.clash_warn)
    print(f"\n         Min host-guest heavy-atom distance : {min_dist:.3f} \u00c5")
    print(f"         Clash check: {n_clashes} pair(s) within {args.clash_warn:.1f} \u00c5"
          + ("  \u2190 tight contact" if n_clashes > 0 else "  \u2713 OK"))
    qc_lines.append(f"Min host-guest distance (heavy): {min_dist:.3f} \u00c5")
    qc_lines.append(f"Clashes (<{args.clash_warn:.1f} \u00c5): {n_clashes}")

    # ── Step 6: summary ────────────────────────────────────────────────────────
    step(6, "Writing reform_summary.txt")
    summary_path = out_dir / "reform_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(f"Reform summary: {args.host_name}  +  {args.guest_name}\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Docking score (top-1 pose)  : {docking_score} kcal/mol\n")
        fh.write(f"Host atoms (heavy+orig. H)  : {n_host_atoms}\n")
        fh.write(f"Guest atoms (PDBQT)         : {n_guest_atoms}\n")
        fh.write(f"Complex atoms (after -h)    : {n_total}\n")
        fh.write(f"Complex H atoms             : {n_h_actual}\n")
        fh.write(f"Complex formula             : {format_formula(formula)}\n")
        fh.write(f"\nNOTE: Polar H atoms have been added to the full complex by obabel.\n")
        fh.write(f"If any do not match the experimental ionisation state, edit\n")
        fh.write(f"complex.xyz (update atom count on line 1 after deleting H lines)\n")
        fh.write(f"before CREST runs. The pipeline pauses for this after reform.\n")
        fh.write("\nQC:\n")
        for line in qc_lines:
            fh.write(f"  - {line}\n")
        # Machine-readable boundary markers for downstream tools (e.g.
        # classify_binding.py). These mark, in atom-index terms valid for
        # complex.xyz / complex_full.mol2 / any CREST output derived from
        # them (atom order is preserved through obabel format conversion,
        # xtb, and CREST -- only NEW H atoms get appended at the very end):
        #   atoms[0 : n_host_atoms]              = host heavy + any original H
        #   atoms[n_host_atoms : n_host_atoms+n_guest_atoms] = guest heavy (united-atom)
        # All newly-added H (from the -h step) are appended after this range.
        fh.write("\nMACHINE-READABLE (for downstream scripts):\n")
        fh.write(f"  n_host_atoms={n_host_atoms}\n")
        fh.write(f"  n_guest_atoms={n_guest_atoms}\n")
        fh.write("\nOutput files:\n")
        fh.write(f"  complex.xyz (CREST input)  : {complex_xyz}\n")
        fh.write(f"  complex_full.mol2 (for QC) : {complex_full}\n")
        fh.write(f"  obabel commands log        : {log_path}\n")
        fh.write(f"  this summary               : {summary_path}\n")

    print(f"         Summary -> {summary_path}")
    print(f"\n[reform] Done: {args.host_name}  +  {args.guest_name}")
    print(f"[reform] {'═'*60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[reform] FATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
