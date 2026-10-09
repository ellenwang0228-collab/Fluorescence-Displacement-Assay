#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: numpy is required.\n"
        "  This is already a pipeline dependency -- activate the same conda\n"
        "  env used for pipeline.sh (e.g. `conda activate crest`) and retry."
    )

try:
    from scipy.spatial import ConvexHull, Delaunay
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


HARTREE_TO_KCAL = 627.5095

MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "Na": 22.990, "K": 39.098, "Mg": 24.305, "Ca": 40.078,
    "Fe": 55.845, "Zn": 65.38, "Cu": 63.546, "Mn": 54.938,
    "Ni": 58.693, "Pd": 106.42, "Pt": 195.084, "Ru": 101.07,
    "Co": 58.933, "Rh": 102.906, "Ir": 192.217, "Ag": 107.868, "Au": 196.967,
    "B": 10.811, "Si": 28.085, "Se": 78.971, "As": 74.922,
}

def parse_mol2_atoms(path: Path):
    """Return list of (element, x, y, z) for every atom in a mol2 file,
    in file order. Element is derived from the mol2 atom-type column
    (e.g. 'C.3' -> 'C', 'N.ar' -> 'N')."""
    atoms = []
    in_atom_block = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) < 6:
                    continue
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                sybyl_type = parts[5]
                elem = sybyl_type.split(".")[0]
                # Normalise common two-letter elements mol2 sometimes writes oddly
                elem = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()
                atoms.append((elem, x, y, z))
    return atoms


def parse_xyz_frames(path: Path):
    """
    Parse a (possibly multi-frame) xyz file. Returns a list of frames, each
    a tuple (comment_line, [(element, x, y, z), ...]).
    Handles CREST's crest_conformers.xyz (many frames) and crest_best.xyz /
    complex.xyz (single frame) identically.
    """
    frames = []
    with open(path, errors="ignore") as fh:
        lines = fh.readlines()

    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        comment = lines[i + 1].strip() if i + 1 < n_lines else ""
        atoms = []
        for j in range(n_atoms):
            idx = i + 2 + j
            if idx >= n_lines:
                break
            parts = lines[idx].split()
            if len(parts) < 4:
                continue
            elem = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((elem, x, y, z))
        if len(atoms) == n_atoms:
            frames.append((comment, atoms))
        i += 2 + n_atoms
    return frames


def parse_reform_summary(path: Path):
    """Extract n_host_atoms, n_guest_atoms, and the Vina docking score from
    a reform_summary.txt file."""
    text = path.read_text(errors="ignore")
    n_host = n_guest = None
    score = None

    m = re.search(r"n_host_atoms=(\d+)", text)
    if m:
        n_host = int(m.group(1))
    m = re.search(r"n_guest_atoms=(\d+)", text)
    if m:
        n_guest = int(m.group(1))
    m = re.search(r"Docking score \(top-1 pose\)\s*:\s*(-?\d+\.?\d*)", text)
    if m:
        score = float(m.group(1))

    return n_host, n_guest, score


def mass_weighted_com(atoms):
    """atoms: list of (element, x, y, z). Returns (cx, cy, cz)."""
    wx = wy = wz = wsum = 0.0
    for elem, x, y, z in atoms:
        m = MASS.get(elem, 12.0)  # unknown elements default to carbon-ish mass
        wx += m * x; wy += m * y; wz += m * z; wsum += m
    if wsum == 0:
        return (0.0, 0.0, 0.0)
    return (wx / wsum, wy / wsum, wz / wsum)


def radius_of_gyration(atoms, com):
    """Mass-weighted Rg about the given COM."""
    wsum = wr2 = 0.0
    cx, cy, cz = com
    for elem, x, y, z in atoms:
        m = MASS.get(elem, 12.0)
        r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        wr2 += m * r2; wsum += m
    if wsum == 0:
        return 1.0  # avoid div-by-zero downstream
    return (wr2 / wsum) ** 0.5


def containment_score(host_atoms, guest_atoms, contact_cutoff: float):
    """
    Compute the composite 0-100 containment score plus its raw components.
    host_atoms / guest_atoms: list of (element, x, y, z), heavy atoms only.
    Returns dict with hull_fraction_pct, contacts_per_atom, com_ratio, score.
    Returns None if either atom list is empty.
    """
    if not host_atoms or not guest_atoms:
        return None

    host_xyz  = np.array([(x, y, z) for _, x, y, z in host_atoms])
    guest_xyz = np.array([(x, y, z) for _, x, y, z in guest_atoms])

    # ── 1. Convex hull containment ───────────────────────────────────────────
    hull_fraction = None
    if HAVE_SCIPY and len(host_xyz) >= 4:
        try:
            hull = Delaunay(host_xyz)
            inside = hull.find_simplex(guest_xyz) >= 0
            hull_fraction = float(np.mean(inside))
        except Exception:
            hull_fraction = None  # degenerate/coplanar host points etc.

    # ── 2. Contact density ───────────────────────────────────────────────────
    # Pairwise distances host x guest; count within cutoff per guest atom.
    diffs = guest_xyz[:, None, :] - host_xyz[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    contacts_per_atom = float(np.mean(np.sum(dists < contact_cutoff, axis=1)))
    contact_score = min(1.0, contacts_per_atom / 3.0)

    # ── 3. COM proximity ──────────────────────────────────────────────────────
    host_com  = mass_weighted_com(host_atoms)
    guest_com = mass_weighted_com(guest_atoms)
    host_rg   = radius_of_gyration(host_atoms, host_com)
    com_dist  = float(np.linalg.norm(np.array(guest_com) - np.array(host_com)))
    com_ratio = com_dist / host_rg if host_rg > 0 else float("inf")
    com_score = max(0.0, 1.0 - min(com_ratio, 1.5) / 1.5)

    # ── Composite ─────────────────────────────────────────────────────────────
    if hull_fraction is not None:
        score = 100 * (0.65 * hull_fraction + 0.25 * contact_score + 0.10 * com_score)
    else:
        # No scipy / degenerate hull: redistribute hull's weight to the
        # other two signals so the score is still meaningful, just less
        # precise (no installation message needed at every call -- emitted
        # once by the caller instead).
        score = 100 * (0.70 * contact_score + 0.30 * com_score)

    return {
        "score": score,
        "hull_fraction_pct": None if hull_fraction is None else round(100 * hull_fraction, 1),
        "contacts_per_atom": round(contacts_per_atom, 2),
        "com_ratio": round(com_ratio, 2),
    }
    
def classify_vina(reform_dir: Path, threshold: float, contact_cutoff: float):
    summary = reform_dir / "reform_summary.txt"
    mol2 = reform_dir / "complex_full.mol2"
    if not summary.exists() or not mol2.exists():
        return None

    n_host, n_guest, vina_score = parse_reform_summary(summary)
    if n_host is None or n_guest is None:
        return None

    atoms = parse_mol2_atoms(mol2)
    if len(atoms) < n_host + n_guest:
        return None

    host_heavy  = [a for a in atoms[:n_host] if a[0] != "H"]
    guest_heavy = [a for a in atoms[n_host:n_host + n_guest] if a[0] != "H"]

    result = containment_score(host_heavy, guest_heavy, contact_cutoff)
    if result is None:
        return None

    result["vina_score"] = vina_score
    result["label"] = "bound" if result["score"] >= threshold else "unbound"
    return result


def classify_crest_ensemble(crest_dir: Path, n_host: int, n_guest: int,
                            threshold: float, contact_cutoff: float,
                            ensemble_bound_frac: float):
    conformers_xyz = crest_dir / "crest_conformers.xyz"
    if not conformers_xyz.exists():
        return None

    frames = parse_xyz_frames(conformers_xyz)
    if not frames:
        return None

    per_conformer = []
    for comment, atoms in frames:
        if len(atoms) < n_host + n_guest:
            continue
        host_heavy  = [a for a in atoms[:n_host] if a[0] != "H"]
        guest_heavy = [a for a in atoms[n_host:n_host + n_guest] if a[0] != "H"]
        result = containment_score(host_heavy, guest_heavy, contact_cutoff)
        if result is None:
            continue
        try:
            energy_hartree = float(comment.split()[0])
        except (ValueError, IndexError):
            energy_hartree = None
        result["energy_hartree"] = energy_hartree
        result["bound"] = result["score"] >= threshold
        per_conformer.append(result)

    if not per_conformer:
        return None

    n_total = len(per_conformer)
    n_bound = sum(1 for c in per_conformer if c["bound"])
    frac_bound = n_bound / n_total

    energies = [c["energy_hartree"] for c in per_conformer if c["energy_hartree"] is not None]
    e_min = min(energies) if energies else None

    def rel_kcal(e):
        if e is None or e_min is None:
            return None
        return (e - e_min) * HARTREE_TO_KCAL

    bound_energies   = [rel_kcal(c["energy_hartree"]) for c in per_conformer if c["bound"]]
    unbound_energies = [rel_kcal(c["energy_hartree"]) for c in per_conformer if not c["bound"]]
    bound_energies   = [e for e in bound_energies if e is not None]
    unbound_energies = [e for e in unbound_energies if e is not None]

    best_bound   = min(bound_energies) if bound_energies else None
    best_unbound = min(unbound_energies) if unbound_energies else None
    penalty = (best_unbound - best_bound) if (best_bound is not None and best_unbound is not None) else None

    return {
        "n_conformers": n_total,
        "n_bound": n_bound,
        "frac_bound": round(frac_bound, 3),
        "best_bound_kcal": None if best_bound is None else round(best_bound, 2),
        "best_unbound_kcal": None if best_unbound is None else round(best_unbound, 2),
        "dissociation_penalty_kcal": None if penalty is None else round(penalty, 2),
        "label": "bound" if frac_bound >= ensemble_bound_frac else "unbound",
    }


def find_pairs(reformed_root: Path):
    pairs = []
    if not reformed_root.is_dir():
        return pairs
    for host_dir in sorted(reformed_root.iterdir()):
        if not host_dir.is_dir():
            continue
        for guest_dir in sorted(host_dir.iterdir()):
            if guest_dir.is_dir() and (guest_dir / "reform_summary.txt").exists():
                pairs.append((host_dir.name, guest_dir.name, guest_dir))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Classify host-guest complexes as bound/unbound by geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--project-dir", default=".",
                    help="Project root containing reformed/, crest_results/ (default: CWD)")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Containment score (0-100) above which a single pose/conformer "
                         "is labelled 'bound' (default: 50)")
    ap.add_argument("--ensemble-bound-frac", type=float, default=0.5,
                    help="Fraction of CREST conformers that must be individually bound "
                         "for the whole ensemble to be labelled 'bound' (default: 0.5)")
    ap.add_argument("--contact-cutoff", type=float, default=4.5,
                    help="Distance (Å) for host-guest heavy-atom contact counting (default: 4.5)")
    ap.add_argument("--no-crest", action="store_true",
                    help="Skip CREST ensemble classification (Vina poses only)")
    ap.add_argument("--output", default=None,
                    help="Output CSV path (default: <project-dir>/binding_classification.csv)")
    args = ap.parse_args()

    if not HAVE_SCIPY:
        print("WARNING: scipy not available -- convex-hull containment is disabled.")
        print("         Falling back to contact-density + COM-proximity only (less precise).")
        print("         Install with: pip install scipy  (or conda install scipy)\n")

    project_root = Path(args.project_dir).resolve()
    reformed_root = project_root / "reformed"
    crest_root = project_root / "crest_results"
    out_csv = Path(args.output).resolve() if args.output \
              else project_root / "binding_classification.csv"

    pairs = find_pairs(reformed_root)
    if not pairs:
        sys.exit(f"No reformed complexes found under {reformed_root}")

    print(f"classify_binding.py")
    print(f"  Project   : {project_root}")
    print(f"  Pairs     : {len(pairs)}")
    print(f"  Threshold : {args.threshold} (single pose)  /  "
          f"{args.ensemble_bound_frac} (ensemble fraction)")
    print(f"  CREST     : {'skipped' if args.no_crest else 'included'}")
    print()

    rows = []
    for host, guest, reform_dir in pairs:
        vina = classify_vina(reform_dir, args.threshold, args.contact_cutoff)
        if vina is None:
            print(f"  [SKIP]  {host} + {guest}  (could not classify Vina pose)")
            continue

        row = {
            "host": host, "guest": guest,
            "vina_score_kcal": vina["vina_score"],
            "vina_containment_pct": round(vina["score"], 1),
            "vina_hull_fraction_pct": vina["hull_fraction_pct"],
            "vina_contacts_per_atom": vina["contacts_per_atom"],
            "vina_com_ratio": vina["com_ratio"],
            "vina_label": vina["label"],
        }

        crest_summary = ""
        if not args.no_crest:
            n_host, n_guest, _ = parse_reform_summary(reform_dir / "reform_summary.txt")
            crest_dir = crest_root / f"{host}_{guest}"
            crest = None
            if n_host is not None and n_guest is not None and crest_dir.is_dir():
                crest = classify_crest_ensemble(
                    crest_dir, n_host, n_guest,
                    args.threshold, args.contact_cutoff, args.ensemble_bound_frac,
                )
            if crest:
                row.update({
                    "crest_n_conformers": crest["n_conformers"],
                    "crest_frac_bound": crest["frac_bound"],
                    "crest_best_bound_kcal": crest["best_bound_kcal"],
                    "crest_best_unbound_kcal": crest["best_unbound_kcal"],
                    "crest_dissociation_penalty_kcal": crest["dissociation_penalty_kcal"],
                    "crest_label": crest["label"],
                })
                crest_summary = (f"  CREST: {crest['n_bound']}/{crest['n_conformers']} bound "
                                 f"({crest['label']})")
            else:
                row.update({
                    "crest_n_conformers": "", "crest_frac_bound": "",
                    "crest_best_bound_kcal": "", "crest_best_unbound_kcal": "",
                    "crest_dissociation_penalty_kcal": "", "crest_label": "",
                })
                crest_summary = "  CREST: (not yet available)"

        print(f"  {host} + {guest}:  Vina={row['vina_containment_pct']}% "
              f"({vina['label']}){crest_summary}")
        rows.append(row)

    if not rows:
        sys.exit("No pairs were successfully classified.")

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(f, "")) for f in fieldnames) + "\n")

    n_vina_bound = sum(1 for r in rows if r["vina_label"] == "bound")
    print(f"\nDone. {len(rows)} pair(s) classified.")
    print(f"  Vina:  {n_vina_bound}/{len(rows)} bound ({100*n_vina_bound/len(rows):.0f}%)")
    if not args.no_crest:
        crest_rows = [r for r in rows if r.get("crest_label")]
        if crest_rows:
            n_crest_bound = sum(1 for r in crest_rows if r["crest_label"] == "bound")
            print(f"  CREST: {n_crest_bound}/{len(crest_rows)} bound "
                  f"({100*n_crest_bound/len(crest_rows):.0f}%)  "
                  f"[{len(rows)-len(crest_rows)} pair(s) had no CREST ensemble yet]")
    print(f"\nWritten to: {out_csv}")


if __name__ == "__main__":
    main()
