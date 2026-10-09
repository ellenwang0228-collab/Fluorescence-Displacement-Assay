#!/usr/bin/env python3
import sys
import os
import csv
import argparse
import subprocess
import tempfile
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")   # silence routine RDKit warnings; we report our own
except ImportError:
    print("[prepare_guests] FATAL: RDKit is not importable in this Python environment.",
          file=sys.stderr)
    print("                  Activate the conda env that has rdkit (same one used for CREST),",
          file=sys.stderr)
    print("                  or:  conda install -c conda-forge rdkit", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SMILES -> 3D full-H mol2 for guest molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True,
                   help="guests_smiles.csv exported by the local app "
                        "(columns: name,category,smiles,charge,pubchem_cid)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for guest mol2 files (./guests)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed for 3D embedding (default 42, for reproducibility)")
    p.add_argument("--mmff-iters", type=int, default=2000,
                   help="Max iterations for force-field optimisation (default 2000)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: SMILES -> 3D mol -> mol2
# ─────────────────────────────────────────────────────────────────────────────

# Multiple seeds to try if the first embedding attempt fails (rare, but happens
# for flexible/macrocyclic molecules with default ETKDG settings).
_EMBED_SEED_OFFSETS = (0, 1, 7, 42, 1234)


def embed_3d(mol, base_seed: int):
    """
    Try to generate a 3D conformer for `mol` (already has explicit Hs).
    Returns the conformer ID on success, or None if all attempts failed.
    """
    params = AllChem.ETKDGv3()
    params.useRandomCoords = False

    for offset in _EMBED_SEED_OFFSETS:
        params.randomSeed = base_seed + offset
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0:
            return cid

    # Last resort: allow random coordinate initialisation (helps some
    # difficult cases, e.g. highly symmetric or strained rings)
    params.useRandomCoords = True
    for offset in _EMBED_SEED_OFFSETS:
        params.randomSeed = base_seed + offset + 9000
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0:
            return cid

    return None


def optimize_3d(mol, conf_id: int, max_iters: int):
    """
    Optimise the geometry of conformer `conf_id`.
    Tries MMFF94 first (more accurate for organics), falls back to UFF
    (broader element coverage — needed for e.g. Pt in cisplatin).

    Returns the name of the force field actually used ('MMFF', 'UFF'),
    or 'none' if both failed (geometry kept as embedded, unoptimised).
    """
    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
            return "MMFF"
        except Exception:
            pass   # fall through to UFF

    if AllChem.UFFHasAllMoleculeParams(mol):
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
            return "UFF"
        except Exception:
            pass

    return "none"


def sdf_to_mol2(sdf_path: str, mol2_path: str, log_path: str) -> bool:
    """
    Convert an SDF file to mol2 via obabel, assigning Gasteiger partial charges
    (consistent with the partial-charge convention used elsewhere in this
    pipeline for ligand preparation).

    Returns True on success (mol2 written and non-empty), False otherwise.
    Any obabel stderr is appended to log_path for debugging.
    """
    try:
        result = subprocess.run(
            ["obabel", sdf_path, "-O", mol2_path, "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=300,
        )
    except FileNotFoundError:
        with open(log_path, "a") as fh:
            fh.write("[prepare_guests] FATAL: 'obabel' not found in PATH.\n")
        return False
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as fh:
            fh.write("[prepare_guests] obabel timed out after 300s.\n")
        return False

    with open(log_path, "a") as fh:
        if result.stdout:
            fh.write(result.stdout)
        if result.stderr:
            fh.write(result.stderr)

    return os.path.exists(mol2_path) and os.path.getsize(mol2_path) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  PER-GUEST PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_guest(row: dict, out_dir: Path, seed: int, mmff_iters: int) -> dict:
    """
    Process one CSV row.  Returns a result dict for the summary table:
        {name, status, detail, n_heavy, n_h, ff_used}
    status is one of: SKIP, OK, FAIL
    """
    name    = row["name"].strip()
    smiles  = row["smiles"].strip()
    csv_chg = row.get("charge", "").strip()

    mol2_out = out_dir / f"{name}.mol2"

    # ── Idempotency: don't clobber an existing/hand-edited structure ─────────
    if mol2_out.exists() and mol2_out.stat().st_size > 0:
        return {"name": name, "status": "SKIP",
                "detail": f"{mol2_out} already exists", "n_heavy": "-", "n_h": "-", "ff_used": "-"}

    if not smiles:
        return {"name": name, "status": "FAIL",
                "detail": "empty SMILES string", "n_heavy": "-", "n_h": "-", "ff_used": "-"}

    print(f"\n[prepare_guests] {name}")
    print(f"                 SMILES: {smiles}")

    # ── Step 1: parse + sanitize ──────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"name": name, "status": "FAIL",
                "detail": "RDKit could not parse SMILES (invalid syntax or valence)",
                "n_heavy": "-", "n_h": "-", "ff_used": "-"}

    n_heavy = mol.GetNumAtoms()
    formal_charge = Chem.GetFormalCharge(mol)

    # ── Cross-check formal charge vs. CSV ─────────────────────────────────────
    if csv_chg:
        try:
            csv_chg_int = int(csv_chg)
            if csv_chg_int != formal_charge:
                print(f"                 NOTE: RDKit formal charge from SMILES = {formal_charge}, "
                      f"but charges.csv will use {csv_chg_int} (your value is kept).")
        except ValueError:
            print(f"                 NOTE: could not parse charge '{csv_chg}' from CSV as integer.")

    # ── Step 2: add explicit hydrogens BEFORE embedding ───────────────────────
    mol = Chem.AddHs(mol)
    n_h = mol.GetNumAtoms() - n_heavy
    print(f"                 Heavy atoms: {n_heavy}   Hydrogens (added): {n_h}   "
          f"Formal charge: {formal_charge}")

    # ── Step 3: 3D embedding ──────────────────────────────────────────────────
    conf_id = embed_3d(mol, seed)
    if conf_id is None:
        return {"name": name, "status": "FAIL",
                "detail": "3D embedding failed after multiple seed attempts "
                          "(molecule may be too large/strained for ETKDG)",
                "n_heavy": n_heavy, "n_h": n_h, "ff_used": "-"}

    # ── Step 4: force-field optimisation ──────────────────────────────────────
    ff_used = optimize_3d(mol, conf_id, mmff_iters)
    if ff_used == "none":
        print(f"                 WARNING: no force field could optimise this molecule "
              f"(unsupported elements) — using raw embedded geometry.")
    else:
        print(f"                 Optimised with: {ff_used}")

    # ── Step 5: write SDF → mol2 (via obabel, Gasteiger charges) ──────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.prep.log"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as tmp:
        tmp_sdf = tmp.name
        tmp.write(Chem.MolToMolBlock(mol, confId=conf_id))

    try:
        ok = sdf_to_mol2(tmp_sdf, str(mol2_out), str(log_path))
    finally:
        os.unlink(tmp_sdf)

    if not ok:
        return {"name": name, "status": "FAIL",
                "detail": f"obabel SDF->mol2 conversion failed — see {log_path}",
                "n_heavy": n_heavy, "n_h": n_h, "ff_used": ff_used}

    print(f"                 -> {mol2_out}")
    return {"name": name, "status": "OK", "detail": str(mol2_out),
            "n_heavy": n_heavy, "n_h": n_h, "ff_used": ff_used}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        print(f"[prepare_guests] FATAL: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[prepare_guests] FATAL: {csv_path} is missing required column(s): "
                  f"{', '.join(sorted(missing))}", file=sys.stderr)
            print(f"                  Found columns: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    if not rows:
        print(f"[prepare_guests] {csv_path} has no data rows — nothing to do.")
        return

    print(f"[prepare_guests] {'='*60}")
    print(f"[prepare_guests]  {len(rows)} guest(s) in {csv_path}")
    print(f"[prepare_guests]  Output directory: {out_dir}")
    print(f"[prepare_guests]  Base random seed:  {args.seed}")
    print(f"[prepare_guests] {'='*60}")

    results = []
    for row in rows:
        try:
            results.append(process_guest(row, out_dir, args.seed, args.mmff_iters))
        except Exception as exc:
            results.append({"name": row.get("name", "?"), "status": "FAIL",
                             "detail": f"unexpected error: {exc}",
                             "n_heavy": "-", "n_h": "-", "ff_used": "-"})

    # ── Summary table ───────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    summary_path = out_dir / "prepare_summary.txt"
    with open(summary_path, "w") as fh:
        header = f"{'name':<28} {'status':<6} {'heavy':>5} {'H':>5} {'FF':>6}  detail"
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")
        for r in results:
            fh.write(f"{r['name']:<28} {r['status']:<6} {str(r['n_heavy']):>5} "
                     f"{str(r['n_h']):>5} {str(r['ff_used']):>6}  {r['detail']}\n")

    print(f"\n[prepare_guests] {'='*60}")
    print(f"[prepare_guests]  Done.  OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}")
    print(f"[prepare_guests]  Summary -> {summary_path}")
    print(f"[prepare_guests] {'='*60}")

    if n_fail > 0:
        print(f"\n[prepare_guests] WARNING: {n_fail} guest(s) failed — see {summary_path}")
        print("                 Failed guests will simply be absent from ./guests/, so")
        print("                 pipeline.sh Phase 0 will report them as missing input files.")
        print("                 Fix the SMILES in guests_smiles.csv and re-run this script,")
        print("                 or drop in a hand-made mol2 with the same <name>.mol2 filename.")


if __name__ == "__main__":
    main()
