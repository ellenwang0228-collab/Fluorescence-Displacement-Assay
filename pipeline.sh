#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

###############################################################################
# pipeline.sh — Host-Guest Docking → Full-H Complex Assembly → CREST
#
# WHAT THIS SCRIPT DOES (in order):
#   Phase 0 — Validate    : check tools, input files, python packages
#   Phase 0.5 — SMILES    : (optional) if ./guests_smiles.csv is present — e.g.
#                            exported by the local Project Builder app — generate
#                            any missing ./guests/*.mol2 via RDKit (SMILES → 3D,
#                            full H, obabel→mol2 with Gasteiger charges).
#   Phase 1 — Prepare     : convert host/guest mol2|xyz → PDBQT via obabel
#                            compute host COM → per-host Vina box configs
#   Phase 2 — Dock        : run Vina for every host × guest pair.
#                            Interactive: runs in the foreground, one pair.
#                            Batch: submits a SLURM array (one task per
#                            pair not already docked), then submits a
#                            second job with --dependency=afterany that
#                            automatically resumes Phases 3-5 once docking
#                            finishes — no manual re-run needed.
#   Phase 3 — Reform      : obabel --join host PDBQT + top-1 docked guest
#                            PDBQT (already co-registered by Vina, no
#                            superposition needed), then -h
#                            --partialcharge gasteiger to perceive bonds
#                            and rehydrogenate -> complex.xyz for CREST.
#   Phase 4 — Manifest    : build crest_manifest.tsv from reformed complexes
#   Phase 5 — CREST       : GFN-FF conformer search (--gfnff --quick --noreftopo)
#
# HOW TO RUN:
#   Interactive (test one pair step-by-step, with checkpoints):
#     ./pipeline.sh --mode interactive --host host1 --guest guest1
#
#   Batch (all hosts × all guests):
#     ./pipeline.sh --mode batch
#   This submits a Vina docking array job, then a dependent "postdock" job
#   (--dependency=afterany) that automatically runs reform + builds the CREST
#   manifest + submits the CREST array once docking finishes — one command,
#   fully chained. If every pair is already docked, it proceeds straight to
#   reform/CREST in this same invocation (no submission needed).
#
#   Skip stages already done (batch only):
#     ./pipeline.sh --mode batch --skip-prep --skip-dock --skip-reform
#
#   Verbose bash tracing:
#     DEBUG=1 ./pipeline.sh --mode interactive --host host1 --guest guest1
#
# DIRECTORY LAYOUT (all paths are configurable below):
#   ./hosts/              ← YOU PROVIDE: host structures (*.mol2 or *.xyz)
#                            These must have the experimentally-correct H count.
#   ./guests/             ← guest structures (*.mol2 or *.xyz). Provide these
#                            directly, OR let Phase 0.5 generate them from
#                            ./guests_smiles.csv below — or mix both: files
#                            already present in ./guests/ are never overwritten.
#   ./guests_smiles.csv   ← OPTIONAL — typically exported by the local Project
#                            Builder app. Columns: name,category,smiles,charge,
#                            pubchem_cid. Phase 0.5 converts each row to
#                            ./guests/<name>.mol2 (3D, full H, via RDKit) unless
#                            that file already exists.
#   ./charges.csv         ← YOU PROVIDE (optional): one line per molecule:
#                            name,charge    e.g.  host1,-6
#                            Complex charge = host_charge + guest_charge.
#                            Molecules not listed default to charge 0.
#                            (Auto-written alongside guests_smiles.csv by the
#                            local app — edit freely.)
#   ./reform_complex.py   ← must sit alongside this script
#   ./crest_worker.sh     ← must sit alongside this script
#   ./dock_worker.sh      ← must sit alongside this script (batch-mode docking)
#   ./prepare_guests.py   ← must sit alongside this script IF guests_smiles.csv is used
#
#   Generated automatically:
#   ./pdbqt/              host PDBQTs for Vina
#   ./ligands_pdbqt/      guest PDBQTs for Vina
#   ./configs/            per-host Vina box configs
#   ./centers.csv         host COM table
#   ./docking_results/    Vina output  (<host>/<guest>_docked.pdbqt)
#                          + dock_manifest.tsv (batch mode job list)
#   ./reformed/           join+rehydrate complexes (<host>/<guest>/)
#   ./crest_manifest.tsv  job list for SLURM array
#   ./crest_results/      CREST output (<host>_<guest>/)
#   ./pipeline.log        full run log (appended each run)
#
# NOTES ON HOST FORMAT:
#   mol2 input is strongly preferred.  xyz input works but obabel must guess
#   bond orders from geometry — this is less reliable for unusual coordination.
#
# NOTES ON CREST:
#   Interactive mode runs CREST in the foreground (assumes you are in an
#   srun interactive session on a compute node, like dock_all_hosts_onego.sh).
#   Batch mode submits a SLURM array job (one task per complex).
###############################################################################



SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

_REQUIRED_SIBLINGS=(dock_worker.sh crest_worker.sh reform_complex.py prepare_guests.py)
_missing_siblings=()
for _s in "${_REQUIRED_SIBLINGS[@]}"; do
    [[ -f "$SCRIPT_DIR/$_s" ]] || _missing_siblings+=("$_s")
done

if [[ ${#_missing_siblings[@]} -gt 0 ]]; then
    echo "[pipeline] Some sibling scripts are missing from $SCRIPT_DIR:"
    printf "[pipeline]   missing: %s\n" "${_missing_siblings[@]}"
    echo "[pipeline] Searching nearby directories..."

    # Search strategy: parent dir and all sibling project dirs
    _search_dirs=()
    _parent="$(dirname "$SCRIPT_DIR")"
    [[ -d "$_parent" ]] && _search_dirs+=("$_parent")
    while IFS= read -r _d; do
        [[ -d "$_d" && "$_d" != "$SCRIPT_DIR" ]] && _search_dirs+=("$_d")
    done < <(find "$_parent" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

    _copied=()
    _still_missing=()
    for _s in "${_missing_siblings[@]}"; do
        _found=0
        for _d in "${_search_dirs[@]}"; do
            if [[ -f "$_d/$_s" ]]; then
                cp "$_d/$_s" "$SCRIPT_DIR/$_s"
                echo "[pipeline]   ✓ copied $_s from $_d"
                _copied+=("$_s")
                _found=1
                break
            fi
        done
        [[ $_found -eq 0 ]] && _still_missing+=("$_s")
    done

    if [[ ${#_still_missing[@]} -gt 0 ]]; then
        echo "[pipeline] ERROR: Could not find the following scripts anywhere nearby:"
        printf "[pipeline]   %s\n" "${_still_missing[@]}"
        echo "[pipeline] Please copy them manually from your original cluster_scripts/ folder:"
        printf "[pipeline]   cp /path/to/cluster_scripts/%s %s/\n" \
            "${_still_missing[@]}" "$SCRIPT_DIR"
        exit 1
    fi
    echo "[pipeline] All missing scripts found and copied — continuing."
fi
unset _REQUIRED_SIBLINGS _missing_siblings _search_dirs _copied _still_missing _s _d _parent _found


HOST_DIR="./hosts"               # host reference structures (*.mol2 or *.xyz)
GUEST_DIR="./guests"             # guest reference structures (*.mol2 or *.xyz)
CHARGES_CSV="./charges.csv"      # optional: name,charge pairs
# Curated host mol2 files with experimentally correct protonation states.
# Place <host_name>.mol2 here (matching the filename stem of your host input).
# If a curated mol2 exists for a host, reform_complex.py uses it AS-IS for
# the host H atoms (no further polar H added by obabel) for EVERY guest
# complex of that host -- edit once, propagates everywhere automatically.
# If absent, the standard join+add-all-H pipeline runs and the post-reform
# checkpoint lets you edit individual complex.xyz files instead.
CURATED_DIR="./curated"

# ── SMILES → 3D guest generation (optional; Phase 0.5) ────────────────────────
# Typically exported by the local Project Builder app alongside charges.csv.
# If GUESTS_SMILES_CSV exists, Phase 0.5 fills in any missing ./guests/*.mol2
# before the rest of the pipeline runs. Existing guest files are never touched.
GUESTS_SMILES_CSV="./guests_smiles.csv"
PREPARE_GUESTS_SCRIPT="$SCRIPT_DIR/prepare_guests.py"
PREPARE_SEED=42                  # base RNG seed for 3D embedding (reproducibility)

# ── Generated directories ──────────────────────────────────────────────────────
PDBQT_DIR="./pdbqt"             # host PDBQTs
LIG_DIR="./ligands_pdbqt"       # guest PDBQTs
CONFIG_DIR="./configs"           # per-host Vina configs
DOCK_DIR="./docking_results"     # Vina outputs
REFORM_DIR="./reformed"          # join+rehydrate complexes (Phase 3)
CREST_DIR="./crest_results"      # CREST outputs
MANIFEST="./crest_manifest.tsv"  # job list for SLURM array
CENTERS_CSV="./centers.csv"      # COM table
RUN_LOG="./pipeline.log"         # full append-mode log

# Resolve everything above to ABSOLUTE paths right now, before anything else
# runs. Rationale: ml/conda setup scripts sourced later (Phase 5) can
# silently `cd` elsewhere (e.g. $HOME) as a side effect, and pushd/popd
# changes CWD deliberately. Relative paths like "./pipeline.log" would then
# resolve against whatever CWD happens to be AT THE TIME each is used --
# RUN_LOG in particular is referenced by log()/info()/etc. throughout the
# script, so a CWD change part-way through silently redirects all further
# logging to the wrong file. `-m` allows paths that don't exist yet.
for _v in PDBQT_DIR LIG_DIR CONFIG_DIR DOCK_DIR REFORM_DIR CREST_DIR \
          MANIFEST CENTERS_CSV RUN_LOG CURATED_DIR; do
    printf -v "$_v" '%s' "$(realpath -m "${!_v}")"
done
unset _v

# ── Vina box settings ──────────────────────────────────────────────────────────
SIZE_X=20
SIZE_Y=20
SIZE_Z=20
EXHAUSTIVENESS=30
NUM_MODES=5
ENERGY_RANGE=8
VINA_CPUS=80                     

# ── Batch docking (SLURM array) ────────────────────────────────────────────────
VINA_CPUS_BATCH=8
VINA_MEM="8G"
VINA_TIME="04:00:00"
VINA_PARTITION="ncpu"
DOCK_WORKER="$SCRIPT_DIR/dock_worker.sh"

# After the docking array finishes (--dependency=afterany, i.e. regardless of
# whether individual pairs failed to dock), a small follow-up job re-invokes
# this script with --skip-prep --skip-dock to run reform -> CREST manifest ->
# CREST array submission automatically. This is that follow-up job's
# allocation, not a per-pair one.
POSTDOCK_MEM="4G"
POSTDOCK_TIME="02:00:00"

# ── Reform settings ────────────────────────────────────────────────────────────
CLASH_WARN_DIST=2.0              # Å: flag host-guest pairs closer than this
REFORM_SCRIPT="$SCRIPT_DIR/reform_complex.py"

# ── CREST / xtb settings ──────────────────────────────────────────────────────
CREST_CHARGE_DEFAULT=0           
CREST_SOLVENT="water"            # ALPB solvent; set to "none" to omit --alpb
CREST_CPUS=64
CREST_MEM="500G"
CREST_TIME="3-00:00:00"
CREST_PARTITION="ncpu"
CREST_WORKER="$SCRIPT_DIR/crest_worker.sh"

# NOTE: a host-guest distance constraint (--cinp) was previously available
# here to prevent the MTD bias potential from dissociating the complex, but
# was REMOVED due to a confirmed, currently-unresolved crash in CREST 3.0.2's
# --cinp parser when combined with --gfnff (uninitialized-memory bug --
# "Error allocating <garbage> bytes: Cannot allocate memory", intermittent).
# Confirmed by a CREST maintainer: github.com/crest-lab/crest/issues/338,
# #367, #381. Dissociation (if it occurs) should instead be handled by
# post-hoc filtering of the resulting conformer ensemble.

# ── Cluster environment (mirrors your xtb.sh) ────────────────────────────────
CONDA_ENV="crest"
CONDA_MODULE="Anaconda3/2023.03"
CONDA_SETUP="/flask/apps/eb/software/Anaconda/conda.env.sh"

# ── Debugging ─────────────────────────────────────────────────────────────────
DEBUG="${DEBUG:-0}"

# ═══════════════════════════════════════════════════════════════════════════════
#  END USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

ts()     { date +"%Y-%m-%d %H:%M:%S"; }
log()    { echo "[$(ts)]       $*" | tee -a "$RUN_LOG" >&2; }
info()   { echo "[$(ts)] INFO  $*" | tee -a "$RUN_LOG" >&2; }
warn()   { echo "[$(ts)] WARN  $*" | tee -a "$RUN_LOG" >&2; }
die()    { echo "[$(ts)] FATAL $*" | tee -a "$RUN_LOG" >&2; exit 1; }

banner() {
    local msg="$*"
    local line; printf -v line '═%.0s' {1..70}
    { echo ""; echo "$line"; printf "  %s\n" "$msg"; echo "$line"; echo ""; } \
        | tee -a "$RUN_LOG" >&2
}

# Trap: print the failing command and line number before exiting
on_err() {
    local ec=$? ln=$1 cmd=$2
    echo "[$(ts)] FATAL  Command failed (exit=$ec) at line $ln" | tee -a "$RUN_LOG" >&2
    echo "         Failed command: $cmd"                         | tee -a "$RUN_LOG" >&2
    exit "$ec"
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

[[ "$DEBUG" == "1" ]] && { set -x; info "DEBUG mode active (bash -x)."; }

# Interactive yes/no prompt — NEVER call this in batch mode
confirm() {
    local prompt="$1" reply
    read -r -p "  >>> $prompt  [y/N] " reply
    [[ "${reply,,}" == "y" ]]
}

# Look up charge for a molecule name in charges.csv.
# Prints the charge (or the default) to stdout so callers can capture it.
get_charge() {
    local name="$1" charge=""
    if [[ -f "$CHARGES_CSV" ]]; then
        charge=$(grep -v '^\s*#' "$CHARGES_CSV" \
                 | awk -F',' -v n="$name" 'NF>=2 && $1==n {gsub(/[[:space:]]/,"",$2); print $2; exit}' \
                 || true)
    fi
    if [[ -z "$charge" ]]; then
        warn "No charge entry for '$name' in $CHARGES_CSV — using default: $CREST_CHARGE_DEFAULT"
        echo "$CREST_CHARGE_DEFAULT"
    else
        echo "$charge"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

MODE=""
INTERACTIVE_HOST=""
INTERACTIVE_GUEST=""
SKIP_PREP=0
SKIP_DOCK=0
SKIP_REFORM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2";               shift 2 ;;
        --host)          INTERACTIVE_HOST="$2";   shift 2 ;;
        --guest)         INTERACTIVE_GUEST="$2";  shift 2 ;;
        --skip-prep)     SKIP_PREP=1;             shift   ;;
        --skip-dock)     SKIP_DOCK=1;             shift   ;;
        --skip-reform)   SKIP_REFORM=1;           shift   ;;
        -h|--help)
            sed -n '2,/^#*$/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown argument: '$1'  Run with --help for usage." ;;
    esac
done

[[ -n "$MODE" ]] || die "Must specify --mode interactive  OR  --mode batch"
[[ "$MODE" == "interactive" || "$MODE" == "batch" ]] \
    || die "Unknown mode '$MODE'. Use 'interactive' or 'batch'."

if [[ "$MODE" == "interactive" ]]; then
    [[ -n "$INTERACTIVE_HOST" && -n "$INTERACTIVE_GUEST" ]] \
        || die "Interactive mode requires --host <name> and --guest <name>"
fi


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
banner "PHASE 0 — Validation"
info "Mode: $MODE"
[[ "$MODE" == "interactive" ]] && info "Pair: $INTERACTIVE_HOST × $INTERACTIVE_GUEST"

# ── Required external tools ───────────────────────────────────────────────────
info "Checking external tools..."
for tool in vina obabel python3; do
    if command -v "$tool" &>/dev/null; then
        info "  ✓ $tool  →  $(command -v "$tool")"
    else
        die "Required tool not found in PATH: $tool\n  Activate your conda environment before running this script."
    fi
done

# ── Python package check (numpy, scipy — needed by prepare_guests.py for ──────
#    RDKit-based 3D embedding in Phase 0.5; reform_complex.py itself now
#    depends only on obabel + stdlib).
info "Checking Python packages..."
python3 - <<'PYCHECK'
import sys
missing = []
for pkg in ["numpy", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[FATAL] Missing Python packages: {', '.join(missing)}", file=sys.stderr)
    print("        Install with:  pip install numpy scipy  (or via conda)", file=sys.stderr)
    sys.exit(1)
print("[INFO]    Python packages OK: numpy, scipy")
PYCHECK

# ── Helper scripts ─────────────────────────────────────────────────────────────
[[ -f "$REFORM_SCRIPT" ]] || die "reform_complex.py not found at $REFORM_SCRIPT\n  It must be in the same directory as this script."
[[ "$MODE" == "batch" ]] && \
    { [[ -f "$CREST_WORKER" ]] || die "crest_worker.sh not found at $CREST_WORKER"; }

# ── Input directories ──────────────────────────────────────────────────────────
[[ -d "$HOST_DIR" ]]  || die "Host directory not found:  $HOST_DIR"

if [[ ! -d "$GUEST_DIR" ]]; then
    if [[ -f "$GUESTS_SMILES_CSV" ]]; then
        info "Guest directory $GUEST_DIR does not exist yet — creating it for Phase 0.5 output."
        mkdir -p "$GUEST_DIR"
    else
        die "Guest directory not found: $GUEST_DIR"
    fi
fi

if [[ ! -f "$CHARGES_CSV" ]]; then
    warn "charges.csv not found at $CHARGES_CSV"
    warn "  All molecules will use default charge=$CREST_CHARGE_DEFAULT"
    warn "  Create $CHARGES_CSV with lines like:  host1,-6"
fi

info "Tool, package, and directory checks passed."


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0.5 — SMILES → 3D GUEST STRUCTURES  (optional)
#
#  If ./guests_smiles.csv exists (e.g. exported by the local Project Builder
#  app), generate any missing ./guests/<name>.mol2 files from SMILES via RDKit
#  (3D embedding + force-field optimisation + obabel→mol2 with Gasteiger
#  charges). Guest files already present in $GUEST_DIR are left untouched, so
#  you can mix hand-prepared structures with app-generated ones, or override an
#  app-generated structure just by dropping a better mol2 into ./guests/ first.
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -f "$GUESTS_SMILES_CSV" ]]; then
    banner "PHASE 0.5 — Generating guest structures from SMILES ($GUESTS_SMILES_CSV)"

    [[ -f "$PREPARE_GUESTS_SCRIPT" ]] \
        || die "$GUESTS_SMILES_CSV found but $PREPARE_GUESTS_SCRIPT is missing.\n  It must be in the same directory as this script."

    info "Checking for RDKit (required to process $GUESTS_SMILES_CSV)..."
    python3 - <<'PYCHECK'
import sys
try:
    import rdkit  # noqa: F401
except ImportError:
    print("[FATAL] RDKit is required to process guests_smiles.csv but is not importable.", file=sys.stderr)
    print("        Install with:  conda install -c conda-forge rdkit", file=sys.stderr)
    sys.exit(1)
print(f"[INFO]    RDKit OK (version {rdkit.__version__})")
PYCHECK

    info "Running: python3 $PREPARE_GUESTS_SCRIPT --csv $GUESTS_SMILES_CSV --out-dir $GUEST_DIR --seed $PREPARE_SEED"
    python3 "$PREPARE_GUESTS_SCRIPT" \
        --csv      "$GUESTS_SMILES_CSV" \
        --out-dir  "$GUEST_DIR" \
        --seed     "$PREPARE_SEED" \
        2>&1 | tee -a "$RUN_LOG"

    info "Phase 0.5 complete — per-guest status in $GUEST_DIR/prepare_summary.txt"
else
    info "No $GUESTS_SMILES_CSV found — skipping Phase 0.5 (SMILES-based guest generation)."
fi

# ── Collect input files ────────────────────────────────────────────────────────
mapfile -t HOST_FILES  < <(find "$HOST_DIR"  -maxdepth 1 \( -name "*.mol2" -o -name "*.xyz" \) | sort)
mapfile -t GUEST_FILES < <(find "$GUEST_DIR" -maxdepth 1 \( -name "*.mol2" -o -name "*.xyz" \) | sort)

[[ ${#HOST_FILES[@]}  -gt 0 ]] || die "No mol2/xyz files found in $HOST_DIR"
[[ ${#GUEST_FILES[@]} -gt 0 ]] || die "No mol2/xyz files found in $GUEST_DIR"

info "Found ${#HOST_FILES[@]} host file(s)  in $HOST_DIR"
info "Found ${#GUEST_FILES[@]} guest file(s) in $GUEST_DIR"

# ── Warn about xyz inputs (bond perception limitations) ───────────────────────
for f in "${HOST_FILES[@]}" "${GUEST_FILES[@]}"; do
    if [[ "${f##*.}" == "xyz" ]]; then
        warn "  xyz input detected: $(basename "$f")"
        warn "    obabel will guess bond orders from geometry — mol2 is preferred for reliability."
    fi
done

# ── In interactive mode, narrow to just the requested pair ────────────────────
if [[ "$MODE" == "interactive" ]]; then
    _hfile=""
    for f in "${HOST_FILES[@]}"; do
        b=$(basename "$f"); b="${b%.*}"
        [[ "$b" == "$INTERACTIVE_HOST" ]] && _hfile="$f" && break
    done
    [[ -n "$_hfile" ]] \
        || die "Host '$INTERACTIVE_HOST' not found in $HOST_DIR (tried mol2 and xyz)"

    _gfile=""
    for f in "${GUEST_FILES[@]}"; do
        b=$(basename "$f"); b="${b%.*}"
        [[ "$b" == "$INTERACTIVE_GUEST" ]] && _gfile="$f" && break
    done
    [[ -n "$_gfile" ]] \
        || die "Guest '$INTERACTIVE_GUEST' not found in $GUEST_DIR (tried mol2 and xyz)"

    HOST_FILES=("$_hfile")
    GUEST_FILES=("$_gfile")
    info "Interactive mode — restricted to: $INTERACTIVE_HOST × $INTERACTIVE_GUEST"
fi

# ── Create output directories ──────────────────────────────────────────────────
mkdir -p "$PDBQT_DIR" "$LIG_DIR" "$CONFIG_DIR" "$DOCK_DIR" "$REFORM_DIR" "$CREST_DIR" "$CURATED_DIR"

info "Validation complete — all checks passed."


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — PREPARATION
#  Convert mol2/xyz → PDBQT; compute host COM; write Vina box configs.
# ═══════════════════════════════════════════════════════════════════════════════
banner "PHASE 1 — Preparation: mol2/xyz → PDBQT, COM computation, Vina configs"

if [[ "$SKIP_PREP" == "1" ]]; then
    info "SKIP_PREP=1 — reusing existing PDBQTs and configs."
    # Still filter GUEST_FILES down to guests with an existing PDBQT, in case
    # a previous run left some guests without one (e.g. they failed
    # conversion last time and were excluded then too).
    kept=()
    missing=()
    for gf in "${GUEST_FILES[@]}"; do
        gname=$(basename "$gf"); gname="${gname%.*}"
        if [[ -s "$LIG_DIR/${gname}.pdbqt" ]]; then
            kept+=("$gf")
        else
            missing+=("$gname")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "  ${#missing[@]} guest(s) have no PDBQT in $LIG_DIR and will be excluded: ${missing[*]}"
        GUEST_FILES=("${kept[@]}")
        [[ ${#GUEST_FILES[@]} -gt 0 ]] \
            || die "No guests with existing PDBQTs in $LIG_DIR. Re-run without SKIP_PREP=1."
    fi
else

# ── 1a: Hosts → PDBQT (rigid receptor; -xr suppresses torsion tree) ───────────
    info "Converting host structures to PDBQT (rigid receptors)..."

    # AutoDock/Vina only recognises a fixed atom-type vocabulary. Many metals
    # that appear in supramolecular cage hosts (Pd, Pt, Ru, Rh, Ir, Co, ...) are
    # absent and will cause Vina to reject the PDBQT entirely. The workaround is
    # to map them to the nearest supported proxy type AFTER obabel writes the
    # file. This only affects how Vina's scoring grid samples those atoms during
    # docking; the downstream CREST/xtb step reads complex.xyz (which comes from
    # reform_complex.py, not from this PDBQT), so the real GFN-FF chemistry
    # around these metals is unaffected.
    #
    # Map: unsupported → supported proxy (Vina built-in types used as keys here)
    #   Pd, Pt, Ni, Cu, Ag, Au → Zn  (all d8/d10 square-planar; Zn is closest
    #                                   in VdW radius and is best-parameterised)
    #   Co, Rh, Ir              → Mn  (octahedral d6/d9, similar radius to Mn)
    #   Ru, Os                  → Fe  (octahedral d6, Fe is the supported proxy)
    #   V, Cr, Mo, W            → Fe  (transition metals; Fe nearest in parameter
    #                                   table)
    #   Ga, Ge, As, Se          → Se (natively supported in Vina 1.2+)
    #   Sr, Ba                  → Ca  (alkaline earths)
    #   Al, Si                  → Mg  (similar radii)
    #
    # Atom types in Vina 1.2 PDBQT: C A N NA NS OA OS S SA H HD HS F CL BR I
    #                                 Mg MN ZN CA FE P Ge As Se  (metals/metalloids)
    #
    # The replacement is done with a targeted sed substitution on the atom-type
    # column (columns 78-79 in a standard PDBQT ATOM/HETATM record), not a
    # naive string replace, to avoid accidentally touching atom names or residues.
    declare -A _METAL_REMAP=(
        [Pd]=Zn  [Pt]=Zn  [Ni]=Zn  [Cu]=Zn  [Ag]=Zn  [Au]=Zn
        [Co]=Mn  [Rh]=Mn  [Ir]=Mn
        [Ru]=Fe  [Os]=Fe
        [V]=Fe   [Cr]=Fe  [Mo]=Fe  [W]=Fe
        [Sr]=Ca  [Ba]=Ca
        [Al]=Mg  [Si]=Mg
    )

    remap_pdbqt_metals() {
        # Usage: remap_pdbqt_metals <file>
        # Rewrites atom types in-place for any entries in _METAL_REMAP.
        # PDBQT ATOM/HETATM format:
        #   columns 1-6   record type
        #   columns 77-78 AutoDock atom type (1 or 2 char, left-justified)
        # sed: match lines starting with ATOM/HETATM and ending with the known
        # bad type (anchored at EOL after optional spaces), replace just the type.
        local f="$1"
        local remapped=()
        for orig in "${!_METAL_REMAP[@]}"; do
            proxy="${_METAL_REMAP[$orig]}"
            # Count occurrences before substitution
            n=$(grep -cP "^(ATOM|HETATM).{70,} ${orig}\s*$" "$f" 2>/dev/null || true)
            if [[ $n -gt 0 ]]; then
                # Replace the atom-type field (last whitespace-delimited token on
                # ATOM/HETATM lines that exactly matches $orig)
                sed -i -E "s/^(ATOM|HETATM)(.{70,} )${orig}(\s*)$/\1\2${proxy}\3/" "$f"
                remapped+=("${n}x ${orig}→${proxy}")
            fi
        done
        if [[ ${#remapped[@]} -gt 0 ]]; then
            info "    → Metal atom-type remapping (Vina proxy): ${remapped[*]}"
            info "      (Pd/Pt/Ni→Zn, Co/Rh/Ir→Mn, Ru/Os→Fe etc. Affects Vina scoring only;"
            info "       CREST/xtb reads complex.xyz and sees the real element.)"
        fi
    }

    for hf in "${HOST_FILES[@]}"; do
        hname=$(basename "$hf"); hname="${hname%.*}"
        out="$PDBQT_DIR/${hname}.pdbqt"
        if [[ -s "$out" ]]; then
            info "  [SKIP]  $hname.pdbqt already exists"
            continue
        fi
        info "  $hname: $(basename "$hf") → $out"

        # Charge-method fallback chain.
        # Gasteiger is the standard for Vina but is an empirical organic-chemistry
        # method that FAILS on metal-coordinated atoms (coordination bonds confuse
        # the valence model it uses). For rigid receptors, Vina's built-in scoring
        # function doesn't use receptor partial charges at all -- only the AD4
        # scoring function does. So falling back to mmff94 or even zero charges
        # is scientifically fine for rigid host docking.
        #
        # Cage hosts also frequently look like multiple disconnected fragments to
        # obabel (one fragment per ligand + one per metal node). The -xr -c flags
        # tell obabel to combine them into a single rigid molecule, which is
        # exactly what a Vina receptor needs.
        host_converted=0
        for charge_method in gasteiger mmff94 none; do
            if [[ "$charge_method" == "none" ]]; then
                charge_args=()
                info "    Attempt 3: no partial charges (all zeros) -- valid for rigid receptor"
            else
                charge_args=(--partialcharge "$charge_method")
                info "    Attempt (--partialcharge $charge_method)..."
            fi

            # -xr : rigid receptor (no ROOT/BRANCH/TORSDOF)
            # -xc : combine separate molecular fragments into one rigid unit
            #       (critical for metal-organic cages: obabel perceives ligand
            #        struts and metal nodes as separate fragments without this)
            if obabel "$hf" -O "$out" "${charge_args[@]}" -xr -xc \
                    >> "$RUN_LOG" 2>&1 && [[ -s "$out" ]]; then
                host_converted=1
                [[ "$charge_method" != "gasteiger" ]] && \
                    warn "    Used --partialcharge $charge_method (Gasteiger failed -- normal for metal-organic cages; safe for rigid receptor docking)"
                break
            fi
            rm -f "$out"
        done

        if [[ $host_converted -eq 0 ]]; then
            die "obabel failed to convert host $hname to PDBQT with all charge methods.
  Input file  : $hf
  Tried       : gasteiger, mmff94, no charges
  See log     : $RUN_LOG
  Tip: open the mol2 in Avogadro/Mercury and check for disconnected atoms,
  missing bonds around the metal, or unusual bond orders (ar/du/un) that
  obabel can't perceive. For metal-organic cages, also try exporting as
  .xyz from your structure viewer first -- obabel handles .xyz → .pdbqt
  more robustly than mol2 → pdbqt for unusual topologies."
        fi

        # Remap any still-unsupported metal atom types to Vina-recognised proxies
        remap_pdbqt_metals "$out"

        n_heavy=$(grep -cE '^(ATOM|HETATM)' "$out" || true)
        info "    → $n_heavy heavy atoms written to $out"
    done

# ── 1b: Guests → PDBQT (flexible ligand; obabel auto-detects rotatable bonds) ─
    info "Converting guest structures to PDBQT (flexible ligands)..."
    FAILED_GUESTS=()
    for gf in "${GUEST_FILES[@]}"; do
        gname=$(basename "$gf"); gname="${gname%.*}"
        out="$LIG_DIR/${gname}.pdbqt"
        if [[ -s "$out" ]]; then
            info "  [SKIP]  $gname.pdbqt already exists"
            continue
        fi
        info "  $gname: $(basename "$gf") → $out"
        if ! obabel "$gf" -O "$out" --partialcharge gasteiger 2>>"$RUN_LOG" || [[ ! -s "$out" ]]; then
            warn "  obabel FAILED on guest $gname ($gf) -- skipping this guest (see $RUN_LOG)"
            rm -f "$out"   # remove any empty/partial output so it can't be mistaken for success later
            FAILED_GUESTS+=("$gname")
            continue
        fi
        n_heavy=$(grep -cE '^(ATOM|HETATM)' "$out" || true)
        n_tors=$(grep -c '^BRANCH'            "$out" || true)
        info "    → $n_heavy heavy atoms, $n_tors rotatable bonds detected"
    done

    # Drop any failed guests from GUEST_FILES entirely -- Phase 2 (docking),
    # Phase 3 (reform) and Phase 4 (manifest) all derive their guest lists
    # from GUEST_FILES, so removing them here is sufficient to exclude them
    # everywhere downstream without special-casing each phase.
    if [[ ${#FAILED_GUESTS[@]} -gt 0 ]]; then
        warn "  ${#FAILED_GUESTS[@]} guest(s) failed PDBQT conversion and will be excluded from the rest of the run: ${FAILED_GUESTS[*]}"
        kept=()
        for gf in "${GUEST_FILES[@]}"; do
            gname=$(basename "$gf"); gname="${gname%.*}"
            skip=0
            for f in "${FAILED_GUESTS[@]}"; do
                [[ "$gname" == "$f" ]] && { skip=1; break; }
            done
            [[ $skip -eq 0 ]] && kept+=("$gf")
        done
        GUEST_FILES=("${kept[@]}")
        [[ ${#GUEST_FILES[@]} -gt 0 ]] \
            || die "All guests failed PDBQT conversion -- nothing left to dock. Check $RUN_LOG for obabel errors."
        info "  Continuing with ${#GUEST_FILES[@]} guest(s)."
    fi

# ── 1c: Compute host COM → Vina box configs ────────────────────────────────────
    info "Computing host centers-of-mass and writing Vina box configs..."
    mapfile -t HOST_PDBQTS < <(
        for hf in "${HOST_FILES[@]}"; do
            hname=$(basename "$hf"); hname="${hname%.*}"
            echo "$PDBQT_DIR/${hname}.pdbqt"
        done
    )

    printf "host,center_x,center_y,center_z\n" > "$CENTERS_CSV"

    # Embedded Python: mass-weighted COM from PDBQT ATOM/HETATM records.
    # AutoDock atom types (e.g. A, NA, OA, HD, SA) are mapped to element symbols
    # for mass lookup.  Unknown types fall back to geometric (unweighted) centre.
    python3 - "$CENTERS_CSV" "$CONFIG_DIR" \
              "$SIZE_X" "$SIZE_Y" "$SIZE_Z" \
              "$EXHAUSTIVENESS" "$NUM_MODES" "$ENERGY_RANGE" \
              "${HOST_PDBQTS[@]}" <<'PY'
import sys, os, re

centers_csv = sys.argv[1]
config_dir  = sys.argv[2]
sx, sy, sz  = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
exhaust     = int(sys.argv[6])
num_modes   = int(sys.argv[7])
e_range     = int(sys.argv[8])
host_paths  = sys.argv[9:]

MASS = {
    "H":1.008,  "C":12.011, "N":14.007, "O":15.999, "F":18.998,
    "P":30.974, "S":32.06,  "CL":35.45, "BR":79.904,"I":126.90,
    "B":10.81,  "SI":28.09, "FE":55.85, "ZN":65.38, "CU":63.55,
    "MN":54.94, "MG":24.31, "CA":40.08,
}

def at_to_elem(at):
    """AutoDock atom type string → element key (uppercase)."""
    s = re.sub(r"[^A-Za-z]", "", at).upper()
    if s == "A":   return "C"
    if s == "HD":  return "H"
    for prefix in ("CL","BR","SI","FE","ZN","CU","MN","MG","CA"):
        if s.startswith(prefix): return prefix
    return s[:1] if s else ""

def com_from_pdbqt(path):
    wtx=wty=wtz=mass=ux=uy=uz=n=0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM","HETATM")): continue
            try:
                x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
            except ValueError:
                p=line.split(); x,y,z=float(p[5]),float(p[6]),float(p[7])
            m = MASS.get(at_to_elem(line.split()[-1]))
            if m: wtx+=m*x; wty+=m*y; wtz+=m*z; mass+=m
            else: ux+=x; uy+=y; uz+=z; n+=1
    if mass > 0: return wtx/mass, wty/mass, wtz/mass
    if n   > 0: return ux/n,    uy/n,    uz/n
    raise ValueError(f"No ATOM/HETATM lines in {path}")

os.makedirs(config_dir, exist_ok=True)
with open(centers_csv,"a") as csv_fh:
    for p in host_paths:
        host = os.path.splitext(os.path.basename(p))[0]
        cx,cy,cz = com_from_pdbqt(p)
        csv_fh.write(f"{host},{cx:.4f},{cy:.4f},{cz:.4f}\n")
        cfg = os.path.join(config_dir, f"{host}.txt")
        with open(cfg,"w") as cf:
            cf.write(f"# Vina config for {host}  (auto-generated by pipeline.sh)\n")
            cf.write(f"# Box centre = mass-weighted COM of host heavy atoms\n\n")
            cf.write(f"center_x = {cx:.4f}\ncenter_y = {cy:.4f}\ncenter_z = {cz:.4f}\n\n")
            cf.write(f"size_x = {sx:g}\nsize_y = {sy:g}\nsize_z = {sz:g}\n\n")
            cf.write(f"exhaustiveness = {exhaust}\nnum_modes = {num_modes}\nenergy_range = {e_range}\n")
        print(f"[INFO]    {host}: COM=({cx:.3f}, {cy:.3f}, {cz:.3f})  config→{cfg}")
PY

    # Sanity check: every host in our list got a config
    for hf in "${HOST_FILES[@]}"; do
        hname=$(basename "$hf"); hname="${hname%.*}"
        cfg="$CONFIG_DIR/${hname}.txt"
        [[ -s "$cfg" ]] \
            || die "Config missing after COM step: $cfg  This should not happen — check Python errors above."
    done

    info "Phase 1 complete — PDBQTs in $PDBQT_DIR / $LIG_DIR,  configs in $CONFIG_DIR"
fi  # end SKIP_PREP


if [[ "$SKIP_DOCK" == "1" ]]; then
    info "SKIP_DOCK=1 — reusing existing docking results."
else

    # Build arrays of host/guest PDBQTs that correspond to our input file list
    declare -a H_PDBQTS=()
    for hf in "${HOST_FILES[@]}"; do
        hname=$(basename "$hf"); hname="${hname%.*}"
        H_PDBQTS+=("$PDBQT_DIR/${hname}.pdbqt")
    done
    declare -a G_PDBQTS=()
    for gf in "${GUEST_FILES[@]}"; do
        gname=$(basename "$gf"); gname="${gname%.*}"
        G_PDBQTS+=("$LIG_DIR/${gname}.pdbqt")
    done

    if [[ "$MODE" == "interactive" ]]; then
        # ── Interactive: foreground, one pair, with checkpoint ─────────────────
        total=$(( ${#H_PDBQTS[@]} * ${#G_PDBQTS[@]} ))
        n=0; n_ok=0; n_skip=0; n_fail=0

        for receptor in "${H_PDBQTS[@]}"; do
            hname=$(basename "$receptor" .pdbqt)
            config="$CONFIG_DIR/${hname}.txt"
            [[ -s "$config" ]] \
                || die "Config not found for host '$hname': $config  Run Phase 1 first."
            host_outdir="$DOCK_DIR/$hname"
            mkdir -p "$host_outdir"

            for ligand in "${G_PDBQTS[@]}"; do
                gname=$(basename "$ligand" .pdbqt)
                (( n++ )) || true
                out_pdbqt="$host_outdir/${gname}_docked.pdbqt"
                log_file="$host_outdir/${gname}.log"

                if [[ -s "$out_pdbqt" ]]; then
                    info "  [$n/$total] SKIP (exists): $hname + $gname"
                    (( n_skip++ )) || true
                    continue
                fi

                info "  [$n/$total] Docking: $hname + $gname"

                # Redirect stdout to log_file; Vina writes its score table to stdout
                if vina --receptor "$receptor" \
                        --ligand   "$ligand" \
                        --config   "$config" \
                        --out      "$out_pdbqt" \
                        --cpu      "$VINA_CPUS" \
                        > "$log_file" 2>&1; then

                    if [[ -s "$out_pdbqt" ]]; then
                        # Pull top-1 score out of the Vina log
                        top_score=$(awk '/VINA RESULT/{print $4; exit}
                                         /minimizedAffinity/{print $2; exit}' \
                                        "$out_pdbqt" "$log_file" 2>/dev/null || echo "unknown")
                        info "    ✓  Top score: ${top_score} kcal/mol  →  $out_pdbqt"
                        (( n_ok++ )) || true
                    else
                        warn "    ✗  Output PDBQT empty after successful Vina run: $out_pdbqt"
                        warn "       (Vina may have found no poses — check $log_file)"
                        (( n_fail++ )) || true
                    fi

                else
                    warn "    ✗  Vina exited non-zero for $hname + $gname  (see $log_file)"
                    (( n_fail++ )) || true
                fi

                # ── INTERACTIVE CHECKPOINT 1: after docking ───────────────────────
                if [[ -s "$out_pdbqt" ]]; then
                    echo ""
                    echo "  ┌── CHECKPOINT: Docking complete ─────────────────────────────────┐"
                    echo "  │  Docked PDBQT  : $out_pdbqt"
                    echo "  │  Vina log      : $log_file"
                    echo "  │  Top score     : ${top_score} kcal/mol"
                    echo "  │"
                    echo "  │  Inspect the pose now (in another terminal):"
                    echo "  │    pymol $receptor $out_pdbqt"
                    echo "  └─────────────────────────────────────────────────────────────────┘"
                    confirm "Pose looks reasonable — proceed to complex reformation?" \
                        || { info "Aborted by user after docking.  Files are in $host_outdir"; exit 0; }
                fi

            done  # guest loop
        done  # host loop

        info "Docking summary:  OK=$n_ok  Skipped=$n_skip  Failed=$n_fail  Total=$total"
        [[ $n_fail -gt 0 ]] && warn "  $n_fail docking job(s) failed — check $RUN_LOG for details"

    else
        # ── Batch: build a docking manifest, submit a SLURM array, and ─────────
        #    auto-chain a "postdock" job that resumes Phases 3-5 once the
        #    array finishes (--dependency=afterany — runs regardless of
        #    whether individual pairs failed to dock; reform/manifest already
        #    skip pairs with no docked output).
        DOCK_MANIFEST="$DOCK_DIR/dock_manifest.tsv"
        printf "job_id\thost\tguest\treceptor\tligand\tconfig\tout_pdbqt\tlog_file\n" > "$DOCK_MANIFEST"

        total_pairs=$(( ${#H_PDBQTS[@]} * ${#G_PDBQTS[@]} ))
        n_dock_jobs=0
        n_already=0

        for receptor in "${H_PDBQTS[@]}"; do
            hname=$(basename "$receptor" .pdbqt)
            config="$CONFIG_DIR/${hname}.txt"
            [[ -s "$config" ]] \
                || die "Config not found for host '$hname': $config  Run Phase 1 first."
            host_outdir="$DOCK_DIR/$hname"
            mkdir -p "$host_outdir"

            for ligand in "${G_PDBQTS[@]}"; do
                gname=$(basename "$ligand" .pdbqt)
                out_pdbqt="$host_outdir/${gname}_docked.pdbqt"
                log_file="$host_outdir/${gname}.log"

                if [[ -s "$out_pdbqt" ]]; then
                    (( n_already++ )) || true
                    continue
                fi

                (( n_dock_jobs++ )) || true
                # receptor/ligand/config/out_pdbqt/log_file are all already
                # absolute (derived from PDBQT_DIR/LIG_DIR/CONFIG_DIR/DOCK_DIR,
                # which were resolved to absolute paths at startup).
                printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$n_dock_jobs" "$hname" "$gname" \
                    "$receptor" "$ligand" "$config" \
                    "$out_pdbqt" "$log_file" >> "$DOCK_MANIFEST"
            done
        done

        info "Docking manifest: $n_dock_jobs job(s) to run, $n_already already done (of $total_pairs total pairs) → $DOCK_MANIFEST"

        if [[ $n_dock_jobs -eq 0 ]]; then
            info "All host-guest pairs already docked — proceeding directly to reform in this run."
        else
            dock_sbatch_cmd=(sbatch
                --parsable
                --job-name="dock"
                --array="1-${n_dock_jobs}"
                --ntasks=1
                --cpus-per-task="$VINA_CPUS_BATCH"
                --mem="$VINA_MEM"
                --time="$VINA_TIME"
                --partition="$VINA_PARTITION"
                --output="${DOCK_DIR}/slurm_%A_%a.out"
                --error="${DOCK_DIR}/slurm_%A_%a.err"
                --export="ALL,\
MANIFEST=$DOCK_MANIFEST,\
CONDA_ENV=$CONDA_ENV,\
CONDA_MODULE=$CONDA_MODULE,\
CONDA_SETUP=$CONDA_SETUP"
                "$DOCK_WORKER"
            )
            info "sbatch command: ${dock_sbatch_cmd[*]}"

            dock_job_id=$("${dock_sbatch_cmd[@]}" 2>&1) \
                || die "sbatch (docking array) failed: $dock_job_id"
            dock_job_id="${dock_job_id%%[!0-9]*}"   # --parsable normally returns just the job ID; strip anything stray

            info "Submitted docking array job $dock_job_id  ($n_dock_jobs task(s), $VINA_CPUS_BATCH CPU(s) each)"
            info "Monitor:   squeue -j $dock_job_id"
            info "Logs:      ${DOCK_DIR}/slurm_${dock_job_id}_<task>.out"

            # ── Auto-chain: resume this pipeline once docking finishes ──────────
            self_script="$(realpath "$0")"
            postdock_cmd=(sbatch
                --parsable
                --job-name="postdock"
                --dependency="afterany:${dock_job_id}"
                --ntasks=1
                --cpus-per-task=1
                --mem="$POSTDOCK_MEM"
                --time="$POSTDOCK_TIME"
                --partition="$VINA_PARTITION"
                --chdir="$PWD"
                --output="${DOCK_DIR}/postdock_%j.out"
                --error="${DOCK_DIR}/postdock_%j.err"
                --wrap="'$self_script' --mode batch --skip-prep --skip-dock"
            )
            info "sbatch command: ${postdock_cmd[*]}"

            if postdock_job_id=$("${postdock_cmd[@]}" 2>&1); then
                postdock_job_id="${postdock_job_id%%[!0-9]*}"
                info "Submitted postdock job $postdock_job_id  (runs after job $dock_job_id finishes, dependency=afterany)"
                info "  -> reform, CREST manifest, and CREST array submission will all happen"
                info "     automatically inside job $postdock_job_id once docking finishes."
                info "Logs:      ${DOCK_DIR}/postdock_${postdock_job_id}.out"
            else
                warn "sbatch (postdock continuation) failed: $postdock_job_id"
                warn "  Docking array $dock_job_id is still queued/running, but Phases 3-5 will"
                warn "  NOT run automatically. Once it finishes, resume manually with:"
                warn "    ./pipeline.sh --mode batch --skip-prep --skip-dock"
            fi

            info ""
            info "Pausing here for this invocation — Phases 3-5 will run in the postdock job."
            exit 0
        fi
    fi
fi  # end SKIP_DOCK


# ─────────────────────────────────────────────────────────────────────────────
#  Harvest Vina scores → docking_scores.csv
#  Scans all docking_results/<host>/<guest>.log files for top-1 scores and
#  writes a CSV with category lookup from the guest library. This is used by
#  the score visualizer (score_viewer.html, generated below).
# ─────────────────────────────────────────────────────────────────────────────
SCORES_CSV="$DOCK_DIR/docking_scores.csv"
GUEST_LIB="$SCRIPT_DIR/../guest_library.json"
[[ -f "$GUEST_LIB" ]] || GUEST_LIB="$SCRIPT_DIR/../../guest_library.json"

info "Harvesting Vina scores → $SCORES_CSV"
python3 - "$DOCK_DIR" "$SCORES_CSV" "$GUEST_LIB" << 'PY'
import sys, os, re, json, pathlib

dock_dir   = pathlib.Path(sys.argv[1])
out_csv    = pathlib.Path(sys.argv[2])
lib_path   = sys.argv[3] if len(sys.argv) > 3 else ""

# Build guest→category lookup from guest_library.json
cat_map = {}
if os.path.isfile(lib_path):
    try:
        lib = json.load(open(lib_path))
        for cat, names in lib.get("categories", {}).items():
            for name in names:
                cat_map[name.lower()] = cat
    except Exception:
        pass

rows = []
for log_file in sorted(dock_dir.rglob("*.log")):
    # Expect: docking_results/<host>/<guest>.log
    parts = log_file.parts
    if len(parts) < 2:
        continue
    host  = parts[-2]
    guest = log_file.stem.replace("_docked", "")
    score = None
    try:
        for line in open(log_file):
            # Vina log: "   1       -8.5      0.000      0.000"
            m = re.match(r'\s+1\s+(-?\d+\.\d+)', line)
            if m:
                score = float(m.group(1))
                break
    except Exception:
        pass
    if score is not None:
        category = cat_map.get(guest.lower().replace("_", " "), "Other")
        rows.append((host, guest, category, score))

rows.sort(key=lambda r: r[3])  # sort by score (most negative first)

with open(out_csv, "w") as f:
    f.write("host,guest,category,score_kcal_mol\n")
    for host, guest, cat, score in rows:
        f.write(f"{host},{guest},{cat},{score}\n")

print(f"  Wrote {len(rows)} score(s) to {out_csv}")
PY


#
#  For each docked pair:
#    1. Extract the top-1 docked pose (MODEL 1) from the guest's docked PDBQT,
#       preserving it exactly as Vina wrote it.
#    2. obabel --join: concatenate the host PDBQT + top-pose guest PDBQT into
#       one structure. NO coordinate transform — both are already in Vina's
#       docking frame, so this is pure concatenation.
#    3. obabel -h --partialcharge gasteiger: perceive bonds/aromaticity from
#       3D geometry and add back the hydrogens PDBQT's united-atom format
#       had stripped, writing complex_full.mol2.
#    4. obabel: complex_full.mol2 -> complex.xyz (CREST input).
#    5. QC: formula/H-count vs curated references (optional), check for any
#       bond perceived ACROSS the host/guest boundary, host-guest clashes.
# ═══════════════════════════════════════════════════════════════════════════════
banner "PHASE 3 — Complex reformation (join host+docked PDBQTs, rehydrogenate)"

if [[ "$SKIP_REFORM" == "1" ]]; then
    info "SKIP_REFORM=1 — reusing existing reformed complexes."
else

    # Build name → reference path maps for fast lookup
    declare -A HOST_REF_MAP GUEST_REF_MAP
    for hf in "${HOST_FILES[@]}"; do
        hname=$(basename "$hf"); hname="${hname%.*}"
        HOST_REF_MAP["$hname"]="$hf"
    done
    for gf in "${GUEST_FILES[@]}"; do
        gname=$(basename "$gf"); gname="${gname%.*}"
        GUEST_REF_MAP["$gname"]="$gf"
    done

    n=0; n_ok=0; n_fail=0

    for host_dock_dir in "$DOCK_DIR"/*/; do
        [[ -d "$host_dock_dir" ]] || continue
        hname=$(basename "$host_dock_dir")

        host_pdbqt="$PDBQT_DIR/${hname}.pdbqt"
        if [[ ! -s "$host_pdbqt" ]]; then
            warn "No host PDBQT for '$hname' at $host_pdbqt (Phase 1 should have created this) — skipping all its pairs."
            continue
        fi

        # host_ref/guest_ref are now OPTIONAL — used only for the formula/H-count
        # QC check, not for the reformation itself (host_pdbqt + docked_pdbqt
        # are already in the same coordinate frame, no superposition needed).
        host_ref="${HOST_REF_MAP[$hname]:-}"

        for docked_pdbqt in "$host_dock_dir"*_docked.pdbqt; do
            [[ -s "$docked_pdbqt" ]] || continue
            fname=$(basename "$docked_pdbqt")
            gname="${fname%_docked.pdbqt}"
            (( n++ )) || true

            guest_ref="${GUEST_REF_MAP[$gname]:-}"

            pose_dir="$REFORM_DIR/$hname/$gname"
            mkdir -p "$pose_dir"
            reform_log="$pose_dir/reform.log"

            info "  Reforming [$n]: $hname + $gname"

            # Build optional QC-reference args (omitted entirely if not found —
            # reform_complex.py just skips the formula-comparison check then)
            ref_args=()
            if [[ -n "$host_ref" && -n "$guest_ref" ]]; then
                ref_args=(--host-ref "$host_ref" --guest-ref "$guest_ref")
            else
                [[ -z "$host_ref"  ]] && warn "  No reference for host '$hname'  — formula QC check will be skipped."
                [[ -z "$guest_ref" ]] && warn "  No reference for guest '$gname' — formula QC check will be skipped."
            fi

            # Check for a curated host mol2 (correct protonation, set once per host).
            # If found, reform uses it AS-IS for the host H atoms -- every guest
            # complex of this host inherits the curated protonation automatically.
            curated_args=()
            curated_mol2="$CURATED_DIR/${hname}.mol2"
            if [[ -s "$curated_mol2" ]]; then
                curated_args=(--curated-host-mol2 "$curated_mol2")
                info "    Using curated host mol2: $curated_mol2"
            else
                info "    No curated mol2 found at $curated_mol2 — using standard pipeline"
                info "    (Tip: place ${hname}.mol2 with correct protonation in $CURATED_DIR/)"
            fi

            if python3 "$REFORM_SCRIPT" \
                    --docked-pdbqt  "$docked_pdbqt" \
                    --host-pdbqt    "$host_pdbqt" \
                    "${ref_args[@]}" \
                    "${curated_args[@]}" \
                    --out-dir       "$pose_dir" \
                    --host-name     "$hname" \
                    --guest-name    "$gname" \
                    --clash-warn    "$CLASH_WARN_DIST" \
                    2>&1 | tee "$reform_log" | tee -a "$RUN_LOG"; then

                if [[ -s "$pose_dir/complex.xyz" ]]; then
                    # Echo the key QC lines reform_complex.py printed
                    formula_line=$(grep -E "Formula matches|H-count differs|HEAVY-ATOM COUNT MISMATCH" "$reform_log" || true)
                    cross_line=$(grep -E "No bonds perceived across|bond\(s\) perceived ACROSS"        "$reform_log" || true)
                    clash_line=$(grep "Clash check" "$reform_log" || true)
                    [[ -n "$formula_line" ]] && info "    ✓  ${formula_line#*✓ }"
                    info "       $cross_line"
                    info "       $clash_line"
                    (( n_ok++ )) || true
                else
                    warn "    ✗  complex.xyz not created — see $reform_log"
                    (( n_fail++ )) || true
                fi

            else
                warn "    ✗  reform_complex.py failed for $hname + $gname — see $reform_log"
                (( n_fail++ )) || true
            fi

            # ── INTERACTIVE CHECKPOINT 2: after reform ─────────────────────────
            if [[ "$MODE" == "interactive" && -s "$pose_dir/complex.xyz" ]]; then
                echo ""
                echo "  ┌── CHECKPOINT: Complex assembly complete ─────────────────────────┐"
                echo "  │  Complex XYZ (CREST input) : $pose_dir/complex.xyz"
                echo "  │  Complex mol2 (visual QC)  : $pose_dir/complex_full.mol2"
                echo "  │  Reform summary            : $pose_dir/reform_summary.txt"
                echo "  │  Reform log                : $reform_log"
                echo "  │"
                echo "  │  Inspect the assembled complex:"
                echo "  │    pymol $pose_dir/complex_full.mol2"
                echo "  └─────────────────────────────────────────────────────────────────┘"
                confirm "Complex looks correct — proceed to CREST?" \
                    || { info "Aborted by user after reform.  All files are in $pose_dir"; exit 0; }
            fi

        done  # docked pdbqt loop
    done  # host dir loop

    info "Reform summary:  OK=$n_ok  Failed=$n_fail  Total=$n"
    [[ $n_fail -gt 0 ]] && warn "  $n_fail reform(s) failed — inspect the reform.log files above."

fi  # end SKIP_REFORM


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT — Protonation curation (interactive mode only)
# ═══════════════════════════════════════════════════════════════════════════════
# obabel -h adds ALL polar hydrogens based on its own perception of the
# structure. For hosts with experimentally confirmed ionisation states
# (e.g. deprotonated pyridyl N, phosphonate groups, Pd-coordinated amines),
# some of these will be wrong and will artificially inflate H-bonding in CREST.
#
# This checkpoint pauses the pipeline so you can:
#   1. Open complex_full.mol2 in PyMOL to identify unwanted H by inspection
#   2. Delete those H atom entries from complex.xyz
#   3. Update the atom count on line 1 of complex.xyz (n_atoms - n_deleted)
#   4. Confirm here — CREST will then run on your curated structure
#
# complex.xyz format reminder:
#   Line 1: <total atom count>    ← decrement this for each H you remove
#   Line 2: comment (free text)
#   Line 3+: <element> <x> <y> <z>   ← delete entire lines for unwanted H

if [[ "$MODE" == "interactive" ]]; then
    # Determine if any host used a curated mol2
    any_curated=0
    any_uncurated=0
    for hf in "${HOST_FILES[@]}"; do
        hname=$(basename "$hf"); hname="${hname%.*}"
        if [[ -s "$CURATED_DIR/${hname}.mol2" ]]; then
            (( any_curated++ )) || true
        else
            (( any_uncurated++ )) || true
        fi
    done

    echo ""
    echo "  ┌── CHECKPOINT: Protonation curation ─────────────────────────────────┐"
    echo "  │"
    if [[ $any_curated -gt 0 && $any_uncurated -eq 0 ]]; then
        echo "  │  All hosts used a curated mol2 — host H is already correct."
        echo "  │  Guest H was added by obabel and should be fine as-is."
        echo "  │  You can confirm immediately unless you need to edit guest H."
    elif [[ $any_curated -gt 0 ]]; then
        echo "  │  Some hosts used a curated mol2 (H correct for those hosts)."
        echo "  │  Uncurated hosts had all H added by obabel — review those below."
    else
        echo "  │  No curated host mol2 found. obabel added all polar H."
        echo "  │  TIP: place curated/<host>.mol2 (correct protonation) in your"
        echo "  │  project directory to fix this automatically for all future runs."
    fi
    echo "  │"
    echo "  │  Reformed complex files:"
    for host_reformed_dir in "$REFORM_DIR"/*/; do
        [[ -d "$host_reformed_dir" ]] || continue
        hname=$(basename "$host_reformed_dir")
        curated_flag=""
        [[ -s "$CURATED_DIR/${hname}.mol2" ]] && curated_flag="  [curated H ✓]"
        for guest_reformed_dir in "$host_reformed_dir"*/; do
            [[ -d "$guest_reformed_dir" ]] || continue
            gname=$(basename "$guest_reformed_dir")
            xyz="$guest_reformed_dir/complex.xyz"
            if [[ -s "$xyz" ]]; then
                n_atoms=$(head -1 "$xyz" | tr -d '[:space:]')
                n_H=$(awk 'NR>2 && $1=="H" {count++} END {print count+0}' "$xyz")
                printf "  │    %-30s + %-20s  (%s atoms, %s H)%s\n" \
                    "$hname" "$gname" "$n_atoms" "$n_H" "$curated_flag"
                printf "  │      complex.xyz      : %s\n" "$xyz"
                printf "  │      complex_full.mol2: %s\n" "$guest_reformed_dir/complex_full.mol2"
            fi
        done
    done
    echo "  │"
    echo "  │  To edit a complex.xyz: delete H lines + decrement count on line 1."
    echo "  │  To fix ALL hosts permanently: create curated/<host>.mol2 and re-run"
    echo "  │  with --skip-prep --skip-dock (reform will use it automatically)."
    echo "  └─────────────────────────────────────────────────────────────────────┘"
    confirm "Protonation confirmed — proceed to CREST manifest and job submission?" \
        || { info "Paused. Re-run with --skip-prep --skip-dock --skip-reform when ready."; exit 0; }
fi



#  Scan ./reformed/ for complete complex.xyz files, resolve per-complex charges
#  (host_charge + guest_charge from charges.csv), and write crest_manifest.tsv.
# ═══════════════════════════════════════════════════════════════════════════════
banner "PHASE 4 — Building CREST job manifest"

printf "job_id\thost\tguest\tcomplex_xyz\tcharge\tsolvent\n" > "$MANIFEST"
n_jobs=0

for host_reformed_dir in "$REFORM_DIR"/*/; do
    [[ -d "$host_reformed_dir" ]] || continue
    hname=$(basename "$host_reformed_dir")
    h_charge=$(get_charge "$hname")

    for guest_reformed_dir in "$host_reformed_dir"*/; do
        [[ -d "$guest_reformed_dir" ]] || continue
        gname=$(basename "$guest_reformed_dir")
        complex_xyz="$guest_reformed_dir/complex.xyz"

        if [[ ! -s "$complex_xyz" ]]; then
            warn "  Skipping $hname + $gname — complex.xyz missing or empty"
            continue
        fi

        g_charge=$(get_charge "$gname")
        complex_charge=$(( h_charge + g_charge ))
        (( n_jobs++ )) || true

        printf "%d\t%s\t%s\t%s\t%d\t%s\n" \
            "$n_jobs" "$hname" "$gname" \
            "$(realpath "$complex_xyz")" \
            "$complex_charge" \
            "$CREST_SOLVENT" >> "$MANIFEST"

        info "  [$n_jobs]  $hname + $gname  charge=$complex_charge  solvent=$CREST_SOLVENT"
    done
done

info "Manifest written: $n_jobs job(s) → $MANIFEST"
[[ $n_jobs -gt 0 ]] \
    || { warn "No CREST jobs in manifest — nothing to submit.  Check Phases 2-3."; exit 0; }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — CREST (GFN-FF, --quick)
#  Interactive: run the single pair in the foreground (assumes srun session).
#  Batch:       submit a SLURM array job (one task per manifest line).
# ═══════════════════════════════════════════════════════════════════════════════
banner "PHASE 5 — CREST conformer search"

if [[ "$MODE" == "interactive" ]]; then

    # Extract the one job from the manifest (line 2, since line 1 is the header)
    line=$(sed -n '2p' "$MANIFEST")
    hname=$(       echo "$line" | cut -f2)
    gname=$(       echo "$line" | cut -f3)
    complex_xyz=$( echo "$line" | cut -f4)
    charge=$(      echo "$line" | cut -f5)
    solvent=$(     echo "$line" | cut -f6)

    crest_outdir="$CREST_DIR/${hname}_${gname}"
    mkdir -p "$crest_outdir"

    # Use however many CPUs are actually available in THIS session (respects
    # srun/cgroup limits), capped at CREST_CPUS -- avoids massively
    # oversubscribing a login node or a small srun allocation.
    avail_cpus=$(nproc)
    if (( avail_cpus < CREST_CPUS )); then
        n_threads=$avail_cpus
    else
        n_threads=$CREST_CPUS
    fi
    n_atoms=$(head -1 "$complex_xyz" | tr -d '[:space:]')

    # Solvation: per-pair 'solvent' column takes precedence over global CREST_SOLVENT
    solvent_args=()
    solvent_display=""
    if [[ "$solvent" != "none" && -n "$solvent" ]]; then
        solvent_args=(--alpb "$solvent")
        solvent_display=" --alpb $solvent"
    fi

    echo ""
    echo "  ┌── CHECKPOINT: About to launch CREST ────────────────────────────┐"
    printf "  │  Complex : %s (%s atoms)\n" "$complex_xyz" "$n_atoms"
    printf "  │  Work dir: %s\n" "$crest_outdir"
    printf "  │  CPUs    : %s (of %s available in this session)\n" "$n_threads" "$avail_cpus"
    echo "  │"
    echo "  │  Steps:"
    echo "  │    1. xtb complex.xyz --gfnff --opt crude --chrg $charge -P $n_threads$solvent_display"
    echo "  │       (pre-relax geometry -- fixes minor topology inconsistencies"
    echo "  │        from obabel's -h step before CREST's stricter check runs)"
    echo "  │    2. crest <step1 output> --gfnff --quick --noreftopo --chrg $charge -P $n_threads$solvent_display"
    echo "  │       (--noreftopo: don't reject conformers for topology drift during"
    echo "  │        GFN-FF optimization -- common for large/flexible complexes)"
    echo "  │"
    echo "  │  NOTE: GFN-FF conformer searches on complexes this size can take"
    echo "  │        minutes to hours, with little/no stdout in between -- a"
    echo "  │        quiet terminal is normal, NOT a hang. tail -f"
    echo "  │        $crest_outdir/crest_log.out from another terminal to watch."
    echo "  │"
    echo "  │  For production runs across many pairs, use --mode batch instead"
    echo "  │  (submits a SLURM array job, one task per manifest line)."
    printf "  │  (Assumes you are in an srun interactive session)\n"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    confirm "Launch CREST now (foreground)?" \
        || { info "User skipped CREST.  To run manually, see $MANIFEST."; exit 0; }

    cp "$complex_xyz" "$crest_outdir/complex.xyz"
    pushd "$crest_outdir" > /dev/null

    info "Loading environment..."
    # /etc/bashrc, Lmod's init scripts, and conda's own setup scripts all
    # reference variables (e.g. $BASHRCSOURCED) without guarding for "unset"
    # -- normally harmless, but fatal under `set -u` (which terminates the
    # shell immediately, before `2>/dev/null || true` can help). Relax
    # strict mode for just this block.
    set +Eeuo pipefail
    ml purge              2>/dev/null || true
    ml "$CONDA_MODULE"    2>/dev/null || true
    source "$CONDA_SETUP" 2>/dev/null || true
    conda activate "$CONDA_ENV"
    set -Eeuo pipefail

    # ml/conda setup above can silently `cd` elsewhere (e.g. $HOME) -- re-assert
    # the work directory before running CREST. crest_outdir is absolute (since
    # CREST_DIR was resolved to an absolute path at startup), so this is safe
    # regardless of where activation left us.
    cd "$crest_outdir"

    ulimit -s unlimited
    export OMP_STACKSIZE=4G
    export OMP_NUM_THREADS="$n_threads"
    export OPENBLAS_NUM_THREADS=1

    # CREST/xtb are gfortran-compiled. gfortran fully buffers stdout when it's
    # not a TTY (i.e. whenever piped/redirected, as here) -- output can sit in
    # an internal buffer for a long time before crest_log.out shows anything,
    # even while CREST is genuinely computing. Force immediate flushing:
    export GFORTRAN_UNBUFFERED_ALL=1

    info "CREST version: $(crest --version 2>&1 | head -1)"

    # ── Step 1: pre-optimize with xtb (GFN-FF, crude) ────────────────────────
    # obabel's -h step places new hydrogens at idealised geometric positions
    # with no energy minimisation, so bond lengths/angles (especially around
    # less-common elements) can be slightly off from what GFN-FF's topology
    # perception expects. A quick xtb optimization settles the geometry into
    # something self-consistent first -- CREST's own documented fix (option A)
    # for "Topology change detected" warnings on its first optimization.
    xtb_cmd=(xtb complex.xyz --gfnff --opt crude --chrg "$charge" -P "$n_threads" "${solvent_args[@]}")
    info "Step 1: ${xtb_cmd[*]}"
    crest_input="complex.xyz"
    if stdbuf -oL -eL "${xtb_cmd[@]}" > xtb_preopt.out 2>&1 && [[ -s xtbopt.xyz ]]; then
        info "  -> xtb pre-optimization OK, using xtbopt.xyz as CREST input"
        crest_input="xtbopt.xyz"
    else
        warn "  -> xtb pre-optimization failed or produced no xtbopt.xyz; falling back to complex.xyz (see $crest_outdir/xtb_preopt.out)"
    fi

    # ── Step 2: CREST conformer search ───────────────────────────────────────
    crest_cmd=(crest "$crest_input"
        --gfnff
        --quick
        --chrg "$charge"
        -P     "$n_threads"
        --noreftopo
        "${solvent_args[@]}"
    )

    info "Step 2: ${crest_cmd[*]}"
    info "(quiet for a while after this is expected -- see NOTE above)"

    stdbuf -oL -eL "${crest_cmd[@]}" | tee crest_log.out

    popd > /dev/null
    info "CREST complete.  Results in $crest_outdir"

else
    # ── Batch mode: SLURM array job ───────────────────────────────────────────

    info "Submitting SLURM array job ($n_jobs tasks)..."

    sbatch_cmd=(sbatch
        --job-name="dock2crest"
        --array="1-${n_jobs}"
        --ntasks=1
        --cpus-per-task="$CREST_CPUS"
        --mem="$CREST_MEM"
        --time="$CREST_TIME"
        --partition="$CREST_PARTITION"
        --output="${CREST_DIR}/slurm_%A_%a.out"
        --error="${CREST_DIR}/slurm_%A_%a.err"
        --export="ALL,\
MANIFEST=$(realpath "$MANIFEST"),\
CREST_DIR=$(realpath "$CREST_DIR"),\
CONDA_ENV=$CONDA_ENV,\
CONDA_MODULE=$CONDA_MODULE,\
CONDA_SETUP=$CONDA_SETUP,\
CREST_SOLVENT=$CREST_SOLVENT"
        "$CREST_WORKER"
    )

    info "sbatch command: ${sbatch_cmd[*]}"

    job_out=$("${sbatch_cmd[@]}" 2>&1) \
        || die "sbatch failed: $job_out"

    job_id=$(echo "$job_out" | grep -oP '(?<=Submitted batch job )\d+' || echo "unknown")

    info "Submitted: array job ID $job_id  ($n_jobs tasks)"
    info "Monitor:   squeue -j $job_id"
    info "Logs:      ${CREST_DIR}/slurm_${job_id}_<task>.out"
    info "Results:   $CREST_DIR/<host>_<guest>/"

fi  # end interactive/batch


# ═══════════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════════
banner "Pipeline complete"
info "Full log: $RUN_LOG"
info "Reformed complexes: $REFORM_DIR"
info "CREST results:      $CREST_DIR"
