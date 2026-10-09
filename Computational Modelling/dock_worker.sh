#!/bin/bash
###############################################################################
# dock_worker.sh — SLURM array task for batch Vina docking.
#
# One task = one host-guest pair from dock_manifest.tsv (written by
# pipeline.sh Phase 2 in batch mode). Submitted as:
#
#   sbatch --array=1-N --export=ALL,MANIFEST=...,CONDA_ENV=...,... dock_worker.sh
#
# Required env vars (set via --export by pipeline.sh):
#   MANIFEST       absolute path to dock_manifest.tsv
#   CONDA_ENV      conda environment name (must contain `vina`)
#   CONDA_MODULE   module to load before sourcing conda setup (e.g. Anaconda3/...)
#   CONDA_SETUP    path to the cluster's conda.env.sh
#
###############################################################################
set -Eeuo pipefail

ts()  { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] TASK=${SLURM_ARRAY_TASK_ID:-?} $*"; }
die() { log "FATAL: $*" >&2; exit 1; }

on_err() {
    local ec=$?
    log "ERROR at line $1: $2  (exit=$ec)" >&2
    exit $ec
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

###############################################################################
# Validate required env vars (injected by pipeline.sh via sbatch --export=ALL,…)
###############################################################################
: "${MANIFEST:?MANIFEST env var not set — was this submitted via pipeline.sh?}"
: "${CONDA_ENV:?CONDA_ENV env var not set}"
: "${CONDA_MODULE:?CONDA_MODULE env var not set}"
: "${CONDA_SETUP:?CONDA_SETUP env var not set}"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — must run as SLURM array job}"

log "Worker started  (array_job=${SLURM_ARRAY_JOB_ID:-?}  task=${task_id})"
log "Manifest      : $MANIFEST"
log "Conda env     : $CONDA_ENV"
log "Conda module  : $CONDA_MODULE"

###############################################################################
# Read manifest line for this task
#   Header = line 1  →  task 1 data is at line 2  →  NR == task_id + 1
###############################################################################
[[ -f "$MANIFEST" ]] || die "Manifest not found: $MANIFEST"

manifest_line=$(awk -v t="$task_id" 'NR == t+1' "$MANIFEST")
[[ -n "$manifest_line" ]] || die "No manifest entry for task $task_id (manifest has $( wc -l < "$MANIFEST" ) lines)"

# Parse tab-separated columns
IFS=$'\t' read -r job_id host guest receptor ligand config out_pdbqt log_file <<< "$manifest_line"

log "job_id=$job_id  host=$host  guest=$guest"
log "receptor : $receptor"
log "ligand   : $ligand"
log "config   : $config"
log "out_pdbqt: $out_pdbqt"

###############################################################################
# Validate inputs, handle idempotent re-runs
###############################################################################
for f in "$receptor" "$ligand" "$config"; do
    [[ -s "$f" ]] || die "Required input missing or empty: $f"
done

if [[ -s "$out_pdbqt" ]]; then
    log "Output already exists (idempotent re-run) — skipping: $out_pdbqt"
    exit 0
fi

mkdir -p "$(dirname "$out_pdbqt")"

###############################################################################
# Load conda environment (same set -u workaround as crest_worker.sh)
###############################################################################
# /etc/bashrc, Lmod's init scripts, and conda's own setup scripts all reference
# variables (e.g. $BASHRCSOURCED) without guarding for "unset" -- normally
# harmless (bash treats unset as empty), but fatal under `set -u`. Relax
# strict mode for just this block, then restore it for the actual Vina run.
set +Eeuo pipefail

ml purge
ml "$CONDA_MODULE"
# Initialise the conda shell function. Older clusters sourced a static
# setup script (CONDA_SETUP); newer conda versions need the shell hook
# evaluated instead. Try the hook first, fall back to the static path.
if conda shell.bash hook &>/dev/null; then
    # shellcheck disable=SC2046
    eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_SETUP" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SETUP"
else
    echo "WARNING: could not initialise conda shell function — trying conda activate anyway" >&2
fi
conda activate "$CONDA_ENV"

set -Eeuo pipefail

log "Environment loaded: $CONDA_ENV"
log "Vina version: $(vina --version 2>&1 | head -1 || echo unknown)"

###############################################################################
# Run Vina
###############################################################################
n_cpus="${SLURM_CPUS_PER_TASK:-1}"

vina_cmd=(vina --receptor "$receptor" --ligand "$ligand" --config "$config" \
               --out "$out_pdbqt" --cpu "$n_cpus")
log "Vina command: ${vina_cmd[*]}"

start_epoch=$(date +%s)

if "${vina_cmd[@]}" > "$log_file" 2>&1; then
    if [[ -s "$out_pdbqt" ]]; then
        # Pull top-1 score out of the Vina log/output
        top_score=$(awk '/VINA RESULT/{print $4; exit}
                         /minimizedAffinity/{print $2; exit}' \
                        "$out_pdbqt" "$log_file" 2>/dev/null || echo "unknown")
        elapsed=$(( $(date +%s) - start_epoch ))
        log "OK -- top score: ${top_score} kcal/mol  (${elapsed}s)  -> $out_pdbqt"
    else
        log "WARNING: Vina exited 0 but produced no/empty output (no poses found?) -- see $log_file"
        log "         This pair will simply be absent from Phase 3 (reform skips missing docked poses)."
    fi
else
    die "Vina failed for $host + $guest (see $log_file)"
fi

log "Task $task_id complete."
