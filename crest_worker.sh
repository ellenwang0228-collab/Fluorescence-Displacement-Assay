#!/usr/bin/env bash

#SBATCH --job-name=crest_gfnff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=ncpu
#SBATCH --output=crest_logs/array_%A_%a.out
#SBATCH --error=crest_logs/array_%A_%a.err

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

: "${MANIFEST:?MANIFEST env var not set — was this submitted via pipeline.sh?}"
: "${CREST_DIR:?CREST_DIR env var not set}"
: "${CONDA_ENV:?CONDA_ENV env var not set}"
: "${CONDA_MODULE:?CONDA_MODULE env var not set}"
: "${CONDA_SETUP:?CONDA_SETUP env var not set}"
CREST_SOLVENT="${CREST_SOLVENT:-none}"   # global default; per-pair solvent column overrides

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — must run as SLURM array job}"

log "Worker started  (array_job=${SLURM_ARRAY_JOB_ID:-?}  task=${task_id})"
log "Manifest      : $MANIFEST"
log "CREST root    : $CREST_DIR"
log "Conda env     : $CONDA_ENV"
log "Conda module  : $CONDA_MODULE"


[[ -f "$MANIFEST" ]] || die "Manifest not found: $MANIFEST"

manifest_line=$(awk -v t="$task_id" 'NR == t+1' "$MANIFEST")
[[ -n "$manifest_line" ]] || die "No manifest entry for task $task_id (manifest has $( wc -l < "$MANIFEST" ) lines)"

# Parse tab-separated columns
IFS=$'\t' read -r job_id host guest complex_xyz charge solvent <<< "$manifest_line"

log "job_id=$job_id  host=$host  guest=$guest  charge=$charge  solvent=$solvent"
log "complex_xyz  : $complex_xyz"

###############################################################################
# Validate input xyz
###############################################################################
[[ -f "$complex_xyz" ]] || die "complex.xyz not found: $complex_xyz"
[[ -s "$complex_xyz" ]] || die "complex.xyz is empty: $complex_xyz"

n_atoms=$(head -1 "$complex_xyz" | tr -d '[:space:]')
[[ "$n_atoms" =~ ^[1-9][0-9]*$ ]] || \
    die "complex.xyz has invalid atom-count header line: '$n_atoms'"

###############################################################################
# Set up work directory
###############################################################################

work_dir="$(realpath -m "${CREST_DIR}/${host}_${guest}")"
mkdir -p "$work_dir"

cp "$complex_xyz" "${work_dir}/complex.xyz"

cd "$work_dir"
log "Working directory: $PWD"


ulimit -s unlimited

set +Eeuo pipefail

ml purge
ml "$CONDA_MODULE"
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

cd "$work_dir"
log "Working directory (re-asserted): $PWD"

log "Environment loaded: $CONDA_ENV"
# Print CREST version for reproducibility; tolerate if --version isn't supported
crest --version 2>&1 | head -5 || true

export OMP_STACKSIZE=4G
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS=1
export GFORTRAN_UNBUFFERED_ALL=1

log "OMP_NUM_THREADS=$OMP_NUM_THREADS  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}"
n_threads="${SLURM_CPUS_PER_TASK:-64}"
eff_solvent="${solvent:-$CREST_SOLVENT}"
eff_solvent="${eff_solvent:-none}"

solvent_args=()
if [[ "$eff_solvent" != "none" && -n "$eff_solvent" ]]; then
    solvent_args=(--alpb "$eff_solvent")
    log "Implicit solvation: --alpb $eff_solvent"
else
    log "Implicit solvation: none (gas phase)"
fi


# optimization step.
xtb_cmd=(xtb complex.xyz --gfnff --opt crude --chrg "$charge" -P "$n_threads" "${solvent_args[@]}")
log "xtb pre-opt command: ${xtb_cmd[*]}"

crest_input="complex.xyz"
if stdbuf -oL -eL "${xtb_cmd[@]}" > xtb_preopt.out 2>&1 && [[ -s xtbopt.xyz ]]; then
    log "xtb pre-optimization OK -- using xtbopt.xyz as CREST input"
    crest_input="xtbopt.xyz"
else
    log "WARNING: xtb pre-optimization failed or produced no xtbopt.xyz -- falling back to complex.xyz (see xtb_preopt.out)"
fi

crest_cmd=(
    crest "$crest_input"
    --gfnff
    --quick
    --chrg "$charge"
    -P "$n_threads"
    --noreftopo
    "${solvent_args[@]}"
)

log "CREST command: ${crest_cmd[*]}"

###############################################################################
# Run CREST
###############################################################################
start_epoch=$(date +%s)

if ! stdbuf -oL -eL "${crest_cmd[@]}" > crest_log.out 2>&1; then
    log "CREST FAILED — last 40 lines of crest_log.out:" >&2
    tail -40 crest_log.out >&2
    # Write a failure summary so harvest_results.sh can flag it
    {
        echo "job_id        : $job_id"
        echo "host          : $host"
        echo "guest         : $guest"
        echo "n_atoms       : $n_atoms"
        echo "charge        : $charge"
        echo "solvent       : $eff_solvent"
        echo "status        : FAILED"
        echo "slurm_job     : ${SLURM_ARRAY_JOB_ID:-?}_${task_id}"
        echo "work_dir      : $work_dir"
    } > crest_summary.txt
    exit 1
fi

end_epoch=$(date +%s)
elapsed=$(( end_epoch - start_epoch ))
log "CREST completed in ${elapsed}s"

conformers_xyz="crest_conformers.xyz"

if [[ ! -s "$conformers_xyz" ]]; then
    log "WARNING: $conformers_xyz missing or empty — 0 conformers found"
    n_conformers=0
else
    total_lines=$(wc -l < "$conformers_xyz")
    block_size=$(( n_atoms + 2 ))
    n_conformers=$(( total_lines / block_size ))
    log "Conformers: $n_conformers  (${total_lines} lines ÷ ${block_size} lines/block)"
fi

summary_file="crest_summary.txt"
{
    echo "job_id        : $job_id"
    echo "host          : $host"
    echo "guest         : $guest"
    echo "complex_xyz   : $complex_xyz"
    echo "n_atoms       : $n_atoms"
    echo "charge        : $charge"
    echo "solvent       : $eff_solvent"
    echo "crest_cmd     : ${crest_cmd[*]}"
    echo "elapsed_s     : $elapsed"
    echo "n_conformers  : $n_conformers"
    echo "slurm_job     : ${SLURM_ARRAY_JOB_ID:-?}_${task_id}"
    echo "work_dir      : $work_dir"
    echo "status        : OK"
} > "$summary_file"

log "Summary → $summary_file"
log "DONE: $host + $guest  —  $n_conformers conformer(s) in ${elapsed}s"
