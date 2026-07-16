#!/bin/bash
#SBATCH --job-name=lrb_dem_single_si1000
#SBATCH --output=lrb_dem_single_si1000_%j.out
#SBATCH --error=lrb_dem_single_si1000_%j.err
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH -t 47:30:00
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --signal=B:USR1@600

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -n "${SUBMIT_DIR}" && -f "${SUBMIT_DIR}/slurm/run_lrb_si1000_all_checks_folded.sh" ]]; then
    SCRIPT_DIR="${SUBMIT_DIR}/slurm"
fi

# This is the single-ancilla counterpart of
# run_lrb_dem_si1000_all_checks_split_folded.sh. Generate the input circuits
# with the exact production metadata used by the split-ancilla DEM run, while
# selecting the conventional folded profile where every stabilizer has one
# measurement ancilla:
#
# python3 scripts/generate_circuits_folded.py \
#     --noise-model si1000 \
#     --ancilla-mode single \
#     --custom-name single-ancilla-si1000 \
#     --n-cliff 30 \
#     --depths 0,2,4,6,10,14,18,20,22 \
#     --n-shots 1000000 \
#     --probabilities 0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.04,0.05 \
#     --stab-checks-const 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
#     --stab-checks-unif 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
#
# Then submit this launcher:
#
# sbatch slurm/run_lrb_dem_si1000_all_checks_folded.sh
#
# After every LRB probability index is complete, generate the matched physical
# RB grid required by the DEM notebook:
#
# sbatch slurm/run_lrb_rb_comparison_folded.sh
#
# DEM only changes the circuit-sampling backend. The generated run retains the
# same SI1000 model, depths, probability grid, shot count, Clifford count, and
# const/unif postselection policies as the split-ancilla run. The code profile
# changes from folded_qutrit_split_unif to folded_qutrit, whose stabilizer
# checks use one ancilla per stabilizer.
export LRB_SIMULATION_BACKEND="dem"
export LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-100000}"
export LRB_RESUBMIT_SCRIPT="${LRB_RESUBMIT_SCRIPT:-${SCRIPT_DIR}/run_lrb_dem_si1000_all_checks_folded.sh}"
export REQUIRE_DETECTORS="${REQUIRE_DETECTORS:-1}"

echo "Submitting single-ancilla folded SI1000 LRB with DEM sampling."
echo "Stabilizer geometry: one measurement ancilla per stabilizer."
echo "Postselection path: unchanged const/unif LRB postprocessing."
echo "Next stage after completion: run_lrb_rb_comparison_folded.sh"
echo "Checkpoint batch size: ${LRB_BATCH_SIZE}"
echo "Delegating validation and srun launch to run_lrb_si1000_all_checks_folded.sh"

exec bash "${SCRIPT_DIR}/run_lrb_si1000_all_checks_folded.sh"
