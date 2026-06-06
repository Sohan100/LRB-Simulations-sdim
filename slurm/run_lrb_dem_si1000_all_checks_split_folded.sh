#!/bin/bash
#SBATCH --job-name=lrb_dem_split_si1000
#SBATCH --output=lrb_dem_split_si1000_%j.out
#SBATCH --error=lrb_dem_split_si1000_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate the detector-annotated input circuits first, for example:
#
# python3 scripts/generate_circuits_folded.py \
#     --noise-model si1000 \
#     --ancilla-mode split-unif \
#     --custom-name split-unif-si1000
#
# Then submit this launcher:
#
# sbatch slurm/run_lrb_dem_si1000_all_checks_split_folded.sh
#
# Optional environment overrides, such as RUN_NAME_OVERRIDE, NUM_SHOTS,
# LRB_RUNS_ROOT, LRB_PYTHON, DRY_RUN, and REQUIRE_SPLIT_SI1000_PARAMETERS,
# are passed through to the underlying all-check split-folded launcher.
#
# DEM only changes the circuit-sampling backend. The downstream
# run_lrb_experiment.py path is the same one used by SDIM runs, so nonzero
# constant checks, const=0 direct-X checks, and uniform-interval checks are
# postprocessed with the existing LRB machinery.
export LRB_SIMULATION_BACKEND="dem"
export REQUIRE_DETECTORS="${REQUIRE_DETECTORS:-1}"

echo "Submitting split-folded SI1000 LRB with DEM sampling."
echo "Postselection path: unchanged const/unif LRB postprocessing."
echo "Delegating validation and srun launch to run_lrb_si1000_all_checks_split_folded.sh"

exec bash "${SCRIPT_DIR}/run_lrb_si1000_all_checks_split_folded.sh"
