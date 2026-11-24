# SLURM launchers

Submit from the repository root.

| launcher | use |
|---|---|
| `run_lrb_dem_circuit_level.sh` | **current.** One node runs every probability index of an SI1000 or SD6 run concurrently, resumes from checkpoints and resubmits itself before its time limit. `sbatch --export=ALL,RUN_NAME_OVERRIDE=<Run-folder> slurm/run_lrb_dem_circuit_level.sh` |
| `run_lrb_dem_si1000_all_checks_folded.sh` | August 2026 one-ancilla SI1000 run (sdim 1.3.4, `lambda = 5p` axis) |
| `run_lrb_dem_si1000_all_checks_split_folded.sh` | June 2026 relay two-ancilla SI1000 DEM run |
| `run_lrb_si1000_all_checks_folded.sh`, `run_lrb_si1000_all_checks_split_folded.sh`, `run_lrb_si1000_all_checks_qgrm.sh` | May to June 2026 SI1000 runs before the DEM backend |
| `run_lrb_rb_comparison_folded.sh`, `run_lrb_rb_comparison_split_folded.sh` | physical RB at `p_RB = 5p` for the August and June runs |
| `run_lrb_slurm_folded.sh`, `run_lrb_slurm_qgrm.sh` | February 2026 depolarizing runs |

The memory study of the folded surface code has its own jobs in
`layouts/folded-surface-code/memory/`.
