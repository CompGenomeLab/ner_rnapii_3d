
# Env param
nproc=42 # Number of parallel processes during parameter search (Running multiple simulations in parallel). Adjust based on your CPU cores.
seed=2026 # Random seed for config generation. Change for different runs. Keep for paper reproducibility.

# Simulation parameters
suffix=10k_4sec # Suffix for output folder and config files. Adjust to your liking.
bp_per_site=250 # Lattice resolution: bp per site. Drives chain length, lesion density, all distances.
chainlength=40000 # Number of sites on lattice, each 250bp. So 40k = 10Mb.
warmup_steps=40000 ### Warmup for equilibration, prior to production (not recorded).
trajectory_length=160000 ### Production simulation (recorded)
tickseconds=4 # Number of seconds per simulation tick. At 250bp sites, a free cohesin extrudes 2 sites/tick = 500bp/tick, so 2s/tick = 250bp/s extrusion speed.
nreplicates=20 # Simulation replicates. All run on the same topology. e.g alleles.
bstr_mult=3.25 # Boundary strength multiplier, increases the probability of LEF stalling at CTCF sites. One time control. It is not checked per simulation tick (resolution-independent).
typea_prob=0.20 # Type A lesion probability, i.e. if a lesion spawns on a gene body, what ratio of those lesions are Type A.
blockProb=0.993 # Probability of Type A (during pre-repair + repair) and Type B (during repair) lesions blocking cohesin extrusion. Evaluated per tick.
prerecognition_seconds=2400 # a.k.a pre-repair seconds, per tick probabilistic. Average time for a lesion to be recognized (and thus start repair). Evaluated per tick. 2400s = 40 minutes average time to recognition.
repair_seconds=360 # Per tick probabilistic. Average time for a lesion to be repaired (after recognition). Evaluated per tick. 360s = 6 minutes average time to repair.


####
outFolder=results_${suffix}
mkdir -p ${outFolder}
mkdir -p configs

uv run python scripts/gen_realistic_configs_variable_tick.py --chain ${chainlength} --bp-per-site ${bp_per_site} --num-chains ${nreplicates} --suffix ${suffix} --out-dir configs/ --seed ${seed} --tick-seconds ${tickseconds} --trajectory-length ${trajectory_length} --warmup-steps ${warmup_steps}

# Run the two configs for comparison Exp.1 vs Baseline
uv run python scripts/compare_config_chain_metrics.py \
      --config1 configs/config1_${suffix}.yaml \
      --config2 configs/config2_${suffix}.yaml \
      --out-dir ${outFolder}/${suffix}_12 \
      --label1 Baseline \
      --label2 'Exp. 1'

# #### Transcription and RNAPII~cohesin QC --- START #####

# Calculate transcription metrics
uv run python scripts/gen_transcription_metrics.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/${suffix}_Baseline_TX

# Cohesin moving-barrier evaluation (Banigan 2023): cohesin accumulation around
# genes, transcription ON vs an auto-derived RNAPII-OFF control (ON-OFF difference)
uv run python scripts/gen_cohesin_barrier_eval.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/${suffix}_Baseline_cohesin_barrier

### Transcription and RNAPII~cohesin QC --- END #####

### Parameter sweeps --- START #####

# Boundary strength sweep (boundary strength multiplier)
uv run python scripts/sweep_rnapoff_boundary_strength_1d.py \
  --config1 configs/config1_${suffix}.yaml \
  --h5-config1 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --multipliers 1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 \
  --out-dir ${outFolder}/boundary_sweep_${suffix} \
  --jobs $nproc

# # Type A probability sweep (TYPE A probability vs lesion density)
uv run python scripts/gen_typea_density_grid.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/typea_density_grid_${suffix} \
  --ta-step 0.05 --ta-lo 0.05 --ta-hi 0.5 \
  --spacings 28 32 40 44 52 56 68 80 100 132 200 400 \
  --bstr-mult ${bstr_mult} \
  --block-prob ${blockProb} \
  --repair-seconds ${repair_seconds} \
  --prerecognition-seconds ${prerecognition_seconds} \
  --jobs ${nproc}

# Block probability sweep (block probability vs lesion density)
# block_prob is a PER-TICK probability; range rescaled for 2s ticks (1-(1-p)/4) to
uv run python scripts/gen_lesion_grid_and_heatmaps.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/block_prob_grid_typea01_${suffix} \
  --p-step 0.0025 --p-lo 0.980 --p-hi 1.0 \
  --spacings 28 32 40 44 52 56 68 80 100 132 200 400 \
  --repair-seconds ${repair_seconds} \
  --prerecognition-seconds ${prerecognition_seconds} \
  --bstr-mult ${bstr_mult} \
  --type-a-prob ${typea_prob} \
  --jobs ${nproc}

# # recognition (pre-repair) and repair sweep in seconds (seconds vs lesion density)
uv run python scripts/gen_lesion_timing_grids.py \
  --config configs/config1_${suffix}.yaml \
  --h5 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/rec_rep_timing \
  --type-a-prob ${typea_prob} \
  --prerecog-lo 300 --prerecog-hi 7200 --prerecog-step 300 \
  --repair-fixed ${repair_seconds} \
  --repair-lo 100 --repair-hi 1000 --repair-step 50 \
  --prerecog-fixed ${prerecognition_seconds} \
  --spacings 28 32 40 44 52 56 68 80 100 132 200 400 \
  --bstr-mult ${bstr_mult} --block-prob ${blockProb} \
  --jobs ${nproc}

# #### Parameter sweeps --- END #####

# #### Free cohesin lifetime vs Cohesin stalled at CTCF sweep --- START #####

uv run python scripts/sweep_lifetime_cohesin_vs_ctcf_2d.py \
  --config1 configs/config1_${suffix}.yaml --h5-config1 ${outFolder}/${suffix}_12/Baseline/LEFPositions.h5 \
  --out-dir ${outFolder}/results_lifetime_grid \
  --cohesin-lifetime-multipliers 0.125,0.25,0.5,1,2,4,6 --ctcf-lifetime-multipliers 0.125,0.25,0.5,1,2,4,6 \
  --jobs ${nproc}

# #### Free cohesin lifetime vs Cohesin stalled at CTCF sweep --- END #####
