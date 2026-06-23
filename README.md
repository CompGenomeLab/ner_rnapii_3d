# ner_rnapii_3d

A standalone **1D loop-extrusion + RNAPII + lesion simulation engine**. Given a
YAML config it simulates, on a 1D lattice:

- **cohesin loop extrusion** with CTCF capture/release boundaries,
- **RNAPII transcription** (load → pre-initiation → elongation → termination, with
  enhancer–promoter gating and cohesin↔RNAPII steric coupling),
- **DNA lesions** (gene-body occurrence, pre-recognition / repair kinetics, cohesin
  and RNAPII blocking),

and writes the trajectory to `LEFPositions.h5`.

## Install / run (uv)

```bash
uv sync
uv run ner-sim run path/to/config.yaml --output run_dir/
# -> run_dir/LEFPositions.h5
```

Generate a ready-to-run config with the bundled generator:

```bash
uv run python scripts/gen_realistic_configs_variable_tick.py \
  --chain 40000 --bp-per-site 250 --num-chains 20 --suffix demo \
  --out-dir configs/ --seed 2026 --tick-seconds 2 \
  --trajectory-length 80000 --warmup-steps 40000
```

Please see the script for the detailed parameter descriptions.

## Reproduce the paper results

```bash
uv sync
bash run_panels.sh
```

`run_panels.sh` regenerates the configs and runs the full pipeline end-to-end —
the transcription-on vs -off comparison, transcription and cohesin-barrier QC, and
the boundary-strength, Type-A/density, block-probability, repair-timing, and
cohesin–CTCF lifetime sweeps — writing everything to `results_<suffix>/`. Adjust
`nproc` (parallel jobs) and `seed` at the top of the script; the defaults reproduce
the published runs.

## Python API

```python
import ner_rnapii_3d as ner

cfg = ner.load_lef_config("config.yaml", output_path="run_dir/")
h5_path = ner.run(cfg)
```

## Layout

```
ner_rnapii_3d/
  config.py    typed config, plugin resolution
  cli.py       `ner-sim` CLI
  engine/      MT19937 port + Numba dynamics kernels + chunked recording
  plugins/     data classes + Python reference rules (topology, cohesin,
               RNAPII, lesions)
```

The `plugins/` modules hold the **Python reference implementation** of every
per-tick rule; the production loop reimplements that logic in `engine/kernel.py`,
with the reference functions kept as the equivalence oracle.

## Tests

```bash
uv run pytest
```
