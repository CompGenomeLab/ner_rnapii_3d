# ner_rnapii_3d

A standalone **1D loop-extrusion + RNAPII + lesion simulation engine**. Given a
YAML config it simulates, on a 1D lattice:

- **cohesin loop extrusion** with CTCF capture/release boundaries,
- **RNAPII transcription** (load → pre-initiation → elongation → termination,
  with enhancer–promoter gating and cohesin↔RNAPII steric coupling),
- **DNA lesions** (occurrence in gene bodies, pre-recognition / repair kinetics,
  cohesin and RNAPII blocking),

and writes the trajectory to `LEFPositions.h5`.

This is a focused **1D-only** engine: there is no 3D polymer, contact-map, QC or
viewer stage. A config carries a single `lef:` section (see
`scripts/gen_realistic_configs_variable_tick.py`); any other top-level sections
are ignored.

## Determinism

The dynamics run on a Numba kernel that embeds a port of NumPy's legacy MT19937
generator, so a given seed yields a **byte-identical** trajectory — independent
of the installed numpy/numba versions. The recording is streamed to HDF5 in
memory-bounded blocks, so arbitrarily long/large configs run without holding the
full trajectory in RAM.

## Output & compression

The trajectory is written to `LEFPositions.h5`. The RNAPII / lesion datasets are
sized to their steric/homeostatic ceiling, so they are mostly `-1` padding — which
compresses extremely well. The HDF5 filter is configurable in the `lef:` section
(lossless — the data is byte-identical regardless):

```yaml
lef:
  compression: gzip        # gzip | lzf | none
  compression_level: 4     # gzip level 1-9 (4 ≈ size/speed knee)
  shuffle: true            # byte-shuffle filter (~15% smaller, ~free)
```

The default `gzip` level 4 + shuffle is **~5× smaller on disk than lzf** with
comparable read speed (gzip is also a standard, portable HDF5 filter); on a
representative RNAPII run the recording dropped from 36 MB (lzf) to 7.5 MB. Use
`lzf` (or a low gzip level) if you want faster writes, `none` for fastest writes
at full size.

## Install / run (uv)

```bash
uv sync
uv run ner-sim run path/to/config.yaml --output run_dir/
# -> run_dir/LEFPositions.h5
```

Generate ready-to-run configs with the bundled generator:

```bash
uv run python scripts/gen_realistic_configs_variable_tick.py \
  --chain 40000 --bp-per-site 250 --num-chains 20 --suffix demo \
  --out-dir configs/ --seed 2026 --tick-seconds 2 \
  --trajectory-length 80000 --warmup-steps 40000
```

It writes 1D-only `lef:` configs whose plugin targets point at this package
(`ner_rnapii_3d.plugins.*`). `run_panels.sh` drives the full set of analysis and
parameter-sweep panels end-to-end.

## Python API

```python
import ner_rnapii_3d as ner

cfg = ner.load_lef_config("config.yaml", output_path="run_dir/")
h5_path = ner.run(cfg)
```

## Layout

```
ner_rnapii_3d/
  config.py            typed config, plugin resolution (1D only)
  cli.py               `ner-sim` command-line interface
  engine/
    rng.py             MT19937 port (reference class + njit free-functions)
    kernel.py          Numba dynamics kernels + chunked recording driver
    simulate.py        run(cfg) -> LEFPositions.h5 (+ Python reference oracle)
    progress.py        progress/ETA logging
  plugins/
    topology.py        builds the simulation args (lattice, CTCF, genes, lesions)
    lef_dynamics.py    Cohesin/Leg data classes + reference cohesin functions
    rnapii.py          Gene/RNAPII data classes + reference transcription functions
    lesions.py         Lesion data class + reference lesion functions
```

The `plugins/` modules carry the data classes and the **Python reference
implementation** of every per-tick rule; the production tick loop reimplements
that logic inside `engine/kernel.py`, and the reference functions remain as the
equivalence oracle (`tests/`).

## Tests

```bash
uv run pytest
```

A self-validating suite (no frozen golden hashes):

- `test_equivalence.py` — the production Numba kernel reproduces the Python
  reference implementation **tick-for-tick** (byte-identical recordings) across
  every regime (cohesin / RNAPII / lesion / combined) and several seeds. This is
  the core correctness guarantee: two independent implementations of the same
  dynamics are compared, so any divergence is caught at its source.
- `test_rng.py` — the embedded MT19937 port is bit-for-bit identical to NumPy's
  legacy `RandomState` (reference class **and** the `@njit` functions).
- `test_chunking.py` — block size and warmup splitting never change the output.
- `test_config.py` — config parsing, plugin resolution, validation.
- `test_run_output.py` — `run()` HDF5 datasets / shapes / dtypes / attrs +
  determinism (same seed identical, different seed differs).
- `test_invariants.py` — physical invariants (cohesin in-bounds & non-crossing,
  RNAPII states/uids valid, lesion population bounded, …).
- `test_eligibility.py` — `numba_eligible` accepts the default plugin set and
  rejects unsupported configs.

The shared harness lives in `tests/conftest.py`.
