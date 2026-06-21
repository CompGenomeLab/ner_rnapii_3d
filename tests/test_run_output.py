"""End-to-end ``ner_rnapii_3d.run()`` + the HDF5 trajectory it writes.

``run()`` IS the production kernel path: it resolves the topology, runs the Numba
engine for ``cfg.warmup_steps`` warmup ticks (no recording) then ``cfg.trajectory_length``
recorded ticks, and streams the result to ``cfg.output_path`` as ``LEFPositions.h5``.

This module is the *output contract* for that file -- complementary to
``test_equivalence.py`` (which proves the kernel == the Python reference tick-for-tick).
Here we assert, per regime:

* exactly the datasets that regime should contain exist, with the right ``(traj, ...)``
  shapes and dtypes (read straight off ``simulate.run``);
* the self-describing attributes (N, LEFNum, chain_length, num_chains, bp_per_site,
  tick_seconds, rnapii_enabled, lesion_enabled, ...) match the config;
* determinism: same cfg+seed -> byte-identical ``positions``; a different seed differs;
* ``run()``'s ``positions`` equals ``conftest.kernel_record(cfg, warmup, traj)['coh']``
  for the warmup/traj/seed ``run()`` itself uses (they are the same code path).

Configs are kept tiny (short trajectory, few warmup ticks) so a full run() per regime
is fast on the already-compiled kernel.
"""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

import ner_rnapii_3d as pkg
from conftest import kernel_record

# Small but non-trivial run params. run() reads these off the cfg, so we pin them
# on the cfg AND reuse the same numbers when calling kernel_record for the
# positions cross-check.
WARMUP = 5
TRAJ = 12
SEED = 5

REGIMES = ["cohesin", "rnapii", "lesion", "combined"]


def _prep(cfg, tmp_path, *, seed=SEED, warmup=WARMUP, traj=TRAJ, name="run.h5"):
    """Deep-copy a regime fixture and pin small, deterministic run params + output."""
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    cfg.warmup_steps = warmup
    cfg.trajectory_length = traj
    cfg.output_path = str(tmp_path / name)
    return cfg


def _run(cfg):
    out = pkg.run(cfg)
    # run() returns the Path it wrote and it must equal the requested output_path.
    assert str(out) == cfg.output_path
    return out


# --------------------------------------------------------------------------- #
# dataset presence + shape + dtype, per regime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", REGIMES)
def test_datasets_present_with_correct_shape_and_dtype(all_regime_cfgs, regime, tmp_path):
    """Each regime writes exactly the datasets it should, at ``(traj, ...)`` with the
    dtypes declared in simulate.run (positions int32; rnapii_states/lesion_* int8; ...)."""
    cfg = _prep(all_regime_cfgs[regime], tmp_path, name=f"{regime}.h5")
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None
    n_lefs = cfg.num_lefs
    rcap = cfg.max_rnapii * cfg.num_chains

    out = _run(cfg)
    with h5py.File(out, "r") as fh:
        # positions: always, (traj, num_lefs, 2) int32.
        assert "positions" in fh
        pos = fh["positions"]
        assert pos.shape == (TRAJ, n_lefs, 2)
        assert pos.dtype == np.int32

        rnapii_dsets = {"rnapii_positions", "rnapii_states", "rnapii_ids"}
        lesion_dsets = {"lesions", "lesion_types", "lesion_states"}

        if rnapii:
            assert rnapii_dsets <= set(fh.keys())
            # The RNAPII columns are resized down to the actual max live width
            # (<= the steric ceiling rnapii_cap); the dropped columns were all -1.
            w = fh["rnapii_positions"].shape[1]
            assert 1 <= w <= rcap
            assert fh["rnapii_positions"].shape == (TRAJ, w, 2)
            assert fh["rnapii_positions"].dtype == np.int32
            assert fh["rnapii_states"].shape == (TRAJ, w)
            assert fh["rnapii_states"].dtype == np.int8
            assert fh["rnapii_ids"].shape == (TRAJ, w)
            assert fh["rnapii_ids"].dtype == np.int32
            # the steric ceiling is still recorded as an attribute.
            assert int(fh.attrs["rnapii_cap"]) == rcap
            # genes table accompanies the RNAPII regime (gene-aware topology).
            assert "genes" in fh
            assert fh["genes"].shape[0] > 0
        else:
            # No RNAPII datasets must leak into a cohesin/lesion-only file.
            assert rnapii_dsets.isdisjoint(fh.keys())
            assert "genes" not in fh

        if lesion:
            assert lesion_dsets <= set(fh.keys())
            lcap = int(fh.attrs["lesion_target"])
            assert lcap > 0
            assert fh["lesions"].shape == (TRAJ, lcap)
            assert fh["lesions"].dtype == np.int32
            assert fh["lesion_types"].shape == (TRAJ, lcap)
            assert fh["lesion_types"].dtype == np.int8
            assert fh["lesion_states"].shape == (TRAJ, lcap)
            assert fh["lesion_states"].dtype == np.int8
        else:
            assert lesion_dsets.isdisjoint(fh.keys())


@pytest.mark.parametrize("regime", REGIMES)
def test_ctcf_anchors_present_for_convergent_topology(all_regime_cfgs, regime, tmp_path):
    """The regime fixtures use a convergent (gene-aware) topology, which exposes a
    per-TAD oriented ``ctcf_anchors`` table -> written with the documented struct dtype."""
    cfg = _prep(all_regime_cfgs[regime], tmp_path, name=f"{regime}_anc.h5")
    out = _run(cfg)
    with h5py.File(out, "r") as fh:
        assert "ctcf_anchors" in fh
        anc = fh["ctcf_anchors"][...]
        assert anc.shape[0] > 0
        # Documented struct fields (simulate.run anchor_arr dtype).
        assert set(anc.dtype.names) == {
            "abs_position", "chain_position", "chain_index",
            "side", "strength", "tad_index", "edge",
        }
        # Self-describing schema attrs go with the anchors.
        assert fh.attrs["boundary_model"] == "per_tad_oriented"
        assert int(fh.attrs["ctcf_anchor_schema_version"]) == 1
        # Anchors are per chain; the recorded count is a whole multiple of num_chains.
        assert anc.shape[0] % cfg.num_chains == 0
        assert int(fh.attrs["ctcf_anchors_per_chain"]) == anc.shape[0] // cfg.num_chains


# --------------------------------------------------------------------------- #
# attributes match the config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", REGIMES)
def test_attrs_match_config(all_regime_cfgs, regime, tmp_path):
    """The self-describing scale/size attributes are exactly the config values, and the
    enabled flags reflect which plugin slots are wired up for this regime."""
    cfg = _prep(all_regime_cfgs[regime], tmp_path, name=f"{regime}_attr.h5")
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None

    out = _run(cfg)
    with h5py.File(out, "r") as fh:
        a = fh.attrs
        assert int(a["N"]) == cfg.num_sites
        assert int(a["LEFNum"]) == cfg.num_lefs
        assert int(a["chain_length"]) == cfg.chain_length
        assert int(a["num_chains"]) == cfg.num_chains
        assert int(a["bp_per_site"]) == int(cfg.bp_per_site)
        assert float(a["tick_seconds"]) == float(cfg.tick_seconds)
        assert int(a["warmup_steps"]) == cfg.warmup_steps
        assert int(a["separation"]) == cfg.separation
        assert int(a["seed"]) == SEED
        assert bool(a["rnapii_enabled"]) is rnapii
        assert bool(a["lesion_enabled"]) is lesion
        if rnapii:
            assert int(a["max_rnapii"]) == cfg.max_rnapii
            assert int(a["rnapii_cap"]) == cfg.max_rnapii * cfg.num_chains
        if lesion:
            # lesion_target drives the recorded lesion width.
            assert int(a["lesion_target"]) == fh["lesions"].shape[1]


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_same_seed_byte_identical_positions(cohesin_cfg, tmp_path):
    """Same cfg + same seed -> byte-identical ``positions`` (and all other datasets)."""
    cfg_a = _prep(cohesin_cfg, tmp_path, seed=11, name="det_a.h5")
    cfg_b = _prep(cohesin_cfg, tmp_path, seed=11, name="det_b.h5")
    out_a = _run(cfg_a)
    out_b = _run(cfg_b)
    with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
        assert np.array_equal(fa["positions"][...], fb["positions"][...])


def test_different_seed_changes_positions(cohesin_cfg, tmp_path):
    """A different seed must change the recorded ``positions`` (RNG actually drives it)."""
    cfg_a = _prep(cohesin_cfg, tmp_path, seed=11, name="seed_a.h5")
    cfg_b = _prep(cohesin_cfg, tmp_path, seed=999, name="seed_b.h5")
    out_a = _run(cfg_a)
    out_b = _run(cfg_b)
    with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
        pa = fa["positions"][...]
        pb = fb["positions"][...]
        assert pa.shape == pb.shape
        assert not np.array_equal(pa, pb)


# --------------------------------------------------------------------------- #
# run() IS the kernel path: positions == kernel_record(...)['coh']
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", REGIMES)
def test_positions_equal_kernel_record(all_regime_cfgs, regime, tmp_path):
    """run()'s on-disk ``positions`` is byte-identical to ``kernel_record(cfg, warmup,
    traj)['coh']`` -- run() warms up ``cfg.warmup_steps`` and records ``cfg.trajectory_length``
    with the same seed, so the two paths must produce the same cohesin trajectory."""
    cfg = _prep(all_regime_cfgs[regime], tmp_path, name=f"{regime}_xcheck.h5")
    out = _run(cfg)

    # kernel_record builds its own args + seeds via the config; must use the SAME
    # warmup/traj that run() consumed off the cfg.
    rec = kernel_record(cfg, cfg.warmup_steps, cfg.trajectory_length)
    with h5py.File(out, "r") as fh:
        on_disk = fh["positions"][...]
    assert on_disk.shape == rec["coh"].shape
    assert np.array_equal(on_disk, rec["coh"])


def test_positions_equal_kernel_record_zero_warmup(rnapii_cfg, tmp_path):
    """The on-disk == kernel_record equivalence also holds with zero warmup, where the
    loading transient is densest (a common spot for off-by-one recording bugs)."""
    cfg = _prep(rnapii_cfg, tmp_path, warmup=0, traj=20, name="zero_warmup.h5")
    out = _run(cfg)
    rec = kernel_record(cfg, 0, 20)
    with h5py.File(out, "r") as fh:
        on_disk = fh["positions"][...]
    assert np.array_equal(on_disk, rec["coh"])
