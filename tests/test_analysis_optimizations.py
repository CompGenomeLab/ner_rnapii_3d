"""Equivalence and scale-edge tests for scripts invoked by run_panels.sh."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare_config_chain_metrics as compare
import gen_cohesin_barrier_eval as barrier
import gen_lesion_grid_and_heatmaps as lesion_grid
import gen_transcription_metrics as transcription
import sweep_blockprob_vs_ctcf_lifetime_2d as block_sweep
import sweep_lifetime_cohesin_vs_ctcf_2d as lifetime_sweep
import sweep_separation_vs_ctcf_lifetime_2d as separation_sweep

from ner_rnapii_3d.engine import simulate


def _run_small(cfg, tmp_path, name, *, warmup=7, trajectory=31):
    cfg = copy.deepcopy(cfg)
    cfg.warmup_steps = warmup
    cfg.trajectory_length = trajectory
    cfg.output_path = str(tmp_path / name)
    return cfg, simulate.run(cfg)


def _assert_nested_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            av, bv = a[key], b[key]
            assert av == bv or (
                isinstance(av, float) and isinstance(bv, float)
                and math.isnan(av) and math.isnan(bv)
            ), (key, av, bv)


def test_hist_percentile_matches_numpy_exactly():
    rng = np.random.default_rng(2026)
    for size in (0, 1, 2, 17, 1000):
        values = rng.integers(0, 101, size=size)
        hist = np.bincount(values, minlength=101)
        for percentile in (0, 1, 25, 50, 75, 90, 95, 99, 100):
            got = compare._hist_percentile(hist, percentile)
            if size:
                assert got == float(np.percentile(values, percentile))
            else:
                assert math.isnan(got)


def test_chain_analysis_is_byte_identical_across_chunks(cohesin_cfg, gen_dir, tmp_path):
    cfg, h5_path = _run_small(cohesin_cfg, tmp_path, "cohesin_analysis.h5")
    import yaml

    raw = yaml.safe_load((gen_dir / "config2_tst.yaml").read_text())
    top = compare._topology(raw)
    one = compare.analyze_run("x", gen_dir / "config2_tst.yaml", h5_path, top, chunk=1)
    whole = compare.analyze_run("x", gen_dir / "config2_tst.yaml", h5_path, top, chunk=10_000)
    with h5py.File(h5_path, "r") as h:
        in_memory = compare.analyze_run(
            "x", gen_dir / "config2_tst.yaml", h5_path, top,
            positions=h["positions"][...], chunk=7,
        )
    for expected, got, memory in zip(one, whole, in_memory):
        _assert_nested_equal(expected, got)
        _assert_nested_equal(expected, memory)


def test_transcription_analysis_is_exact_across_chunks(rnapii_cfg, tmp_path):
    _, h5_path = _run_small(rnapii_cfg, tmp_path, "rnapii_analysis.h5")
    one = transcription.measure_transcription(h5_path, rnapii_cfg.tick_seconds, chunk=1)
    whole = transcription.measure_transcription(h5_path, rnapii_cfg.tick_seconds, chunk=10_000)
    pd.testing.assert_frame_equal(one[0], whole[0], check_exact=True)
    assert one[1].keys() == whole[1].keys()
    for key in one[1]:
        a, b = one[1][key], whole[1][key]
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))
    for a, b in zip(one[2:], whole[2:]):
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.names is None else np.array_equal(a, b)


def test_lesion_and_barrier_analysis_are_exact_across_chunks(lesion_cfg, tmp_path):
    _, h5_path = _run_small(lesion_cfg, tmp_path, "lesion_analysis.h5")
    one = lesion_grid.measure_lesion_stall(h5_path, lesion_cfg.tick_seconds, chunk=1)
    whole = lesion_grid.measure_lesion_stall(h5_path, lesion_cfg.tick_seconds, chunk=10_000)
    assert one.keys() == whole.keys()
    for key in one:
        a, b = one[key], whole[key]
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), key

    occ_one = barrier.read_cohesin_occupancy(h5_path, chunk=1)
    occ_whole = barrier.read_cohesin_occupancy(h5_path, chunk=10_000)
    assert np.array_equal(occ_one[0], occ_whole[0])
    assert occ_one[1:] == occ_whole[1:]


def test_permutation_samples_and_small_lifetimes_are_scale_safe():
    a = np.linspace(0.0, 1.0, 20)
    b = np.linspace(0.2, 1.4, 20)
    assert compare._two_sample_permutation_pvalue(a, b) == 0.009295
    assert "Monte Carlo" in compare.STAT_TEST_NAME

    base_lifetimes = {
        "lifetime": 2.0, "lifetime_stalled": 2.0,
        "lifetime_rnapii_stalled": 2.0, "lifetime_ctcf": 2.0,
    }
    cfg, values = lifetime_sweep.make_grid_config(
        {"lef": {"topology_kwargs": {}}}, base_lifetimes, 0.1, 0.1
    )
    assert values["lifetime"] == values["lifetime_stalled"] == 1
    assert values["lifetime_rnapii_stalled"] == values["lifetime_ctcf"] == 1
    assert cfg["lef"]["lifetime_ctcf"] == 1

    cfg, values = block_sweep.make_grid_config(
        {"lef": {"topology_kwargs": {}}}, 2.0, 0.9, 0.1
    )
    assert values["lifetime_ctcf"] == cfg["lef"]["lifetime_ctcf"] == 1
    cfg, values = separation_sweep.make_grid_config(
        {"lef": {}}, 2.0, 2.0, 0.1, 0.1
    )
    assert values["separation"] == values["lifetime_ctcf"] == 1
