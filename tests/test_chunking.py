"""Block-streaming correctness of the chunked Numba recorder.

The production engine streams its recording to HDF5 in memory-bounded BLOCKS
(``init_chunked`` -> repeated ``advance_chunked``), reusing the exact per-tick
functions across block boundaries. The block size is purely a memory/IO knob: it
must NOT change the recorded trajectory. These tests pin that invariant directly
on the kernel (independent of the Python oracle, which ``test_equivalence`` covers):

* The recording is IDENTICAL for many ``block_len`` values (1, 3, 7, 50, traj) --
  including the regimes whose per-tick state crosses block boundaries in
  non-trivial ways: RNAPII (population load/compaction) and lesions (insertion-order
  population streaming).
* Warmup is just un-recorded ticks: splitting the warmup into several
  ``advance_chunked(do_record=False)`` calls yields the SAME subsequent recording
  as one warmup call (RNG + all engine state persist in the ``ChunkedRun``).
* ``block_len_for(run, traj)`` returns a sensible positive, ``<= traj`` bound that
  grows with the recorded width.
"""
import copy

import numpy as np
import pytest

from conftest import build_args, kernel_record, assert_records_equal
from ner_rnapii_3d.engine import kernel


WARMUP, TRAJ = 12, 60
BLOCK_LENS = [1, 3, 7, 50, TRAJ]


# --------------------------------------------------------------------------- #
# block_len must not change the recording
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", ["rnapii", "lesion"])
@pytest.mark.parametrize("block_len", BLOCK_LENS)
def test_block_len_invariant(all_regime_cfgs, regime, block_len):
    """Recording in ``block_len``-sized blocks equals recording in one block.

    The single-block (``block_len=traj``) run is the reference; every other block
    size must reproduce it byte-for-byte. Run on the RNAPII regime (its Pol
    population is loaded/compacted every tick) and the lesion regime (its lesion
    population is streamed in insertion order), where block boundaries are most
    likely to perturb state if the engine were not persisting it correctly.
    """
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 2024
    ref = kernel_record(cfg, WARMUP, TRAJ, block_len=TRAJ)
    got = kernel_record(cfg, WARMUP, TRAJ, block_len=block_len)
    assert_records_equal(ref, got, label=f"{regime}/block_len={block_len}")


@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
def test_all_block_lens_agree_pairwise(all_regime_cfgs, regime):
    """For every regime, ALL block sizes in ``BLOCK_LENS`` produce one identical
    recording (sanity that this holds for cohesin-only and combined too, and that
    no single block size is silently special)."""
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 11
    recs = {bl: kernel_record(cfg, WARMUP, TRAJ, block_len=bl) for bl in BLOCK_LENS}
    base = recs[TRAJ]
    for bl, rec in recs.items():
        assert_records_equal(base, rec, label=f"{regime}/block_len={bl}")


def test_block_len_invariant_is_meaningful(rnapii_cfg):
    """Guard against a vacuous block test: a DIFFERENT seed must actually change the
    recording. If two seeds produced the same trajectory, equality across block
    sizes would prove nothing."""
    a = copy.deepcopy(rnapii_cfg); a.seed = 1
    b = copy.deepcopy(rnapii_cfg); b.seed = 2
    ra = kernel_record(a, WARMUP, TRAJ, block_len=7)
    rb = kernel_record(b, WARMUP, TRAJ, block_len=7)
    assert not np.array_equal(ra["coh"], rb["coh"]), \
        "two seeds gave identical cohesin trajectories; block test would be vacuous"


# --------------------------------------------------------------------------- #
# warmup = un-recorded ticks: splitting the warmup must not change the recording
# --------------------------------------------------------------------------- #
def _record_with_warmup_calls(cfg, warmup_chunks, traj, block_len):
    """Build a ChunkedRun, burn the warmup via the given sequence of un-recorded
    ``advance_chunked`` calls, then record ``traj`` ticks -- returning the same
    dict layout as :func:`conftest.kernel_record`."""
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None
    args = build_args(cfg)
    run = kernel.init_chunked(cfg, args)

    for n in warmup_chunks:
        if n > 0:
            kernel.advance_chunked(run, n, do_record=False)

    blocks = []
    done = 0
    while done < traj:
        bl = min(block_len, traj - done)
        blocks.append(kernel.advance_chunked(run, bl, do_record=True))
        done += bl

    def cat(i):
        return np.concatenate([b[i] for b in blocks], axis=0)

    out = {"coh": cat(0)}
    if rnapii:
        out["rp"] = cat(1); out["rs"] = cat(2); out["ri"] = cat(3)
    if lesion:
        lp = cat(4)
        out["lp"] = lp[:, :, 0]; out["lt"] = lp[:, :, 1]; out["ls"] = cat(5)
    return out


@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
@pytest.mark.parametrize("warmup_chunks", [[5, 4, 3], [1] * 12, [11, 1], [0, 12, 0]])
def test_warmup_split_equals_single(all_regime_cfgs, regime, warmup_chunks):
    """Splitting the 12-tick warmup into several un-recorded ``advance_chunked``
    calls yields the SAME recording as one 12-tick warmup call. Warmup ticks carry
    no recording but DO advance RNG + engine state, which persists in the
    ChunkedRun, so the split is purely cosmetic. (Includes a ``[0, 12, 0]`` case to
    confirm zero-length warmup calls are no-ops.)"""
    assert sum(warmup_chunks) == WARMUP
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 99
    single = _record_with_warmup_calls(cfg, [WARMUP], TRAJ, block_len=7)
    split = _record_with_warmup_calls(cfg, warmup_chunks, TRAJ, block_len=7)
    assert_records_equal(single, split, label=f"{regime}/warmup={warmup_chunks}")


def test_split_warmup_matches_kernel_record(rnapii_cfg):
    """The manual split-warmup driver reproduces conftest's ``kernel_record``
    (which does a single warmup call), cross-checking that the local helper is a
    faithful stand-in for the shared harness."""
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.seed = 7
    harness = kernel_record(cfg, WARMUP, TRAJ, block_len=7)
    manual = _record_with_warmup_calls(cfg, [4, 4, 4], TRAJ, block_len=7)
    assert_records_equal(harness, manual, label="rnapii/split-vs-harness")


def test_warmup_actually_changes_recording(lesion_cfg):
    """Sanity: warmup is not a no-op overall. A run with warmup must differ from one
    with zero warmup (otherwise 'splitting warmup' would be trivially invariant)."""
    cfg = copy.deepcopy(lesion_cfg)
    cfg.seed = 5
    warm = _record_with_warmup_calls(cfg, [WARMUP], TRAJ, block_len=TRAJ)
    cold = _record_with_warmup_calls(cfg, [0], TRAJ, block_len=TRAJ)
    assert not np.array_equal(warm["coh"], cold["coh"]), \
        "warmup did not change the recording; warmup-split test would be vacuous"


# --------------------------------------------------------------------------- #
# block_len_for: a sensible positive memory bound
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
def test_block_len_for_is_sensible_bound(all_regime_cfgs, regime):
    """``block_len_for(run, traj)`` is a positive block size that never exceeds
    ``traj`` (you cannot record more ticks than requested in one buffer)."""
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 0
    run = kernel.init_chunked(cfg, build_args(cfg))
    for traj in (1, 10, 1000):
        bl = kernel.block_len_for(run, traj)
        assert isinstance(bl, int) and bl >= 1, f"{regime}: block_len_for={bl!r} not >=1"
        assert bl <= traj, f"{regime}: block_len_for({traj})={bl} exceeds traj"


def test_block_len_for_caps_at_traj_when_small(rnapii_cfg):
    """When ``traj`` is tiny the memory budget is not the binding constraint, so the
    bound is exactly ``traj`` (the whole short trajectory fits in one block)."""
    cfg = copy.deepcopy(rnapii_cfg); cfg.seed = 0
    run = kernel.init_chunked(cfg, build_args(cfg))
    assert kernel.block_len_for(run, 3) == 3


def test_block_len_for_recommended_block_reproduces_single(rnapii_cfg):
    """The block size the engine would actually pick (``block_len_for``) records the
    same trajectory as one big block -- i.e. the production block choice is safe."""
    cfg = copy.deepcopy(rnapii_cfg); cfg.seed = 13
    run = kernel.init_chunked(cfg, build_args(cfg))
    bl = kernel.block_len_for(run, TRAJ)
    ref = kernel_record(cfg, WARMUP, TRAJ, block_len=TRAJ)
    got = kernel_record(cfg, WARMUP, TRAJ, block_len=bl)
    assert_records_equal(ref, got, label=f"rnapii/recommended block_len={bl}")


def test_block_len_for_grows_with_recorded_width(all_regime_cfgs):
    """The bound accounts for ALL recorded streams: at a fixed large ``traj`` (so the
    memory budget binds), a config that additionally records RNAPII/lesions has a
    per-tick footprint >= the cohesin-only config, hence a block bound <= it. This
    pins that ``block_len_for`` is width-aware, not a constant."""
    big = 10_000_000  # large enough that the byte budget, not traj, is binding
    bounds = {}
    for regime in ("cohesin", "rnapii", "lesion", "combined"):
        cfg = copy.deepcopy(all_regime_cfgs[regime]); cfg.seed = 0
        run = kernel.init_chunked(cfg, build_args(cfg))
        bounds[regime] = kernel.block_len_for(run, big)
        assert bounds[regime] < big, f"{regime}: budget should bind at traj={big}"
    # extra recorded streams cost memory -> never a LARGER block than cohesin-only
    assert bounds["rnapii"] <= bounds["cohesin"]
    assert bounds["lesion"] <= bounds["cohesin"]
    assert bounds["combined"] <= bounds["rnapii"]
