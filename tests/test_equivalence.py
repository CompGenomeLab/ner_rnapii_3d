"""The core correctness guarantee: the production Numba kernel reproduces the
Python reference implementation TICK-FOR-TICK (byte-identical recordings), across
every regime and several seeds.

This is a self-validating check -- it compares two independent implementations of
the same dynamics rather than a frozen hash -- so it catches any divergence in the
kernel (RNG order, draw conditions, state transitions, recording) at its source.
"""
import copy

import pytest

from conftest import oracle_record, kernel_record, assert_records_equal

SEEDS = [0, 1, 7, 42, 2026]
REGIMES = ["cohesin", "rnapii", "lesion", "combined"]


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_kernel_matches_oracle(all_regime_cfgs, regime, seed):
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = seed
    warmup, traj = 40, 150
    ref = oracle_record(cfg, warmup, traj)
    got = kernel_record(cfg, warmup, traj)
    assert_records_equal(ref, got, label=f"{regime}/seed={seed}")


def test_kernel_matches_oracle_zero_warmup(rnapii_cfg):
    """Equivalence must hold from tick 0 (no warmup), where loading transients are
    densest -- a common place for draw-order bugs to surface."""
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.seed = 3
    ref = oracle_record(cfg, 0, 120)
    got = kernel_record(cfg, 0, 120)
    assert_records_equal(ref, got, label="rnapii/zero-warmup")
