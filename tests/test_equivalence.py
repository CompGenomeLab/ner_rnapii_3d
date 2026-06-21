"""The core correctness guarantee: the production Numba kernel reproduces the
Python reference implementation TICK-FOR-TICK (byte-identical recordings), across
every regime and several seeds.

This is a self-validating check -- it compares two independent implementations of
the same dynamics rather than a frozen hash -- so it catches any divergence in the
kernel (RNG order, draw conditions, state transitions, recording) at its source.
"""
import copy

import numpy as np
import pytest

from conftest import oracle_record, kernel_record, assert_records_equal
from ner_rnapii_3d.engine import kernel

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


@pytest.mark.parametrize(
    "chain_length,num_chains,nlef,n_genes",
    [(17, 1, 0, 5), (31, 2, 11, 13), (101, 7, 83, 47), (503, 3, 401, 71)],
)
def test_indexed_ep_contacts_match_brute_force_at_multiple_scales(
    chain_length, num_chains, nlef, n_genes
):
    """The indexed hot path is the exact same containment predicate at any scale."""
    rng = np.random.default_rng(20260621 + nlef)
    N = chain_length * num_chains
    lpos = np.empty(nlef, np.int64)
    rpos = np.empty(nlef, np.int64)
    for i in range(nlef):
        chain = int(rng.integers(num_chains))
        a, b = sorted(rng.integers(chain_length, size=2).tolist())
        lpos[i] = chain * chain_length + a
        rpos[i] = chain * chain_length + b
        if i % 7 == 0:  # endpoint order is not assumed by either implementation
            lpos[i], rpos[i] = rpos[i], lpos[i]

    gene_tss = rng.integers(N, size=n_genes, dtype=np.int64)
    enh_off = np.arange(0, 3 * n_genes + 1, 3, dtype=np.int64)
    enh_pos = np.empty(3 * n_genes, np.int64)
    for g, tss in enumerate(gene_tss):
        chain = int(tss) // chain_length
        enh_pos[3 * g:3 * g + 3] = (
            chain * chain_length + rng.integers(chain_length, size=3)
        )
    tol = 2

    expected = np.zeros(n_genes, np.int64)
    for g, tss in enumerate(gene_tss):
        gchain = int(tss) // chain_length
        for enhancer in enh_pos[enh_off[g]:enh_off[g + 1]]:
            lo = min(int(tss), int(enhancer)) + tol
            hi = max(int(tss), int(enhancer)) - tol
            for left, right in zip(lpos, rpos):
                left, right = sorted((int(left), int(right)))
                if left // chain_length == gchain and left <= lo and right >= hi:
                    expected[g] += 1
                    break

    got = np.empty(n_genes, np.int64)
    scratch = [np.empty(nlef, np.int64) for _ in range(3)]
    kernel._compute_ep_contacts(
        lpos, rpos, nlef, enh_pos, enh_off, gene_tss, n_genes,
        tol, chain_length, got, *scratch,
    )
    assert np.array_equal(got, expected)
