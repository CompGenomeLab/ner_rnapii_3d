"""Physical / biological invariants on a recorded trajectory.

The equivalence suite (``test_equivalence.py``) already proves the Numba kernel
and the Python reference agree tick-for-tick. This module instead asserts that a
recorded trajectory is internally *consistent* with the physics the engine claims
to model -- properties that must hold for ANY correct trajectory regardless of
seed:

* cohesin legs live inside the lattice, never cross with each other (left<=right),
  never straddle a chain boundary, and no two cohesin legs share a site;
* RNAPII positions/states/uids/gene-ids are well-formed and unique-per-frame;
* the lesion population is capped at the homeostatic target, well-typed, and the
  recorder emits it sorted ascending by site with no duplicate sites.

These are recorded mainly with :func:`kernel_record` (the production engine), and
the cohesin invariants are additionally cross-checked against the Python
:func:`oracle_record` so the property is shown to hold for BOTH implementations
(not merely "the kernel matches itself"). ``-1`` padding of inactive slots is
skipped everywhere.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import kernel_record, oracle_record
from ner_rnapii_3d.config import resolve_plugin
from ner_rnapii_3d.engine import kernel

# Short trajectory: invariants are per-frame, so a handful of frames after a
# small warmup already exercises a loaded, churning population in every regime.
WARMUP, TRAJ = 15, 25


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _topology(cfg):
    """Resolve the topology plugin to read N / chain_length / num_chains exactly
    as the engine does (rather than re-deriving from cfg, which the engine trusts
    via these args)."""
    args = resolve_plugin(cfg.plugins.topology)(cfg, **cfg.topology_kwargs)
    return int(args["N"]), int(args["chain_length"]), int(args["num_chains"]), args


def _active_coh(coh):
    """Boolean mask of occupied cohesin slots (both legs >= 0).

    Cohesin slots are always live in this engine (reload-into-slot), but mask
    defensively so the invariants never accidentally read padding.
    """
    return (coh[:, :, 0] >= 0) & (coh[:, :, 1] >= 0)


# --------------------------------------------------------------------------- #
# cohesin invariants (checked for BOTH engines)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
@pytest.mark.parametrize("recorder", ["kernel", "oracle"])
def test_cohesin_legs_in_bounds_and_ordered(all_regime_cfgs, regime, recorder):
    """Every cohesin leg site lies in ``[0, N)`` and ``left <= right`` each frame.

    A loop-extrusion factor's two legs bracket a loop, so the left (minus-strand)
    leg can never overtake the right (plus-strand) leg; both must stay on-lattice.
    """
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 11
    N, _, _, _ = _topology(cfg)
    rec = (kernel_record if recorder == "kernel" else oracle_record)(cfg, WARMUP, TRAJ)
    coh = rec["coh"]
    occ = _active_coh(coh)
    left = coh[:, :, 0][occ]
    right = coh[:, :, 1][occ]

    assert occ.all(), f"{regime}/{recorder}: a cohesin slot was unexpectedly empty"
    assert left.min() >= 0 and right.max() < N, (
        f"{regime}/{recorder}: leg out of [0,{N}): "
        f"min={left.min()} max={right.max()}"
    )
    assert np.all(left <= right), f"{regime}/{recorder}: found a left>right cohesin"


@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
@pytest.mark.parametrize("recorder", ["kernel", "oracle"])
def test_cohesin_both_legs_same_chain(all_regime_cfgs, regime, recorder):
    """Neither leg crosses a chain boundary: both legs of a LEF sit in the same
    chain block of size ``chain_length`` (``pos // chain_length`` equal).

    Chains are independent polymers concatenated on one lattice; a LEF that
    bridged two chains would be physically impossible, and the engine's
    ``pos // chain_length`` boundary guard exists precisely to forbid it.
    """
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 11
    _, chain_length, num_chains, _ = _topology(cfg)
    rec = (kernel_record if recorder == "kernel" else oracle_record)(cfg, WARMUP, TRAJ)
    coh = rec["coh"]
    occ = _active_coh(coh)
    left_chain = coh[:, :, 0][occ] // chain_length
    right_chain = coh[:, :, 1][occ] // chain_length

    assert np.all(left_chain == right_chain), (
        f"{regime}/{recorder}: a LEF straddles two chains"
    )
    # and every leg lands in a real chain index [0, num_chains)
    assert left_chain.min() >= 0 and left_chain.max() < num_chains, (
        f"{regime}/{recorder}: chain index out of range"
    )


@pytest.mark.parametrize("regime", ["cohesin", "rnapii", "lesion", "combined"])
def test_cohesin_legs_occupy_distinct_sites(all_regime_cfgs, regime):
    """No two occupied cohesin legs share a site within a frame.

    The dense ``occupied`` grid enforces steric exclusion: a site holds at most
    one cohesin leg. We flatten all legs (left+right of every LEF) per frame and
    require every site value to be unique.
    """
    cfg = copy.deepcopy(all_regime_cfgs[regime])
    cfg.seed = 11
    rec = kernel_record(cfg, WARMUP, TRAJ)
    coh = rec["coh"]  # (traj, nlef, 2)
    traj = coh.shape[0]
    for t in range(traj):
        occ = _active_coh(coh)[t]
        legs = np.concatenate([coh[t, :, 0][occ], coh[t, :, 1][occ]])
        assert len(np.unique(legs)) == len(legs), (
            f"{regime}: two cohesin legs share a site at frame {t}: "
            f"{sorted(legs.tolist())}"
        )


# --------------------------------------------------------------------------- #
# RNAPII invariants
# --------------------------------------------------------------------------- #
_VALID_STATES = {
    kernel.STATE_PRE_INITIATION,
    kernel.STATE_PAUSED,
    kernel.STATE_ELONGATING,
    kernel.STATE_TERMINATING,
    kernel.STATE_STALLED,
}


def test_rnapii_positions_states_genes_uids(rnapii_cfg):
    """Every recorded Pol is well-formed: pos in ``[0, N)``, state in the valid
    STATE_* set, gene_id a real gene, and live-Pol uids unique within each frame.

    A Pol II is one molecule on the template; it occupies a real lattice site, is
    always in exactly one of the five modelled kinetic states, transcribes exactly
    one gene, and carries a unique identity (no two live Pols may share a uid).
    """
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.seed = 5
    N, _, _, args = _topology(cfg)
    n_genes = len(args["genes"])
    assert n_genes > 0, "rnapii config has no genes"

    rec = kernel_record(cfg, WARMUP, TRAJ)
    rp, rs, ri = rec["rp"], rec["rs"], rec["ri"]
    active = rp[:, :, 0] >= 0  # a slot is live iff its position is set

    # at least some Pol must be present, else the test is vacuous
    assert active.any(), "no RNAPII was ever loaded -- config too quiet to test"

    pos = rp[:, :, 0][active]
    gene = rp[:, :, 1][active]
    state = rs[active]

    assert pos.min() >= 0 and pos.max() < N, (
        f"RNAPII pos out of [0,{N}): min={pos.min()} max={pos.max()}"
    )
    assert gene.min() >= 0 and gene.max() < n_genes, (
        f"RNAPII gene_id out of [0,{n_genes}): min={gene.min()} max={gene.max()}"
    )
    bad = set(np.unique(state).tolist()) - _VALID_STATES
    assert not bad, f"RNAPII in unknown state(s) {bad}; valid={sorted(_VALID_STATES)}"

    # an active slot must have a real uid (>=0) and uids are unique per frame
    for t in range(rp.shape[0]):
        uids = ri[t][active[t]]
        assert np.all(uids >= 0), f"frame {t}: active Pol with no uid"
        assert len(np.unique(uids)) == len(uids), (
            f"frame {t}: duplicate RNAPII uid {sorted(uids.tolist())}"
        )


def test_rnapii_distinct_sites_per_frame(rnapii_cfg):
    """No two live Pol II share a lattice site within a frame.

    Pol II also obeys steric exclusion on the ``occupied`` grid (the engine refuses
    a step onto an ``RNAPII_CELL``), so recorded positions must be distinct per
    frame.
    """
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.seed = 5
    rec = kernel_record(cfg, WARMUP, TRAJ)
    rp = rec["rp"]
    active = rp[:, :, 0] >= 0
    for t in range(rp.shape[0]):
        pos = rp[t, :, 0][active[t]]
        assert len(np.unique(pos)) == len(pos), (
            f"frame {t}: two RNAPII share a site {sorted(pos.tolist())}"
        )


# --------------------------------------------------------------------------- #
# lesion invariants
# --------------------------------------------------------------------------- #
_VALID_LTYPE = {kernel.LESION_TYPE_A, kernel.LESION_TYPE_B}
_VALID_LSTATE = {kernel.LESION_PRE, kernel.LESION_REPAIR}


def test_lesion_population_capped_and_well_typed(lesion_cfg):
    """Active lesions per frame <= ``lesion_target``; sites in ``[0, N)``; types in
    {A,B}; states in {PRE,REPAIR}.

    The damage model holds a homeostatic lesion population: it refills toward the
    target each tick but the recorder array (width ``lesion_target`` = lcap) can
    never overflow it. Each lesion is one of two photoproduct classes and in one
    of two stages of the repair cycle.
    """
    cfg = copy.deepcopy(lesion_cfg)
    cfg.seed = 5
    N, _, _, args = _topology(cfg)
    lcap = int(args["lesion_target"])
    assert lcap > 0, "lesion config has no lesion_target"

    rec = kernel_record(cfg, WARMUP, TRAJ)
    lp, lt, ls = rec["lp"], rec["lt"], rec["ls"]
    assert lp.shape[1] == lcap, (
        f"recorder width {lp.shape[1]} != lesion_target {lcap}"
    )
    active = lp >= 0

    # population is capped by lcap (it can be < lcap transiently, never more)
    per_frame = active.sum(axis=1)
    assert per_frame.max() <= lcap, (
        f"active lesions {per_frame.max()} exceed cap {lcap}"
    )
    assert active.any(), "no lesion was ever present -- config too quiet to test"

    site = lp[active]
    assert site.min() >= 0 and site.max() < N, (
        f"lesion site out of [0,{N}): min={site.min()} max={site.max()}"
    )
    bad_t = set(np.unique(lt[active]).tolist()) - _VALID_LTYPE
    assert not bad_t, f"unknown lesion type(s) {bad_t}; valid={sorted(_VALID_LTYPE)}"
    bad_s = set(np.unique(ls[active]).tolist()) - _VALID_LSTATE
    assert not bad_s, f"unknown lesion state(s) {bad_s}; valid={sorted(_VALID_LSTATE)}"


def test_lesion_sites_sorted_and_distinct_per_frame(lesion_cfg):
    """Within each frame the recorder emits lesion sites sorted ascending with no
    duplicate site (the engine ``np.sort``s the active sites before recording, and
    a site holds at most one lesion via the ``lesion_at`` index).
    """
    cfg = copy.deepcopy(lesion_cfg)
    cfg.seed = 5
    rec = kernel_record(cfg, WARMUP, TRAJ)
    lp = rec["lp"]
    active = lp >= 0
    for t in range(lp.shape[0]):
        sites = lp[t][active[t]]
        assert np.all(np.diff(sites) >= 0), (
            f"frame {t}: lesion sites not sorted ascending: {sites.tolist()}"
        )
        assert len(np.unique(sites)) == len(sites), (
            f"frame {t}: duplicate lesion site {sorted(sites.tolist())}"
        )
