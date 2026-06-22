"""Promoter-proximal occlusion (the pause-initiation limit): a paused/pre-init Pol
in the window [TSS .. TSS+dir*pause_offset] suppresses new initiation. The kernel
and the Python reference must agree in BOTH modes, and the mechanism must actually
reduce Pol loading vs occlusion-off (Gressel 2019; Bartman 2019)."""
import copy

import numpy as np

from conftest import oracle_record, kernel_record, assert_records_equal

WARMUP, TRAJ = 40, 150


def _cfg(base, occlusion):
    cfg = copy.deepcopy(base)
    cfg.seed = 7
    cfg.topology_kwargs = dict(cfg.topology_kwargs)
    cfg.topology_kwargs["rnapii_promoter_occlusion"] = occlusion
    return cfg


def test_occlusion_on_kernel_matches_oracle(rnapii_cfg):
    cfg = _cfg(rnapii_cfg, True)
    assert_records_equal(oracle_record(cfg, WARMUP, TRAJ),
                         kernel_record(cfg, WARMUP, TRAJ), "occlusion-on")


def test_occlusion_off_kernel_matches_oracle(rnapii_cfg):
    cfg = _cfg(rnapii_cfg, False)
    assert_records_equal(oracle_record(cfg, WARMUP, TRAJ),
                         kernel_record(cfg, WARMUP, TRAJ), "occlusion-off")


def test_occlusion_suppresses_loading(rnapii_cfg):
    # Crank loading high so the pause-initiation limit is the BINDING constraint. The invariant
    # (occlusion can only reduce loading) is a property of the MECHANISM, not the loading level; at
    # the production loading anchor (deliberately low, for a bottom-heavy Zipf output) only a few Pol
    # load at this small test locus, so the live-Pol comparison is noise-dominated. High, unconditional
    # loading makes the suppression unambiguous.
    base = copy.deepcopy(rnapii_cfg)
    base.topology_kwargs = dict(base.topology_kwargs)
    base.topology_kwargs["genes"] = [dict(g, load_prob=0.5, requires_enhancer=False,
                                          load_requires_enhancer=False)
                                     for g in base.topology_kwargs["genes"]]
    on = kernel_record(_cfg(base, True), WARMUP, TRAJ)
    off = kernel_record(_cfg(base, False), WARMUP, TRAJ)
    live_on = (on["rp"][:, :, 0] >= 0).sum(1)
    live_off = (off["rp"][:, :, 0] >= 0).sum(1)
    # the pause-initiation limit can only reduce (never increase) loading
    assert live_on.mean() < live_off.mean(), (live_on.mean(), live_off.mean())
    assert not np.array_equal(on["rp"], off["rp"])   # mechanism changes the output
