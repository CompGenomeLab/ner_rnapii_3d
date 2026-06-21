"""Gatekeeping tests for ``kernel.numba_eligible``.

``numba_eligible(cfg, args) -> (eligible: bool, reason: str)`` is the single
guard that decides whether a config may run on the production Numba engine. It
accepts ONLY the validated default plugin set and the validated arg regimes;
anything else must fall back (here: be rejected with a non-empty reason). This
module pins down both directions:

* every default regime fixture is accepted with reason ``""``;
* swapping any per-tick plugin to a non-default target, or turning on an
  unported arg path (targeted load / loading_sites / Tei pause restraint), is
  rejected with the exact reason string the function emits.

We mirror ``numba_eligible``'s own checks (read from
``ner_rnapii_3d/engine/kernel.py``) rather than re-deriving them, so the
asserted reason strings are the real ones. ``run`` on an eligible cfg is
smoke-tested once to confirm the accept path actually produces a trajectory.
"""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from conftest import build_args

import ner_rnapii_3d
from ner_rnapii_3d.config import PluginSpec
from ner_rnapii_3d.engine import kernel

# The fully-qualified default plugin module prefixes (mirrors config.py). Used to
# build *plausible* swapped targets so the rejection is about the attr name, not
# an import error.
LEF_DYN = "ner_rnapii_3d.plugins.lef_dynamics"
RNAPII = "ner_rnapii_3d.plugins.rnapii"
LESIONS = "ner_rnapii_3d.plugins.lesions"

REGIMES = ("cohesin", "rnapii", "lesion", "combined")


# --------------------------------------------------------------------------- #
# accept path: all four default regimes are eligible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", REGIMES)
def test_default_regimes_are_eligible(regime, all_regime_cfgs):
    """Each shipped default regime is accepted with an empty reason."""
    cfg = all_regime_cfgs[regime]
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is True, f"{regime}: expected eligible, got reason {why!r}"
    assert why == "", f"{regime}: eligible reason must be empty, got {why!r}"


# --------------------------------------------------------------------------- #
# reject path: a swapped per-tick plugin -> (False, exact reason)
# --------------------------------------------------------------------------- #
def test_reject_wrong_unload_name(cohesin_cfg):
    """A plausible-but-wrong unload_prob target (the task's ``...:capture`` swap)
    is rejected. The attr resolves to a real callable, so this is a name-policy
    rejection, not an import failure."""
    cfg = copy.deepcopy(cohesin_cfg)
    cfg.plugins.unload_prob = PluginSpec(target=f"{LEF_DYN}:capture")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom unload_prob"


def test_reject_custom_load_name(cohesin_cfg):
    """A custom (non-default) load plugin name is rejected; the reason names it.

    ``_DEFAULT_LOAD`` is ``("load_one", "load_targeted")``; ``capture`` is not in
    it, so the load check fires first and reports the offending short name."""
    cfg = copy.deepcopy(cohesin_cfg)
    cfg.plugins.load = PluginSpec(target=f"{LEF_DYN}:capture")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom load plugin capture"


def test_reject_custom_capture_and_release(cohesin_cfg):
    """A non-default capture (or release) plugin is rejected with the shared
    capture/release reason."""
    cfg = copy.deepcopy(cohesin_cfg)
    cfg.plugins.capture = PluginSpec(target=f"{LEF_DYN}:release")  # wrong attr
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom capture/release"


def test_reject_custom_translocate(cohesin_cfg):
    """A translocate target outside ``_DEFAULT_TRANSLOCATE`` is rejected and the
    reason quotes its short name."""
    cfg = copy.deepcopy(cohesin_cfg)
    cfg.plugins.translocate = PluginSpec(target=f"{LEF_DYN}:capture")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom translocate capture"


def test_reject_custom_rnapii_load(rnapii_cfg):
    """With RNAPII active, a non-default rnapii_load target is rejected."""
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.plugins.rnapii_load = PluginSpec(target=f"{RNAPII}:stateful_translocate_rnapii")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom rnapii_load"


def test_reject_nonstateful_rnapii_translocate(rnapii_cfg):
    """The v1 (non-stateful) ``translocate_rnapii`` is unported -> rejected."""
    cfg = copy.deepcopy(rnapii_cfg)
    cfg.plugins.rnapii_translocate = PluginSpec(target=f"{RNAPII}:translocate_rnapii")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "rnapii_translocate not stateful (v1 unsupported)"


def test_reject_custom_lesion_plugin(lesion_cfg):
    """A non-default lesion plugin target is rejected when lesions are active."""
    cfg = copy.deepcopy(lesion_cfg)
    cfg.plugins.lesion = PluginSpec(target=f"{LESIONS}:update_lesions_v2")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom lesion plugin"


def test_rnapii_plugin_swap_inert_when_rnapii_off(cohesin_cfg):
    """RNAPII plugin identity is only checked when BOTH rnapii slots are set.

    On a cohesin-only cfg the rnapii_* slots are ``None``; setting only one of
    them leaves ``rnapii`` False, so the rnapii name checks are skipped and the
    cfg stays eligible. This guards the ``rnapii = load is not None and
    translocate is not None`` gate in numba_eligible."""
    cfg = copy.deepcopy(cohesin_cfg)
    assert cfg.plugins.rnapii_load is None and cfg.plugins.rnapii_translocate is None
    # Set only one slot to a bogus target -> rnapii stays disabled.
    cfg.plugins.rnapii_load = PluginSpec(target=f"{RNAPII}:translocate_rnapii")
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is True and why == ""


# --------------------------------------------------------------------------- #
# reject path: unported arg regimes (mutate the args dict from build_args)
# --------------------------------------------------------------------------- #
def test_reject_targeted_load_prob_positive(cohesin_cfg):
    """``targeted_load_prob > 0`` selects the unported targeted-placement draw."""
    cfg = copy.deepcopy(cohesin_cfg)
    args = build_args(cfg)
    # Default regime arrives with targeted_load_prob == 0 (eligible)...
    assert float(args.get("targeted_load_prob", 0.0)) == 0.0
    ok, _ = kernel.numba_eligible(cfg, args)
    assert ok is True
    # ...flip it on and the same cfg becomes ineligible.
    args["targeted_load_prob"] = 0.25
    ok, why = kernel.numba_eligible(cfg, args)
    assert ok is False
    assert why == "targeted_load_prob>0 (targeted placement not ported)"


def test_reject_nonempty_loading_sites(cohesin_cfg):
    """A non-empty ``loading_sites`` list triggers the unported targeted-load draw,
    independent of ``targeted_load_prob``."""
    cfg = copy.deepcopy(cohesin_cfg)
    args = build_args(cfg)
    assert not (args.get("loading_sites") or [])  # default regime: empty/absent
    args["loading_sites"] = [10, 20, 30]
    ok, why = kernel.numba_eligible(cfg, args)
    assert ok is False
    assert why == "non-empty loading_sites (targeted-load draw not ported)"


def test_reject_tei_pause_restraint(rnapii_cfg):
    """``rnapii_pause_cohesin_restraint < 1.0`` (Tei restraint) is unported."""
    cfg = copy.deepcopy(rnapii_cfg)
    args = build_args(cfg)
    # Default value is the no-restraint sentinel 1.0 -> eligible.
    assert float(args.get("rnapii_pause_cohesin_restraint", 1.0)) == 1.0
    args["rnapii_pause_cohesin_restraint"] = 0.5
    ok, why = kernel.numba_eligible(cfg, args)
    assert ok is False
    assert why == "Tei pause restraint not ported"


def test_check_order_load_before_unload(cohesin_cfg):
    """When several checks would fail, the load check fires first (it is evaluated
    before unload_prob). Pins the documented short-circuit order so the reported
    reason is stable."""
    cfg = copy.deepcopy(cohesin_cfg)
    cfg.plugins.load = PluginSpec(target=f"{LEF_DYN}:capture")       # fails first
    cfg.plugins.unload_prob = PluginSpec(target=f"{LEF_DYN}:capture")  # would also fail
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok is False
    assert why == "custom load plugin capture"


# --------------------------------------------------------------------------- #
# accept path actually runs: light end-to-end smoke
# --------------------------------------------------------------------------- #
def test_eligible_cfg_run_smoke(cohesin_cfg, tmp_path):
    """An eligible cfg runs end-to-end via the public ``ner_rnapii_3d.run`` and
    writes a non-empty ``positions`` trajectory. Kept tiny: short warmup + traj."""
    cfg = copy.deepcopy(cohesin_cfg)
    # sanity: this is the accept path we are smoke-testing
    ok, why = kernel.numba_eligible(cfg, build_args(cfg))
    assert ok and why == ""

    cfg.warmup_steps = 3
    cfg.trajectory_length = 8
    cfg.seed = 123
    out = tmp_path / "LEFPositions.h5"
    cfg.output_path = str(out)

    returned = ner_rnapii_3d.run(cfg)
    assert returned == out
    assert out.exists()

    with h5py.File(out, "r") as fh:
        pos = fh["positions"][:]
    # shape is (trajectory_length, num_lefs, 2); cohesin legs are in-range sites.
    assert pos.shape == (8, cfg.num_lefs, 2)
    assert pos.dtype == np.int32
    assert pos.min() >= 0 and pos.max() < cfg.num_sites
    # legs are placed (no padding rows for the cohesin dataset) and ordered left<right.
    assert np.all(pos[..., 0] < pos[..., 1])
