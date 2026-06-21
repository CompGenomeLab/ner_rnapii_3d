"""Shared test harness for the 1D engine.

Two recorders that both consume a :class:`LEFConfig` and return the SAME dict of
recorded arrays, so any test can assert the production Numba kernel matches the
Python reference implementation tick-for-tick:

* :func:`oracle_record` -- drives the Python reference (``simulate._advance_one_step``
  + the plugin functions) and records each frame. This is the ground truth.
* :func:`kernel_record` -- drives the production Numba engine
  (``init_chunked`` / ``advance_chunked``), optionally in fixed-size blocks.

Regime configs (cohesin-only / rnapii / lesion / combined) are generated once per
session by the bundled generator at a small size, then loaded as LEFConfig objects.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "gen_realistic_configs_variable_tick.py"

from ner_rnapii_3d.config import load_lef_config, resolve_plugin, PluginSpec
from ner_rnapii_3d.engine import simulate, kernel


# --------------------------------------------------------------------------- #
# args + recorders
# --------------------------------------------------------------------------- #
def build_args(cfg):
    """Resolve the topology plugin into a fresh ``args`` dict (deterministic, so the
    oracle and the kernel each build their own and still agree)."""
    args = resolve_plugin(cfg.plugins.topology)(cfg, **cfg.topology_kwargs)
    args["num_chains"] = cfg.num_chains
    args["rnapii_enabled"] = (
        cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    )
    return args


def _caps(cfg, args):
    rcap = int(cfg.max_rnapii) * int(cfg.num_chains)
    lcap = int(args.get("lesion_target", 0))
    return rcap, lcap


def oracle_record(cfg, warmup, traj):
    """Run the Python reference engine and record every frame.

    Returns a dict with keys ``coh`` (always) and, when active, ``rp``/``rs``/``ri``
    (RNAPII pos+gene / state / uid) and ``lp``/``lt``/``ls`` (lesion site / type / state).
    """
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None
    args = build_args(cfg)
    rcap, lcap = _caps(cfg, args)

    np.random.seed(int(cfg.seed))
    args["lesions"] = {}
    args["rnapii_by_pos"] = {}
    args["cohesin_leg_by_pos"] = {}
    args["_rnapii_uid_next"] = 0
    load_fn = resolve_plugin(cfg.plugins.load)
    occ, coh = simulate.initial_state(cfg, args, load_fn)
    rnp = []
    fns = dict(
        rnapii_load_fn=resolve_plugin(cfg.plugins.rnapii_load) if rnapii else None,
        rnapii_translocate_fn=resolve_plugin(cfg.plugins.rnapii_translocate) if rnapii else None,
        translocate_fn=resolve_plugin(cfg.plugins.translocate),
        unload_fn=resolve_plugin(cfg.plugins.unload_prob),
        load_fn=load_fn,
        capture_fn=resolve_plugin(cfg.plugins.capture),
        release_fn=resolve_plugin(cfg.plugins.release),
        lesion_update_fn=resolve_plugin(cfg.plugins.lesion) if lesion else None,
    )
    step = lambda: simulate._advance_one_step(
        rnapii_enabled=rnapii, rnapiis=rnp, cohesins=coh, occupied=occ, args=args, **fns
    )
    for _ in range(warmup):
        step()

    nlef = len(coh)
    out = {"coh": np.empty((traj, nlef, 2), np.int32)}
    if rnapii:
        out["rp"] = np.full((traj, rcap, 2), -1, np.int32)
        out["rs"] = np.full((traj, rcap), -1, np.int32)
        out["ri"] = np.full((traj, rcap), -1, np.int32)
    if lesion:
        out["lp"] = np.full((traj, lcap), -1, np.int32)
        out["lt"] = np.full((traj, lcap), -1, np.int32)
        out["ls"] = np.full((traj, lcap), -1, np.int32)
    for t in range(traj):
        step()
        for i, c in enumerate(coh):
            out["coh"][t, i, 0] = c.left.pos
            out["coh"][t, i, 1] = c.right.pos
        if rnapii:
            for i, r in enumerate(rnp[:rcap]):
                out["rp"][t, i, 0] = r.pos
                out["rp"][t, i, 1] = r.gene_id
                out["rs"][t, i] = r.state if r.state is not None else -1
                out["ri"][t, i] = r.uid
        if lesion:
            cur = args["lesions"]
            for i, s in enumerate(sorted(cur)[:lcap]):
                out["lp"][t, i] = s
                out["lt"][t, i] = cur[s].ltype
                out["ls"][t, i] = cur[s].state
    return out


def kernel_record(cfg, warmup, traj, block_len=None):
    """Run the production Numba engine (optionally in fixed ``block_len`` blocks) and
    record every frame in the same layout as :func:`oracle_record`."""
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None
    args = build_args(cfg)
    run = kernel.init_chunked(cfg, args)
    if block_len is None:
        block_len = traj
    if warmup:
        kernel.advance_chunked(run, warmup, do_record=False)

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
        out["rp"] = cat(1)
        out["rs"] = cat(2)
        out["ri"] = cat(3)
    if lesion:
        lp = cat(4)            # (traj, lcap, 2) = [site, type]
        out["lp"] = lp[:, :, 0]
        out["lt"] = lp[:, :, 1]
        out["ls"] = cat(5)
    return out


def assert_records_equal(a, b, label=""):
    """Assert two recordings (oracle vs kernel) are byte-identical, key by key."""
    assert set(a) == set(b), f"{label}: stream sets differ {set(a) ^ set(b)}"
    for k in a:
        if not np.array_equal(a[k], b[k]):
            diff = np.argwhere(np.asarray(a[k]) != np.asarray(b[k]))
            first = tuple(diff[0]) if len(diff) else None
            raise AssertionError(
                f"{label}: stream '{k}' differs at {first} "
                f"(oracle={a[k][first] if first else '?'} kernel={b[k][first] if first else '?'}; "
                f"{len(diff)} elements)"
            )


# --------------------------------------------------------------------------- #
# regime configs (generated once per session)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def gen_dir(tmp_path_factory):
    """Generate the four config families once at a small size (chain 6000)."""
    work = tmp_path_factory.mktemp("cfgs")
    subprocess.run(
        [sys.executable, str(GEN), "--out-dir", str(work), "--suffix", "tst",
         "--chain", "6000", "--bp-per-site", "250", "--num-chains", "2",
         "--seed", "7", "--tick-seconds", "2",
         "--trajectory-length", "150", "--warmup-steps", "40"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return work


@pytest.fixture(scope="session")
def cohesin_cfg(gen_dir):
    """config2: cohesin-only (RNAPII off, no lesions)."""
    return load_lef_config(gen_dir / "config2_tst.yaml")


@pytest.fixture(scope="session")
def rnapii_cfg(gen_dir):
    """config1: cohesin + RNAPII (enhancer-gated transcription)."""
    return load_lef_config(gen_dir / "config1_tst.yaml")


@pytest.fixture(scope="session")
def lesion_cfg(gen_dir):
    """config4: cohesin + lesions (RNAPII off)."""
    return load_lef_config(gen_dir / "config4_tst.yaml")


@pytest.fixture(scope="session")
def combined_cfg(gen_dir):
    """RNAPII + lesions together: config1 with config4's lesion knobs + lesion plugin."""
    base = load_lef_config(gen_dir / "config1_tst.yaml")
    les = load_lef_config(gen_dir / "config4_tst.yaml")
    cfg = copy.deepcopy(base)
    for k, v in les.topology_kwargs.items():
        if k.startswith("lesion_"):
            cfg.topology_kwargs[k] = v
    cfg.plugins.lesion = PluginSpec(target=les.plugins.lesion.target)
    return cfg


@pytest.fixture(scope="session")
def all_regime_cfgs(cohesin_cfg, rnapii_cfg, lesion_cfg, combined_cfg):
    return {"cohesin": cohesin_cfg, "rnapii": rnapii_cfg,
            "lesion": lesion_cfg, "combined": combined_cfg}
