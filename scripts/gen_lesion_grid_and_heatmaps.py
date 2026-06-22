#!/usr/bin/env python
"""Lesion grid-search generator + per-metric heatmaps vs a baseline 1D sim.

From ONE config + its baseline ``LEFPositions.h5``, derive a 2-D grid of configs
that share the baseline TAD skeleton/genes but ablate RNAPII (``max_rnapii: 0``,
``rnapii_load`` / ``rnapii_translocate`` nulled) and boost every boundary strength
by ``--bstr-mult`` (capped at 1.0). Grid axes:

  Y -- ``lesion_block_prob`` (``p``): per-tick probability a lesion holds a cohesin
       leg (bypass ``1 - p``). The realized stall saturates (the lesion is REMOVED
       on repair and a held leg can unload), so no closed form is used -- the stall
       is MEASURED directly; analytic seconds columns are kept only as reference.
  X -- lesion DENSITY in lesions/Mbp via ``lesion_spacing`` (lattice sites);
       density = ``1e6 / (spacing * bp_per_site)`` (independent of chain length),
       ``bp_per_site`` read from the config.

For each grid config the 1D LEF stage is run (or a matching H5 reused), the
``compare_config_chain_metrics.py`` per-chain metrics are computed, and each
metric's chain-mean is divided by the BASELINE chain-mean. Figures: fold heatmaps
(centred at 1.0) plus, unless ``--no-measure``, measured cohesin-stall heatmaps
(per-lesion occupancy and dwell DURATION in seconds; see
:func:`measure_lesion_stall`).
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported so --jobs sims do not
# oversubscribe cores; parallelism comes from concurrent sims, not threaded BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm

# Diverging fold-change colormap: slate-blue (below baseline) -> white
# (fold = 1.0, the TwoSlopeNorm centre) -> brick-red (above baseline).
FOLD_CMAP = LinearSegmentedColormap.from_list(
    "fold_slate_red", ["#465775", "#ffffff", "#A63446"]
)
# Sequential map for the non-centered (no natural midpoint) heatmaps.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_white_red", ["#ffffff", "#A63446"]
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the comparison script's topology + metric machinery so grid folds are
# computed identically to the pairwise comparison.
from compare_config_chain_metrics import (  # noqa: E402
    BOUNDARY_LABELS,
    CORE_LABELS,
    _resolve_h5,
    _safe_label,
    _topology,
    analyze_run,
)

LESION_PLUGIN = "ner_rnapii_3d.plugins.lesions:update_lesions"

# Biological stage durations (seconds) used only when the baseline config has no
# lesion kwargs to inherit; mirrors gen_realistic_configs_variable_tick.py.
LESION_PRERECOGNITION_SECONDS = 1200.0
LESION_REPAIR_SECONDS = 300.0
DEFAULT_TYPE_A_PROB = 0.25
DEFAULT_TAD_SIZE_EXPONENT = 1.0
DEFAULT_TAD_REPAIR_EXPONENT = 1.0

# Metrics to draw, in display order: the de-duplicated union of the core +
# boundary sets used by compare_config_chain_metrics.py.
METRIC_LABELS: dict[str, str] = {}
for _src in (CORE_LABELS, BOUNDARY_LABELS):
    for _k, _v in _src.items():
        METRIC_LABELS.setdefault(_k, _v.replace("\n", " "))


def raw_bypass_seconds(p: float, tick_seconds: float) -> float:
    """Naive mean time to stochastically bypass a persistent lesion: tick/(1-p)
    (reference only; assumes the lesion never disappears, so it diverges at p=1)."""
    if p >= 1.0:
        return float("inf")
    return float(tick_seconds) / (1.0 - p)


def effective_stall_seconds(
    p: float,
    tick_seconds: float,
    repair_ticks: float,
    lifetime_stalled_ticks: float | None = None,
) -> float:
    """Mean time a cohesin leg is actually held at a lesion (s).

    Three independent memory-less escapes (leaky bypass ``1 - p``, lesion repair
    removal at mean ``repair_ticks``, and optional cohesin unload at
    ``lifetime_stalled_ticks``) have additive per-tick hazards; the stall is their
    reciprocal sum, so it saturates as ``p -> 1`` (capped by repair) rather than
    diverging like :func:`raw_bypass_seconds`."""
    rate = (1.0 - p) / float(tick_seconds) + 1.0 / (float(repair_ticks) * float(tick_seconds))
    if lifetime_stalled_ticks:
        rate += 1.0 / (float(lifetime_stalled_ticks) * float(tick_seconds))
    return 1.0 / rate if rate > 0 else float("inf")


def density_per_mbp(spacing: int, bp_per_site: int = 1000) -> float:
    """Lesions/Mbp for a spacing in lattice sites: ``1e6/(spacing*bp_per_site)``."""
    return 1e6 / (float(spacing) * float(bp_per_site))


def gene_body_mass(base: dict, tad_size_exponent: float) -> float:
    """Fraction of the TAD-weighted placement mass landing in gene bodies.

    Mirrors ``lesions.precompute_lesion_fields`` (sum of ``lesion_site_p`` over
    gene-body sites). With Type-B disabled only gene-body lesions can be Type A, so
    realised Type-A density = ``total_density * gene_body_mass * type_a_prob``."""
    from ner_rnapii_3d.plugins.lesions import _tad_length_per_site

    lef = base["lef"]
    tk = lef.get("topology_kwargs", {})
    chain_length = int(lef["chain_length"])
    num_chains = int(lef["num_chains"])
    N = chain_length * num_chains
    tad_len, _ = _tad_length_per_site(num_chains, chain_length, tk.get("tad_positions") or [])
    weights = tad_len ** (-float(tad_size_exponent))
    total = float(weights.sum())
    p = (weights / total) if total > 0 else np.full(N, 1.0 / N)
    mask = np.zeros(N, dtype=bool)
    replicate = bool(tk.get("replicate_genes_across_chains", False))
    offsets = [c * chain_length for c in range(num_chains)] if replicate else [0]
    for g in tk.get("genes") or []:
        tss, tes = int(g["tss"]), int(g["tes"])
        lo, hi = (tss, tes) if tss <= tes else (tes, tss)
        for off in offsets:
            a, b = off + lo, off + hi
            if 0 <= a < N:
                mask[a: min(b + 1, N)] = True
    return float(p[mask].sum())


def lesion_density_axis(
    sp_sorted: list[int],
    *,
    type_b_enabled: bool,
    gbm: float = 1.0,
    type_a_prob: float = 1.0,
    bp_per_site: int = 1000,
) -> list[float]:
    """Per-column x-axis lesion density.

    Type-B enabled -> TOTAL density. Type-B disabled -> Type-A density
    ``total * gbm * type_a_prob`` (only gene-body lesions can be Type A)."""
    factor = 1.0 if type_b_enabled else float(gbm) * float(type_a_prob)
    return [density_per_mbp(sp, bp_per_site) * factor for sp in sp_sorted]


def _p_values(p_hi: float, p_lo: float, step: float) -> list[float]:
    """p from p_hi down to p_lo (inclusive) in `step` decrements."""
    if step <= 0:
        raise ValueError("grid step must be positive")
    if p_hi < p_lo:
        raise ValueError(f"grid high bound {p_hi} is below low bound {p_lo}")
    n = int(round((p_hi - p_lo) / step))
    return [round(p_hi - i * step, 6) for i in range(n + 1)]


def build_grid_config(
    base: dict,
    *,
    p: float,
    spacing: int,
    bstr_mult: float,
    type_a_prob: float,
    tick_seconds: float,
    lesion_defaults: dict,
    type_b_enabled: bool = True,
) -> dict:
    """Deep-copied config with RNAPII off, boosted boundaries, and the lesion
    block-prob / spacing set for one grid point. ``type_a_prob`` is a strict
    override applied to every config; ``type_b_enabled=False`` keeps only Type-A
    lesions (Type-A count unchanged, no backfill)."""
    cfg = copy.deepcopy(base)
    if int(spacing) <= 0:
        raise ValueError(f"lesion spacing must be positive, got {spacing}")
    lef = cfg["lef"]

    # --- RNAPII fully off -------------------------------------------------
    lef["max_rnapii"] = 0
    plugins = lef.setdefault("plugins", {})
    plugins["rnapii_load"] = None
    plugins["rnapii_translocate"] = None
    # Enable the lesion plugin (config may not have had it).
    plugins["lesion"] = LESION_PLUGIN

    tk = lef["topology_kwargs"]

    # --- boundary strengths x bstr_mult, capped at 1.0 -------------------
    cap = lambda v: round(min(float(v) * bstr_mult, 1.0), 4)
    for tad in tk.get("tads", []):
        if "left_strength" in tad:
            tad["left_strength"] = cap(tad["left_strength"])
        if "right_strength" in tad:
            tad["right_strength"] = cap(tad["right_strength"])
    if "default_boundary_strength" in tk:
        tk["default_boundary_strength"] = cap(tk["default_boundary_strength"])

    # --- lesion grid point ------------------------------------------------
    tk["lesion_spacing"] = int(spacing)
    tk["lesion_block_prob"] = round(float(p), 4)
    # Strict overrides: always set, never inherited. lesion_defaults already
    # resolves each to the CLI override, else the config value, else the default.
    tk["lesion_type_a_prob"] = round(float(type_a_prob), 4)
    tk["lesion_type_b_enabled"] = bool(type_b_enabled)
    tk["lesion_prerecognition_ticks"] = int(lesion_defaults["prerecognition_ticks"])
    tk["lesion_repair_ticks"] = int(lesion_defaults["repair_ticks"])
    # TAD-size exponents are not (yet) CLI-controlled; carry from the baseline.
    tk.setdefault("lesion_tad_size_exponent", lesion_defaults["tad_size_exponent"])
    tk.setdefault("lesion_tad_repair_exponent", lesion_defaults["tad_repair_exponent"])
    return cfg


def metric_means(rows: list[dict]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    return df.groupby("metric")["raw_value"].mean().to_dict()


def analyze_h5(label: str, cfg_path: Path, h5_path: Path, topo: dict,
               *, positions: np.ndarray | None = None) -> dict[str, float]:
    core, boundary, _aux, _lengths = analyze_run(label, cfg_path, h5_path, topo,
                                                 positions=positions)
    means = metric_means(core)
    means.update(metric_means(boundary))  # identical for shared names
    return means


# Lesion codes (must match plugins/lesions.py).
_LES_TYPE_A, _LES_TYPE_B, _LES_STATE_PRE, _LES_STATE_REPAIR = 0, 1, 0, 1

# Blocking-regime category codes: exactly the (type, state) rows of
# lesions.lesion_stalls_cohesin that stall cohesin (B-PRE never blocks; A-PRE only
# blocks when RNAPII is absent).
_CAT_A_PRE, _CAT_A_REPAIR, _CAT_B_REPAIR = 1, 2, 3
_CATEGORIES = {"a_pre": _CAT_A_PRE, "a_repair": _CAT_A_REPAIR, "b_repair": _CAT_B_REPAIR}

MEASURED_LABELS = {
    "stalled_per_lesion": "Cohesin legs stalled at a lesion\n(per lesion)",
    "stalled_per_blocking_lesion": "Cohesin legs stalled at a lesion\n(per REPAIR-state lesion)",
}

MEASURED_SECONDS_LABELS = {
    "dwell_mean_s": "Mean cohesin stall at a lesion\n(measured, s)",
    "dwell_median_s": "Median cohesin stall at a lesion\n(measured, s)",
}

# Per-(type, state) mean stall durations -- the detailed breakdown.
MEASURED_BYTYPE_LABELS = {
    "a_pre_dwell_mean_s": "Type A pre-recognition stall\n(mean, s)",
    "a_repair_dwell_mean_s": "Type A repair stall\n(mean, s)",
    "b_repair_dwell_mean_s": "Type B repair stall\n(mean, s)",
}

# Per-(type, state) stall as a FRACTION of the lesion's lifetime in that stage:
# per-encounter mean stall / stage-window seconds (PRE -> pre-recognition window,
# REPAIR -> repair window). 1.0 = a held leg occupies the whole stage on average.
MEASURED_STAGEFRAC_LABELS = {
    "a_pre_stall_frac_of_stage": "Type A pre-recognition\nstall / stage lifetime",
    "a_repair_stall_frac_of_stage": "Type A repair\nstall / stage lifetime",
    "b_repair_stall_frac_of_stage": "Type B repair\nstall / stage lifetime",
}


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _dwell_summary(run_ticks: list[int], tick_seconds: float, prefix: str,
                   legframes: int, frames: int, n_lefs: int) -> dict[str, float]:
    """Mean/median/p90/max/n + leg-time fraction for one set of stall segments."""
    d = np.asarray(run_ticks, dtype=float) * float(tick_seconds)
    return {
        f"{prefix}_dwell_mean_s": float(d.mean()) if d.size else float("nan"),
        f"{prefix}_dwell_median_s": _pct(d, 50),
        f"{prefix}_dwell_p90_s": _pct(d, 90),
        f"{prefix}_dwell_max_s": float(d.max()) if d.size else float("nan"),
        f"{prefix}_n_events": int(d.size),
        f"{prefix}_legtick_fraction": legframes / max(frames * n_lefs * 2, 1),
    }


def _stage_frac(out: dict[str, float], tick_seconds: float,
                prerecognition_ticks: float | None, repair_ticks: float | None) -> None:
    """Add ``<cat>_stall_frac_of_stage`` to ``out``: per-encounter mean stall over
    the lesion's mean lifetime in that stage (PRE -> pre-recognition window,
    REPAIR -> repair window). NaN when the window is unknown or there was no stall."""
    stage_ticks = {"a_pre": prerecognition_ticks,
                   "a_repair": repair_ticks, "b_repair": repair_ticks}
    for cat in _CATEGORIES:
        win_t = stage_ticks.get(cat)
        win_s = float(win_t) * float(tick_seconds) if win_t else 0.0
        dwell = out.get(f"{cat}_dwell_mean_s", float("nan"))
        out[f"{cat}_stall_frac_of_stage"] = (
            dwell / win_s if win_s > 0 and np.isfinite(dwell) else float("nan"))


def measure_lesion_stall(
    h5_path: Path, tick_seconds: float, chunk: int = 2000,
    *, prerecognition_ticks: float | None = None, repair_ticks: float | None = None,
    positions: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure lesion-induced cohesin stalling directly from a 1D trajectory.

    A leg is lesion-stalled in a frame when the site it would extrude into holds a
    blocking lesion: the left leg (``positions[:,j,0]``) steps to ``pos-1``, the
    right (``positions[:,j,1]``) to ``pos+1`` (left.pos <= right.pos always; slot
    ``j`` is a stable column so a reload jump ends a run). A lesion blocks cohesin
    when in REPAIR (either type) or PRE as Type A with RNAPII absent -- exactly
    ``lesions.lesion_stalls_cohesin``.

    Two readouts, neither using an analytic formula: time-averaged occupancy
    (``stalled_per_lesion`` etc.), and dwell DURATION -- a stall event is a maximal
    run of consecutive blocked frames for one ``(slot, leg)``, length x
    ``tick_seconds``. Runs touching frame 0 (left-censored) or open at the final
    frame (right-censored) are dropped. Each frame is one tick, so this needs FULL
    frames (no striding); frames stream in blocks of ``chunk`` to bound memory.

    Returns occupancy keys, the OVERALL stall duration (``dwell_*`` /
    ``n_stall_events`` / ``stalled_legtick_fraction``), and a per-regime breakdown
    (``a_pre`` / ``a_repair`` / ``b_repair`` with ``<cat>_dwell_*`` / ``_n_events``
    / ``_legtick_fraction``). The overall run is the full continuous stall; the
    per-category segments split it where the blocking lesion's (type, state) changes.
    """
    keys = ("stalled_legs_per_frame", "lesions_per_frame", "blocking_lesions_per_frame",
            "stalled_per_lesion", "stalled_per_blocking_lesion",
            "dwell_mean_s", "dwell_median_s", "dwell_p90_s", "dwell_max_s",
            "n_stall_events", "stalled_legtick_fraction")
    with h5py.File(h5_path, "r") as h:
        N = int(h.attrs["N"])
        chain = int(h.attrs["chain_length"])
        rnapii = bool(h.attrs.get("rnapii_enabled", False))
        if "lesions" not in h:
            base = {k: float("nan") for k in keys}
            for cat in _CATEGORIES:
                base.update(_dwell_summary([], tick_seconds, cat, 0, 1, 1))
            _stage_frac(base, tick_seconds, prerecognition_ticks, repair_ticks)
            return base
        # ``positions`` may be supplied by a caller that already loaded the dataset,
        # sharing one decompress; same bytes -> every stall count/dwell is identical.
        frames = int(positions.shape[0]) if positions is not None else int(h["positions"].shape[0])
        n_lefs = int(positions.shape[1]) if positions is not None else int(h["positions"].shape[1])
        ds_sites, ds_states, ds_types = h["lesions"], h["lesion_states"], h["lesion_types"]
        ds_pos = h["positions"] if positions is None else None
        if frames == 0:
            raise ValueError(f"{h5_path}: lesion-stall metrics require at least one frame")
        if positions is not None and (
            positions.ndim != 3 or positions.shape[2] != 2
            or positions.shape[:2] != h["positions"].shape[:2]
        ):
            raise ValueError(
                f"positions shape {positions.shape} does not match H5 positions "
                f"shape {h['positions'].shape}"
            )
        widest_row = max(int(ds_sites.shape[1]), n_lefs, 1)
        chunk = min(max(1, int(chunk)), max(1, 2_000_000 // widest_row))

        stalled = lesions_tot = blocking_tot = 0
        # Overall continuous stall (any blocking lesion). Completed run lengths are
        # collected as per-frame arrays and concatenated once at the end (same values
        # in the same order, so every dwell statistic is byte-identical).
        runs: list[np.ndarray] = []
        cur = np.zeros((n_lefs, 2), dtype=np.int64)
        from_zero = np.zeros((n_lefs, 2), dtype=bool)
        # Per-category segments (split when the blocking category changes).
        cat_runs: dict[int, list[np.ndarray]] = {c: [] for c in _CATEGORIES.values()}
        cat_legframes: dict[int, int] = {c: 0 for c in _CATEGORIES.values()}
        cur_cat = np.zeros((n_lefs, 2), dtype=np.int8)
        cur_len = np.zeros((n_lefs, 2), dtype=np.int64)
        from_zero_cat = np.zeros((n_lefs, 2), dtype=bool)
        gf = 0
        # Category lookup indexed by ``type*2 + state``: 0=(A,PRE) 1=(A,REPAIR)
        # 2=(B,PRE) 3=(B,REPAIR). A-PRE blocks only when RNAPII is absent; B-PRE
        # never blocks (matches lesions.lesion_stalls_cohesin).
        cat_lut = np.array(
            [_CAT_A_PRE if not rnapii else 0, _CAT_A_REPAIR, 0, _CAT_B_REPAIR],
            dtype=np.int8,
        )
        for start in range(0, frames, chunk):
            end = min(start + chunk, frames)
            sites = ds_sites[start:end]
            states = ds_states[start:end]
            types = ds_types[start:end]
            pos = positions[start:end] if positions is not None else ds_pos[start:end]

            # Blocking category per recorded lesion via a table lookup on
            # (type*2 + state), with -1 padding / out-of-range sites cleared.
            # Recorded lesions are site-sorted with live entries packed into the low
            # columns and -1 padding in the tail, so a frame's valid set is its
            # leading ``row_nv[k]`` columns (sliced below, no per-frame mask).
            valid_blk = (sites >= 0) & (sites < N)
            code = types.astype(np.int16)
            code *= 2
            code += states
            np.clip(code, 0, 3, out=code)
            block_cat = cat_lut[code]
            block_cat[~valid_blk] = 0
            row_nv = valid_blk.sum(axis=1)
            lesions_tot += int(valid_blk.sum())
            blocking_tot += int((block_cat > 0).sum())

            for k in range(end - start):
                nv = int(row_nv[k])
                s = sites[k, :nv]
                site_categories = block_cat[k, :nv]
                # Re-sort only if the site-sorted invariant is somehow violated, so
                # the searchsorted leg queries below stay correct.
                if nv > 1 and np.any(s[1:] < s[:-1]):
                    order = np.argsort(s, kind="stable")
                    s = s[order]
                    site_categories = site_categories[order]

                left = pos[k, :, 0]
                right = pos[k, :, 1]
                legcat = np.zeros((n_lefs, 2), dtype=np.int8)  # blocking category at each leg's next site
                l_ok = (left % chain) != 0           # left leg's -1 step stays in-chain
                r_ok = ((right + 1) % chain) != 0    # right leg's +1 step stays in-chain
                for target, ok, side in ((left - 1, l_ok, 0), (right + 1, r_ok, 1)):
                    query = target[ok]
                    idx = np.searchsorted(s, query)
                    found = idx < s.size
                    if found.any():
                        found[found] = s[idx[found]] == query[found]
                    dest = np.flatnonzero(ok)
                    legcat[dest[found], side] = site_categories[idx[found]]
                blocked = legcat > 0
                stalled += int(blocked.sum())
                for c in _CATEGORIES.values():
                    cat_legframes[c] += int((legcat == c).sum())

                # --- overall continuous-stall runs (any category) ---
                ended = (~blocked) & (cur > 0)
                keep = ended & (~from_zero)
                if keep.any():
                    runs.append(cur[keep])
                cur[ended] = 0
                from_zero[ended] = False
                cur[blocked] += 1
                if gf == 0:
                    from_zero[blocked] = True

                # --- per-category segments (split on category change) ---
                cont = (legcat == cur_cat) & (legcat > 0)
                seg_ended = (cur_len > 0) & (~cont)
                for c in _CATEGORIES.values():
                    sel = seg_ended & (cur_cat == c) & (~from_zero_cat)
                    if sel.any():
                        cat_runs[c].append(cur_len[sel])
                cur_len[cont] += 1
                cur_len[seg_ended] = 0
                cur_cat[seg_ended] = 0
                from_zero_cat[seg_ended] = False
                new_seg = (legcat > 0) & (~cont)
                cur_len[new_seg] = 1
                cur_cat[new_seg] = legcat[new_seg]
                if gf == 0:
                    from_zero_cat[new_seg] = True
                gf += 1

        # open runs at trajectory end are right-censored -> dropped
        runs_arr = np.concatenate(runs) if runs else np.empty(0, dtype=np.int64)
        dwell = runs_arr.astype(float) * float(tick_seconds)
        out = {
            "stalled_legs_per_frame": stalled / frames,
            "lesions_per_frame": lesions_tot / frames,
            "blocking_lesions_per_frame": blocking_tot / frames,
            "stalled_per_lesion": stalled / max(lesions_tot, 1),
            "stalled_per_blocking_lesion": stalled / max(blocking_tot, 1),
            "dwell_mean_s": float(dwell.mean()) if dwell.size else float("nan"),
            "dwell_median_s": _pct(dwell, 50),
            "dwell_p90_s": _pct(dwell, 90),
            "dwell_max_s": float(dwell.max()) if dwell.size else float("nan"),
            "n_stall_events": int(dwell.size),
            "stalled_legtick_fraction": stalled / max(frames * n_lefs * 2, 1),
        }
        for cat, code in _CATEGORIES.items():
            parts = cat_runs[code]
            cat_arr = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            out.update(_dwell_summary(cat_arr, tick_seconds, cat,
                                      cat_legframes[code], frames, n_lefs))
        _stage_frac(out, tick_seconds, prerecognition_ticks, repair_ticks)
        return out


def grid_label(p: float, spacing: int) -> str:
    return _safe_label(f"grid_p{p:.3f}_s{spacing}")


def _run_grid_point(task: dict) -> dict:
    """Worker: run (or reuse) the 1D LEF stage for one grid point, returning its H5
    path. Top-level + picklable for the process pool; each process re-seeds numpy
    from the config seed, so results are deterministic and worker-independent."""
    h5_path = _resolve_h5(
        cfg_path=Path(task["cfg_path"]),
        requested_h5=None,
        out_dir=Path(task["h5_dir"]),
        label=task["label"],
        force_1d=task["force_1d"],
    )
    return {"ri": task["ri"], "ci": task["ci"], "label": task["label"], "h5": str(h5_path)}


def _measure_grid_point(task: dict, h5_path: Path) -> dict:
    """Compute one grid point's fold means + (optionally) the measured lesion-stall
    dict from its trajectory. ``task`` carries the analysis inputs (``topo`` and,
    when ``measure`` is set, ``tick_seconds`` / ``measure_chunk`` / ``prerec_ticks``
    / ``repair_ticks``). The parent accumulates results order-independently via
    ``ri``/``ci``, so output is byte-identical to computing them inline."""
    means = analyze_h5(task["label"], Path(task["cfg_path"]), Path(h5_path), task["topo"])
    ms = None
    if task.get("measure"):
        ms = measure_lesion_stall(
            Path(h5_path), task["tick_seconds"], chunk=task["measure_chunk"],
            prerecognition_ticks=task["prerec_ticks"], repair_ticks=task["repair_ticks"])
    return {"ri": task["ri"], "ci": task["ci"], "label": task["label"],
            "h5": str(h5_path), "means": means, "ms": ms}


def run_and_measure_grid_point(task: dict) -> dict:
    """Worker: run/reuse one grid point's 1D H5 AND analyse it in the same process,
    so the measurement parallelises with the 1D stage. Top-level + picklable;
    deterministic and per-point independent -> byte-identical to the serial path."""
    res = _run_grid_point(task)
    return _measure_grid_point(task, Path(res["h5"]))


def plot_heatmaps(
    grids: dict[str, np.ndarray],
    labels: dict[str, str],
    *,
    y_values: list[float],
    y_label: str,
    density_axis: list[float],
    out_path: Path,
    title: str,
    value_label: str,
    y_fmt: str = "{:.3f}",
    cell_fmt: str = "{:.2f}",
    cell_text: "Callable[[float], str] | None" = None,
    cell_fontsize: float = 6.0,
    center: float | None = 1.0,
    log_norm: bool = False,
    cmap=FOLD_CMAP,
    x_label: str = "Lesion density (lesions/Mbp)",
    y_ticklabels: list[str] | None = None,
) -> None:
    """Grid of heatmaps, one per metric in ``grids``. Rows = ``y_values``
    (ascending), cols = density (ascending).

    ``center``: a float centres a diverging map there (e.g. fold = 1.0); ``None``
    uses a sequential scale spanning the data, and with ``log_norm=True`` a log
    scale over positive data (cell annotations always show raw values). ``cell_text``
    overrides only the in-cell text (callable on finite values; else
    ``cell_fmt.format(value)``); the colour scale is unaffected."""
    metrics = [m for m in labels if m in grids]
    ncols = min(3, max(1, len(metrics)))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    if y_ticklabels is None:
        y_ticklabels = [("∞" if not np.isfinite(v) else y_fmt.format(v)) for v in y_values]
    x_ticklabels = [f"{d:.1f}" for d in density_axis]

    for idx, metric in enumerate(metrics):
        ax = axes.flat[idx]
        grid = grids[metric]
        finite = grid[np.isfinite(grid)]
        dmin, dmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if center is not None:
            vmin, vmax = min(dmin, center - 1e-6), max(dmax, center + 1e-6)
            norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        elif log_norm:
            pos = finite[finite > 0]
            lo = float(pos.min()) if pos.size else 1e-6
            norm = LogNorm(vmin=lo, vmax=dmax if dmax > lo else lo * 10.0)
        else:
            norm = Normalize(vmin=dmin, vmax=dmax if dmax > dmin else dmin + 1e-6)
        cmap_obj = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
        cmap_obj.set_bad("#f1f3f5")
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap_obj, norm=norm)

        ax.set_xticks(range(len(density_axis)))
        ax.set_xticklabels(x_ticklabels, fontsize=8)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels(y_ticklabels, fontsize=8)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(labels[metric], fontsize=10, pad=6)

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if not np.isfinite(v):
                    txt = "n/a"
                elif cell_text is not None:
                    txt = cell_text(v)
                else:
                    txt = cell_fmt.format(v)
                ax.text(c, r, txt, ha="center", va="center", fontsize=cell_fontsize, color="#111111")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(value_label, fontsize=8)

    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def write_grid_tsv(
    folds: dict[str, np.ndarray],
    raws: dict[str, np.ndarray],
    *,
    p_values: list[float],
    spacings: list[int],
    stall_axis: list[float],
    raw_bypass_axis: list[float],
    density_axis: list[float],
    tick_seconds: float,
    baseline: dict[str, float],
    out_path: Path,
) -> None:
    rows = []
    for ri, p in enumerate(p_values):
        for ci, sp in enumerate(spacings):
            for metric in folds:
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "lesion_block_prob": p,
                        "lesion_spacing": sp,
                        # Realized stall used for the y-axis (repair-/lifetime-capped)
                        # plus the naive bypass time for reference/comparison.
                        "effective_stall_seconds": stall_axis[ri],
                        "raw_bypass_seconds": raw_bypass_axis[ri],
                        "lesion_density_per_mbp": density_axis[ci],
                        "tick_seconds": tick_seconds,
                        "baseline_mean": baseline.get(metric, float("nan")),
                        "grid_mean": raws[metric][ri, ci],
                        "fold_vs_baseline": folds[metric][ri, ci],
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True, help="baseline config YAML")
    ap.add_argument("--h5", type=Path, required=True, help="baseline LEFPositions.h5 for --config")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory (configs, h5s, heatmaps)")
    ap.add_argument("--bstr-mult", type=float, default=3.0, help="boundary-strength multiplier (capped at 1.0)")
    ap.add_argument(
        "--type-a-prob",
        type=float,
        default=DEFAULT_TYPE_A_PROB,
        help="strict lesion_type_a_prob override applied to every grid config "
             "(P(Type A | in a gene body); off-gene lesions are always Type B)",
    )
    ap.add_argument(
        "--lesion-type-b-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include Type-B (GG-NER) lesions (default). --no-lesion-type-b-enabled "
             "keeps only Type-A lesions and switches the heatmap x-axis to Type-A density.",
    )
    ap.add_argument(
        "--repair-seconds",
        type=float,
        default=None,
        help="lesion repair window in SECONDS (REPAIR->removed; the window in which a "
             "Type-B lesion blocks cohesin). Converted to lesion_repair_ticks via "
             "round(seconds / tick_seconds). Default: inherit from --config, else ~300s.",
    )
    ap.add_argument(
        "--prerecognition-seconds",
        type=float,
        default=None,
        help="lesion pre-recognition window in SECONDS (PRE->REPAIR). Converted to "
             "lesion_prerecognition_ticks via round(seconds / tick_seconds). Longer PRE "
             "means a smaller steady-state REPAIR (blocking) fraction. Default: inherit "
             "from --config, else ~1200s.",
    )
    ap.add_argument(
        "--block-probs", type=float, nargs="+", default=None,
        help="explicit lesion_block_prob values (rows). When given, these are used "
             "directly and --p-hi/--p-lo/--p-step are ignored.",
    )
    ap.add_argument("--p-hi", type=float, default=1.0, help="highest lesion_block_prob (top row)")
    ap.add_argument("--p-lo", type=float, default=0.8, help="lowest lesion_block_prob (bottom row)")
    ap.add_argument("--p-step", type=float, default=0.025, help="lesion_block_prob decrement")
    ap.add_argument(
        "--spacings",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 150, 200],
        help="lesion_spacing values in lattice SITES (columns); "
             "density = 1e6/(spacing*bp_per_site) lesions/Mbp (bp_per_site from config)",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="only generate the grid configs; do not run 1D or plot",
    )
    ap.add_argument(
        "--force-1d",
        action="store_true",
        help="rerun the 1D LEF stage for every grid point even if a matching H5 exists",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of grid-point 1D sims to run concurrently (process pool); "
             "1 = sequential. Each sim is an independent process writing its own "
             "H5, so results are identical to sequential.",
    )
    ap.add_argument(
        "--no-lifetime-cap",
        action="store_true",
        help="effective-stall reference column: cap only by lesion repair (bypass + "
             "repair), excluding the cohesin stalled-lifetime route. Default includes it.",
    )
    ap.add_argument(
        "--no-measure",
        action="store_true",
        help="skip the measured lesion-stall heatmap (only emit the fold heatmaps)",
    )
    ap.add_argument(
        "--measure-chunk",
        type=int,
        default=2000,
        help="frames per streamed block when measuring lesion stalls from the H5 "
             "(memory knob; dwell durations need full frames so striding is not used)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    h5_dir = out_dir / "h5"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(args.config.read_text())
    lef = base["lef"]
    tick_seconds = float(lef.get("tick_seconds") or 0.0)
    if tick_seconds <= 0:
        raise SystemExit("config lef.tick_seconds must be a positive number to convert stall times")
    bp_per_site = int(lef.get("bp_per_site", 1000))   # lattice resolution; lesion density depends on it

    tk = lef["topology_kwargs"]
    lesion_defaults = {
        "prerecognition_ticks": tk.get(
            "lesion_prerecognition_ticks", max(1, int(round(LESION_PRERECOGNITION_SECONDS / tick_seconds)))
        ),
        "repair_ticks": tk.get(
            "lesion_repair_ticks", max(1, int(round(LESION_REPAIR_SECONDS / tick_seconds)))
        ),
        "tad_size_exponent": tk.get("lesion_tad_size_exponent", DEFAULT_TAD_SIZE_EXPONENT),
        "tad_repair_exponent": tk.get("lesion_tad_repair_exponent", DEFAULT_TAD_REPAIR_EXPONENT),
    }
    # Strict CLI overrides given in SECONDS, converted to ticks (floored at 1) and
    # applied uniformly across every grid config.
    if args.repair_seconds is not None:
        lesion_defaults["repair_ticks"] = max(1, int(round(args.repair_seconds / tick_seconds)))
    if args.prerecognition_seconds is not None:
        lesion_defaults["prerecognition_ticks"] = max(1, int(round(args.prerecognition_seconds / tick_seconds)))

    if args.block_probs is not None:
        p_values = sorted(set(float(p) for p in args.block_probs))
    else:
        p_values = _p_values(args.p_hi, args.p_lo, args.p_step)
    spacings = list(args.spacings)
    # Axis orderings: ascending stall (rows) and ascending density (cols).
    p_sorted = sorted(p_values)  # ascending p -> ascending stall (0.8..1.0)
    sp_sorted = sorted(spacings, reverse=True)  # 200..10 -> ascending density

    # Effective cohesin stall (s): bounded by lesion repair and -- unless
    # --no-lifetime-cap -- the cohesin stalled lifetime. repair_ticks is uniform
    # across the grid (not swept); lifetime_stalled is read from the config.
    repair_ticks = float(lesion_defaults["repair_ticks"])
    lifetime_stalled = float(lef.get("lifetime_stalled") or lef.get("lifetime") or 0) or None
    life_term = None if args.no_lifetime_cap else lifetime_stalled
    stall_axis = [effective_stall_seconds(p, tick_seconds, repair_ticks, life_term) for p in p_sorted]
    raw_bypass_axis = [raw_bypass_seconds(p, tick_seconds) for p in p_sorted]
    # X-axis density: total (B on) or Type-A density (B off, scaled by the
    # gene-body mass and the fixed type_a_prob).
    type_b_enabled = bool(args.lesion_type_b_enabled)
    gbm = gene_body_mass(base, float(lesion_defaults["tad_size_exponent"])) if not type_b_enabled else 1.0
    density_axis = lesion_density_axis(
        sp_sorted, type_b_enabled=type_b_enabled, gbm=gbm, type_a_prob=args.type_a_prob,
        bp_per_site=bp_per_site)
    x_label = ("Lesion density (lesions/Mbp)" if type_b_enabled
               else "Type-A lesion density (lesions/Mbp)")

    print(f"baseline config : {args.config}")
    print(f"baseline H5     : {args.h5}")
    print(f"tick_seconds    : {tick_seconds:g}")
    print(f"grid            : {len(p_sorted)} stall x {len(sp_sorted)} density = "
          f"{len(p_sorted) * len(sp_sorted)} configs")
    repair_seconds = repair_ticks * tick_seconds
    life_note = (
        f"+ cohesin stalled lifetime {lifetime_stalled * tick_seconds:.0f}s"
        if life_term else "(repair only; cohesin lifetime excluded)"
    )
    print(f"  lesion_block_prob : {p_sorted}")
    print(f"  eff stall (s)     : {[round(s, 1) for s in stall_axis]}")
    print(f"    capped by lesion repair {repair_seconds:.0f}s {life_note}")
    print(f"  raw bypass (old)  : {['inf' if not np.isfinite(s) else round(s, 1) for s in raw_bypass_axis]}  <- diverges; not used")
    print(f"  spacing           : {sp_sorted}")
    if type_b_enabled:
        print(f"  density /Mbp      : {[round(d,2) for d in density_axis]}  (total; Type-B enabled)")
    else:
        print(f"  Type-B DISABLED -> x-axis = Type-A density (gene_body_mass={gbm:.3f} x type_a={args.type_a_prob:g})")
        print(f"  Type-A density /Mbp: {[round(d,2) for d in density_axis]}")
    prerec_ticks = int(lesion_defaults["prerecognition_ticks"])
    repair_t = int(lesion_defaults["repair_ticks"])
    # Type-B lesions only block while in REPAIR; the steady-state REPAIR fraction
    # is repair/(prerec+repair) (uniform-TAD approximation).
    repair_frac = repair_t / (prerec_ticks + repair_t)
    rsrc = "CLI(s)" if args.repair_seconds is not None else "config"
    psrc = "CLI(s)" if args.prerecognition_seconds is not None else "config"
    print(f"  lesion_repair_ticks       : {repair_t} (~{repair_t * tick_seconds:.0f}s) [{rsrc}]")
    print(f"  lesion_prerecognition_ticks: {prerec_ticks} (~{prerec_ticks * tick_seconds:.0f}s) [{psrc}]")
    print(f"    -> steady-state REPAIR (blocking) fraction ~ {repair_frac:.2f}")
    print(f"  boundary x{args.bstr_mult:g} (cap 1.0), lesion_type_a_prob={args.type_a_prob:g}, "
          f"max_rnapii=0, rnapii_load/translocate=null")

    # 1) generate every grid config (always) ------------------------------
    grid_cfg_paths: dict[tuple[float, int], Path] = {}
    for p in p_sorted:
        for sp in sp_sorted:
            cfg = build_grid_config(
                base, p=p, spacing=sp, bstr_mult=args.bstr_mult,
                type_a_prob=args.type_a_prob,
                tick_seconds=tick_seconds, lesion_defaults=lesion_defaults,
                type_b_enabled=type_b_enabled,
            )
            path = cfg_dir / f"{grid_label(p, sp)}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
            grid_cfg_paths[(p, sp)] = path
    print(f"wrote {len(grid_cfg_paths)} grid configs to {cfg_dir}")

    if args.skip_run:
        print("--skip-run set: configs generated, skipping 1D + heatmaps")
        return

    # Topology is identical across baseline + grid (strengths don't move masks),
    # so compute it once and reuse for every analyze_run -> topology-matched folds.
    topo = _topology(base)

    # 2) baseline metric means -------------------------------------------
    baseline = analyze_h5("baseline", args.config, args.h5.resolve(), topo)

    # 3) run + analyze every grid point ----------------------------------
    metrics_seen = list(METRIC_LABELS)
    shape = (len(p_sorted), len(sp_sorted))
    raws = {m: np.full(shape, np.nan) for m in metrics_seen}
    folds = {m: np.full(shape, np.nan) for m in metrics_seen}
    # Measured (not vs baseline): lesion-induced cohesin stall, read from the H5 --
    # both time-averaged occupancy and realized dwell DURATION in seconds.
    measured_keys = [
        "stalled_per_lesion", "stalled_per_blocking_lesion",
        "stalled_legs_per_frame", "lesions_per_frame", "blocking_lesions_per_frame",
        "dwell_mean_s", "dwell_median_s", "dwell_p90_s", "dwell_max_s",
        "n_stall_events", "stalled_legtick_fraction",
    ]
    # Per-(type, state) breakdown: a_pre / a_repair / b_repair x stat.
    for cat in _CATEGORIES:
        measured_keys += [f"{cat}_dwell_mean_s", f"{cat}_dwell_median_s",
                          f"{cat}_dwell_p90_s", f"{cat}_dwell_max_s",
                          f"{cat}_n_events", f"{cat}_legtick_fraction"]
    # Per-(type, state) stall fraction of the lesion's stage lifetime.
    measured_keys += list(MEASURED_STAGEFRAC_LABELS)
    measured = {k: np.full(shape, np.nan) for k in measured_keys}

    # Each task carries the per-point analysis inputs (topo + measurement knobs) so
    # the worker can run the 1D stage and measure in one process.
    tasks = [
        {
            "ri": ri, "ci": ci, "p": p, "sp": sp,
            "label": grid_label(p, sp),
            "cfg_path": str(grid_cfg_paths[(p, sp)]),
            "h5_dir": str(h5_dir),
            "force_1d": args.force_1d,
            "topo": topo,
            "measure": not args.no_measure,
            "tick_seconds": tick_seconds,
            "measure_chunk": args.measure_chunk,
            "prerec_ticks": prerec_ticks,
            "repair_ticks": repair_t,
        }
        for ri, p in enumerate(p_sorted)
        for ci, sp in enumerate(sp_sorted)
    ]
    total = len(tasks)
    jobs = max(1, int(args.jobs))

    def _accumulate(result: dict) -> None:
        ri, ci = result["ri"], result["ci"]
        means = result["means"]
        for m in metrics_seen:
            gm = means.get(m, float("nan"))
            bm = baseline.get(m, float("nan"))
            raws[m][ri, ci] = gm
            # Fold is undefined when the baseline mean is 0 or non-finite (e.g. a
            # metric that never fires in the baseline run); leave NaN so the
            # heatmap masks that cell instead of drawing inf.
            folds[m][ri, ci] = gm / bm if np.isfinite(bm) and bm != 0 else float("nan")
        if not args.no_measure:
            ms = result["ms"]
            for k in measured:
                measured[k][ri, ci] = ms[k]

    if jobs == 1:
        for n, task in enumerate(tasks, 1):
            print(f"[{n}/{total}] {task['label']}  "
                  f"(p={task['p']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
            _accumulate(run_and_measure_grid_point(task))
    else:
        # Process pool: each grid point re-seeds numpy from the config and writes
        # its own H5 -> results are bitwise-identical to sequential. A fork context
        # avoids re-importing this module (and matplotlib) in every worker.
        # ``ex.map`` preserves order, so zipping with ``tasks`` recovers axis values.
        print(f"running {total} grid points with --jobs {jobs} (process pool, fork context)")
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for n, (task, result) in enumerate(
                zip(tasks, ex.map(run_and_measure_grid_point, tasks)), 1
            ):
                print(f"[{n}/{total}] done {result['label']}  "
                      f"(p={task['p']:.3f}, density={density_axis[task['ci']]:.1f}/Mbp)")
                _accumulate(result)

    # 4) outputs ----------------------------------------------------------
    plots_dir = out_dir
    svg_path = plots_dir / "lesion_grid_heatmaps.svg"
    tsv_path = plots_dir / "lesion_grid_folds.tsv"
    y_label = "Lesion block probability p (per tick)"
    # Fold heatmaps: y-axis is the raw swept knob (lesion_block_prob); colour bar
    # centred at fold = 1.0.
    plot_heatmaps(
        folds, METRIC_LABELS,
        y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
        out_path=svg_path,
        title=(
            f"Lesion grid vs baseline 1D sim (fold = grid mean / baseline mean)\n"
            f"RNAPII off, boundaries x{args.bstr_mult:g} (cap 1.0), tick={tick_seconds:g}s"
        ),
        value_label="fold vs baseline mean",
        center=1.0,
    )
    write_grid_tsv(
        folds, raws,
        p_values=p_sorted, spacings=sp_sorted,
        stall_axis=stall_axis, raw_bypass_axis=raw_bypass_axis, density_axis=density_axis,
        tick_seconds=tick_seconds, baseline=baseline, out_path=tsv_path,
    )
    print(f"wrote {svg_path}")
    print(f"wrote {tsv_path}")

    # Measured lesion-stall heatmaps (standalone; NOT vs baseline) ---------
    if not args.no_measure:
        # (a) per-lesion occupancy: how busy each lesion is.
        occ_svg = plots_dir / "lesion_grid_measured_stall.svg"
        plot_heatmaps(
            measured, MEASURED_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=occ_svg,
            title=(
                "Measured lesion-induced cohesin stall (from 1D trajectory, not vs baseline)\n"
                f"per-lesion normalised; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stalled cohesin legs per lesion",
            cell_fmt="{:.3f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (b) realized dwell DURATION in seconds: how long a stall lasts.
        sec_svg = plots_dir / "lesion_grid_measured_stall_seconds.svg"
        plot_heatmaps(
            measured, MEASURED_SECONDS_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=sec_svg,
            title=(
                "Measured cohesin stall DURATION at a lesion (from 1D trajectory)\n"
                f"per stall event, in seconds; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall duration (s)",
            cell_fmt="{:.0f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (c) per-(type, state) breakdown: A-PRE / A-REPAIR / B-REPAIR mean stall.
        bytype_svg = plots_dir / "lesion_grid_measured_stall_by_type.svg"
        plot_heatmaps(
            measured, MEASURED_BYTYPE_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=bytype_svg,
            title=(
                "Measured cohesin stall by lesion type/state (from 1D trajectory)\n"
                f"mean per-regime stall, in seconds; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall duration (s)",
            cell_fmt="{:.0f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # (d) per-(type, state) stall as a FRACTION of the lesion's stage lifetime.
        frac_svg = plots_dir / "lesion_grid_measured_stall_fraction.svg"
        plot_heatmaps(
            measured, MEASURED_STAGEFRAC_LABELS,
            y_values=p_sorted, y_label=y_label, density_axis=density_axis, x_label=x_label,
            out_path=frac_svg,
            title=(
                "Measured cohesin stall as a fraction of the lesion's stage lifetime\n"
                f"per-encounter mean stall / stage window; RNAPII off, tick={tick_seconds:g}s"
            ),
            value_label="stall / stage lifetime",
            cell_fmt="{:.2f}",
            center=None,
            cmap=SEQ_CMAP,
        )
        # measured TSV: occupancy + dwell distribution + per-type + analytic ref.
        m_tsv = plots_dir / "lesion_grid_measured_stall.tsv"
        m_rows = []
        for ri, p in enumerate(p_sorted):
            for ci, sp in enumerate(sp_sorted):
                m_rows.append({
                    "lesion_block_prob": p,
                    "lesion_spacing": sp,
                    "lesion_density_per_mbp": density_axis[ci],
                    **{k: measured[k][ri, ci] for k in measured},
                    "effective_stall_seconds_analytic": stall_axis[ri],
                })
        pd.DataFrame(m_rows).to_csv(m_tsv, sep="\t", index=False)
        print(f"wrote {occ_svg}")
        print(f"wrote {sec_svg}")
        print(f"wrote {bytype_svg}")
        print(f"wrote {frac_svg}")
        print(f"wrote {m_tsv}")


if __name__ == "__main__":
    main()
