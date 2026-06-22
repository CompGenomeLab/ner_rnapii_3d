#!/usr/bin/env python
"""Transcription-kinetics metrics for a transcription-ON (RNAPII-enabled) config.

Given ONE config (and optionally its ``LEFPositions.h5``; the 1D stage is run or
reused if absent), this measures per-gene + locus-aggregate transcription rates
DIRECTLY from the RNAPII trajectory -- no analytic formula. It is the
transcription counterpart to ``gen_lesion_grid_and_heatmaps.py``'s measured
cohesin-stall readout, and reuses the same ``_resolve_h5`` 1D runner.

The trajectory exposes (see ner_rnapii_3d/engine/simulate.py):

  * ``rnapii_positions`` (T, R, 2) = [global_site, gene_id], -1 = absent.
  * ``rnapii_states``    (T, R)    = PRE-INITIATION/PAUSED/ELONGATING/TERMINATING/STALLED.
  * ``rnapii_ids``       (T, R)    = stable per-Pol uid (survives column reshuffle
                                     on unload) -> reconstructs each polymerase's
                                     load -> unload life, the basis for completion
                                     counts, per-molecule velocity and dwell times.
  * ``genes``            struct    = (gene_id, tss, tes, direction, load_prob).
  * ``lesions`` / ``positions``    (optional) -> co-transcriptional stall attribution.

Each recorded frame is one tick; ``tick_seconds`` (from the config) converts ticks
to seconds. Four metric families are computed per gene and as a locus aggregate:

  CORE RATES & DENSITY
    initiation_rate_per_min    new Pol II loaded at the TSS / time (bursty input)
    completion_rate_per_min    Pol II reaching the TES (= transcript production rate)
    synthesis_rate_nt_per_s    sites transcribed / time (nascent RNA throughput)
    pol2_occupancy             mean Pol II present per tick (ChIP analog, per allele; incl. the 3' window)
    pol2_density_per_kb        occupancy / gene length (kb)
    nascent_signal             mean ELONGATING|STALLED Pol II per tick (engaged)
    term_zone_occupancy        mean Pol II in the 3' termination window past the TES (per tick)
    elongation_velocity_bp_s   sites advanced / elongating-tick (bp/s; excludes the 3' termination crawl)

  Position classes are window-aware: rel = (pos-TSS)*direction; rel<=pause_offset is the
  promoter/pause, pause_offset<rel<=gene_length is the gene body (sets the pausing-index
  denominator), and rel>gene_length is the 3' termination window (Schwalb 2016) -- the last
  is excluded from gene-body density so the pausing index is not diluted by terminating Pol.

  PAUSING & STATE KINETICS
    pausing_index              (TSS..pause-site) / gene-body density (travelling ratio)
    frac_{pre_initiation,paused,elongating,stalled,terminating}   state-time fractions
    mean_{pre_initiation,pause,terminating}_dwell_s               per-state residence (s)
    mean_residence_s           full load->unload transcription time (s)
    pause_release_efficiency   completed / initiated

  CO-TRANSCRIPTIONAL INTERFERENCE
    stall_frac                 STALLED / (ELONGATING+STALLED) engaged-time
    lesion_stall_frac          of STALLED ticks, fraction with a blocking lesion
                               immediately ahead (NaN if no ``lesions`` dataset)
    cohesin_stall_frac         of STALLED ticks, fraction with a cohesin leg
                               immediately ahead (NaN if no ``positions`` dataset)

  TRANSCRIPTIONAL BURSTING
    active_fraction            fraction of ticks the gene holds >=1 Pol II
    burst_rate_per_min         onsets of activity / time
    mean_burst_size            new Pol II initiated per active burst
    mean_burst_duration_s      mean active-interval length (s)
    mean_interburst_s          mean silent-interval length (s)

Outputs (to ``--out-dir``, default alongside the H5):

  * ``transcription_metrics.tsv``        one row per gene + an ``ALL`` aggregate row.
  * ``transcription_metrics.svg``        per-gene panels (rates, density, pausing,
                                         state fractions, velocity, bursting).
  * ``transcription_tracks.svg``         Pol II ChIP + nascent per-site tracks,
                                         folded across alleles onto one locus.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported (matches the sister
# scripts; the 1D stage is numpy-light and we never want threaded BLAS here).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_config_chain_metrics import _resolve_h5, _safe_label  # noqa: E402
from ner_rnapii_3d.plugins.lesions import (  # noqa: E402
    LESION_REPAIR,
    LESION_TYPE_A,
)
from ner_rnapii_3d.plugins.rnapii import (  # noqa: E402
    STATE_ELONGATING,
    STATE_PAUSED,
    STATE_PRE_INITIATION,
    STATE_STALLED,
    STATE_TERMINATING,
)

# Engaged Pol II carries a nascent transcript (productive or transiently
# obstacle-stalled in the gene body); same convention as polii_chip_nascent_track.py.
_ENGAGED = (STATE_ELONGATING, STATE_STALLED)
_STATE_ORDER = [STATE_PRE_INITIATION, STATE_PAUSED, STATE_ELONGATING, STATE_STALLED, STATE_TERMINATING]
_STATE_NAMES = {
    STATE_PRE_INITIATION: "pre_initiation", STATE_PAUSED: "paused", STATE_ELONGATING: "elongating",
    STATE_STALLED: "stalled", STATE_TERMINATING: "terminating",
}

# Per-gene metric columns, in TSV/display order (the aggregate row fills the same).
_RATE_KEYS = [
    "initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
    "pol2_occupancy", "pol2_density_per_kb", "nascent_signal", "term_zone_occupancy",
    "elongation_velocity_bp_s",
]
_PAUSE_KEYS = [
    "pausing_index", "frac_pre_initiation", "frac_paused", "frac_elongating", "frac_stalled",
    "frac_terminating", "mean_pre_initiation_dwell_s", "mean_pause_dwell_s",
    "apparent_pause_duration_s", "mean_terminating_dwell_s", "mean_residence_s",
    "pause_release_efficiency", "productive_fraction", "premature_termination_frac",
]
_INTERFERENCE_KEYS = ["stall_frac", "lesion_stall_frac", "cohesin_stall_frac"]

# Minimum initiation events needed to ESTIMATE a per-gene productive fraction. Below this, pf =
# completions/initiations is dominated by counting noise (one completion flips it by ~1/init), so a pf==0
# there is undersampling, NOT biology. Used to (1) separate undefined/low-count/estimable in the productive-
# fraction panel and (2) compute the realism-table pf median over estimable genes only. ~10 keeps the binomial
# 95% CI half-width on pf~0.24 under ~0.27. (Genes with NO initiation are already NaN, mathematically undefined.)
PF_MIN_EVENTS = 10
_BURST_KEYS = [
    "active_fraction", "burst_rate_per_min", "mean_burst_size",
    "mean_burst_duration_s", "mean_interburst_s",
]

# --- human-biology reference ranges -----------------------------------------
# Accepted in-vivo mammalian ranges for the trackable observables, in the SAME
# native units the metric is measured in (so the per-gene median compares
# directly). `scale`/`unit` are display-only. `soft=True` marks activity-
# dependent quantities with a deliberately wide band (plausibility, not a tight
# law).
HUMAN_RANGES: dict[str, dict] = {
    "elongation_velocity_bp_s": dict(
        lo=8.3, hi=100.0, scale=0.06, unit="kb/min", disp="0.5-6 kb/min", soft=False,
        src="genome-wide in-vivo: 0.5-4 kb/min mESC mean ~2; median ~1.5 human; "
            "2-6 kb/min HeLa mode 3.5; cf. typical-gene 1.3-4.3"),
    "mean_pause_dwell_s": dict(
        lo=5.0, hi=60.0, scale=1 / 60, unit="min", disp="~0.1-1 min (single-Pol)", soft=True,
        src="single-Pol kinetic dwell 1/(k_release+k_term): ~42 s human live-cell residence "
            "(Steurer 2018, high conf); mammalian PRO-seq models imply faster, sub-minute (~6-20 s, "
            "less certain). NOT the minutes-scale apparent pause."),
    "apparent_pause_duration_s": dict(
        lo=60.0, hi=900.0, scale=1 / 60, unit="min", disp="~1-15 min (occ/output)", soft=True,
        src="apparent = pause occupancy / PRODUCTIVE initiation (NOT single-Pol dwell): human mRNA "
            "median ~1 min (Gressel 2019); MOUSE 6.9 min (Jonkers 2014); >15 min for many genes "
            "(Shao & Zeitlinger; Krebs). = single-Pol dwell / productive_fraction"),
    "mean_terminating_dwell_s": dict(
        lo=60.0, hi=480.0, scale=1 / 60, unit="min", disp="1-8 min", soft=True,
        src="DERIVED, not directly measured in seconds (medium confidence): HEK293 3' decel to "
            "0.7-0.9 kb/min over ~1-4 kb past the PAS => ~1-6 min (Cortazar 2019; Fong 2015)"),
    "mean_residence_s": dict(
        lo=180.0, hi=2400.0, scale=1 / 60, unit="min", disp="~3-40 min", soft=True,
        src="avg human gene ~24kb at ~1.7-3.8 kb/min -> ~7-15min "
            "elongation + pause + 3' dwell; direct total dwell 179-357s for short genes"),
    "pol2_occupancy": dict(
        lo=0.3, hi=10.0, scale=1.0, unit="Pol/gene", disp="~0.5-1.5 typ, <=10 active", soft=True,
        src="~1.1 elongating Pol/gene body; promoter occupancy <10%; "
            "~2/gene derived from model rates (~0.7 elongating + 3'-stalled barrier)"),
    "pol2_density_per_kb": dict(
        lo=0.005, hi=0.25, scale=1.0, unit="Pol/kb", disp="~0.005-0.25 Pol/kb", soft=True,
        src="derived (occupancy / gene length)"),
    "initiation_rate_per_min": dict(
        lo=0.005, hi=3.0, scale=60.0, unit="/h", disp="~0.3-180 /h (loading; bursty)", soft=True,
        src="LOADING rate incl. ~75-80% abortive Pol -- NOT productive output (compare completion_rate). "
            "Human PRODUCTIVE initiation: median ~1/min mRNA, 0.3 lincRNA, 0.15 eRNA, up to ~87/min (HSPA1A) "
            "(Gressel/Cramer 2019 K562); loading runs ~1/productive_fraction higher. lo=0.005/min floor for least-active expressed genes"),
    "completion_rate_per_min": dict(
        lo=0.005, hi=1.5, scale=60.0, unit="/h", disp="~0.3-90 /h (productive)", soft=True,
        src="PRODUCTIVE initiation = mRNA output (Pol reaching the 3' end): human median ~1/min mRNA, "
            "0.3 lincRNA, 0.15 eRNA, up to ~87/min (Gressel/Cramer 2019 K562). The literature-anchored "
            "band (vs loading above); lo=0.005/min floor for least-active expressed genes"),
    "pausing_index": dict(
        lo=1.5, hi=20.0, scale=1.0, unit="", disp="~2-20 (TSS->pause)", soft=True,
        src="travelling ratio = promoter-proximal (TSS..pause site) / gene-body Pol density; paused"),
    "productive_fraction": dict(
        lo=0.05, hi=0.30, scale=1.0, unit="", disp="~0.15-0.25 (premat. term.)", soft=False,
        src="~75-80% of paused Pol II prematurely terminate -> ~0.20-0.25 productive "
            "(STL-seq Drosophila ~20%; Mukherjee & Guertin mammalian ~25%; cross-species, not direct human)"),
}


def realism_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene median of each trackable metric vs its human-biology range.

    Flags each as OK / LOW / HIGH (median inside / below / above the band) and
    reports the fraction of genes inside the band."""
    rows = []
    ev = pd.to_numeric(df["initiation_events"], errors="coerce") if "initiation_events" in df.columns else None
    for key, r in HUMAN_RANGES.items():
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce")
        else:
            continue
        # Productive fraction is undefined with no initiation (NaN, dropped below) and dominated by counting
        # noise below PF_MIN_EVENTS initiations -- exclude those low-count genes so the median/in-range reflect
        # genes where the ratio is actually estimable, not undersampling artefacts (guarded for old TSVs).
        if ev is not None and key == "productive_fraction":
            s = s.where(ev >= PF_MIN_EVENTS)
        s = s.dropna()
        if s.empty:
            continue
        med = float(s.median())
        frac_in = float(((s >= r["lo"]) & (s <= r["hi"])).mean())
        flag = "OK" if r["lo"] <= med <= r["hi"] else ("LOW" if med < r["lo"] else "HIGH")
        rows.append({
            "metric": key, "median_native": med, "median_display": med * r["scale"],
            "unit": r["unit"], "human_range": r["disp"], "flag": flag,
            "frac_genes_in_range": frac_in, "activity_dependent": r["soft"], "source": r["src"],
        })
    return pd.DataFrame(rows)


def print_realism_report(rr: pd.DataFrame) -> None:
    """Pretty-print the realism table (a '*' marks activity-dependent soft bands)."""
    glyph = {"OK": "OK  ✓", "LOW": "LOW ↓", "HIGH": "HIGH↑"}
    print("\n=== human-biology range check (per-gene median; * = activity-dependent, wide band) ===")
    print(f"  {'metric':26s} {'median':>16s}   {'human range':24s} {'flag':6s} {'in-rng':>6s}  source")
    for _, r in rr.iterrows():
        name = r["metric"] + ("*" if r["activity_dependent"] else "")
        med = f"{r['median_display']:.3g} {r['unit']}".strip()
        print(f"  {name:26s} {med:>16s}   {r['human_range']:24s} "
              f"{glyph[r['flag']]:6s} {r['frac_genes_in_range']:>5.0%}  {r['source']}")
    hard = rr[~rr["activity_dependent"]]
    nok = int((hard["flag"] == "OK").sum())
    print(f"  -> hard-range metrics in human range: {nok}/{len(hard)} "
          f"({int((hard['flag']=='HIGH').sum())} high, {int((hard['flag']=='LOW').sum())} low); "
          f"soft (activity-dependent) shown with *")


def activity_report(df: pd.DataFrame) -> dict:
    """Measure the per-gene transcriptional ACTIVITY distribution (the expression
    heavy-tail), which the per-metric range check does NOT capture. A typical
    locus is mostly QUIET with a minority of active genes, spanning orders of
    magnitude (log-normal / Zipf). Expressed-gene initiation should fall in the
    human PRO-seq band 0.2-1.5/min."""
    occ = pd.to_numeric(df["pol2_occupancy"], errors="coerce")
    load = pd.to_numeric(df["initiation_rate_per_min"], errors="coerce")   # LOADING (incl. abortive)
    prod = pd.to_numeric(df["completion_rate_per_min"], errors="coerce")   # PRODUCTIVE (reach 3' = transcript)
    m = occ.notna()
    occ, load, prod = occ[m], load[m], prod[m]
    f = lambda mask: 100.0 * float(mask.mean())
    silent = occ < 0.3; quiet = (occ >= 0.3) & (occ < 1.0)
    mod = (occ >= 1.0) & (occ < 3.0); act = occ >= 3.0
    nz = occ[occ > 1e-3]
    spread = (np.percentile(nz, 95) / max(np.percentile(nz, 5), 1e-9)) if len(nz) > 5 else float("nan")
    expr = occ >= 1.0
    # The 0.2-1.5/min band is the PRODUCTIVE/effective rate (omega), so compare it to the
    # completion (3'-arrival = transcript) rate, NOT loading (which counts the ~75% that prematurely
    # terminate at the pause).
    in_prod_band = 100.0 * float(((prod >= 0.2) & (prod <= 1.5) & expr).sum()) / max(int(expr.sum()), 1)
    med_expr_prod = float(prod[expr].median()) if expr.any() else float("nan")
    med_expr_load = float(load[expr].median()) if expr.any() else float("nan")
    print("\n=== transcriptional ACTIVITY distribution (expression heavy-tail) ===")
    print(f"  Pol-occupancy classes: silent(<0.3)={f(silent):.0f}%  quiet(0.3-1)={f(quiet):.0f}%  "
          f"moderate(1-3)={f(mod):.0f}%  active(>3)={f(act):.0f}%")
    print(f"    (a QUIET majority + a small active tail -- log-normal / Zipf)")
    print(f"  activity spread (occupancy p95/p5) = {spread:.0f}x   "
          f"(~10-1000x; expression spans 3-4 orders genome-wide)")
    print(f"  expressed genes: loading median {med_expr_load:.2f}/min; PRODUCTIVE median "
          f"{med_expr_prod:.2f}/min; {in_prod_band:.0f}% PRODUCTIVE in 0.2-1.5/min band")
    quiet_maj = (f(silent) + f(quiet)) >= 30.0
    heavy_tail = (spread >= 10.0) if spread == spread else False
    verdict = ("OK (quiet majority + heavy tail)" if (quiet_maj and heavy_tail)
               else "CHECK: distribution too narrow/uniform" if not heavy_tail
               else "CHECK: too few quiet genes (locus uniformly active)")
    print(f"  -> activity distribution: {verdict}")
    return {"silent_pct": f(silent), "quiet_pct": f(quiet), "moderate_pct": f(mod),
            "active_pct": f(act), "spread_p95_p5": spread,
            "expr_load_median_per_min": med_expr_load,
            "expr_prod_median_per_min": med_expr_prod, "expr_in_prod_band_pct": in_prod_band}


# Per-class productive-output references (Gressel/Cramer 2019 K562): housekeeping/active genes
# approach the mRNA median ~1/min; cell-type-specific genes sit at lincRNA level ~0.3/min; poised
# developmental genes are mostly off, ~0.15/min. Printed as guidance, not a pass/fail flag -- the
# point is to judge output PER CLASS rather than against a single band that conflates the
# correctly-quiet cell-type majority with genuinely under-active housekeeping genes.
_CLASS_OUTPUT_TARGET = {
    "hk_const": "~1/min (mRNA median)", "hk_high": "~1/min+ (active mRNA)",
    "celltype": "~0.3/min (lincRNA / cell-type)", "dev": "~0.15/min (poised / developmental)",
}


def class_breakdown(df: pd.DataFrame, gene_table: np.ndarray) -> "pd.DataFrame | None":
    """Per regulatory-class transcription summary, keyed by the ``gene_class`` tag carried on the
    H5 ``genes`` dataset (emitted by the config generator). Returns None for trajectories that
    predate the tag. Reports, per class: gene count, % silent / % expressed (by occupancy),
    median output (all + expressed), and median Pol II occupancy, alongside the per-class
    literature target."""
    names = getattr(gene_table.dtype, "names", None) or ()
    if "gene_class" not in names:
        return None

    def _dec(v):
        return v.decode("ascii", "ignore") if isinstance(v, bytes) else str(v)

    cls_by_id = {int(r["gene_id"]): _dec(r["gene_class"]) for r in gene_table}
    g = df.copy()
    g["gene_class"] = g["gene_id"].astype(int).map(cls_by_id).fillna("")
    occ = pd.to_numeric(g["pol2_occupancy"], errors="coerce")
    comp = pd.to_numeric(g["completion_rate_per_min"], errors="coerce")
    known = ["hk_const", "hk_high", "celltype", "dev"]
    order = [c for c in known if (g["gene_class"] == c).any()]
    order += sorted(c for c in g["gene_class"].unique() if c and c not in order)
    rows = []
    for c in order:
        m = g["gene_class"] == c
        o, p = occ[m], comp[m]
        expr = o >= 1
        rows.append({
            "gene_class": c, "n": int(m.sum()),
            "pct_silent": round(100.0 * float((o < 0.3).mean()), 1),
            "pct_expressed": round(100.0 * float(expr.mean()), 1),
            "median_completion_per_min": round(float(p.median()), 4),
            "median_completion_expressed": (round(float(p[expr].median()), 4) if expr.any() else float("nan")),
            "median_pol2_occupancy": round(float(o.median()), 3),
            "literature_target": _CLASS_OUTPUT_TARGET.get(c, ""),
        })
    return pd.DataFrame(rows)


def print_class_breakdown(cb: pd.DataFrame) -> None:
    """Pretty-print the per-class table; output is completion_rate_per_min (productive, /min)."""
    print("\n=== per regulatory-class transcription (productive output vs per-class literature) ===")
    print(f"  {'class':10s} {'n':>4s} {'%sil':>5s} {'%expr':>6s} {'med_out':>8s} {'out_expr':>9s}  literature target")
    for _, r in cb.iterrows():
        oe = r["median_completion_expressed"]
        print(f"  {r['gene_class']:10s} {int(r['n']):>4d} {r['pct_silent']:>4.0f}% {r['pct_expressed']:>5.0f}% "
              f"{r['median_completion_per_min']:>8.3f} {oe:>9.3f}  {r['literature_target']}")
    print("  -> housekeeping should approach ~1/min; cell-type ~0.3; developmental ~0.15 "
          "(per-class medians, not a single band)")


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs where the boolean timeseries is True."""
    if not active.any():
        return []
    d = np.diff(active.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [active.size]
    return list(zip(starts, ends))


def _isin_sorted(query: np.ndarray, sorted_vals: np.ndarray) -> np.ndarray:
    """Membership of each ``query`` in the ascending-sorted, unique ``sorted_vals``.

    Equivalent to ``np.isin(query, sorted_vals)`` but skips np.isin's internal
    re-sort/unique of the second argument -- used for the recorded lesion sites,
    which are already site-sorted and unique (one lesion per site)."""
    if sorted_vals.size == 0 or query.size == 0:
        return np.zeros(query.shape, dtype=bool)
    idx = np.searchsorted(sorted_vals, query)
    idx = np.minimum(idx, sorted_vals.size - 1)
    return sorted_vals[idx] == query


def measure_transcription(
    h5_path: Path, tick_seconds: float, *, pause_offset: int = 1, chunk: int = 2000
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Per-gene transcription metrics measured from a 1D RNAPII trajectory.

    Returns ``(per_gene_df, aggregate_row, chip_track, nascent_track, gene_table)``.
    The two tracks are per-allele mean occupancy folded onto one locus (length
    ``chain_length``); ``gene_table`` is the structured ``genes`` dataset.
    """
    with h5py.File(h5_path, "r") as h:
        if not bool(h.attrs.get("rnapii_enabled", False)):
            raise SystemExit(f"{h5_path}: RNAPII not enabled in this run (tx-OFF config).")
        if "genes" not in h:
            raise SystemExit(f"{h5_path}: no 'genes' dataset; nothing to transcribe.")
        N = int(h.attrs["N"])
        chain = int(h.attrs["chain_length"])
        num_chains = int(h.attrs.get("num_chains", 1))
        # Lattice resolution: bp per site. New runs stamp this attr; a missing
        # attr means a pre-refactor trajectory authored at the 1 kb/site scale.
        bp_per_site = int(h.attrs.get("bp_per_site", 1000))
        kb_per_site = bp_per_site / 1000.0
        genes = h["genes"][:]
        n_genes = genes.shape[0]
        gids = genes["gene_id"].astype(int)
        tss = genes["tss"].astype(int)
        tes = genes["tes"].astype(int)
        direction = genes["direction"].astype(int)
        glen = np.abs(tes - tss)                       # gene length in lattice sites
        gidx = np.full(int(gids.max()) + 1, -1, dtype=np.int64)
        gidx[gids] = np.arange(n_genes)

        T = int(h["rnapii_positions"].shape[0])
        if T == 0:
            raise ValueError(f"{h5_path}: transcription metrics require at least one frame")
        has_lesions = "lesions" in h
        has_cohesin = "positions" in h

        # Streaming occupancy/burst accumulators. Keeping the former per-gene x
        # per-frame matrices cost O(genes * frames) RAM (about 2.5 GiB for the
        # panel config) even though only their sums and boolean runs are needed.
        present_ticks = np.zeros(n_genes, dtype=np.int64)
        engaged_ticks = np.zeros(n_genes, dtype=np.int64)
        active_ticks = np.zeros(n_genes, dtype=np.int64)
        inactive_ticks = np.zeros(n_genes, dtype=np.int64)
        burst_count = np.zeros(n_genes, dtype=np.int64)
        gap_count = np.zeros(n_genes, dtype=np.int64)
        previous_active = np.zeros(n_genes, dtype=bool)
        have_previous_frame = False
        state_ticks = np.zeros((n_genes, 5), dtype=np.int64)   # cols indexed by state code
        tss_pause_ct = np.zeros(n_genes, dtype=np.int64)   # Pol in [TSS, pause site]: rel in [0, pause_offset]
        body_ct = np.zeros(n_genes, dtype=np.int64)        # Pol in the gene body: pause_offset < rel <= glen
        term_zone_ct = np.zeros(n_genes, dtype=np.int64)   # Pol in the 3' termination window: rel > glen (past the TES)
        stall_total = np.zeros(n_genes, dtype=np.int64)
        stall_lesion = np.zeros(n_genes, dtype=np.int64)
        stall_cohesin = np.zeros(n_genes, dtype=np.int64)
        chip_fold = np.zeros(chain, dtype=np.float64)
        nascent_fold = np.zeros(chain, dtype=np.float64)
        # Per-molecule lifecycle state is needed only while a UID is live. UIDs
        # are stable and a polymerase appears in consecutive frames until unload,
        # so vanished UIDs can be reduced immediately instead of retaining five
        # arrays containing every (frame, Pol) record for a global sort.
        empty_i8 = np.empty(0, dtype=np.int64)
        live_uid = empty_i8
        live_gene = empty_i8
        live_first = empty_i8
        live_last = empty_i8
        live_prev_pos = empty_i8
        live_n_pre = empty_i8
        live_n_paused = empty_i8
        live_n_elong = empty_i8
        live_n_term = empty_i8
        live_pause_ep = empty_i8
        live_adv = empty_i8
        live_prev_paused = np.empty(0, dtype=bool)

        init_ct = np.zeros(n_genes, dtype=np.int64)
        completed_ct = np.zeros(n_genes, dtype=np.int64)
        aborted_ct = np.zeros(n_genes, dtype=np.int64)
        adv_sites = np.zeros(n_genes, dtype=np.int64)
        pause_ticks = np.zeros(n_genes, dtype=np.int64)
        pause_episodes = np.zeros(n_genes, dtype=np.int64)
        pre_n_sum = np.zeros(n_genes, dtype=np.int64)
        term_n_sum = np.zeros(n_genes, dtype=np.int64)
        resid_fr_sum = np.zeros(n_genes, dtype=np.int64)
        resid_ct = np.zeros(n_genes, dtype=np.int64)

        def finalize_lifecycles(mask: np.ndarray) -> None:
            """Reduce vanished/current-live UID rows into per-gene integer totals."""
            if not mask.any():
                return
            ug = live_gene[mask]
            first = live_first[mask]
            last = live_last[mask]
            npre = live_n_pre[mask]
            npaused = live_n_paused[mask]
            nelong = live_n_elong[mask]
            nterm = live_n_term[mask]
            pep = live_pause_ep[mask]
            adv = live_adv[mask]
            new = first > 0
            reached = nterm > 0
            nr = new & reached

            adv_sites[:] += np.bincount(ug, weights=adv, minlength=n_genes).astype(np.int64)
            pause_ticks[:] += np.bincount(ug, weights=npaused, minlength=n_genes).astype(np.int64)
            pause_episodes[:] += np.bincount(ug, weights=pep, minlength=n_genes).astype(np.int64)
            init_ct[:] += np.bincount(ug[new], minlength=n_genes).astype(np.int64)
            completed_ct[:] += np.bincount(ug[reached], minlength=n_genes).astype(np.int64)
            abort = new & (nelong == 0) & (npaused > 0) & ~reached
            aborted_ct[:] += np.bincount(ug[abort], minlength=n_genes).astype(np.int64)
            pre_n_sum[:] += np.bincount(
                ug[new], weights=npre[new], minlength=n_genes
            ).astype(np.int64)
            term_n_sum[:] += np.bincount(
                ug[reached], weights=nterm[reached], minlength=n_genes
            ).astype(np.int64)
            resid_fr_sum[:] += np.bincount(
                ug[nr], weights=(last[nr] - first[nr]), minlength=n_genes
            ).astype(np.int64)
            resid_ct[:] += np.bincount(ug[nr], minlength=n_genes).astype(np.int64)

        ds_pos, ds_st, ds_id = h["rnapii_positions"], h["rnapii_states"], h["rnapii_ids"]
        ds_les = h["lesions"] if has_lesions else None
        ds_lty = h["lesion_types"] if has_lesions else None
        ds_lst = h["lesion_states"] if has_lesions else None
        ds_lef = h["positions"] if has_cohesin else None
        widest_row = max(
            int(ds_pos.shape[1]),
            int(ds_les.shape[1]) if ds_les is not None else 0,
            int(ds_lef.shape[1]) if ds_lef is not None else 0,
            1,
        )
        chunk = min(max(1, int(chunk)), max(1, 2_000_000 // widest_row))

        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            pos = ds_pos[start:end]
            st = ds_st[start:end]
            ids = ds_id[start:end]
            les = ds_les[start:end] if has_lesions else None
            lty = ds_lty[start:end] if has_lesions else None
            lst = ds_lst[start:end] if has_lesions else None
            lef = ds_lef[start:end] if has_cohesin else None

            for k in range(end - start):
                gf = start + k
                site = pos[k, :, 0]
                present = site >= 0
                if present.any():
                    sp = site[present]
                    gp = gidx[pos[k, present, 1]]
                    stp = st[k][present].astype(int)
                    idp = ids[k][present].astype(np.int64, copy=False)
                    per_gene_present = np.bincount(gp, minlength=n_genes)
                else:
                    sp = gp = stp = idp = empty_i8
                    per_gene_present = np.zeros(n_genes, dtype=np.int64)

                present_ticks += per_gene_present
                active = per_gene_present > 0
                active_ticks += active
                inactive_ticks += ~active
                if have_previous_frame:
                    burst_count += active & ~previous_active
                    gap_count += ~active & previous_active
                else:
                    burst_count += active
                    gap_count += ~active
                    have_previous_frame = True
                previous_active = active

                if not present.any():
                    finalize_lifecycles(np.ones(live_uid.size, dtype=bool))
                    live_uid = live_gene = live_first = live_last = empty_i8
                    live_prev_pos = live_n_pre = live_n_paused = live_n_elong = empty_i8
                    live_n_term = live_pause_ep = live_adv = empty_i8
                    live_prev_paused = np.empty(0, dtype=bool)
                    continue

                # Histogram per gene/state with bincount (same result as np.add.at,
                # but unbuffered scatter-add is ~10x slower).
                eng = (stp == STATE_ELONGATING) | (stp == STATE_STALLED)   # == _ENGAGED
                engaged_ticks += np.bincount(gp[eng], minlength=n_genes)
                state_ticks += np.bincount(gp * 5 + stp, minlength=n_genes * 5).reshape(n_genes, 5)
                # pausing index from 1D position: rel = distance past TSS toward TES.
                # The pause sits at rel == pause_offset; the body is pause_offset < rel <= glen.
                # Pol with rel > glen are PAST the TES in the 3' termination window (they crawl
                # there before release) -- they must NOT count toward gene-body density, else the
                # pausing index is diluted by the terminating Pol the window now lets exist.
                rel = (sp - tss[gp]) * direction[gp]
                glen_p = glen[gp]
                tss_pause_ct += np.bincount(gp[rel <= pause_offset], minlength=n_genes)
                body_ct += np.bincount(gp[(rel > pause_offset) & (rel <= glen_p)], minlength=n_genes)
                term_zone_ct += np.bincount(gp[rel > glen_p], minlength=n_genes)
                # per-site ChIP / nascent tracks, folded onto one locus
                chip_fold += np.bincount(sp % chain, minlength=chain)
                nascent_fold += np.bincount(sp[eng] % chain, minlength=chain)

                # Co-transcriptional stall attribution: look one site ahead of each
                # STALLED Pol (it stalls just upstream of the obstacle) and test those
                # few sites against the active lesion/cohesin coordinates. This avoids
                # allocating two N-wide boolean landscapes per stalled frame.
                any_stall = bool((stp == STATE_STALLED).any())
                blocking_lesion_sites = empty_i8
                if has_lesions and any_stall:
                    s = les[k]
                    v = (s >= 0) & (s < N)
                    sv, tv, stv = s[v], lty[k][v], lst[k][v]
                    block = (stv == LESION_REPAIR) | (tv == LESION_TYPE_A)
                    blocking_lesion_sites = sv[block]
                cohesin_sites = empty_i8
                if has_cohesin and any_stall:
                    legs = lef[k].reshape(-1)
                    cohesin_sites = legs[(legs >= 0) & (legs < N)]

                # Stall attribution (vectorised over the stalled Pols in this frame):
                # a STALLED Pol stalls just upstream of its obstacle, so look one site
                # ahead. Lesion takes priority over cohesin (matches the original elif).
                if any_stall:
                    stmask = stp == STATE_STALLED
                    g_st = gp[stmask]
                    p_st = sp[stmask]
                    stall_total += np.bincount(g_st, minlength=n_genes)
                    ahead = p_st + direction[g_st]
                    valid = (ahead >= 0) & (ahead < N)
                    gv, av = g_st[valid], ahead[valid]
                    if gv.size:
                        # Each membership test is computed only when its dataset is
                        # present (the absent-dataset case was an all-False np.isin +
                        # zero bincount -> a no-op feeding a NaN-reported metric).
                        # Lesion sites are recorded site-sorted+unique, so a
                        # searchsorted membership avoids np.isin's re-sort; cohesin
                        # legs are unsorted and keep np.isin. Byte-identical results.
                        les_hit = _isin_sorted(av, blocking_lesion_sites) if has_lesions else None
                        if has_lesions:
                            stall_lesion += np.bincount(gv[les_hit], minlength=n_genes)
                        if has_cohesin:
                            coh_hit = np.isin(av, cohesin_sites)
                            if les_hit is not None:
                                coh_hit = coh_hit & ~les_hit
                            stall_cohesin += np.bincount(gv[coh_hit], minlength=n_genes)

                # Merge the current live UID set with the previous frame. Valid
                # engine output is UID-sorted; tolerate external H5 writers by
                # sorting only when needed.
                if idp.size > 1 and np.any(idp[1:] < idp[:-1]):
                    uid_order = np.argsort(idp, kind="stable")
                    idp = idp[uid_order]
                    sp = sp[uid_order]
                    gp = gp[uid_order]
                    stp = stp[uid_order]

                loc = np.searchsorted(live_uid, idp)
                survived = loc < live_uid.size
                if survived.any():
                    survived[survived] = live_uid[loc[survived]] == idp[survived]
                retained = np.zeros(live_uid.size, dtype=bool)
                retained[loc[survived]] = True
                finalize_lifecycles(~retained)

                n_live = idp.size
                cur_first = np.full(n_live, gf, dtype=np.int64)
                cur_n_pre = np.zeros(n_live, dtype=np.int64)
                cur_n_paused = np.zeros(n_live, dtype=np.int64)
                cur_n_elong = np.zeros(n_live, dtype=np.int64)
                cur_n_term = np.zeros(n_live, dtype=np.int64)
                cur_pause_ep = np.zeros(n_live, dtype=np.int64)
                cur_adv = np.zeros(n_live, dtype=np.int64)
                previous_position = sp.copy()
                was_paused = np.zeros(n_live, dtype=bool)

                ci = np.flatnonzero(survived)
                if ci.size:
                    pi = loc[ci]
                    cur_first[ci] = live_first[pi]
                    cur_n_pre[ci] = live_n_pre[pi]
                    cur_n_paused[ci] = live_n_paused[pi]
                    cur_n_elong[ci] = live_n_elong[pi]
                    cur_n_term[ci] = live_n_term[pi]
                    cur_pause_ep[ci] = live_pause_ep[pi]
                    cur_adv[ci] = live_adv[pi]
                    previous_position[ci] = live_prev_pos[pi]
                    was_paused[ci] = live_prev_paused[pi]

                cur_n_pre += stp == STATE_PRE_INITIATION
                cur_n_paused += stp == STATE_PAUSED
                cur_n_elong += stp == STATE_ELONGATING
                cur_n_term += stp == STATE_TERMINATING
                paused_now = stp == STATE_PAUSED
                cur_pause_ep += paused_now & ~was_paused
                advancement = (sp - previous_position) * direction[gp]
                advance_mask = survived & (advancement > 0) & (stp == STATE_ELONGATING)
                cur_adv[advance_mask] += advancement[advance_mask]

                live_uid = idp.copy()
                live_gene = gp.copy()
                live_first = cur_first
                live_last = np.full(n_live, gf, dtype=np.int64)
                live_prev_pos = sp.copy()
                live_n_pre = cur_n_pre
                live_n_paused = cur_n_paused
                live_n_elong = cur_n_elong
                live_n_term = cur_n_term
                live_pause_ep = cur_pause_ep
                live_adv = cur_adv
                live_prev_paused = paused_now

    # Reduce polymerases still live in the final recorded frame. Molecules that
    # vanished earlier were finalized as soon as they left the live UID set.
    finalize_lifecycles(np.ones(live_uid.size, dtype=bool))

    # dwell means (nan where no contributing Pol); tick_seconds applied once so the
    # result equals mean([count*tick_seconds, ...]) for the per-gene Pol set.
    with np.errstate(invalid="ignore"):
        mean_pre = np.where(init_ct > 0, pre_n_sum / np.maximum(init_ct, 1) * tick_seconds, np.nan)
        mean_term = np.where(completed_ct > 0, term_n_sum / np.maximum(completed_ct, 1) * tick_seconds, np.nan)
        mean_resid = np.where(resid_ct > 0, resid_fr_sum / np.maximum(resid_ct, 1) * tick_seconds, np.nan)

    t_obs = T * tick_seconds
    elong_ticks = state_ticks[:, STATE_ELONGATING].astype(float)
    total_ticks = state_ticks.sum(axis=1).astype(float)

    rows = []
    for g in range(n_genes):
        occ = present_ticks[g] / T
        nasc = engaged_ticks[g] / T
        prox_window = pause_offset + 1                      # TSS through the pause site (sites)
        body_len = max(glen[g] - prox_window, 1)
        body_d = body_ct[g] / body_len
        st_tot = total_ticks[g] if total_ticks[g] > 0 else float("nan")
        engaged_t = elong_ticks[g] + state_ticks[g, STATE_STALLED]
        row = {
            "gene_id": int(gids[g]),
            "tss": int(tss[g]), "tes": int(tes[g]),
            "chain_relative_tss": int(tss[g] % chain),
            "direction": int(direction[g]),
            "gene_length_kb": float(glen[g] * kb_per_site),
            # core rates & density
            "initiation_rate_per_min": init_ct[g] / t_obs * 60.0,
            "completion_rate_per_min": completed_ct[g] / t_obs * 60.0,
            "synthesis_rate_nt_per_s": adv_sites[g] * bp_per_site / t_obs,
            "pol2_occupancy": occ,
            "pol2_density_per_kb": occ / max(glen[g] * kb_per_site, kb_per_site),
            "nascent_signal": nasc,
            # mean Pol II in the 3' termination window (past the TES). The window lets a terminating
            # Pol vacate the single TES slot, so the 3' end no longer jams; this is the 3' Pol
            # accumulation that also reinforces the cohesin barrier downstream of the gene.
            "term_zone_occupancy": term_zone_ct[g] / T,
            "elongation_velocity_bp_s": (adv_sites[g] / elong_ticks[g] * bp_per_site / tick_seconds
                                         if elong_ticks[g] > 0 else float("nan")),
            # pausing & state kinetics
            # travelling ratio over the TSS->pause-site window.
            "pausing_index": ((tss_pause_ct[g] / prox_window) / body_d if body_d > 0 else float("nan")),
            "mean_pre_initiation_dwell_s": float(mean_pre[g]),
            # single-Pol pause dwell: real time one Pol II stays paused before it
            # releases OR terminates (1/(k_release+k_termination); ~0.4 min, short).
            "mean_pause_dwell_s": (pause_ticks[g] / pause_episodes[g] * tick_seconds
                                   if pause_episodes[g] > 0 else float("nan")),
            # apparent pause duration (occupancy/output): pause occupancy
            # divided by PRODUCTIVE initiation -- inflated ~1/productive_fraction-fold
            # vs the single-Pol dwell because terminated Pol II are dropped from output.
            "apparent_pause_duration_s": (state_ticks[g, STATE_PAUSED] * tick_seconds / completed_ct[g]
                                          if completed_ct[g] > 0 else float("nan")),
            "mean_terminating_dwell_s": float(mean_term[g]),
            "mean_residence_s": float(mean_resid[g]),
            "pause_release_efficiency": (completed_ct[g] / init_ct[g]
                                         if init_ct[g] > 0 else float("nan")),
            "productive_fraction": (completed_ct[g] / init_ct[g]
                                    if init_ct[g] > 0 else float("nan")),
            "premature_termination_frac": (aborted_ct[g] / init_ct[g]
                                           if init_ct[g] > 0 else float("nan")),
            # Raw event counts behind the ratio metrics, so downstream plotting/reporting can separate an
            # UNDEFINED productive fraction (no initiation -> NaN) from a TRUE near-zero (enough events to
            # estimate it) from an undersampled pf==0 (too few initiation events; see PF_MIN_EVENTS). They
            # also drive a physical-plausibility guard: under promoter occlusion a gene loads at most once per
            # frame, so init_ct<=T and completed_ct<=init_ct -- a violation is a metric UID double-count on a
            # degenerate/short-window allele (the spurious thousands-per-hour outliers), not real output.
            "initiation_events": int(init_ct[g]),
            "completion_events": int(completed_ct[g]),
            "events_implausible": bool(completed_ct[g] > init_ct[g] or init_ct[g] > T),
            # co-transcriptional interference
            "stall_frac": (state_ticks[g, STATE_STALLED] / engaged_t
                           if engaged_t > 0 else float("nan")),
            "lesion_stall_frac": (stall_lesion[g] / stall_total[g]
                                  if has_lesions and stall_total[g] > 0 else float("nan")),
            "cohesin_stall_frac": (stall_cohesin[g] / stall_total[g]
                                   if has_cohesin and stall_total[g] > 0 else float("nan")),
        }
        for code in _STATE_ORDER:
            row[f"frac_{_STATE_NAMES[code]}"] = (state_ticks[g, code] / st_tot
                                                 if st_tot == st_tot else float("nan"))
        # Bursts/gaps were counted online from the same active = occupancy > 0
        # series. Mean run length is total True/False ticks divided by run count.
        row["active_fraction"] = active_ticks[g] / T
        row["burst_rate_per_min"] = burst_count[g] / t_obs * 60.0
        row["mean_burst_size"] = (init_ct[g] / burst_count[g]
                                  if burst_count[g] else float("nan"))
        row["mean_burst_duration_s"] = (
            active_ticks[g] / burst_count[g] * tick_seconds
            if burst_count[g] else float("nan")
        )
        row["mean_interburst_s"] = (
            inactive_ticks[g] / gap_count[g] * tick_seconds
            if gap_count[g] else float("nan")
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # Locus aggregate: rates/counts sum, intensive quantities are occupancy-weighted.
    agg = _aggregate(df, n_alleles=n_genes)
    chip_track = chip_fold / (T * max(num_chains, 1))
    nascent_track = nascent_fold / (T * max(num_chains, 1))
    return df, agg, chip_track, nascent_track, genes


def _aggregate(df: pd.DataFrame, *, n_alleles: int) -> dict:
    """Locus-level summary row. Extensive rates are summed across genes; intensive
    quantities (densities, fractions, dwell times, velocity) are averaged so the
    row reads as 'the typical gene' rather than a sum of incommensurate ratios."""
    agg: dict = {"gene_id": "ALL", "tss": -1, "tes": -1, "chain_relative_tss": -1,
                 "direction": 0, "gene_length_kb": float(df["gene_length_kb"].mean())}
    summed = {"initiation_rate_per_min", "completion_rate_per_min",
              "synthesis_rate_nt_per_s", "burst_rate_per_min"}
    for col in df.columns:
        if col in agg:
            continue
        if col in summed:
            agg[col] = float(df[col].sum())
        else:
            values = df[col].to_numpy(dtype=float)
            non_nan = values[~np.isnan(values)]
            agg[col] = float(non_nan.mean()) if non_nan.size else float("nan")
    agg["n_genes"] = n_alleles
    return agg


# Each panel: (metric column, descriptive title, x-axis label) — the band/scale come
# from HUMAN_RANGES so every panel shades the human range and is self-explaining.
# (column, title, x-label, log-x, display-scale [None -> HUMAN_RANGES scale])
_DIST_PANELS = [
    ("elongation_velocity_bp_s", "Elongation velocity\n(Pol II speed along the gene)", "kb / min", False, None),
    ("mean_pause_dwell_s", "Promoter-proximal pause\n(time Pol II pauses near the TSS)", "minutes", False, None),
    ("mean_terminating_dwell_s", "3′ termination dwell\n(time Pol II spends at the gene 3′ end)", "minutes", False, None),
    ("initiation_rate_per_min", "Initiation rate\n(new Pol II loaded)", "events / h", False, None),
    ("pol2_occupancy", "Pol II per gene\n(occupancy ≈ Pol II ChIP)", "polymerases / gene", True, None),
    ("pausing_index", "Pausing index\n(promoter ÷ gene-body density)", "ratio", False, None),
    ("completion_rate_per_min", "Transcript output\n(productive Pol II reaching the 3′ end)", "transcripts / h", False, 60.0),
    ("synthesis_rate_nt_per_s", "Nascent RNA synthesis\n(engaged Pol II throughput ≈ GRO/PRO-seq)", "nt / s", True, 1.0),
]


def _dist_panel(ax, vals, key, title, xlabel, logx, scale=None):
    """One distribution panel: histogram + median line. ``scale`` overrides the
    HUMAN_RANGES display scale for metrics not in that table (e.g. transcript output)."""
    sc = scale if scale is not None else HUMAN_RANGES.get(key, {}).get("scale", 1.0)
    v = np.asarray(pd.to_numeric(vals, errors="coerce"), float) * sc
    v = v[np.isfinite(v)]
    if logx:
        v = v[v > 0]
        ax.set_xscale("log")
        bins = np.logspace(np.log10(max(v.min(), 1e-3)), np.log10(v.max()), 28) if v.size else 12
        vplot = v
    else:
        # clip a heavy tail to the 99th pct so a few outliers don't stretch the axis
        hi_clip = np.percentile(v, 99) if v.size else 1.0
        vplot = np.clip(v, None, hi_clip)
        bins = np.linspace(min(v.min(), 0) if v.size else 0, hi_clip, 26)
    ax.hist(vplot, bins=bins, color="#6f8fbf", edgecolor="white", linewidth=0.3, zorder=2)
    med = float(np.median(v)) if v.size else np.nan
    ax.axvline(med, color="#A63446", lw=2.0, zorder=3)
    left = med < (np.nanmean(ax.get_xlim()) if not logx else 10 ** np.mean(np.log10(ax.get_xlim())))
    ax.annotate(f"median\n{med:.2g}", xy=(med, 0.97), xycoords=("data", "axes fraction"),
                ha="left" if left else "right", va="top", fontsize=9, color="#A63446", fontweight="bold")
    ax.set_title(title, fontsize=11.5, pad=5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("number of genes", fontsize=9.5)
    ax.tick_params(labelsize=9)


def _zipf_panel(ax, out) -> None:
    """Rank plot: genes sorted by transcript output (descending), rank vs output on log-log axes -- scatter only,
    no fitted slope or reference lines."""
    out = np.asarray(out, float)
    rf = np.sort(out[np.isfinite(out) & (out > 0)])[::-1]
    if rf.size:
        rank = np.arange(1, rf.size + 1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.scatter(rank, rf, s=5, alpha=0.35, color="#3b6b8a", edgecolor="none", zorder=2)
    ax.set_title("Transcript output by gene rank", fontsize=11.5, pad=5)
    ax.set_xlabel("gene rank (by output)", fontsize=10)
    ax.set_ylabel("transcripts / h", fontsize=9.5); ax.tick_params(labelsize=9)


def _pi_activity_panel(ax, df) -> None:
    """Pausing index vs gene activity: each gene a point (pausing index vs productive
    output, transcripts/h) on log-log axes, with a binned-median trend (red)."""
    out = pd.to_numeric(df.get("completion_rate_per_min"), errors="coerce").to_numpy() * 60.0  # transcripts/h
    pi = pd.to_numeric(df.get("pausing_index"), errors="coerce").to_numpy()
    m = np.isfinite(out) & np.isfinite(pi) & (out > 0) & (pi > 0)
    out, pi = out[m], pi[m]
    ax.set_xscale("log"); ax.set_yscale("log")
    if out.size:
        ax.scatter(out, pi, s=5, alpha=0.28, color="#3b6b8a", edgecolor="none", zorder=2)
        # binned-median trend: deciles of output, geo-mean-centred, drawn where a bin has >=5 genes
        edges = np.unique(np.quantile(out, np.linspace(0, 1, 9)))
        cx, cy = [], []
        for i in range(edges.size - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = (out >= lo) & (out <= hi) if i == edges.size - 2 else (out >= lo) & (out < hi)
            if sel.sum() >= 5:
                cx.append(np.exp(np.mean(np.log(out[sel])))); cy.append(np.median(pi[sel]))
        if len(cx) >= 2:
            ax.plot(cx, cy, color="#A63446", lw=2.0, marker="o", ms=4, zorder=3, label="binned median")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Pausing index vs activity\n(traveling ratio vs output)", fontsize=11.5, pad=5)
    ax.set_xlabel("transcripts / h (activity)", fontsize=10)
    ax.set_ylabel("pausing index", fontsize=9.5); ax.tick_params(labelsize=9)


def _pf_panel(ax, df) -> None:
    """Productive-fraction panel that SEPARATES three populations instead of lumping every pf==0 into one bar:
    (i) ESTIMABLE genes (>= PF_MIN_EVENTS initiations) -> the real histogram + median line, with the human band
    shaded (the only genes pf is trustworthy for); (ii) LOW-COUNT genes (1..PF_MIN_EVENTS-1 inits) where a
    pf==0 is undersampling, not biology; (iii) UNDEFINED genes (no initiation -> NaN). (ii) and (iii) are
    tallied as a text annotation, NOT plotted as zeros, so the panel no longer shows a spurious spike at 0."""
    pf = pd.to_numeric(df.get("productive_fraction"), errors="coerce").to_numpy()
    ev = (pd.to_numeric(df.get("initiation_events"), errors="coerce").to_numpy()
          if "initiation_events" in df.columns else np.where(np.isfinite(pf), PF_MIN_EVENTS, 0))
    est = np.isfinite(pf) & (ev >= PF_MIN_EVENTS)
    lowc = np.isfinite(pf) & (ev > 0) & (ev < PF_MIN_EVENTS)
    undef = ~np.isfinite(pf)
    r = HUMAN_RANGES.get("productive_fraction", {})
    if r:
        ax.axvspan(r["lo"], r["hi"], color="#dfe7c8", alpha=0.6, zorder=0)   # shade the human band
    v = pf[est]
    if v.size:
        ax.hist(v, bins=np.linspace(0, 1, 26), color="#6f8fbf", edgecolor="white", linewidth=0.3, zorder=2)
        med = float(np.median(v))
        ax.axvline(med, color="#A63446", lw=2.0, zorder=3)
        ax.annotate(f"median\n{med:.2g}", xy=(med, 0.97), xycoords=("data", "axes fraction"),
                    ha="right" if med > 0.5 else "left", va="top", fontsize=9, color="#A63446", fontweight="bold")
    ax.annotate(f"estimable (≥{PF_MIN_EVENTS} inits): {int(est.sum())}\n"
                f"low-count (1–{PF_MIN_EVENTS - 1}): {int(lowc.sum())}\n"
                f"undefined (no init): {int(undef.sum())}",
                xy=(0.96, 0.6), xycoords="axes fraction", ha="right", va="top", fontsize=8, color="#555")
    ax.set_title("Productive fraction\n(estimable genes only; ≥%d inits)" % PF_MIN_EVENTS, fontsize=11.5, pad=5)
    ax.set_xlabel("fraction", fontsize=10)
    ax.set_ylabel("number of genes", fontsize=9.5); ax.tick_params(labelsize=9)


# Rate/ratio metrics that go non-physical on a metric UID double-count (events_implausible); masked out of the
# PLOTS and the locus aggregate so one degenerate allele can't dominate the rank plot or inflate the ALL row.
# The raw values + the events_implausible flag are still written to the per-gene TSV (a complete, honest record).
_GUARDED_METRICS = ["initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
                    "productive_fraction", "premature_termination_frac", "pause_release_efficiency",
                    "pol2_occupancy", "pol2_density_per_kb", "nascent_signal"]


def _mask_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the guarded rate/ratio metrics NaN'd for events_implausible genes (UID double-counts
    on degenerate/short-window alleles -> spurious thousands-per-hour outliers), so plots and the aggregate
    ignore them. Old TSVs without the flag pass through unchanged."""
    if "events_implausible" not in df.columns:
        return df
    bad = df["events_implausible"].to_numpy(dtype=bool)
    if not bad.any():
        return df
    out = df.copy()
    for c in _GUARDED_METRICS:
        if c in out.columns:
            out.loc[bad, c] = np.nan
    print(f"  [guard] masked {int(bad.sum())} gene(s) with physically-impossible event counts "
          f"(completion>initiation or initiation>frames) from plots/aggregate")
    return out


# Publication figure: 9 curated panels (key transcription kinetics vs the human range) in a 3x3 grid.
# (col, title, xlabel, logx, scale) -- scale None => HUMAN_RANGES display scale; each panel shades the band.
_PUB_PANELS = [
    ("initiation_rate_per_min", "Initiation (Pol II loaded,\nincl. abortive)", "events / h", True, 60.0),
    ("completion_rate_per_min", "Transcript output\n(productive, reaching 3′)", "transcripts / h", True, 60.0),
    "ZIPF",  # -> rank-vs-output scatter
    ("mean_pause_dwell_s", "Promoter-proximal pause\n(single-Pol dwell)", "minutes", True, None),
    "PI_ACTIVITY",  # -> pausing index vs activity (output) scatter
    ("elongation_velocity_bp_s", "Elongation velocity", "kb / min", False, None),
    "PF",  # -> productive-fraction panel separating undefined / low-count / estimable genes
    ("pol2_occupancy", "Pol II per gene\n(occupancy ≈ ChIP)", "polymerases / gene", True, None),
    ("mean_terminating_dwell_s", "3′ termination dwell", "minutes", False, None),
]


def plot_transcription_distributions_pub(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Publication 3x3: per-gene distribution of each key transcription kinetic (model median in red).
    One panel is the rank-vs-output scatter."""
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5))
    out = pd.to_numeric(df.get("completion_rate_per_min"), errors="coerce").to_numpy() * 60.0  # transcripts/h
    for i, (ax, spec) in enumerate(zip(axes.flat, _PUB_PANELS)):
        if spec == "ZIPF":
            _zipf_panel(ax, out)
        elif spec == "PI_ACTIVITY":
            _pi_activity_panel(ax, df)
        elif spec == "PF":
            _pf_panel(ax, df)
        else:
            key, ttl, xl, logx, sc = spec
            _dist_panel(ax, df.get(key), key, ttl, xl, logx, sc)
        # panel letter (a, b, c, ...) matching the legend .txt, top-left outside the axes
        ax.text(-0.10, 1.07, f"{chr(97 + i)}", transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="right")
    fig.suptitle(title, fontsize=15, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.6, w_pad=1.8)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    """Publication figure: per-gene distributions of each transcription metric
    (median marked in red), including transcript production and nascent-RNA synthesis,
    an occupancy->output scatter (Pol II density vs transcripts), where Pol II spends its
    time, and the activity (expression) distribution. Each panel is self-contained:
    the title says what it is and the x-axis gives the unit."""
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 17))
    for ax, (key, ttl, xl, logx, sc) in zip(axes.flat[:8], _DIST_PANELS):
        _dist_panel(ax, df.get(key), key, ttl, xl, logx, sc)

    # Panel 9: occupancy -> output (does Pol II density translate into transcripts, or saturate?).
    ax = axes.flat[8]
    occ_s = pd.to_numeric(df.get("pol2_occupancy"), errors="coerce")
    out_s = pd.to_numeric(df.get("completion_rate_per_min"), errors="coerce") * 60.0
    m = np.isfinite(occ_s) & np.isfinite(out_s) & (occ_s > 0) & (out_s > 0)
    ax.scatter(occ_s[m], out_s[m], s=6, alpha=0.25, color="#3b6b8a", edgecolor="none", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Occupancy → output\n(does Pol II density yield transcripts?)", fontsize=11.5, pad=5)
    ax.set_xlabel("Pol II per gene (occupancy)", fontsize=10)
    ax.set_ylabel("transcripts / h", fontsize=9.5); ax.tick_params(labelsize=9)

    # Panel 10: where Pol II spends its time (mean state-time fractions, one stacked bar).
    ax = axes.flat[9]
    palette = [("pre_initiation", "#c9d6df"), ("paused", "#f2c14e"), ("elongating", "#3b8a5a"),
               ("stalled", "#A63446"), ("terminating", "#6b4e9e")]
    left = 0.0
    for name, color in palette:
        frac = float(np.nanmean(pd.to_numeric(df.get(f"frac_{name}"), errors="coerce")))
        ax.barh(0, frac, left=left, color=color, edgecolor="white")
        if frac > 0.04:
            ax.text(left + frac / 2, 0, f"{name}\n{frac*100:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if name != "pre_initiation" else "#333")
        left += frac
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_title("Where Pol II spends its time\n(mean fraction per state)", fontsize=11.5, pad=5)
    ax.set_xlabel("fraction of polymerase-time", fontsize=10); ax.tick_params(labelsize=9)

    # Panel 11: transcript-OUTPUT distribution across genes -- a HISTOGRAM of gene counts vs productive output
    # (completion rate, transcripts/h, log-x) with silent/low/moderate/high output bands shaded. Complements the
    # occupancy histogram (panel 5): occupancy is length/speed-confounded, this is the actual transcript yield.
    # Bands (transcripts/h) anchor on the human productive band ~0.3-90/h: silent below it, then log-spaced x10.
    ax = axes.flat[10]
    out = pd.to_numeric(df.get("completion_rate_per_min"), errors="coerce").dropna() * 60.0   # -> transcripts/h
    floor = 0.01                                               # plot floor so silent (~0-output) genes show
    bounds = [0.3, 3.0, 30.0]                                  # silent | low | moderate | high (transcripts/h)
    cat_colors = ["#c9d6df", "#9ab0cc", "#6f8fbf", "#A63446"]
    cat_names = ["silent", "low", "moderate", "high"]
    if len(out):
        vplot = np.clip(out, floor, None)                     # silent / zero-output genes pile at the floor
        ax.set_xscale("log")
        bins = np.logspace(np.log10(floor), np.log10(max(float(vplot.max()), floor * 10)), 30)
        counts, edges, patches = ax.hist(vplot, bins=bins, edgecolor="white", linewidth=0.3, zorder=2)
        for patch, lo, hi in zip(patches, edges[:-1], edges[1:]):
            patch.set_facecolor(cat_colors[sum(np.sqrt(lo * hi) >= b for b in bounds)])  # colour bar by band
        ymax = counts.max() if counts.size else 1.0
        hi_x = max(float(out.max()), 30.0)
        masks = [out < 0.3, (out >= 0.3) & (out < 3), (out >= 3) & (out < 30), out >= 30]
        label_x = [np.sqrt(floor * 0.3), np.sqrt(0.3 * 3), np.sqrt(3 * 30), np.sqrt(30 * hi_x)]
        for b in bounds:
            ax.axvline(b, color="#444", lw=0.8, ls="--", alpha=0.6, zorder=3)
        for name, mask, lx in zip(cat_names, masks, label_x):
            ax.text(lx, ymax * 0.96, f"{name}\n{100*mask.mean():.0f}%", ha="center", va="top",
                    fontsize=8, color="#333", zorder=4)
        ax.axvline(max(float(out.median()), floor), color="#A63446", lw=2.0, zorder=5)
    ax.set_xlabel("transcripts produced per gene (transcripts / h)", fontsize=10)
    ax.set_ylabel("number of genes", fontsize=9.5)
    ax.set_title("Transcript-output distribution across genes\n(gene counts; silent/low/moderate/high output)",
                 fontsize=11.5, pad=5); ax.tick_params(labelsize=9)

    # Panel 12: rank-vs-output scatter.
    _zipf_panel(axes.flat[11], out)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


# Genome-browser metric tracks: (column, label, unit, display scale, colour, log-y).
# Heavy-tailed metrics (occupancy, pause, pausing index) use a LOG y-axis so a few
# outliers don't squash the typical genes.
_GENOME_TRACKS = [
    ("pol2_occupancy", "Pol II / gene", "Pol", 1.0, "#465775", True),
    ("initiation_rate_per_min", "initiation", "/min", 1.0, "#1b6ca8", False),
    ("elongation_velocity_bp_s", "elongation", "kb/min", 0.06, "#3b8a5a", False),
    ("mean_pause_dwell_s", "pause", "min", 1 / 60, "#c77d18", True),
    ("pausing_index", "pausing index", "ratio", 1.0, "#6b4e9e", True),
]


def _draw_boundaries(ax, boundaries, *, label=False, top=False, bp_per_site=1000):
    """Mark TAD boundaries as vertical dashed lines (drawn in site coordinates);
    optionally label them in kb."""
    kb_per_site = bp_per_site / 1000.0
    for b in boundaries or []:
        ax.axvline(b, color="#c0392b", lw=0.9, ls="--", alpha=0.55, zorder=1)
    if label and top and boundaries and len(boundaries) <= 16:
        for b in boundaries:
            ax.annotate(f"{b * kb_per_site:g}", xy=(b, 1.02), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7.5, color="#c0392b")


def plot_genome_tracks(df: pd.DataFrame, out_path: Path, *, title: str,
                       boundaries=None, chain: int | None = None,
                       bp_per_site: int = 1000) -> None:
    """Publication figure (genome-browser view): each transcription metric plotted
    ALONG the genome coordinate -- one marker per gene at its TSS (alleles averaged) --
    with TAD boundaries marked (red dashed). Shows WHERE active genes and metric
    outliers sit relative to TADs."""
    g = df.copy()
    g["pos"] = pd.to_numeric(g["chain_relative_tss"], errors="coerce")
    agg = g.dropna(subset=["pos"]).groupby("pos").mean(numeric_only=True).reset_index().sort_values("pos")
    pos = agg["pos"].to_numpy()
    n = len(_GENOME_TRACKS)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.9 * n + 0.8), sharex=True)
    for ax, (key, lab, unit, sc, col, logy) in zip(axes, _GENOME_TRACKS):
        if key not in agg.columns:
            continue
        y = pd.to_numeric(agg[key], errors="coerce").to_numpy() * sc
        if logy:
            yy = np.where(y > 0, y, np.nan)
            ax.set_yscale("log")
            base = np.nanmin(yy) if np.isfinite(yy).any() else 1.0
            ax.vlines(pos, base, yy, color=col, lw=0.8, alpha=0.35, zorder=2)
            ax.scatter(pos, yy, s=18, color=col, zorder=3, edgecolor="white", linewidth=0.3)
            ax.set_ylabel(f"{lab}\n({unit}, log)", fontsize=10)
        else:
            ax.vlines(pos, 0, y, color=col, lw=1.0, alpha=0.5, zorder=2)
            ax.scatter(pos, y, s=18, color=col, zorder=3, edgecolor="white", linewidth=0.3)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"{lab}\n({unit})", fontsize=10)
        _draw_boundaries(ax, boundaries, bp_per_site=bp_per_site)
        ax.tick_params(labelsize=8.5); ax.margins(x=0.01)
    _draw_boundaries(axes[0], boundaries, label=True, top=True, bp_per_site=bp_per_site)
    axes[0].annotate("red dashed = TAD boundary",
                     xy=(0.99, 1.16), xycoords="axes fraction", ha="right", fontsize=9,
                     style="italic", color="#555")
    axes[-1].set_xlabel("genome coordinate (kb; all alleles folded onto one chain)", fontsize=11.5)
    # data is in site coordinates; display tick labels in kb
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * bp_per_site / 1000.0:g}"))
    if chain:
        axes[-1].set_xlim(0, chain)
    fig.suptitle(title, fontsize=14.5, y=0.995)
    fig.text(0.5, 0.004, "one point per gene at its TSS (alleles averaged); height = metric value",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight"); fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight"); plt.close(fig)


def plot_tracks(chip: np.ndarray, nascent: np.ndarray, genes: np.ndarray,
                chain: int, out_path: Path, *, title: str, boundaries=None,
                bp_per_site: int = 1000) -> None:
    """Pol II ChIP (all present) and nascent (engaged) per-site tracks, folded
    across alleles onto one locus; gene bodies shaded; TAD boundaries marked."""
    xs = np.arange(chain)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    a1.fill_between(xs, chip, color="#465775", lw=0)
    a1.set_ylabel("Pol II per site\n(per allele)", fontsize=10.5)
    a1.set_title("Pol II occupancy  —  all polymerases (ChIP-seq analog)", fontsize=12, loc="left")
    a2.fill_between(xs, nascent, color="#3b8a5a", lw=0)
    a2.set_ylabel("nascent RNA per site\n(per allele)", fontsize=10.5)
    a2.set_title("Nascent RNA  —  engaged (elongating + stalled) Pol II (GRO/PRO-seq analog)",
                 fontsize=12, loc="left")
    for g in genes:
        lo, hi = sorted((int(g["tss"]) % chain, int(g["tes"]) % chain))
        for ax in (a1, a2):
            ax.axvspan(lo, hi, color="#000000", alpha=0.06, lw=0)
    for ax in (a1, a2):
        _draw_boundaries(ax, boundaries, bp_per_site=bp_per_site)
    _draw_boundaries(a1, boundaries, label=True, top=True, bp_per_site=bp_per_site)
    a2.set_xlabel("genome coordinate (kb; all alleles folded onto one chain)", fontsize=11)
    # xs is in site coordinates; display tick labels in kb
    a2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * bp_per_site / 1000.0:g}"))
    for ax in (a1, a2):
        ax.tick_params(labelsize=9)
        ax.margins(x=0)
    a1.annotate("grey = gene bodies · red dashed = TAD boundary · sharp 5′ peaks = promoter Pol II",
                xy=(0.5, 1.20), xycoords="axes fraction", ha="center", fontsize=9,
                style="italic", color="#555")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="transcription-ON config YAML")
    ap.add_argument("--h5", type=Path, default=None,
                    help="LEFPositions.h5 for --config (default: run/reuse next to --out-dir)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for TSV + figures (default: the H5's parent)")
    ap.add_argument("--force-1d", action="store_true",
                    help="rerun the 1D stage even if a matching H5 exists")
    ap.add_argument("--pause-offset", type=int, default=1,
                    help="pause site is this many lattice sites downstream of the TSS "
                         "(matches gene.pause_offset); the pausing index uses TSS->pause "
                         "and pause-site-only windows")
    ap.add_argument("--measure-chunk", type=int, default=2000,
                    help="frames per streamed block when reading the trajectory")
    ap.add_argument("--no-plot", action="store_true", help="write only the TSV")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.config.read_text())
    lef = base["lef"]
    tick_seconds = float(lef.get("tick_seconds") or 0.0)
    if tick_seconds <= 0:
        raise SystemExit("config lef.tick_seconds must be a positive number")

    label = _safe_label(args.config.stem)
    out_dir = args.out_dir or (args.h5.parent if args.h5 else args.config.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = _resolve_h5(
        cfg_path=args.config, requested_h5=args.h5.resolve() if args.h5 else None,
        out_dir=out_dir, label=label, force_1d=args.force_1d,
    )

    print(f"config       : {args.config}")
    print(f"trajectory   : {h5_path}")
    print(f"tick_seconds : {tick_seconds:g}")
    df, agg, chip, nascent, genes = measure_transcription(
        h5_path, tick_seconds, pause_offset=args.pause_offset, chunk=args.measure_chunk)

    # Drop physically-impossible event-count alleles (UID double-counts) from the SUMMARIES and PLOTS so a
    # single degenerate gene cannot dominate the rank plot or inflate the locus aggregate. The raw per-gene
    # rows (with the events_implausible flag) are still written to the TSV -- a complete, honest record.
    dfp = _mask_implausible(df)
    if dfp is not df:
        agg = _aggregate(dfp, n_alleles=int(agg.get("n_genes", len(df))))

    tsv = out_dir / "transcription_metrics.tsv"
    out_df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    out_df.to_csv(tsv, sep="\t", index=False)
    print(f"\nmeasured {len(df)} gene(s); locus aggregate:")
    for k in ("initiation_rate_per_min", "completion_rate_per_min", "synthesis_rate_nt_per_s",
              "pol2_occupancy", "elongation_velocity_bp_s", "pausing_index",
              "stall_frac", "active_fraction"):
        print(f"  {k:28s}: {agg[k]:.4g}")
    print(f"wrote {tsv}")

    # Human-biology range check (the headline "are my transcription numbers in range?").
    rr = realism_report(dfp)
    print_realism_report(rr)
    act = activity_report(dfp)
    realism_tsv = out_dir / "transcription_realism.tsv"
    rr.to_csv(realism_tsv, sep="\t", index=False)
    pd.DataFrame([act]).to_csv(out_dir / "transcription_activity.tsv", sep="\t", index=False)
    print(f"wrote {realism_tsv} + transcription_activity.tsv")

    # Per regulatory-class breakdown (when the trajectory carries the gene_class tag).
    cb = class_breakdown(dfp, genes)
    if cb is not None:
        print_class_breakdown(cb)
        class_tsv = out_dir / "transcription_class_breakdown.tsv"
        cb.to_csv(class_tsv, sep="\t", index=False)
        print(f"wrote {class_tsv}")

    if not args.no_plot:
        with h5py.File(h5_path, "r") as h:
            chain = int(h.attrs["chain_length"])
            bp_per_site = int(h.attrs.get("bp_per_site", 1000))
        # TAD boundaries (chain-relative) for the genome-coordinate plots. Internal
        # boundaries appear as adjacent edge pairs; merge those within ~5 kb and drop
        # the chromosome ends (~3 kb) so each boundary is one labelled line. The
        # thresholds are physical, so they convert to site counts via bp_per_site.
        edge = max(1, round(3000 / bp_per_site))
        merge = max(1, round(5000 / bp_per_site))
        tads = lef.get("topology_kwargs", {}).get("tads", [])
        raw_b = sorted({int(t["left"]) for t in tads} | {int(t["right"]) for t in tads})
        boundaries: list[int] = []
        for b in raw_b:
            if edge < b < chain - edge and (not boundaries or b - boundaries[-1] > merge):
                boundaries.append(b)
        sub = f"{args.config.name}, tick={tick_seconds:g}s"
        gene_svg = out_dir / "transcription_metrics.svg"
        plot_genome_tracks(dfp, gene_svg, title=f"Transcription metrics along the genome\n{sub}",
                           boundaries=boundaries, chain=chain, bp_per_site=bp_per_site)
        dist_svg = out_dir / "transcription_distributions.svg"
        plot_distributions(dfp, dist_svg,
                           title=f"Transcription metric distributions\n{sub}")
        n = len(df)
        nloci = int(df["chain_relative_tss"].nunique()) if "chain_relative_tss" in df else n
        nall = n // max(nloci, 1)
        pub_svg = out_dir / "transcription_distributions_pub.svg"
        plot_transcription_distributions_pub(
            dfp, pub_svg,
            title=f"Per-gene transcription kinetics\n"
                  f"n = {n} genes ({nloci} loci × {nall} alleles), {sub}")
        # publication figure legend (auto-filled with this run's n / loci / alleles)
        (out_dir / "transcription_distributions_pub_legend.txt").write_text(
            "Figure. Per-gene transcription kinetics across the simulated gene population.\n"
            f"Distributions over n = {n} genes ({nloci} genes in the modeled locus x {nall} "
            "replicate chains, i.e. alleles). Red line, model median.\n"
            "(a) Initiation rate (Pol II loading, including the ~75-80% that prematurely terminate "
            "at the pause). (b) Productive transcript output (Pol II reaching the 3' end). (c) transcript output by "
            "gene rank: genes ranked by output (descending) on log-log axes. (d) Single-polymerase "
            "promoter-proximal pause dwell. (e) Pausing index vs gene activity: per-gene pausing "
            "index (promoter-proximal / gene-body Pol II density) against productive output "
            "(transcripts/h), log-log, with the binned-median trend (red); highly expressed genes "
            "show a lower pausing index (traveling ratio). (f) Elongation velocity. (g) Productive "
            f"fraction (initiations reaching the 3' end), shown only for genes with >={PF_MIN_EVENTS} "
            "initiations (where the ratio is estimable); genes with fewer (low-confidence) or zero "
            "(undefined) initiations are tallied separately, not plotted as a spike at zero. "
            "(h) Pol II occupancy per gene. (i) 3' termination dwell. Heavy-tailed metrics use a "
            "logarithmic x-axis.\n")
        track_svg = out_dir / "transcription_tracks.svg"
        plot_tracks(chip, nascent, genes, chain, track_svg,
                    title=f"Pol II ChIP & nascent RNA tracks\n{sub}", boundaries=boundaries,
                    bp_per_site=bp_per_site)
        print(f"wrote {gene_svg}, {dist_svg}, {pub_svg} (+legend.txt), {track_svg}")


if __name__ == "__main__":
    main()
