#!/usr/bin/env python
"""Generate human-calibrated config1-4 for the loop-extrusion + transcription
pipeline at an arbitrary 1D tick size (1 site = BP_PER_SITE bp).

Parameters are authored as TICK-INDEPENDENT biology (rates /s, durations s,
sizes kb); ``--tick-seconds`` converts them to per-tick YAML probabilities via
p = 1-exp(-k*t). Values are human (or closest mammalian) in-vivo. Key anchors:

  ELONGATION  ~2.1 kb/min model anchor; in-vivo median ~1.5 kb/min (Veloso 2014),
              rising to 2-5 kb/min after ~15 kb (Jonkers & Lis 2015).
  PAUSING     single-Pol dwell ~42 s (Steurer 2018); ~80% premature termination
              -> productive fraction ~0.15-0.25 (Zimmer STL-seq 2021; Gressel 2019).
  INITIATION  productive ~1/min median for mRNA, ~0.3 lincRNA, up to ~87/min (HSPA1A);
              realized rate is pause-escape-limited, not loading-limited (Gressel 2019).
  3' DWELL    ~3 min over a ~3.3 kb termination window (Schwalb 2016; Cortazar 2019).
  TADs        classical scale (mean ~833 kb); convergent-CTCF + cohesin mechanism (Rao 2014).
  CTCF        per-encounter stall ~0.125 (Gabriele 2022); config3 strengthens it x2.5.
  COHESIN     1/240 kb; residence ~22.5 min (Hansen 2017 ~22; Gabriele 2022 ~20);
              CTCF-stabilized x4 (Gabriele 2022, "stabilized fourfold").
  CLASSES     ~3800 housekeeping genes, ~20% (Eisenberg & Levanon 2013);
              E-P median ~24 kb (Gasperini 2019; drawn here at ~30 kb).

config1 = txn ON; config2 = txn OFF (same skeleton/genes); config3 = config2 +
stronger CTCF (x bstr-mult); config4 = config3 + UV lesions. Deterministic (seeded).
"""

from __future__ import annotations
import argparse
import math
import numpy as np

# ============================== Locus & RNG ==============================
# Lattice resolution: bp per site (== one 3D monomer). 250 bp ~ 1 Pol/site ~ the real
# ~3.7 Pol/kb packing of active genes. --bp-per-site overrides; kb -> sites via kb*1000/BP_PER_SITE.
BP_PER_SITE = 250
LOCUS_BP = 10_000_000         # physical locus span (10 Mb); CHAIN sites = LOCUS_BP // BP_PER_SITE
CHAIN = LOCUS_BP // BP_PER_SITE   # lattice length in SITES; --chain overrides
GENE_DENSITY = 16.0           # target genes/Mb before no-overlap packing
SEED = 7                      # default RNG seed (--seed overrides)


def kb2sites(kb: float) -> int:
    """Convert a genomic distance in kb to lattice sites at the current BP_PER_SITE."""
    return max(1, int(round(float(kb) * 1000.0 / BP_PER_SITE)))


def bp2sites(bp: float) -> int:
    """Convert a genomic distance in bp to lattice sites at the current BP_PER_SITE."""
    return max(1, int(round(float(bp) / BP_PER_SITE)))

# ========================= Timing / tick rescaling =======================
# Per-tick probabilities below are authored at REFERENCE_TICK_SECONDS and rescaled
# to the requested --tick-seconds via continuous-time rates.
REFERENCE_TICK_SECONDS = 20.0        # authoring tick for the prob ranges
REFERENCE_TRAJECTORY_LENGTH = 12096  # recorded LEF steps (rescaled by tick)
REFERENCE_WARMUP_STEPS = 10000       # discarded warm-up ticks (rescaled)
REFERENCE_MD_STEPS_PER_BLOCK = 5000  # 3D MD steps per recorded block (rescaled)
RESTART_EVERY_BLOCKS = 5000          # polymer restart interval (traj snapped to a multiple)

# ========================== Cohesin (loop extrusion) =====================
# Mean cohesin residence /s (lifetime_ticks = this/tick); 22.5 min live-imaging base
# (Gabriele 2022 ~20 min; Hansen 2017 ~22 min mESC FRAP).
COHESIN_LIFETIME_SECONDS = 22.5 * 60
# CTCF-anchored lifetime = this * free lifetime; x4 verbatim Gabriele 2022 ("stabilized fourfold").
CTCF_LIFETIME_BOOST = 4

# ====================== RNAPII transcription kinetics ====================
# Expressed-gene loading rate /s (geometric-mean anchor; load_prob = 1-exp(-this*tick)). Sets the overall
# transcription LEVEL among expressed genes (silenced genes get a graded near-0 load; see SILENT_PROB).
# Output is bounded by two soft ceilings on the hottest genes: gene-body density (<=~0.25 Pol/kb) and
# promoter pause-site clearance (~1/dwell). 0.0013 -> detected median ~1.2-1.3 transcripts/h, ~69% detected.
RNAP_LOAD_RATE = 0.0013
# Hard cap on a single gene's loading rate /s. With the pause-initiation limit enforced in the engine
# (rnapii_promoter_occlusion; Gressel 2019, Bartman 2019), realized initiation is gated by pause-escape,
# not this cap, so it stays open to let fast-pause genes reach the hot tail (up to HSPA1A ~87/min).
LOAD_RATE_MAX = 0.30
# Pre-initiation dwell (s), coupled to activity rank (log-linear, set in the activity post-pass) so active
# promoters re-fire fast (pre-loaded PIC) and quiet ones stay slow.
PRE_INIT_DWELL_FAST_S = 2.0    # active genes: pre-init at the tick floor
PRE_INIT_DWELL_SLOW_S = 30.0   # quietest genes: slow de-novo PIC assembly
# Premature-termination rate /s at the pause; productive_fraction = k_release/(k_release+this).
# 0.025 centers productive_fraction at ~0.20-0.25 (STL-seq; Mukherjee & Guertin 2025, ~1 in 4 productive).
RNAP_PAUSE_TERM_RATE = 0.025
# 3'-termination/unbind rate /s for a Pol in the termination window (and the legacy single-slot dwell).
RNAP_UNBIND_RATE = 0.005677
# 3' TERMINATION WINDOW: a Pol reaching the TES walks downstream through a per-gene window (lognormal) before
# releasing, vacating the TES on the first step so the 3' end no longer caps output at 1/dwell. Schwalb 2016
# (TT-seq): median read-through window ~3.3 kb past the pA site; the >10 kb tail is the model's draw. 0 = slot.
TERM_WINDOW_KB_MEDIAN = 3.3        # median 3' termination-window width (Schwalb 2016)
TERM_WINDOW_SIGMA = 0.8            # lognormal sigma -> model draws ~8% of windows >10 kb
# Decelerated speed of a terminating Pol crawling the window (Cortazar 2019: ~0.7-0.9 kb/min); sets
# rnapii_termination_step_prob -> ~1-3 min total termination time.
TERM_SPEED_KB_PER_SECOND = 0.017
# Elongation speed MEDIAN anchor (kb/min = 60*this); per gene drawn log-normally and soft-saturated into
# [ELONG_MIN, ELONG_MAX], so elongation is GENE-VARIABLE. In-vivo median ~1.5 kb/min (Veloso 2014), 0.5->2-5
# kb/min position-dependent (Jonkers & Lis 2015), up to ~6 in HeLa (Fuchs 2014). 0.035 kb/s = 2.1 kb/min.
RNAP_SPEED_KB_PER_SECOND = 0.035
# Per-gene log-normal spread of elongation speed (ln sigma); 0.45 with the soft knee below -> ~0.5-6 kb/min.
ELONG_LOGNORM_SIGMA = 0.45
# Clamp (kb/min) on per-gene speed (in-vivo range): MIN keeps slow genes resolvable; MAX caps the fast tail
# and sizes the global stride so step_prob <= 1. Applied as a SMOOTH (tanh) saturation above ELONG_SOFT_KNEE
# (not a hard clip) so the fast tail asymptotes toward MAX instead of piling up on it.
ELONG_MIN_KB_PER_MIN = 0.5
ELONG_MAX_KB_PER_MIN = 6.0
ELONG_SOFT_KNEE_KB_PER_MIN = 4.5   # speeds above this saturate smoothly toward ELONG_MAX
# POSITIVE activity->elongation coupling (log-linear tilt by activity rank, in the post-pass): active genes
# elongate faster (Danko 2013, elongation-rate vs body-density rho~0.46; Kwak & Lis 2013). Kept weak: faster
# elongation lowers body density and so inflates the pausing index. mult = exp(COUPLING*(rank-0.5)); 0 = off.
ELONG_ACTIVITY_COUPLING = 0.3
# ACTIVITY-DEPENDENT SINGLE-POL PAUSE DWELL = 1/(k_release + k_term). Both pause-exit rates are scaled per
# gene by a log-linear factor of activity RANK so dwell is ANTI-correlated with activity (CDK9/P-TEFb shortens
# the pause: Gressel 2017; pausing is prominent at poised genes: Adelman & Lis 2012). Scaling BOTH rates by
# the same factor leaves productive_fraction unchanged. After the ramp all rates are renormalised so the
# median lands at PAUSE_DWELL_MEDIAN_S. 42 s = human live-cell paused residence (Steurer 2018).
PAUSE_DWELL_MEDIAN_S = 42.0
# Systematic activity trend (ln full width across the rank range): 2.2 -> most active genes pause ~3x shorter
# than median, least active ~3x longer.
PAUSE_DWELL_ACTIVITY_SPREAD = 2.2
# Extra ln-shortening applied ONLY over rank 0.9->1.0 so the very top genes keep stretching toward PAUSE_MIN
# (re-initiation ~1/dwell) and the output tail stays a smooth heavy tail instead of a flat ceiling. 0 = off.
PAUSE_DWELL_TOPTAIL_BOOST = 1.3
# Per-gene log-normal scatter on the dwell ramp: real pausing is set mostly by promoter architecture and only
# loosely by activity (Day 2016 Spearman ~0.18; Zimmer 2021). Mean-zero in log. 0 = deterministic ramp.
PAUSE_DWELL_ACTIVITY_NOISE = 0.2
# Hard clamp on the single-Pol dwell (s), applied to the common rate scale (never perturbs productive_fraction)
# AFTER the median pin, so it only trims the extreme tails.
PAUSE_MIN_S = 2     # tick floor: let the hottest pauses collapse into the steric regime
PAUSE_MAX_S = 90
# ACTIVITY-DEPENDENT PRODUCTIVE FRACTION (gradient on the k_release/k_term RATIO): real pf is activity-graded
# (quiet ~0.1-0.25, active ~0.4-0.8; Mukherjee & Guertin 2025 k_pre/k_rel~2.6 -> ~1/4 productive for a typical
# gene, high-output genes terminate less prematurely -- Gressel 2019). The dwell ramp scales k_release and
# k_term together (ratio-invariant); this changes the RATIO: k_term reduced for ACTIVE genes (pf up), mildly
# raised for QUIET genes (pf down). Deterministic in rank. Keep the realism MEDIAN <= 0.30.
PF_QUIET_GAIN = 0.2       # ln k_term boost at rank 0 (quiet -> slightly lower pf)
PF_ACTIVE_GAIN = 1.1      # ln k_term reduction at rank 1 before the top boost (active -> higher pf)
PF_ACTIVE_TOPBOOST = 0.8  # extra convex ln k_term reduction over rank 0.9->1.0 (hottest genes -> pf ~0.5-0.7)

# ============== Activity distribution (loading heterogeneity) ============
# Per-gene loading is drawn from a POWER-LAW (Zipf): the real transcriptome is power-law -- most genes lowly
# expressed, a thin heavy tail high (Furusawa & Kaneko 2003). The heavy tail pulls the GEOMETRIC MEAN up, so
# renormalising it to RNAP_LOAD_RATE leaves the mode far below -> a bottom-heavy 'low'-dominant output.
LOAD_POWERLAW_ALPHA = 2.4     # steeper -> more bottom-heavy at fixed geo-mean
LOAD_POWERLAW_RANGE = 3000.0  # max/min multiplier (dynamic range)
LOAD_LOGNORM_SIGMA = 1.3      # on/off gate for loading heterogeneity (>0 = on)
# Per-class fraction of genes forced NOT-DETECTED (cell-type/developmental off in this cell); sized so the
# not-detected total lands ~30-40% genome-wide -> ~60-70% detected.
SILENT_PROB = {"hk_const": 0.0, "hk_high": 0.0, "celltype": 0.27, "dev": 0.56}
LOAD_SILENT_FRAC = 0.0         # DEPRECATED/unused (silent genes split into hard-off + ultra-low, below).
# Not-detected structure: of all alleles ~15-25% TRULY OFF (load 0), ~10-20% ULTRA-LOW (rare leakage
# ~0.001-0.05 completions/h), ~60-70% DETECTED. SILENT_PROB sets the not-detected pool; SILENT_HARD_FRAC
# splits it into truly-off vs ultra-low (the latter draw a log-uniform load_prob over LOAD_ULTRALOW_RANGE).
# Real expression is a continuum, not a hard spike at 0 (Ramskold 2009).
SILENT_HARD_FRAC = 0.5               # of force-silenced genes: fraction TRULY OFF (load 0); rest ultra-low
LOAD_ULTRALOW_RANGE = (2e-6, 1e-4)   # load_prob (log-uniform) for ultra-low genes -> ~0.001-0.05 completions/h
LOAD_TADSIZE_EXPONENT = 0.2    # loading factor = (median_TAD_size / TAD_size)^this (small TADs more active)

# ============== RNAPII <-> cohesin coupling (moving barrier) =============
# Cohesin-bypass time (s); block_prob = exp(-tick/this), k_bypass = 1/this. Bigger = stronger Pol barrier.
RNAP_BYPASS_SECONDS = 100.0                   # elongating/paused Pol
RNAP_TERM_BYPASS_SECONDS = 100.0              # terminating Pol at the 3' end
RNAP_PRE_INITIATION_BYPASS_SECONDS = 12.0     # pre-initiation Pol (leaky barrier)
# Cohesin-bypass time (s) at a stalling lesion; default lesion_block_prob = exp(-tick/this). 262.6 s
# reproduces the legacy 0.97 prob at 8 s/tick, holding the physical barrier dwell constant across ticks.
LESION_BYPASS_SECONDS = 262.6
RNAP_HEADON_PUSH_PROB = 1.0   # prob elongating Pol pushes a converging (head-on) cohesin leg (Banigan)
RNAP_PUSH_PROB = 0.0          # prob it pushes a co-directional (trailing) leg (0 = impede only)
RNAP_STALL_PROB = 0.0         # intrinsic prob Pol stalls on any cohesin contact (0 = off; non-Banigan)
RNAP_PAUSE_RESTRAINT = 1.0    # cohesin next to a paused Pol scales its pause-release (1 = off)

# ============================= Lesions (config4) =========================
LESION_PRERECOGNITION_SECONDS = 2400.0   # mean time (s) a UV lesion sits unrecognized before repair starts
LESION_REPAIR_SECONDS = 360.0            # mean repair duration (s) once recognized

# ======================= TAD architecture & boundaries ===================
# A smooth gene-density field is the causal latent -> TAD size, class mix, and boundary strength all
# follow from it (size is a consequence, not an input).
SHORT_TAD_KB = 750.0           # target median size of dense/short TADs (diagnostic; SHORT_SPACING is the knob)
LONG_TAD_KB  = 1250.0          # target median size of sparse/long TADs (diagnostic; LONG_SPACING is the knob)
SHORT_SPACING = 600.0          # dense-region boundary-spacing endpoint (kb)
LONG_SPACING  = 1450.0         # sparse-region boundary-spacing endpoint (kb)
DENS_ACTIVE = 20.0             # genes/Mb at max density
DENS_POOR   = 7.0              # genes/Mb at min density
TXN_NOISE   = 0.12             # transcription<->density/size decoupling
TAD_SIZE_TXN_WEIGHT = 0.78     # txn = this*(inverse TAD size) + (1-this)*density + noise
BSTR_RANGE  = (0.03, 0.15)     # CTCF boundary per-encounter cohesin-stall prob; centered on Gabriele 2022
                               # (12.5% per-encounter stall, Fbn2 mESC; plausible band ~0.125-0.25)
BSTR_NOISE  = 0.001            # jitter added to boundary strength
BSTR_TXN_WEIGHT = 0.5          # boundary strength = this*flank_txn + (1-this)*flank_density

# =========================== Gene classes & mix ==========================
# frac = genome fraction (documentation-only; class is drawn per TAD from FRAC_ACTIVE/FRAC_POOR below);
# init = PRE-INITIATION->PAUSED prob range (ref 20 s); prel = pause-release rate range k_release (ref 20 s);
# req = enhancer-gated; nenh = (min,max) enhancers. hk_const/hk_high = housekeeping (~3800 genes, ~20%;
# Eisenberg & Levanon 2013 list 3804), celltype = cell-type-specific, dev = developmental. init is HIGH (fast
# pre-init) so the PAUSE -- not pre-initiation -- gates the occlusion re-initiation cycle.
CLASSES = {
    "hk_const": dict(frac=0.15, init=(0.93, 0.97), prel=(0.098, 0.145), req=False, nenh=(0, 0)),
    "hk_high":  dict(frac=0.05, init=(0.95, 0.98), prel=(0.123, 0.193), req=False, nenh=(0, 1)),
    "celltype": dict(frac=0.55, init=(0.90, 0.95), prel=(0.075, 0.123), req=True,  nenh=(1, 3)),
    "dev":      dict(frac=0.25, init=(0.85, 0.92), prel=(0.050, 0.098), req=True,  nenh=(2, 5)),
}
ORDER = list(CLASSES)
FRAC = np.array([CLASSES[c]["frac"] for c in ORDER])   # documentation-only; see FRAC_ACTIVE/FRAC_POOR
# Per-TAD class mix [hk_const, hk_high, celltype, dev], interpolated by the TAD's transcription level.
# Housekeeping is enriched in active TADs but still a minority (~35%); gene-poor TADs are developmental +
# off cell-type. The activity-weighted blend nets ~20-25% housekeeping genome-wide.
FRAC_ACTIVE = np.array([0.22, 0.06, 0.64, 0.08])   # mix in max-txn TADs
FRAC_POOR   = np.array([0.08, 0.01, 0.48, 0.43])   # mix in min-txn TADs (celltype-heavy + dev silent)

# ===================== Per-class output stratification ===================
# Three class-conditioned levers lift housekeeping output toward the ~1/min mRNA median (cell-type/dev stay
# at lincRNA/poised level):
#   (1) LOADING -- hk promoters are strongly loaded (CLASS_LOAD_MULT; relative, renormalized).
#   (2) PREMATURE TERMINATION -- hk genes terminate LESS at the pause (higher productive_fraction; per-class
#       k_term in CLASS_PAUSE_TERM_RATE; engine reads per-gene pause_term_prob, falls back to global).
#   (3) 3' TERMINATION DWELL -- active genes terminate faster (CLASS_UNBIND_RATE). The readthrough WINDOW is
#       left uniform (CLASS_TERM_WINDOW_MULT all 1.0).
CLASS_LOAD_MULT = {"hk_const": 1.8, "hk_high": 2.5, "celltype": 1.0, "dev": 0.7}
# Per-gene log-normal jitter (ln sigma) on the EMITTED loading rate, applied AFTER the rank/dwell/pf post-pass
# (perturbs only load_prob, never the activity rank / silent mask / dwell rng -- all stay byte-identical).
# Turns the 4 CLASS_LOAD_MULT levels into overlapping lobes so the realized initiation distribution is
# continuous rather than bimodal. Mean-zero in log. Silent genes exempt. 0 = legacy hard class steps.
CLASS_LOAD_MULT_LOGSIGMA = 0.5
CLASS_PAUSE_TERM_RATE = {"hk_const": 0.020, "hk_high": 0.016, "celltype": 0.028, "dev": 0.038}
CLASS_TERM_WINDOW_MULT = {"hk_const": 1.0, "hk_high": 1.0, "celltype": 1.0, "dev": 1.0}
# Per-class gene-length multiplier on the lognormal median: housekeeping COMPACT, developmental LONGER
# (Eisenberg & Levanon 2003).
CLASS_GENE_LEN_MULT = {"hk_const": 0.7, "hk_high": 0.7, "celltype": 1.15, "dev": 1.6}
# Per-class 3' termination/unbind rate /s, anti-correlated with activity: hk terminate faster (~1.5-2 min
# dwell), dev slower (~3.7 min). Sets per-gene termination_prob (falls back to RNAP_UNBIND_RATE).
CLASS_UNBIND_RATE = {"hk_const": 0.007, "hk_high": 0.0085, "celltype": 0.006, "dev": 0.0045}


def _rate_to_prob(rate_per_second: float, tick_seconds: float) -> float:
    """Convert a continuous-time rate to a per-tick event probability."""
    return 1.0 - math.exp(-float(rate_per_second) * float(tick_seconds))


def _ref_prob_to_rate(prob: float, ref_tick_seconds: float = REFERENCE_TICK_SECONDS) -> float:
    """Infer a continuous rate from a reference per-tick probability."""
    p = min(max(float(prob), 0.0), 1.0 - 1e-12)
    return -math.log1p(-p) / float(ref_tick_seconds)


def _scale_ref_prob(prob: float, tick_seconds: float) -> float:
    """Convert a 20 s reference probability to the requested tick size."""
    return _rate_to_prob(_ref_prob_to_rate(prob), tick_seconds)


def _scale_ref_range(bounds, tick_seconds: float):
    return tuple(_scale_ref_prob(x, tick_seconds) for x in bounds)


def _round_prob(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _kbmin_to_sites_per_tick(kb_per_min: float, tick_seconds: float) -> float:
    """Elongation speed kb/min -> lattice sites advanced per tick (resolution-general)."""
    return kb_per_min / 60.0 * 1000.0 / BP_PER_SITE * tick_seconds


def _elong_step_prob(kb_per_min: float, stride: int, tick_seconds: float) -> float:
    """Per-gene elongation_step_prob: speed (kb/min) over the global stride; <=1 because stride is
    sized for ELONG_MAX. stride * step_prob preserves the target speed on average for any tick."""
    return min(1.0, _kbmin_to_sites_per_tick(kb_per_min, tick_seconds) / stride)


def _soft_clamp_elong(kb_per_min: float) -> float:
    """Bound per-gene elongation speed to [ELONG_MIN, ELONG_MAX] with a SMOOTH (tanh) saturation above
    ELONG_SOFT_KNEE instead of a hard np.clip: speeds below the knee are untouched, speeds above it are
    compressed asymptotically toward ELONG_MAX, so the fast tail no longer piles up as a delta-spike exactly
    on the ceiling. Median/bulk and the realism band are unchanged (only genes above the knee move)."""
    v = float(kb_per_min)
    knee = ELONG_SOFT_KNEE_KB_PER_MIN
    if v > knee and ELONG_MAX_KB_PER_MIN > knee:
        v = knee + (ELONG_MAX_KB_PER_MIN - knee) * math.tanh((v - knee) / (ELONG_MAX_KB_PER_MIN - knee))
    return min(max(v, ELONG_MIN_KB_PER_MIN), ELONG_MAX_KB_PER_MIN)


def make_calibration(tick_seconds: float, trajectory_length=None, warmup_steps=None) -> dict:
    """Compute all tick-dependent defaults from biological rates/durations."""
    if tick_seconds <= 0:
        raise ValueError("--tick-seconds must be positive")
    # sites/tick = (kb/s)*(1000 bp/kb)/(bp/site)*(s/tick). At 1kb/8s and 250bp/2s this is the same
    # 0.424 sites/tick (3.18 kb/min), but the 1000/BP_PER_SITE factor makes it resolution-general.
    # Size the (global) stride for the FASTEST per-gene speed (ELONG_MAX) so every gene's per-gene
    # elongation_step_prob stays <= 1; the median anchor sets the default step_prob (per-gene draws in
    # _make_gene override it). At 250 bp/2 s both give stride 1, so this does not change production configs.
    rnapii_stride = max(1, int(math.ceil(_kbmin_to_sites_per_tick(ELONG_MAX_KB_PER_MIN, tick_seconds) - 1e-12)))
    target_sites_per_tick = RNAP_SPEED_KB_PER_SECOND * 1000.0 / BP_PER_SITE * tick_seconds
    elongation_step_prob = min(1.0, target_sites_per_tick / rnapii_stride)
    lifetime = max(1, int(round(COHESIN_LIFETIME_SECONDS / tick_seconds)))
    traj = (
        max(1, int(round(REFERENCE_TRAJECTORY_LENGTH * REFERENCE_TICK_SECONDS / tick_seconds)))
        if trajectory_length is None else int(trajectory_length)
    )
    # The polymer stage requires the trajectory to be an exact multiple of
    # restart_every_blocks (it errors out otherwise). Snap to the nearest
    # multiple so generated configs are always runnable; never below one block.
    traj = max(RESTART_EVERY_BLOCKS, int(round(traj / RESTART_EVERY_BLOCKS)) * RESTART_EVERY_BLOCKS)
    warm = (
        max(0, int(round(REFERENCE_WARMUP_STEPS * REFERENCE_TICK_SECONDS / tick_seconds)))
        if warmup_steps is None else int(warmup_steps)
    )
    md_steps = max(1, int(round(REFERENCE_MD_STEPS_PER_BLOCK * tick_seconds / REFERENCE_TICK_SECONDS)))
    return {
        "tick_seconds": float(tick_seconds),
        "lifetime": lifetime,
        "lifetime_ctcf": CTCF_LIFETIME_BOOST * lifetime,
        "trajectory_length": traj,
        "restart_every_blocks": RESTART_EVERY_BLOCKS,
        "warmup_steps": warm,
        "md_steps_per_block": md_steps,
        # Decelerated 3' termination crawl: TERM_SPEED sites/s -> per-tick step prob (stride 1;
        # term speed is slow enough that term_speed*tick < 1 for any realistic tick).
        "rnapii_termination_step_prob": min(1.0, TERM_SPEED_KB_PER_SECOND * 1000.0 / BP_PER_SITE * tick_seconds),
        "rnapii_stride": rnapii_stride,
        "elongation_step_prob": elongation_step_prob,
        "rnap_load_prob": _rate_to_prob(RNAP_LOAD_RATE, tick_seconds),
        "load_prob_max": _rate_to_prob(LOAD_RATE_MAX, tick_seconds),
        "rnap_termination_prob": _rate_to_prob(RNAP_UNBIND_RATE, tick_seconds),
        "rnap_pause_term_prob": _rate_to_prob(RNAP_PAUSE_TERM_RATE, tick_seconds),
        "rnap_block_prob": math.exp(-tick_seconds / RNAP_BYPASS_SECONDS),
        "rnap_term_block_prob": math.exp(-tick_seconds / RNAP_TERM_BYPASS_SECONDS),
        "rnap_pre_initiation_block_prob": math.exp(-tick_seconds / RNAP_PRE_INITIATION_BYPASS_SECONDS),
        "class_ranges": {
            cls: {
                "init": _scale_ref_range(spec["init"], tick_seconds),
                "prel": _scale_ref_range(spec["prel"], tick_seconds),
            }
            for cls, spec in CLASSES.items()
        },
    }


def _density_field(rng):
    """Smooth gene/promoter-density profile along the chain, normalized to 0..1.

    Built from a handful of random anchors (interpolated + boxcar-smoothed) so
    the chain has coherent dense and sparse stretches rather than per-site noise.
    """
    n = max(5, int(round(CHAIN * BP_PER_SITE / 1e6)))   # ~1 density anchor per Mb (resolution-independent)
    anchors = rng.random(n)
    field = np.interp(np.arange(CHAIN), np.linspace(0, CHAIN - 1, n), anchors)
    w = max(1, CHAIN // (n * 4))                          # smoothing window in sites scales with CHAIN
    field = np.convolve(field, np.ones(2 * w + 1) / (2 * w + 1), mode="same")
    return (field - field.min()) / (field.max() - field.min() + 1e-9)


def gen_tads(rng, s_small_kb=SHORT_SPACING, s_large_kb=LONG_SPACING):
    """Density-driven TADs: a gene-density field sets local boundary spacing, so dense regions emerge as
    small TADs and sparse as large (size is a consequence of density). Returns ``(bounds, dens)`` with
    ``dens[i]`` the mean density (0..1) of interval i. ``s_small_kb``/``s_large_kb`` are the dense/sparse
    spacing endpoints (kb -> sites); realized medians are compressed vs these -- verify the printed sizes."""
    field = _density_field(rng)
    s_small = kb2sites(s_small_kb)          # dense/sparse spacing endpoints in SITES
    s_large = kb2sites(s_large_kb)
    lo_c = min(kb2sites(300), 0.5 * s_small)    # min TAD clip (sites), <= half the dense median
    hi_c = float(min(kb2sites(3000), max(s_large * 1.9, CHAIN / 3)))   # max TAD clip (sites)
    bounds, x = [], 0.0
    while x < CHAIN - lo_c:
        d = float(field[int(min(x, CHAIN - 1))])
        med = s_large * (1.0 - d) + s_small * d        # denser -> shorter step
        step = int(np.clip(rng.lognormal(np.log(med), 0.35), lo_c, hi_c))
        x += step
        if x < CHAIN - lo_c:
            bounds.append(int(x))
    intervals = list(zip([0, *bounds], [*bounds, CHAIN]))
    dens = [float(field[lo:hi].mean()) for lo, hi in intervals]
    return bounds, dens


def gene_length(rng, cls=""):
    """Genomic span (TSS->TES) in SITES: lognormal median 24 kb x CLASS_GENE_LEN_MULT, sigma 1.3
    (model mean ~56 kb, tail >100 kb; cf. human median ~26 kb, Piovesan 2019)."""
    med = 24.0 * CLASS_GENE_LEN_MULT.get(cls, 1.0) * 1000.0 / BP_PER_SITE   # median in SITES
    return int(np.clip(rng.lognormal(mean=np.log(med), sigma=1.3), kb2sites(3), kb2sites(800)))


def term_window(rng, cls=""):
    """3' termination-window length in SITES downstream of the TES: lognormal median ~3.3 kb (Schwalb 2016
    TT-seq) x CLASS_TERM_WINDOW_MULT, model tail to >10 kb; clipped to [1, 20] kb."""
    med = TERM_WINDOW_KB_MEDIAN * CLASS_TERM_WINDOW_MULT.get(cls, 1.0) * 1000.0 / BP_PER_SITE
    return int(np.clip(rng.lognormal(mean=np.log(med), sigma=TERM_WINDOW_SIGMA),
                       kb2sites(1), kb2sites(20)))


def _make_gene(rng, tss, tes, cls, calibration):
    """Assemble one gene dict for class ``cls`` (fields match the YAML schema)."""
    C = CLASSES[cls]
    ranges = calibration["class_ranges"][cls]
    # Draw per-gene randoms in a FIXED order (init, pause-release, elongation base) so the RNG stream --
    # and gene composition -- is identical with or without the activity coupling. The elongation base is
    # stashed in _elong_kbmin; the post-pass applies the activity tilt + clamp. (Reordering reshuffles the
    # stream and changes which genes end up silent.)
    init_p = rng.uniform(*ranges["init"])
    prel_p = rng.uniform(*ranges["prel"])
    elong_kbmin = float(rng.lognormal(mean=np.log(RNAP_SPEED_KB_PER_SECOND * 60.0), sigma=ELONG_LOGNORM_SIGMA))
    g = dict(tss=int(tss), tes=int(tes),
             load_prob=_round_prob(calibration["rnap_load_prob"], 4),
             requires_enhancer=C["req"], load_requires_enhancer=C["req"],
             initiation_prob=_round_prob(init_p, 4),
             pause_release_prob=_round_prob(prel_p, 4),
             # GENE-VARIABLE elongation: no-tilt fallback (used only if the post-pass is skipped, which
             # overrides it with the activity-coupled soft-clamped speed). stride * step_prob keeps the speed.
             elongation_step_prob=round(_elong_step_prob(
                 _soft_clamp_elong(elong_kbmin),
                 calibration["rnapii_stride"], calibration["tick_seconds"]), 6),
             # pause_offset=1: paused Pol sits 1 site downstream of the TSS (~the real +30-60 bp pause at
             # 250 bp), freeing the TSS for the next loader (offset 0 would block re-loading for the whole pause).
             pause_offset=1,
             # Class-specific 3' termination dwell (CLASS_UNBIND_RATE; falls back to RNAP_UNBIND_RATE).
             termination_prob=_round_prob(
                 _rate_to_prob(CLASS_UNBIND_RATE.get(cls, RNAP_UNBIND_RATE),
                               calibration["tick_seconds"]), 4),
             # 3' termination window (sites downstream of TES) the Pol walks before release (vacates the TES fast).
             termination_window=term_window(rng, cls))
    g["_cls"] = cls   # transient class tag for the activity post-pass (not emitted; see fmt_gene)
    g["_elong_kbmin"] = elong_kbmin   # transient base elongation speed (kb/min) for the post-pass tilt
    return g, C


def _add_enhancers(rng, g, C, lo, hi, strand):
    """Attach E-P enhancers (the cohesin loading sites, ``target_enhancers``) in the TAD interior, >= M
    sites from the flank (overshoots scattered uniformly, not clamped to the boundary). Enhancer side is
    50/50 up/downstream of the TSS, independent of strand (distal enhancers are orientation-independent),
    so ``strand`` is unused here."""
    if not C["req"]:
        return
    tss = g["tss"]
    M = kb2sites(10)                            # keep loaders >= 10 kb (in sites) from the flank
    lo_i, hi_i = lo + M, hi - M
    edge = kb2sites(5)                          # 5 kb chromosome-end margin, in sites
    ne = max(1, int(rng.integers(C["nenh"][0], C["nenh"][1] + 1)))
    enh = []
    med_dist = 30.0 * 1000.0 / BP_PER_SITE      # median E-P distance ~30 kb, in SITES
    for _e in range(ne):
        dist = int(np.clip(rng.lognormal(np.log(med_dist), 1.0),
                           kb2sites(5), kb2sites(800)))  # model median ~30 kb (cf. measured ~24 kb, Gasperini 2019)
        sign = 1 if rng.random() < 0.5 else -1     # up/downstream with equal prob
        e = tss + sign * dist
        if rng.random() < 0.10:               # ~10% cross a TAD boundary (modeling assumption)
            e = int(np.clip(e, edge, CHAIN - edge))
        elif hi_i <= lo_i:                     # tiny TAD: centre
            e = (lo + hi) // 2
        elif not (lo_i <= e <= hi_i):          # overshoot -> scatter into interior, not the flank
            e = int(rng.integers(lo_i, hi_i))
        enh.append(int(e))
    g["enhancers"] = sorted(set(enh)); g["enhancer_logic"] = "additive"


def gen_genes(rng, bounds, dens_per, txn_per, calibration):
    intervals = list(zip([0, *bounds], [*bounds, CHAIN]))
    genes = []
    EDGE = kb2sites(3)                          # minimal end-margin (~3 kb), in sites
    min_gap = kb2sites(12)                      # min gene-to-gene spacing (~12 kb), in sites
    for (lo, hi), d, t in zip(intervals, dens_per, txn_per):
        span = hi - lo
        if span < kb2sites(40):
            continue
        # DENSITY sets gene count; TRANSCRIPTION (t) sets the expressed-class mix.
        # span_mb = span_sites * bp_per_site / 1e6.
        gdens = DENS_POOR + d * (DENS_ACTIVE - DENS_POOR)
        n = int(np.clip(round(gdens * span * BP_PER_SITE / 1e6 * rng.lognormal(0, 0.3)), 0, span // min_gap))
        if n == 0:
            continue
        fr = (1.0 - t) * FRAC_POOR + t * FRAC_ACTIVE
        fr = fr / fr.sum()
        occ = np.zeros(span, bool); occ[:EDGE] = True; occ[-EDGE:] = True
        remaining = n

        # --- boundary-flank promoter: TAD boundaries are MODESTLY hk-enriched (~1.5-2.6x; Dixon 2012),
        # not saturated -- so at most ONE flank promoter per TAD, probabilistically (FLANK_PROB), over a
        # wide window (3-25 kb), only ~70% housekeeping.
        FLANK_PROB = 0.5
        avail = [s for s in ("L", "R") if (lo > 0 if s == "L" else hi < CHAIN)]
        if avail and n > 2 and rng.random() < FLANK_PROB:
            side = avail[int(rng.integers(len(avail)))]
            off = int(rng.integers(kb2sites(3), kb2sites(26)))   # 3-25 kb into the flank, in sites
            # Flank-promoter class first (boundaries are hk-ENRICHED, not exclusive), so the gene
            # length can be class-specific (housekeeping compact).
            if rng.random() < 0.70:
                cls = "hk_high" if rng.random() < (0.15 + 0.5 * t) else "hk_const"
            else:
                cls = ORDER[rng.choice(len(ORDER), p=fr)]
            L = min(gene_length(rng, cls), max(3, span // 4))
            if side == "L":
                s = off
                tss_pos, tes_pos, strand = lo + s, lo + s + L, True
            else:
                s = span - off - L
                tss_pos, tes_pos, strand = hi - off, hi - off - L, False
            if EDGE <= s and s + L <= span - EDGE and not occ[s:s + L].any():
                occ[max(0, s - kb2sites(5)):s + L + kb2sites(5)] = True
                g, C = _make_gene(rng, tss_pos, tes_pos, cls, calibration)
                _add_enhancers(rng, g, C, lo, hi, strand)
                genes.append(g); remaining -= 1

        # --- interior genes (classes drawn by TAD activity), largest first so
        # long genes get space (avoids short-gene packing bias) ---
        draws = []
        for _ in range(remaining):
            cls = ORDER[rng.choice(len(ORDER), p=fr)]
            draws.append((cls, min(gene_length(rng, cls), span - 2 * EDGE - 4)))
        draws.sort(key=lambda d: -d[1])
        for cls, L in draws:
            if L < 3:
                continue
            placed = False
            for _t in range(40):
                hi_s = span - L - EDGE
                if hi_s <= EDGE:
                    break
                s = int(rng.integers(EDGE, hi_s))
                if not occ[s:s + L].any():
                    occ[max(0, s - kb2sites(5)):s + L + kb2sites(5)] = True; placed = True; break
            if not placed:
                continue
            strand = rng.random() < 0.5
            a, b = lo + s, lo + s + L
            tss_pos, tes_pos = (a, b) if strand else (b, a)
            g, C = _make_gene(rng, tss_pos, tes_pos, cls, calibration)
            _add_enhancers(rng, g, C, lo, hi, strand)
            genes.append(g)
    return genes, intervals


GENE_KEYS = ["tss", "tes", "load_prob", "requires_enhancer", "load_requires_enhancer",
             "enhancers", "enhancer_logic", "initiation_prob", "pause_release_prob",
             "elongation_step_prob", "pause_offset", "termination_prob", "termination_window",
             "pause_term_prob", "gene_class"]


def fmt_gene(g):
    parts = []
    for k in GENE_KEYS:
        if k not in g:
            continue
        v = g[k]
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, list):
            v = "[" + ", ".join(str(x) for x in v) + "]"
        parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


DEFAULT_BSTR = 0.15  # default_boundary_strength emitted below (near the BSTR_RANGE center); ends forced to 1.0


def make_tad_records(bounds, bstrength, chain, gap_frac=0.0):
    """Per-TAD oriented-boundary records from shared boundaries + per-boundary
    strengths.

    TAD ``i`` takes its left_strength from the boundary on its left
    (``bstrength[i-1]``) and its right_strength from the boundary on its right
    (``bstrength[i]``); the two chromosome-end outer sides use the default (the
    topology forces them to hard 1.0 walls regardless). This reproduces the
    legacy ``tad_positions`` + ``boundary_strength`` ``ctcfCapture`` exactly.

    With ``gap_frac > 0`` each interior TAD sheds ``gap_frac`` of its span on the
    right, opening an anchor-free spacer before the next TAD. ``gap_frac == 0``
    (default) keeps the anchors byte-identical to the legacy abutting layout.
    """
    edges = [0, *list(bounds), chain]
    n = len(edges) - 1
    records = []
    for i in range(n):
        left = edges[i]
        right = edges[i + 1] - 1
        ls = float(bstrength[i - 1]) if i > 0 else DEFAULT_BSTR
        rs = float(bstrength[i]) if i < len(bounds) else DEFAULT_BSTR
        if gap_frac > 0.0 and i < n - 1:
            gw = int(round(gap_frac * (right - left)))
            if 1 <= gw < right - left:
                right -= gw
        records.append((left, right, ls, rs))
    return records


def build(txn_on, bounds, bstrength, genes, calibration, num_chains=4, separation=240,
          lesions=None, gap_frac=0.0):
    # max_rnapii is PER CHAIN and sets the recording-buffer width (total = max_rnapii * num_chains; do NOT
    # scale by num_chains here). lef.py records only rnapiis[:rnapii_cap] and SILENTLY drops the surplus,
    # which biases genes to zero. Size it to the HARD physical ceiling so it can never truncate: with steric
    # exclusion (<=1 Pol/site) and non-overlapping genes, the max simultaneous Pol per chain is the total
    # (gene-body + 3' termination-window) length in sites. The buffer is -1-padded + lzf-compressed, so the
    # on-disk cost tracks the real Pol count, not the cap.
    max_rnapii = (max(64, sum(abs(g["tes"] - g["tss"]) + 1 + int(g.get("termination_window", 0))
                              for g in genes)) if txn_on else 0)
    lifetime = calibration["lifetime"]
    life_ctcf = calibration["lifetime_ctcf"]
    life_rnapii_stalled = lifetime
    rl = ("ner_rnapii_3d.plugins.rnapii:load_rnapii" if txn_on else "null")
    rt = ("ner_rnapii_3d.plugins.rnapii:stateful_translocate_rnapii" if txn_on else "null")
    records = make_tad_records(bounds, bstrength, CHAIN, gap_frac)
    tads_block = "\n".join(
        f"      - {{left: {l}, right: {r}, left_strength: {ls:.2f}, right_strength: {rs:.2f}}}"
        for l, r, ls, rs in records
    )
    gene_lines = "\n".join(f"      - {fmt_gene(g)}" for g in genes)
    # Lesion block: when ``lesions`` is given (config4) emit the full UV-damage
    # knob set + enable the lesion plugin; otherwise lesions are off. (We always
    # emit ``lesion_spacing`` -- never the retired ``lesion_prob`` -- so the
    # topology plugin accepts the kwargs.)
    if lesions:
        ls = lesions
        lesion_kwargs_block = "\n".join([
            f"    lesion_spacing: {ls['spacing']}              # steady-state lesions = num_sites // spacing",
            f"    lesion_type_a_prob: {ls['type_a_prob']}      # P(Type A | in a gene body); off-gene always Type B",
            f"    lesion_prerecognition_ticks: {ls['prerecognition_ticks']}   # ~{ls['prerecognition_seconds']:g}s mean at an average-sized TAD",
            f"    lesion_repair_ticks: {ls['repair_ticks']}            # ~{ls['repair_seconds']:g}s mean at an average-sized TAD",
            f"    lesion_block_prob: {ls['block_prob']}      # per-tick prob a stalling lesion blocks a cohesin leg",
            f"    lesion_tad_size_exponent: {ls['tad_size_exponent']}   # L_TAD**(-alpha): shorter TADs carry MORE lesions",
            f"    lesion_tad_repair_exponent: {ls['tad_repair_exponent']} # (L_mean/L_TAD)**beta: shorter TADs recognise/repair FASTER",
            f"    lesion_type_b_enabled: {str(ls['type_b_enabled']).lower()}  # false -> only Type-A lesions kept (Type-A count unchanged, no backfill)",
        ])
        lesion_plugin_line = (
            "\n    lesion:      ner_rnapii_3d.plugins.lesions:update_lesions"
        )
    else:
        lesion_kwargs_block = "    lesion_spacing: 0            # lesions off (config1/2/3)"
        lesion_plugin_line = ""
    return f"""lef:
  bp_per_site: {BP_PER_SITE}   # bp represented by one lattice site (== one 3D monomer)
  chain_length: {CHAIN}
  num_chains: {num_chains}
  separation: {separation}
  lifetime: {lifetime}
  lifetime_stalled: {lifetime}
  lifetime_ctcf: {life_ctcf} # Cohesin lifetime when its captured by a CTCF site (boosted from the base lifetime)
  tick_seconds: {calibration["tick_seconds"]:g}
  warmup_steps: {calibration["warmup_steps"]}
  trajectory_length: {calibration["trajectory_length"]}
  chunk_size: 50
  seed: 42
  max_rnapii: {max_rnapii}

  topology_kwargs:
    tads:
{tads_block}
    default_boundary_strength: {DEFAULT_BSTR:g}
    release_prob: 0.0
    include_chromosome_ends: true
    # RNAP is a permeable moving barrier (Banigan), NOT a fast cohesin-eviction state: keep RNAP-stalled
    # cohesin at the base lifetime (a short eviction lifetime would DEPLETE cohesin at genes, verified 1D).
    lifetime_rnapii_stalled: {life_rnapii_stalled}
    rnapii_stride: {calibration["rnapii_stride"]}
    # Decelerated step prob for a Pol crawling the 3' termination window (Cortazar 2019 ~0.7-0.9 kb/min);
    # vacates the TES quickly so the 3' end stops capping output at 1/dwell.
    rnapii_termination_step_prob: {round(calibration["rnapii_termination_step_prob"], 6)}
    # Banigan moving barrier (per-encounter probs, tick-independent): an elongating RNAP wins head-on and
    # pushes the converging cohesin leg; co-directional (trailing) contact only impedes it.
    rnapii_stall_prob: {RNAP_STALL_PROB}
    rnapii_push_prob: {RNAP_PUSH_PROB}
    rnapii_headon_push_prob: {RNAP_HEADON_PUSH_PROB}
    rnapii_pause_cohesin_restraint: {RNAP_PAUSE_RESTRAINT}
    rnapii_pause_restraint_window: 1
    # Promoter-proximal premature termination competing with productive release (pause_release_prob);
    # productive fraction ~0.1-0.25 (activity-dependent).
    rnapii_pause_term_prob: {_round_prob(calibration["rnap_pause_term_prob"], 4)}
    # Per-tick cohesin bypass of a Pol II barrier (tbypass ~100 s, Banigan); pre-initiation Pol is a
    # LEAKY barrier (~8x faster bypass), paused/elongating/terminating are strong.
    rnapii_pre_initiation_block_prob: {_round_prob(calibration["rnap_pre_initiation_block_prob"], 3)}
    rnapii_paused_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_elongating_block_prob: {_round_prob(calibration["rnap_block_prob"], 3)}
    rnapii_terminating_block_prob: {_round_prob(calibration["rnap_term_block_prob"], 3)}
    # Pause-initiation limit: a paused Pol in the promoter-proximal window suppresses new
    # initiation (Gressel 2019; Bartman 2019). Output is then gated by pause-escape, ~realistic.
    rnapii_promoter_occlusion: true
    ep_contact_tolerance: 1
    replicate_genes_across_chains: true
    targeted_load_prob: 0.0
    loading_window: 1
    target_enhancers: false
    target_tss: false
    weight_loading_by_activity: false
{lesion_kwargs_block}
    genes:
{gene_lines}

  plugins:
    topology:    ner_rnapii_3d.plugins.topology:gene_aware_convergent_tad_topology
    load:        ner_rnapii_3d.plugins.lef_dynamics:load_targeted
    unload_prob: ner_rnapii_3d.plugins.lef_dynamics:unload_prob
    capture:     ner_rnapii_3d.plugins.lef_dynamics:capture
    release:     ner_rnapii_3d.plugins.lef_dynamics:release
    translocate: ner_rnapii_3d.plugins.lef_dynamics:translocate_with_rnapii
    rnapii_load:        {rl}
    rnapii_translocate: {rt}{lesion_plugin_line}
"""


def main():
    global CHAIN, BP_PER_SITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--tick-seconds", type=float, required=True,
                    help="biological seconds represented by one 1D lattice tick")
    ap.add_argument("--bp-per-site", type=int, default=BP_PER_SITE,
                    help="bp represented by one lattice site (== one 3D monomer); finer -> more "
                         "realistic Pol II / cohesin density, more compute (default 250)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chain", type=int, default=None,
                    help="chain length in SITES; default = 10 Mb locus / bp_per_site")
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--suffix", default="real", help="config name suffix: config1_<suffix>.yaml")
    ap.add_argument("--short-spacing", type=float, default=SHORT_SPACING,
                    help="dense-region TAD spacing endpoint (kb); realized dense median is a bit higher")
    ap.add_argument("--long-spacing", type=float, default=LONG_SPACING,
                    help="sparse-region TAD spacing endpoint (kb); realized sparse median is a bit lower")
    ap.add_argument("--separation", type=int, default=240,
                    help="kb per cohesin (converted to sites); 240 kb matches the Gabriele C36 density estimate")
    ap.add_argument("--trajectory-length", type=int, default=None,
                    help="number of recorded LEF trajectory steps; default preserves the 20 s generator's real duration")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="discarded 1D warmup ticks; default preserves the 20 s generator's real duration")
    ap.add_argument("--bstr-mult", type=float, default=2.5,
                    help="config3 = config2 with all boundary strengths X times larger "
                         "(2.5x BSTR_RANGE -> ~0.15-0.45, a strong but not deterministic CTCF perturbation)")
    ap.add_argument("--gap-frac", type=float, default=0.0,
                    help="fraction of each interior TAD's span left anchor-free as an "
                         "inter-TAD gap (0 = abutting TADs, byte-identical to legacy)")
    # config4 (= config3 + UV lesions) knobs
    ap.add_argument("--lesion-spacing", type=int, default=10,
                    help="config4: lesion spacing in kb (converted to sites); steady-state count = num_sites // spacing_sites (0 disables)")
    ap.add_argument("--lesion-type-a-prob", type=float, default=0.25,
                    help="config4: P(Type A | in a gene body); off-gene lesions are always Type B")
    ap.add_argument("--lesion-prerecognition-seconds", type=float, default=LESION_PRERECOGNITION_SECONDS,
                    help="config4: mean pre-recognition duration at an average-sized TAD")
    ap.add_argument("--lesion-repair-seconds", type=float, default=LESION_REPAIR_SECONDS,
                    help="config4: mean repair duration at an average-sized TAD")
    ap.add_argument("--lesion-block-prob", type=float, default=None,
                    help="config4: per-tick prob a stalling lesion blocks a cohesin leg "
                         "(default: tick-aware exp(-tick/LESION_BYPASS_SECONDS), ~0.97 at 8 s/tick)")
    ap.add_argument("--lesion-tad-size-exponent", type=float, default=1.0,
                    help="config4: placement weight L_TAD**(-alpha); larger -> shorter TADs carry more lesions")
    ap.add_argument("--lesion-tad-repair-exponent", type=float, default=0.25,
                    help="config4: rate * (L_mean/L_TAD)**beta; larger -> shorter TADs recognise/repair faster")
    ap.add_argument("--lesion-type-b-enabled", action=argparse.BooleanOptionalAction, default=True,
                    help="config4: include Type-B (GG-NER) lesions. DEFAULT OFF -> only Type-A lesions "
                         "(the Type-A count is unchanged, no backfill). Pass --lesion-type-b-enabled to add Type-B.")

    args = ap.parse_args()
    # Set the lattice resolution FIRST -- kb2sites/bp2sites and CHAIN derivation depend on it.
    BP_PER_SITE = int(args.bp_per_site)
    if BP_PER_SITE <= 0:
        raise ValueError("--bp-per-site must be positive")
    CHAIN = int(args.chain) if args.chain else LOCUS_BP // BP_PER_SITE
    separation_sites = kb2sites(args.separation)          # 240 kb -> sites
    lesion_spacing_sites = kb2sites(args.lesion_spacing) if args.lesion_spacing > 0 else 0
    calibration = make_calibration(
        args.tick_seconds,
        trajectory_length=args.trajectory_length,
        warmup_steps=args.warmup_steps,
    )
    rng = np.random.default_rng(args.seed)
    bounds, dens = gen_tads(rng, args.short_spacing, args.long_spacing)
    # transcription level per TAD: primarily inverse TAD size (longer TADs = less transcription), blended
    # with the density latent + noise (strong but not deterministic). rng.normal drawn once per TAD.
    _iv0 = list(zip([0, *bounds], [*bounds, CHAIN]))
    _sz0 = np.array([hi - lo for lo, hi in _iv0], dtype=float)
    _inv_size = 1.0 - (_sz0 - _sz0.min()) / (_sz0.max() - _sz0.min() + 1e-9)  # 1 = smallest TAD
    txn = np.clip(
        [TAD_SIZE_TXN_WEIGHT * iv + (1.0 - TAD_SIZE_TXN_WEIGHT) * d + rng.normal(0, TXN_NOISE)
         for iv, d in zip(_inv_size, dens)],
        0.0, 1.0,
    )
    # boundary strength blends flank TRANSCRIPTION (active domains insulate more) and flank DENSITY (dense
    # domains pack into small, strongly-bounded TADs), both mapped into BSTR_RANGE via the more-extreme
    # flank. The density term is what makes corr(flankTADsize, bstrength) reliably negative.
    lo_b, hi_b = BSTR_RANGE
    tb = [max(txn[i], txn[i + 1]) for i in range(len(bounds))]
    db = [max(dens[i], dens[i + 1]) for i in range(len(bounds))]
    drive = [BSTR_TXN_WEIGHT * t + (1.0 - BSTR_TXN_WEIGHT) * d for t, d in zip(tb, db)]
    bstrength = np.round(np.clip([lo_b + (hi_b - lo_b) * x + rng.normal(0, BSTR_NOISE) for x in drive], lo_b, hi_b), 2)
    genes, intervals = gen_genes(rng, bounds, dens, txn, calibration)
    # Per-gene loading heterogeneity -> a heavy-tailed (Zipf) ACTIVITY distribution: most genes quiet, a
    # few hotspots, a silent fraction. Drivers: pause-escape rate (pause_release_prob), TAD size
    # ((1/size)^exp), a power-law multiplier, and per-class SILENT_PROB. The expressed genes' GEOMETRIC
    # mean is held at RNAP_LOAD_RATE.
    if genes and LOAD_LOGNORM_SIGMA > 0:
        cls = [g.pop("_cls", "") for g in genes]              # consume the transient class tag
        prel = np.array([g["pause_release_prob"] for g in genes], dtype=float)
        gpos = np.array([min(g["tss"], g["tes"]) for g in genes])
        gsize = np.array([next((hi - lo for lo, hi in intervals if lo <= p < hi), np.nan)
                          for p in gpos], dtype=float)
        gsize[~np.isfinite(gsize)] = np.nanmedian(gsize)
        size_factor = (np.median(gsize) / gsize) ** LOAD_TADSIZE_EXPONENT
        class_mult = np.array([CLASS_LOAD_MULT.get(c, 1.0) for c in cls])
        # POWER-LAW (Zipf) per-gene loading multiplier m ~ m^(-ALPHA) on [1, RANGE], inverse-CDF from one
        # uniform per gene. Most genes draw m~1 (low), a thin tail high.
        a1 = 1.0 - LOAD_POWERLAW_ALPHA
        u = rng.random(len(genes))
        powerlaw = (1.0 + u * (LOAD_POWERLAW_RANGE ** a1 - 1.0)) ** (1.0 / a1)
        raw = prel * size_factor * powerlaw
        silent = np.array([rng.random() < SILENT_PROB.get(c, 0.0) for c in cls])
        target = calibration["rnap_load_prob"]
        expressed = ~silent
        geo = np.exp(np.log(raw[expressed]).mean()) if expressed.any() else raw.mean()
        # Class-blind renorm to the geo-mean target, THEN an ABSOLUTE per-class multiplier so lifting
        # housekeeping does NOT sink cell-type/dev (cell-type stays ~target, housekeeping above it).
        loads = raw / geo * target * class_mult
        # Hard-cap the loading tail at the gene-body density ceiling so genes saturate near the
        # density ceiling instead of piling Pol II up in the body. The 1e-4 LOWER clip is a
        # body-density floor for EXPRESSED genes only; force-silenced genes are set to 0 AFTER the
        # clip so the floor cannot lift them back to ~16 leaked inits over the run. _round_prob(.,5)
        # then emits load_prob==0.0 exactly, the engine's `nb_random()>=0` gate never fires (no
        # load), init_ct==0, and the metrics map that to NaN (auto-dropped from pf/PI). Silencing is
        # deterministic in the precomputed `silent` mask -> no new rng draw, expressed ranks/loads
        # (hence dwell/pf/elong couplings) byte-identical.
        loads = np.clip(loads, 1e-4, calibration["load_prob_max"])
        loads[silent] = 0.0                                   # placeholder; split into hard-off + ultra-low below
        exprm = ~silent                                       # transcribing genes
        # Activity rank computed over EXPRESSED genes ONLY (silent genes don't transcribe), so rank 0.5 == the
        # MEDIAN EXPRESSED gene -- the pf gradient, dwell ramp, elong tilt and pre-init dwell all pivot correctly
        # even when a large fraction (~35-40%) is force-silenced. (If silent genes were ranked too, they'd fill the
        # bottom 40% of ranks and shove every expressed gene above rank 0.5, lifting all their pf -> median pf HIGH.)
        # Silent genes get rank 0; their kinetics are emitted but never used (load_prob 0). Deterministic -> no rng draw.
        rank = np.zeros(len(loads))
        if exprm.any():
            rank[exprm] = (np.argsort(np.argsort(loads[exprm])) + 0.5) / int(exprm.sum())
        ts = calibration["tick_seconds"]
        # Per-gene k_release (from the class prel draw) and per-class k_term (-> class productive_fraction:
        # hk terminate less -> more productive). Both continuous rates /s.
        k_rel = np.array([_ref_prob_to_rate(g["pause_release_prob"], ts) for g in genes])
        k_term = np.array([CLASS_PAUSE_TERM_RATE.get(c, RNAP_PAUSE_TERM_RATE) for c in cls])
        # Full-range, activity-graded productive fraction (pivot at rank 0.5): RAISE k_term mildly for quiet
        # genes (pf down) and LOWER it for active genes (pf up), with a convex extra reduction over the top
        # decile so the hottest genes reach pf ~0.5-0.7. Changes the k_rel/k_term RATIO (unlike the dwell ramp),
        # so it genuinely spreads pf with activity and lets the hottest genes exceed the output plateau.
        # Deterministic in rank -> no rng draw; base=k_rel+k_term is recomputed and the dwell-median bisection
        # re-pins the bulk dwell, so the pausing_index median is preserved.
        if PF_QUIET_GAIN > 0.0 or PF_ACTIVE_GAIN > 0.0:
            _pf_fac = np.where(rank < 0.5,
                               np.exp(PF_QUIET_GAIN * (0.5 - rank) * 2.0),
                               np.exp(-PF_ACTIVE_GAIN * (rank - 0.5) * 2.0))
            _pf_fac = _pf_fac * np.exp(-PF_ACTIVE_TOPBOOST * np.clip((rank - 0.9) / 0.1, 0.0, 1.0))
            k_term = k_term * _pf_fac
        base = k_rel + k_term                                            # un-scaled pause-exit rate /s
        # Activity-dependent single-Pol pause dwell: a log-linear ramp of the activity rank makes dwell
        # anti-correlated with activity (active -> shorter), plus per-gene scatter for a realistic trend.
        # A convex top-tail boost (PAUSE_DWELL_TOPTAIL_BOOST, over rank 0.8->1.0) stops the active genes
        # saturating at the linear ramp's ~3x ceiling, so they keep stretching toward PAUSE_MIN (active-subset
        # pause ~25 s, matching active/cancer genes 10-30 s) and the output tail spreads instead of plateauing.
        # Deterministic in rank -> no new rng draw, composition unperturbed; the median pin holds the global median.
        top_boost = np.exp(PAUSE_DWELL_TOPTAIL_BOOST * np.clip((rank - 0.8) / 0.2, 0.0, 1.0))
        dwell_mult = np.exp(PAUSE_DWELL_ACTIVITY_SPREAD * (rank - 0.5)
                            + PAUSE_DWELL_ACTIVITY_NOISE * rng.standard_normal(len(genes))) * top_boost  # >1 shorter
        # PIN THE MEDIAN, but only on the bulk so the active tail keeps its fast pause escape: apply the
        # median slow-down with weight pin_weight = 1 for rank<=0.5 ramping to 0 at rank=1 (active tail
        # exempt). The bulk median stays at PAUSE_DWELL_MEDIAN_S; hot genes keep high k_release + low PI.
        # Ratio-invariant per gene, so productive_fraction is untouched.
        pin_weight = 1.0 - np.clip((rank - 0.5) / 0.5, 0.0, 1.0)          # 1 below median rank -> 0 at rank 1
        # Exempting the active tail would pull the median below target, so solve for the pin factor f
        # (applied as f**pin_weight) giving realized post-clamp median = PAUSE_DWELL_MEDIAN_S; bisect in log.
        def _median_dwell(fv):
            sc = np.clip(dwell_mult * fv ** pin_weight,
                         1.0 / (PAUSE_MAX_S * base), 1.0 / (PAUSE_MIN_S * base))
            d = 1.0 / (base * sc)
            return float(np.median(d[exprm])) if exprm.any() else float(np.median(d))   # pin the EXPRESSED median
        lo_f, hi_f = 1e-3, 1e3
        for _ in range(50):
            mid = math.sqrt(lo_f * hi_f)
            if _median_dwell(mid) > PAUSE_DWELL_MEDIAN_S:
                lo_f = mid                                               # median too long -> raise f (shorten)
            else:
                hi_f = mid
        f = math.sqrt(lo_f * hi_f)
        scale = np.clip(dwell_mult * f ** pin_weight,
                        1.0 / (PAUSE_MAX_S * base), 1.0 / (PAUSE_MIN_S * base))
        # Positive activity->elongation tilt on the base speed (_elong_kbmin): faster at active genes,
        # log-linear, then clamp to [ELONG_MIN, ELONG_MAX].
        elong_mult = np.exp(ELONG_ACTIVITY_COUPLING * (rank - 0.5))
        # Activity-coupled pre-initiation dwell -> initiation_prob (overrides the _make_gene class draw,
        # kept only to hold the RNG stream stable). Active re-fire fast, quiet slow.
        pre_init_dwell = np.exp(np.log(PRE_INIT_DWELL_SLOW_S) * (1.0 - rank)
                                + np.log(PRE_INIT_DWELL_FAST_S) * rank)
        # Smooth the bimodal emitted-loading distribution (initiation is otherwise class-bimodal). Per-gene
        # log-normal jitter on the EXPRESSED loads only, drawn HERE -- after the silent mask (line ~820) and
        # the dwell_mult rng draw (above) -- so the force-silent composition and the dwell stream stay
        # byte-identical, and rank was computed from the un-jittered loads so dwell/pf/elong couplings are
        # unchanged. This is the last rng draw in the post-pass (nothing downstream consumes the stream).
        if CLASS_LOAD_MULT_LOGSIGMA > 0.0:
            load_jit = np.exp(rng.normal(0.0, CLASS_LOAD_MULT_LOGSIGMA, len(genes)))
            load_jit[silent] = 1.0                               # silent genes don't take the expressed jitter
            loads = np.clip(loads * load_jit, 1e-4, calibration["load_prob_max"])
        # THREE-CLASS not-detected structure: split the force-silenced pool into TRULY-OFF (load 0) and
        # ULTRA-LOW (log-uniform basal leakage over LOAD_ULTRALOW_RANGE -> ~0.001-0.05/h output), instead of one
        # hard spike at 0. These are the LAST rng draws; silent loads feed only the emit step (rank/dwell/pf
        # exclude silent genes via exprm), so nothing upstream is perturbed.
        if silent.any():
            sidx = np.where(silent)[0]
            hard = rng.random(sidx.size) < SILENT_HARD_FRAC
            loads[sidx[hard]] = 0.0                                          # truly off (no meaningful loading)
            n_ulo = int((~hard).sum())
            if n_ulo:
                lo_l, hi_l = LOAD_ULTRALOW_RANGE
                loads[sidx[~hard]] = np.exp(rng.uniform(math.log(lo_l), math.log(hi_l), n_ulo))  # ultra-low leakage
        for g, lp, kr, kt, s, em, pid, c in zip(genes, loads, k_rel, k_term, scale, elong_mult,
                                                pre_init_dwell, cls):
            g["load_prob"] = _round_prob(float(lp), 6)   # 6 dp so ultra-low loads (down to ~2e-6) survive rounding
            g["initiation_prob"] = _round_prob(_rate_to_prob(1.0 / float(pid), ts), 4)
            g["pause_release_prob"] = _round_prob(_rate_to_prob(kr * s, ts), 4)
            g["pause_term_prob"] = _round_prob(_rate_to_prob(kt * s, ts), 4)
            v = _soft_clamp_elong(g.pop("_elong_kbmin") * em)
            g["elongation_step_prob"] = round(_elong_step_prob(v, calibration["rnapii_stride"], ts), 6)
            g["gene_class"] = c            # emit the regulatory class for per-class QC (engine ignores it)
    else:
        for g in genes:
            c = g.pop("_cls", "")
            g.pop("_elong_kbmin", None)    # keep the un-tilted fallback elongation_step_prob from _make_gene
            g["gene_class"] = c
            g["pause_term_prob"] = _round_prob(
                _rate_to_prob(CLASS_PAUSE_TERM_RATE.get(c, RNAP_PAUSE_TERM_RATE),
                              calibration["tick_seconds"]), 4)
    from pathlib import Path
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix
    (od / f"config1_{sfx}.yaml").write_text(build(True, bounds, bstrength, genes, calibration, args.num_chains, separation_sites, gap_frac=args.gap_frac))
    (od / f"config2_{sfx}.yaml").write_text(build(False, bounds, bstrength, genes, calibration, args.num_chains, separation_sites, gap_frac=args.gap_frac))
    # config3 = byte-identical to config2 (txn OFF) except every boundary strength is
    # X times larger (--bstr-mult). Stronger insulation, same domain skeleton/genes.
    bstrength_x = np.round(np.minimum(bstrength * args.bstr_mult, 1.0), 2)
    (od / f"config3_{sfx}.yaml").write_text(build(False, bounds, bstrength_x, genes, calibration, args.num_chains, separation_sites, gap_frac=args.gap_frac))
    # config4 = config3 + UV lesions. Stage durations are biological seconds -> ticks; shorter TADs carry
    # more lesions (size exponent) but repair faster (repair exponent).
    ts = calibration["tick_seconds"]
    # Tick-aware default lesion block prob: hold the physical barrier dwell constant across resolutions
    # (exp(-tick/LESION_BYPASS_SECONDS); ~0.97 at 8 s/tick).
    lesion_block_prob = (args.lesion_block_prob if args.lesion_block_prob is not None
                         else math.exp(-ts / LESION_BYPASS_SECONDS))
    lesion_params = dict(
        spacing=lesion_spacing_sites,
        type_a_prob=args.lesion_type_a_prob,
        prerecognition_seconds=args.lesion_prerecognition_seconds,
        repair_seconds=args.lesion_repair_seconds,
        prerecognition_ticks=max(1, int(round(args.lesion_prerecognition_seconds / ts))),
        repair_ticks=max(1, int(round(args.lesion_repair_seconds / ts))),
        block_prob=lesion_block_prob,
        tad_size_exponent=args.lesion_tad_size_exponent,
        tad_repair_exponent=args.lesion_tad_repair_exponent,
        type_b_enabled=args.lesion_type_b_enabled,
    )
    (od / f"config4_{sfx}.yaml").write_text(build(False, bounds, bstrength_x, genes, calibration, args.num_chains, separation_sites, lesions=lesion_params, gap_frac=args.gap_frac))
    # stats
    lens = [abs(g["tes"] - g["tss"]) for g in genes]
    nenh = [len(g.get("enhancers", [])) for g in genes]
    perTAD = [sum(1 for g in genes if a <= min(g['tss'], g['tes']) < b) for a, b in intervals]
    sizes = np.array([b - a for a, b in intervals], float)
    dA = np.array(dens)
    def corr(x, y): return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    med = np.median(dA)
    flank_size = np.array([min(sizes[i], sizes[i + 1]) for i in range(len(bounds))]) if bounds else np.array([])
    flank_txn = np.array(tb) if bounds else np.array([])
    kbf = BP_PER_SITE / 1000.0   # sites -> kb for display
    print(f"CHAIN={CHAIN} sites ({CHAIN*BP_PER_SITE/1e6:.1f} Mb, bp_per_site={BP_PER_SITE}) "
          f"num_chains={args.num_chains}  TADs: {len(intervals)} "
          f"size mean={sizes.mean()*kbf:.0f}kb min={int(sizes.min()*kbf)} max={int(sizes.max()*kbf)}")
    def _med_kb(sub):   # NaN-safe median (empty subset -> "n/a"), in kb
        return f"{int(np.median(sub) * kbf)}kb" if sub.size else "n/a"
    print(f"  dense(top 50%) median size={_med_kb(sizes[dA>=med])}  "
          f"sparse median size={_med_kb(sizes[dA<med])}")
    print(f"  corr(density,size)={corr(dA,sizes):+.2f}  corr(txn,density)={corr(np.array(txn),dA):+.2f}")
    if len(bounds) > 2:
        print(f"  corr(flank_txn,bstrength)={corr(flank_txn,bstrength):+.2f} (active->stronger)  "
              f"corr(flankTADsize,bstrength)={corr(flank_size,bstrength):+.2f} (smaller->stronger)")
    if bstrength.size:
        print(f"  boundary_strength: min={bstrength.min():.2f} median={np.median(bstrength):.2f} max={bstrength.max():.2f}")
    else:
        print("  boundary_strength: n/a (no interior boundaries)")
    expected_kb_min = (calibration["rnapii_stride"] * calibration["elongation_step_prob"]
                       / calibration["tick_seconds"] * 60.0 * BP_PER_SITE / 1000.0)
    pause_ranges = [v["prel"] for v in calibration["class_ranges"].values()]
    pause_min = min(lo for lo, _hi in pause_ranges)
    pause_max = max(hi for _lo, hi in pause_ranges)
    print("calibration:")
    print(f"  tick_seconds={calibration['tick_seconds']:g}")
    print(f"  lifetime={calibration['lifetime']} ticks ({calibration['lifetime'] * calibration['tick_seconds'] / 60:.1f} min)  "
          f"lifetime_ctcf={calibration['lifetime_ctcf']}")
    print(f"  warmup_steps={calibration['warmup_steps']} ({calibration['warmup_steps'] * calibration['tick_seconds'] / 3600:.2f} h)  "
          f"trajectory_length={calibration['trajectory_length']} ({calibration['trajectory_length'] * calibration['tick_seconds'] / 3600:.2f} h)")
    print(f"  RNAP load_prob={calibration['rnap_load_prob']:.6f}  "
          f"pause_release_prob_range={pause_min:.6f}-{pause_max:.6f}  "
          f"termination_prob={calibration['rnap_termination_prob']:.6f}  "
          f"block_prob={calibration['rnap_block_prob']:.6f}")
    # Realized single-Pol pause dwell = 1/(k_release + k_term) per gene (target median PAUSE_DWELL_MEDIAN_S,
    # anti-correlated with activity by PAUSE_DWELL_ACTIVITY_SPREAD, clamped to [PAUSE_MIN_S, PAUSE_MAX_S]).
    _ts = calibration["tick_seconds"]
    _dw = np.array([1.0 / (_ref_prob_to_rate(g["pause_release_prob"], _ts)
                           + _ref_prob_to_rate(g["pause_term_prob"], _ts))
                    for g in genes if "pause_term_prob" in g])
    if _dw.size:
        print(f"  pause dwell s: min={_dw.min():.0f} p5={np.percentile(_dw,5):.0f} "
              f"median={np.median(_dw):.0f} p95={np.percentile(_dw,95):.0f} max={_dw.max():.0f}  "
              f"(<=20s {100*np.mean(_dw<=20):.0f}%, >=60s {100*np.mean(_dw>=60):.0f}%)")
    # Realized per-gene elongation speed = stride * elongation_step_prob * bp_per_site / tick (kb/min).
    _ev = np.array([calibration["rnapii_stride"] * g["elongation_step_prob"]
                    / _ts * 60.0 * BP_PER_SITE / 1000.0 for g in genes if "elongation_step_prob" in g])
    if _ev.size:
        print(f"  elongation kb/min: min={_ev.min():.1f} p5={np.percentile(_ev,5):.1f} "
              f"median={np.median(_ev):.1f} p95={np.percentile(_ev,95):.1f} max={_ev.max():.1f}")
    print(f"  RNAP stride={calibration['rnapii_stride']}  "
          f"elongation_step_prob={calibration['elongation_step_prob']:.6f}  "
          f"expected_elongation={expected_kb_min:.2f} kb/min")
    print(f"  md_steps_per_block={calibration['md_steps_per_block']}")
    print(f"genes: {len(genes)}  = {len(genes)/(CHAIN*BP_PER_SITE/1e6):.1f}/Mb")
    if lens:
        print(f"gene len kb: median={int(np.median(lens)*kbf)} mean={int(np.mean(lens)*kbf)} "
              f">100kb={100*np.mean(np.array(lens)>kb2sites(100)):.0f}%")
    else:
        print("gene len kb: n/a (no genes at this chain size)")
    print(f"E-P genes: {sum(1 for g in genes if g.get('enhancers'))}  enh/EPgene mean={np.mean([n for n in nenh if n]) if any(nenh) else 0:.1f}")
    print(f"genes/TAD: min={min(perTAD)} median={int(np.median(perTAD))} max={max(perTAD)}")
    if bstrength_x.size:
        print(f"  config3 boundary_strength (x{args.bstr_mult:g}): min={bstrength_x.min():.2f} median={np.median(bstrength_x):.2f} max={bstrength_x.max():.2f}")
    else:
        print(f"  config3 boundary_strength (x{args.bstr_mult:g}): n/a (no interior boundaries at this chain size)")
    n_sites = CHAIN * args.num_chains
    target = (n_sites // lesion_spacing_sites) if lesion_spacing_sites > 0 else 0
    print(f"config4 lesions: spacing={lesion_spacing_sites} sites ({args.lesion_spacing} kb) -> ~{target} lesions held "
          f"({100.0 * target / n_sites:.1f}% of {n_sites} sites); "
          f"pre-recognition={lesion_params['prerecognition_ticks']} ticks (~{args.lesion_prerecognition_seconds:g}s), "
          f"repair={lesion_params['repair_ticks']} ticks (~{args.lesion_repair_seconds:g}s); "
          f"size_exp(alpha)={args.lesion_tad_size_exponent:g}, repair_exp(beta)={args.lesion_tad_repair_exponent:g}")
    print(f"wrote {od}/config1_{sfx}.yaml , config2_{sfx}.yaml , config3_{sfx}.yaml , config4_{sfx}.yaml")


if __name__ == "__main__":
    main()
