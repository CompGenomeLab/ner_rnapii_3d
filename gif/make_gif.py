#!/usr/bin/env python
"""Render a presentation GIF from a ner_rnapii_3d demo trajectory.

Reads the demo config (for enhancer / TAD / gene layout) and the LEFPositions.h5
trajectory, and draws a genome-browser-style animation with three feature tracks
that together demonstrate every mechanic of the engine:

  Track A  cohesin loop extrusion  -- arcs grow, get captured at oriented CTCF
                                      barriers, bypass a leaky boundary, turn over
  Track B  genes / RNA Pol II      -- gene classes, enhancer-promoter contact
                                      (loop-gated), single-molecule Pol II traffic
                                      (pre-init / pause / elongation / termination
                                      / stall) with comet trails + active/poised/
                                      silent state badges
  Track C  UV damage / NER         -- Type-A (TC-NER) & Type-B (GG-NER) lesions
                                      cycling pre-recognition -> repair; a stalled
                                      Pol II over a Type-A lesion is flagged TC-NER

Usage
-----
  python make_gif.py --frame 120 --png preview.png   # single-frame preview (fast)
  python make_gif.py --out demo.gif --frames 300 --fps 15
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, FancyBboxPatch
from PIL import Image

# ---------------------------------------------------------------------------
# Palette (dark scientific theme)
# ---------------------------------------------------------------------------
BG       = "#0e0f14"       # figure / axes surface
INK      = "#f4f5f2"       # primary text (titles, stat numbers)
INK2     = "#eceded"       # secondary text (near-white on the black surface)
CAPTION  = "#dddee4"       # sub-captions + legend labels
MUTED    = "#c4c6ce"       # axis ticks, TAD labels, meta
SILENT   = "#b3b6bf"       # 'silent' gene-state word (dimmest state, still legible)
DIM      = "#6b6e7b"       # faint glyph accent (inactive enhancer edge)
BACKBONE = "#3c3f4b"       # the DNA fibre
GRID     = "#22242e"

COHESIN     = "#2f9aa8"    # extruding cohesin loop (dim, thin, round legs)
COHESIN_CAP = "#7ff0f7"    # CTCF-captured (stable) loop (bright, thick, square legs)
CTCF        = "#e9e9ec"    # oriented barrier arrowhead

# gene classes (categorical identity)
CLS = {"hk_const": "#4f8ef0", "celltype": "#e879b0", "dev": "#a68bf2"}
CLS_LABEL = {"hk_const": "housekeeping", "celltype": "cell-type", "dev": "developmental"}

# RNA Pol II states (widened hue gaps: yellow -> orange -> red for the warm trio)
RN = {0: "#aab0bd", 1: "#ffcf2e", 2: "#2ec96b", 3: "#ff8a34", 4: "#f5455f"}
RN_NAME = {0: "pre-init", 1: "paused", 2: "elongating", 3: "terminating", 4: "stalled"}

STATE_COL = {"active": "#2ec96b", "poised": "#ffcf2e", "silent": SILENT}

TEAL = "#22d3c4"           # enhancer-promoter COMMUNICATION (distinct from activity-green)
LES_A = "#ff4d4d"          # Type A = transcription-coupled (TC-NER)
LES_B = "#b07bff"          # Type B = global-genome (GG-NER)

STATE_PRE, STATE_PAUSED, STATE_ELONG, STATE_TERM, STATE_STALL = 0, 1, 2, 3, 4
LTYPE_A, LTYPE_B = 0, 1
LSTATE_PRE, LSTATE_REPAIR = 0, 1


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load(config_path, h5_path):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)["lef"]
    tk = cfg["topology_kwargs"]
    genes_cfg = tk["genes"]

    f = h5py.File(h5_path, "r")
    d = dict(
        pos=f["positions"][:], rp=f["rnapii_positions"][:], rs=f["rnapii_states"][:],
        rid=f["rnapii_ids"][:], les=f["lesions"][:], ltype=f["lesion_types"][:],
        lstate=f["lesion_states"][:], genes=f["genes"][:], anchors=f["ctcf_anchors"][:],
        N=int(f.attrs["N"]), chain=int(f.attrs["chain_length"]),
        bp=int(f.attrs["bp_per_site"]), tick_s=float(f.attrs["tick_seconds"]),
    )
    f.close()

    gtab = []
    for g in d["genes"]:
        gid = int(g["gid"] if "gid" in g.dtype.names else g[0])
        spec = genes_cfg[gid]
        lo, hi = sorted((int(g["tss"]), int(g["tes"])))
        gtab.append(dict(gid=gid, tss=int(g["tss"]), tes=int(g["tes"]),
                         direction=int(g["direction"]), cls=g["gene_class"].decode(),
                         enhancers=[int(e) for e in spec.get("enhancers", [])], lo=lo, hi=hi))
    d["gtab"] = gtab
    d["tol"] = int(tk.get("ep_contact_tolerance", 2))
    d["tads"] = [(int(t["left"]), int(t["right"])) for t in tk["tads"]]
    d["anchor_set"] = set(int(a["abs_position"]) for a in d["anchors"])
    return d


def gene_of(d, site):
    for g in d["gtab"]:
        if g["lo"] <= site <= g["hi"]:
            return g
    return None


# ---------------------------------------------------------------------------
# Per-frame derived quantities
# ---------------------------------------------------------------------------
def derive(d, frames):
    pos, rp, rs, rid = d["pos"], d["rp"], d["rs"], d["rid"]
    gtab, tol = d["gtab"], d["tol"]
    T = pos.shape[0]

    # transcripts completed = cumulative distinct Pol uids that reached TERMINATING
    seen, comp = set(), np.zeros(T, dtype=int)
    for t in range(T):
        ids, st = rid[t], rs[t]
        m = (st == STATE_TERM) & (ids >= 0)
        if m.any():
            seen.update(int(u) for u in ids[m])
        comp[t] = len(seen)

    # per-sampled-frame: gene state + E-P contacts, and uid->x for comet trails
    per, uidpos = {}, {}
    for t in frames:
        coh = pos[t]
        lo = np.minimum(coh[:, 0], coh[:, 1]); hi = np.maximum(coh[:, 0], coh[:, 1])
        gstate, contacts = {}, {}
        gid_row, st_row = rp[t, :, 1], rs[t]
        for g in gtab:
            on = (gid_row == g["gid"]) & (st_row >= 0)
            if np.any(on & (st_row == STATE_ELONG)):
                gstate[g["gid"]] = "active"
            elif np.any(on & ((st_row == STATE_PAUSED) | (st_row == STATE_PRE))):
                gstate[g["gid"]] = "poised"
            else:
                gstate[g["gid"]] = "silent"
            ec = []
            for e in g["enhancers"]:
                a, b = min(g["tss"], e), max(g["tss"], e)
                if np.any((lo <= a + tol) & (hi >= b - tol)):
                    ec.append(e)
            contacts[g["gid"]] = ec
        per[t] = dict(gstate=gstate, contacts=contacts)
        ids, xs = rid[t], rp[t, :, 0]
        uidpos[t] = {int(u): int(x) for u, x in zip(ids, xs) if u >= 0}
    return comp, per, uidpos


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
YMAX = 100
BACK_Y   = 82.0      # DNA backbone (track A)
ARC_MAXH = 16.5
LBL_Y    = 66.5      # gene class label row
GENE_Y   = 58.0      # gene body centre (track B)
GENE_H   = 4.6
ENH_Y    = 48.5      # enhancer row
LES_Y    = 31.0      # lesion line (track C)
TAD_TOP  = 90.0
TAD_BOT  = 26.5


def arc_height(span, chain):
    return min(15.0, 3.0 + ARC_MAXH * np.sqrt(min(span, chain) / chain))


def _figy(main_ax, data_y):
    p = main_ax.get_position(); y0, y1 = main_ax.get_ylim()
    return p.y0 + (data_y - y0) / (y1 - y0) * p.height


def draw_static(ax, d):
    chain = d["chain"]
    ax.set_xlim(-0.012 * chain, chain * 1.008)
    ax.set_ylim(14, YMAX)
    ax.axis("off")

    for i, (l, r) in enumerate(d["tads"]):
        ax.add_patch(patches.Rectangle(
            (l, TAD_BOT), r - l + 1, TAD_TOP - TAD_BOT,
            facecolor=("#ffffff" if i % 2 else "#8fa0c0"),
            alpha=(0.028 if i % 2 else 0.045), edgecolor="none", zorder=0))
        ax.text((l + r) / 2, 98.0, f"TAD {i+1}", ha="center", va="center",
                color=MUTED, fontsize=8.0, zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor="none", alpha=0.65))
        if i > 0:
            ax.plot([l - 0.5, l - 0.5], [TAD_BOT, TAD_TOP], color=GRID, lw=0.8, zorder=0.5)

    ax.plot([0, chain], [BACK_Y, BACK_Y], color=BACKBONE, lw=3.0, solid_capstyle="round", zorder=2)
    ax.plot([0, chain], [LES_Y, LES_Y], color=BACKBONE, lw=2.0, solid_capstyle="round", zorder=2)

    # oriented CTCF barriers: arrowhead points INWARD (into its own TAD); length ~ strength.
    # small inward offset separates the back-to-back anchors that meet at a boundary.
    for a in d["anchors"]:
        x = int(a["abs_position"]); edge = int(a["edge"]); s = float(a["strength"])
        d_in = 1 if edge == 0 else -1
        w = 5 + 6 * s
        base = x + d_in * 2.0; tip = base + d_in * w
        tri = [(tip, BACK_Y), (base, BACK_Y + 3.3), (base, BACK_Y - 3.3)]
        ax.add_patch(patches.Polygon(tri, closed=True, facecolor=CTCF, edgecolor="none",
                     alpha=0.42 + 0.5 * s, zorder=3))

    # genes: promoter (bent TSS arrow) + body + directional head, coloured by class
    for g in d["gtab"]:
        c = CLS[g["cls"]]; a, b = g["lo"], g["hi"]
        ax.add_patch(FancyBboxPatch(
            (a, GENE_Y - GENE_H / 2), b - a, GENE_H,
            boxstyle="round,pad=0,rounding_size=1.6", mutation_aspect=0.5,
            facecolor=c, edgecolor="none", alpha=0.45, zorder=3))
        hx, dirn = g["tes"], g["direction"]
        head = [(hx, GENE_Y), (hx - dirn * 7, GENE_Y + GENE_H / 2 + 1.3),
                (hx - dirn * 7, GENE_Y - GENE_H / 2 - 1.3)]
        ax.add_patch(patches.Polygon(head, closed=True, facecolor=c, edgecolor="none",
                     alpha=0.85, zorder=3))
        tx = g["tss"]
        ax.plot([tx, tx], [GENE_Y + GENE_H / 2, GENE_Y + GENE_H / 2 + 4.5], color=c, lw=1.6, zorder=4)
        ax.annotate("", xy=(tx + dirn * 9, GENE_Y + GENE_H / 2 + 4.5),
                    xytext=(tx, GENE_Y + GENE_H / 2 + 4.5),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6), zorder=4)


def _stagger_y(xs):
    """Bead-string: give co-located points (sorted) a small vertical stagger."""
    order = np.argsort(xs)
    offs = np.zeros(len(xs)); levels = [0.0, 2.0, -2.0]
    lvl = 0; prev = None
    for k in order:
        if prev is not None and xs[k] - prev < 6:
            lvl = (lvl + 1) % 3
        else:
            lvl = 0
        offs[k] = levels[lvl]; prev = xs[k]
    return offs


def draw_frame(ax, d, t, comp, info, uidpos, frames, fi, dyn):
    for art in dyn:
        art.remove()
    dyn.clear()
    chain = d["chain"]; anchors = d["anchor_set"]
    gstate, contacts = info["gstate"], info["contacts"]

    # ---- active-gene glow (persistent activity cue) -------------------------
    for g in d["gtab"]:
        if gstate[g["gid"]] == "active":
            dyn.append(ax.add_patch(FancyBboxPatch(
                (g["lo"] - 3, GENE_Y - GENE_H / 2 - 2.2), g["hi"] - g["lo"] + 6, GENE_H + 4.4,
                boxstyle="round,pad=0,rounding_size=2.4", mutation_aspect=0.5,
                facecolor=STATE_COL["active"], edgecolor="none", alpha=0.10, zorder=2.5)))

    # ---- enhancers (upstream regulatory elements, outside the gene body) -----
    # idle: faint dotted line to the promoter it regulates; on cohesin-loop-mediated
    # E-P contact: a bright teal arc from the enhancer to the promoter.
    for g in d["gtab"]:
        hot_set = contacts[g["gid"]]; tss_x = g["tss"]
        for e in g["enhancers"]:
            hot = e in hot_set
            if hot:
                dyn.append(ax.scatter([e], [ENH_Y], s=190, marker="D", color=TEAL,
                           alpha=0.22, zorder=4, linewidths=0))
            else:
                dyn.append(ax.plot([e, tss_x], [ENH_Y + 1.6, GENE_Y - GENE_H / 2],
                           color=GRID, lw=0.8, ls=":", alpha=0.7, zorder=3)[0])
            dyn.append(ax.scatter([e], [ENH_Y], s=48, marker="D",
                       facecolor=(TEAL if hot else "#3a3d49"),
                       edgecolor=(TEAL if hot else DIM), linewidths=1.0, zorder=5))
        for e in hot_set:
            mx = (e + tss_x) / 2
            dep = min(4.5, 1.8 + 0.06 * abs(e - tss_x))
            verts = [(e, ENH_Y + 1.8), (mx, ENH_Y - dep), (tss_x, GENE_Y - GENE_H / 2)]
            dyn.append(ax.add_patch(PathPatch(
                MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
                fill=False, edgecolor=TEAL, lw=1.5, alpha=0.85, zorder=4)))

    # ---- gene class labels + activity state ---------------------------------
    for g in d["gtab"]:
        st = gstate[g["gid"]]; mid = (g["tss"] + g["tes"]) / 2
        mx = min(max(mid, 46), chain - 46)
        dyn.append(ax.text(mx, LBL_Y + 2.4, CLS_LABEL[g["cls"]], ha="center", va="center",
                   color=CLS[g["cls"]], fontsize=8.4, fontweight="bold", zorder=6))
        dyn.append(ax.text(mx, LBL_Y - 2.5, st, ha="center", va="center",
                   color=STATE_COL[st], fontsize=7.8, zorder=6))

    # ---- cohesin loops (arcs) + legs + capture glow -------------------------
    for j in range(d["pos"].shape[1]):
        xl, xr = int(d["pos"][t, j, 0]), int(d["pos"][t, j, 1])
        lo, hi = min(xl, xr), max(xl, xr); span = hi - lo
        captured = (lo in anchors) and (hi in anchors)
        h = arc_height(span, chain)
        if captured:  # bright solid arc + glow + square anchor caps (categorical cue)
            dyn.append(ax.scatter([lo, hi], [BACK_Y, BACK_Y], s=150, color=COHESIN_CAP,
                       alpha=0.16, zorder=5, linewidths=0))
            dyn.append(ax.add_patch(patches.Arc(((lo + hi) / 2, BACK_Y), width=max(span, 1),
                       height=2 * h, angle=0, theta1=0, theta2=180, color=COHESIN_CAP,
                       lw=2.6, alpha=0.97, zorder=6.5)))
            dyn.append(ax.scatter([lo, hi], [BACK_Y, BACK_Y], s=42, marker="s",
                       color=COHESIN_CAP, edgecolor=BG, linewidths=0.8, zorder=7))
        else:         # dim thin arc + small round legs = actively extruding
            dyn.append(ax.add_patch(patches.Arc(((lo + hi) / 2, BACK_Y), width=max(span, 1),
                       height=2 * h, angle=0, theta1=0, theta2=180, color=COHESIN,
                       lw=1.4, alpha=0.62, zorder=6)))
            dyn.append(ax.scatter([lo, hi], [BACK_Y, BACK_Y], s=15, color=COHESIN,
                       edgecolor=BG, linewidths=0.5, zorder=7))

    # ---- RNA Pol II: comet trails, bead-string stagger, stalled on top -------
    rp, rs, rid = d["rp"], d["rs"], d["rid"]
    live = rs[t] >= 0
    xs = rp[t, live, 0].astype(float); stt = rs[t, live]; uids = rid[t, live]
    # per-gene vertical stagger so co-located Pols stay distinct
    yoff = np.zeros(len(xs))
    gid_live = rp[t, live, 1]
    for gid in np.unique(gid_live):
        m = gid_live == gid
        yoff[m] = _stagger_y(xs[m])
    ys = GENE_Y + yoff
    # comet trails for moving (elongating) polymerases
    for k in range(len(xs)):
        if stt[k] != STATE_ELONG:
            continue
        u = int(uids[k]); pts = [xs[k]]
        for back in (1, 2, 3):
            if fi - back < 0:
                break
            pv = uidpos[frames[fi - back]].get(u)
            if pv is None:
                break
            pts.append(pv)
        if len(pts) >= 2:
            for s in range(len(pts) - 1):
                dyn.append(ax.plot([pts[s], pts[s + 1]], [ys[k], ys[k]], color=RN[2],
                           lw=2.2, alpha=0.30 * (1 - s / len(pts)), solid_capstyle="round",
                           zorder=7.5)[0])
    moving = stt != STATE_STALL
    if np.any(moving):
        dyn.append(ax.scatter(xs[moving], ys[moving], s=44,
                   c=[RN[int(s)] for s in stt[moving]], edgecolors=BG, linewidths=0.5, zorder=8))
    if np.any(~moving):  # stalled: drawn on top, white ring
        dyn.append(ax.scatter(xs[~moving], ys[~moving], s=58, c=RN[4],
                   edgecolors="#ffffff", linewidths=1.3, zorder=9))

    # ---- lesions (track C): stagger co-located, dark stroke ------------------
    les, ltype, lstate = d["les"], d["ltype"], d["lstate"]
    lm = les[t] >= 0
    lsites = les[t, lm].astype(float); ltyp = ltype[t, lm]; lst = lstate[t, lm]
    lyoff = _stagger_y(lsites) * 1.4
    stalled_x = set(int(v) for v in xs[~moving]) if np.any(~moving) else set()
    for site, typ, stt2, dy in zip(lsites, ltyp, lst, lyoff):
        col = LES_A if typ == LTYPE_A else LES_B
        repair = (stt2 == LSTATE_REPAIR); y = LES_Y + dy
        if repair:
            dyn.append(ax.scatter([site], [y], s=330, marker="*", color=col, alpha=0.22,
                       zorder=5, linewidths=0))
        dyn.append(ax.scatter([site], [y], s=170, marker="*",
                   facecolor=(col if repair else "none"), edgecolor=col, linewidths=1.6,
                   alpha=(1.0 if repair else 0.95), zorder=6))
        # TC-NER coupling: Type-A lesion in a gene body with a stalled Pol just upstream
        if typ == LTYPE_A and gene_of(d, int(site)) is not None and \
                any(abs(int(site) - sx) <= 2 for sx in stalled_x):
            dyn.append(ax.plot([site, site], [y + 2.2, GENE_Y - GENE_H / 2 - 2],
                       color=LES_A, lw=1.0, ls=":", alpha=0.7, zorder=4)[0])
            dyn.append(ax.text(site, (y + GENE_Y) / 2, "TC-NER", ha="center", va="center",
                       color=LES_A, fontsize=6.4, rotation=90, alpha=0.85, zorder=6,
                       bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none",
                                 alpha=0.6)))
    return dyn


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------
def _chip(ax, x, y, color, label, marker="o", hollow=False, size=9):
    if marker == "arc":
        ax.add_patch(patches.Arc((x + 0.010, y + 0.004), 0.024, 0.05, angle=0, theta1=0,
                     theta2=180, color=color, lw=2.0, transform=ax.transAxes))
    elif marker == "tri":
        ax.scatter([x + 0.006], [y], marker=">", s=size * 5, color=color,
                   transform=ax.transAxes, edgecolor="none")
    elif marker == "pair":  # convergent CTCF pair  > <
        ax.scatter([x + 0.003], [y], marker=">", s=size * 5, color=color,
                   transform=ax.transAxes, edgecolor="none")
        ax.scatter([x + 0.013], [y], marker="<", s=size * 5, color=color,
                   transform=ax.transAxes, edgecolor="none")
    elif marker == "star":
        ax.scatter([x + 0.007], [y], marker="*", s=size * 11,
                   facecolor=("none" if hollow else color), edgecolor=color,
                   linewidths=1.3, transform=ax.transAxes)
    else:
        ax.scatter([x + 0.006], [y], marker=marker, s=size * 6,
                   facecolor=("none" if hollow else color), edgecolor=color,
                   linewidths=1.3, transform=ax.transAxes)
    off = 0.020 if marker in ("pair", "star") else 0.015
    ax.text(x + off, y, label, transform=ax.transAxes, color=CAPTION, fontsize=7.9,
            va="center", ha="left")


def setup_chrome(fig, main_ax, foot_ax, d):
    fig.text(0.012, 0.965, "Simulation Engine", color=INK, fontsize=17, fontweight="bold",
             ha="left", va="top")
    fig.text(0.012, 0.925,
             "single-molecule loop extrusion  +  RNA Pol II transcription  +  nucleotide-excision repair",
             color=INK2, fontsize=9.5, ha="left", va="top")
    kb = d["chain"] * d["bp"] / 1000
    fig.text(0.988, 0.958, f"{kb:.0f} kb locus   ·   {d['bp']} bp / site",
             color=MUTED, fontsize=8.5, ha="right", va="top")

    def put(datay, title, sub):
        fy = _figy(main_ax, datay)
        fig.text(0.010, fy + 0.006, title, ha="left", va="bottom", color=INK2,
                 fontsize=8.8, fontweight="bold", linespacing=1.1)
        fig.text(0.010, fy - 0.006, sub, ha="left", va="top", color=CAPTION,
                 fontsize=7.1, linespacing=1.42)
    put(BACK_Y, "LOOP\nEXTRUSION", "loading · extrusion\nCTCF capture\nbypass · turnover")
    put(GENE_Y, "GENES /\nRNA Pol II", "classes · enhancers\nE–P loop gating\npause·elongate·terminate")
    put(LES_Y, "DNA DAMAGE\n& REPAIR", "TC-NER · GG-NER\nstalled Pol II\nrepair barriers")

    chain = d["chain"]
    for kbx in range(0, int(kb) + 1, 25):
        sx = kbx * 1000 / d["bp"]
        main_ax.plot([sx, sx], [TAD_BOT - 1.5, TAD_BOT - 3.0], color=MUTED, lw=0.8)
        main_ax.text(sx, TAD_BOT - 5.0, f"{kbx}", ha="center", va="top", color=MUTED, fontsize=7.0)
    main_ax.text(chain, TAD_BOT - 8.6, "kb", ha="right", va="top", color=MUTED, fontsize=7.0)

    # ---- footer legend: six grouped legends ---------------------------------
    foot_ax.axis("off")

    def header(x, txt):
        foot_ax.text(x, 0.92, txt, transform=foot_ax.transAxes, color=INK2,
                     fontsize=7.4, fontweight="bold")

    header(0.010, "COHESIN")
    _chip(foot_ax, 0.010, 0.60, COHESIN, "extruding", marker="arc")
    _chip(foot_ax, 0.010, 0.24, COHESIN_CAP, "CTCF-captured", marker="arc")

    header(0.128, "CTCF BARRIER")
    _chip(foot_ax, 0.128, 0.60, CTCF, "convergent, oriented", marker="pair")
    foot_ax.text(0.128, 0.24, "arrow = capture side", transform=foot_ax.transAxes,
                 color=CAPTION, fontsize=7.9, va="center")

    header(0.268, "GENE CLASS")
    for i, k in enumerate(["hk_const", "celltype", "dev"]):
        _chip(foot_ax, 0.268, 0.60 - 0.29 * i, CLS[k], CLS_LABEL[k], marker="s")

    header(0.404, "RNA Pol II")
    order = [0, 1, 2, 3, 4]
    for i, s in enumerate(order):
        x = 0.404 + 0.088 * (i % 2)
        y = 0.60 - 0.29 * (i // 2)
        _chip(foot_ax, x, y, RN[s], RN_NAME[s], marker="o", hollow=False)

    header(0.612, "REGULATION")
    _chip(foot_ax, 0.612, 0.60, "#3a3d49", "enhancer (idle)", marker="D")
    _chip(foot_ax, 0.612, 0.24, TEAL, "E–P contact", marker="D")

    header(0.792, "UV LESION")
    _chip(foot_ax, 0.792, 0.60, LES_A, "Type A · TC-NER", marker="star", hollow=True)
    _chip(foot_ax, 0.792, 0.24, LES_B, "Type B · GG-NER", marker="star", hollow=True)
    _chip(foot_ax, 0.905, 0.60, LES_A, "★ filled = repair", marker="star", hollow=False)


def stats_text(fig, d, t, comp, info, holder):
    for h in holder:
        h.remove()
    holder.clear()
    tsec = t * d["tick_s"]; mm, ss = int(tsec // 60), int(tsec % 60)
    anchors = d["anchor_set"]; coh = d["pos"][t]
    lo = np.minimum(coh[:, 0], coh[:, 1]); hi = np.maximum(coh[:, 0], coh[:, 1])
    captured = int(np.sum([(int(lo[j]) in anchors) and (int(hi[j]) in anchors)
                           for j in range(len(lo))]))
    rs = d["rs"][t]
    n_el = int(np.sum(rs == STATE_ELONG)); n_st = int(np.sum(rs == STATE_STALL))
    lst = d["lstate"][t]; lm = d["les"][t] >= 0
    n_rep = int(np.sum(lst[lm] == LSTATE_REPAIR))
    n_active = sum(1 for v in info["gstate"].values() if v == "active")

    items = [
        (f"t = {mm:02d}:{ss:02d}", "biological time", INK),
        (f"{captured}/{d['pos'].shape[1]}", "cohesins CTCF-anchored", COHESIN_CAP),
        (f"{n_active}/{len(d['gtab'])}", "genes active", STATE_COL["active"]),
        (f"{n_el}", "Pol II elongating", RN[2]),
        (f"{n_st}", "Pol II stalled", RN[4]),
        (f"{n_rep}", "lesions in repair", LES_B),
        (f"{int(comp[t])}", "transcripts completed", INK),
    ]
    x = 0.012
    for val, lab, col in items:
        holder.append(fig.text(x, 0.052, val, color=col, fontsize=13, fontweight="bold",
                               ha="left", va="baseline"))
        holder.append(fig.text(x, 0.016, lab, color=CAPTION, fontsize=7.5, ha="left", va="baseline"))
        x += 0.140


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_figure(d, dpi=100):
    # figsize is fixed (16:9); dpi sets the pixel resolution. Everything is sized in
    # points, so a higher dpi scales the whole render uniformly:
    #   dpi=100 -> 1280x720   dpi=150 -> 1920x1080   dpi=200 -> 2560x1440 (2K/QHD)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    fig.patch.set_facecolor(BG)
    main = fig.add_axes([0.118, 0.205, 0.872, 0.675]); main.set_facecolor(BG)
    foot = fig.add_axes([0.008, 0.075, 0.984, 0.112]); foot.set_facecolor(BG)
    return fig, main, foot


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--config", default=str(here / "config_demo.yaml"))
    ap.add_argument("--h5", default=str(here / "run" / "LEFPositions.h5"))
    ap.add_argument("--out", default=str(here / "demo.gif"))
    ap.add_argument("--png", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=100,
                    help="pixel resolution: 100=720p, 150=1080p, 200=2560x1440 (2K)")
    args = ap.parse_args()

    d = load(args.config, args.h5)
    T = d["pos"].shape[0]
    stride = max(1, (T - args.start) // args.frames)
    frames = list(range(args.start, T, stride))[:args.frames]
    comp, per, uidpos = derive(d, frames)

    fig, main_ax, foot_ax = build_figure(d, args.dpi)
    draw_static(main_ax, d)
    setup_chrome(fig, main_ax, foot_ax, d)
    dyn, stat_holder = [], []

    def render(fi):
        t = frames[fi]
        draw_frame(main_ax, d, t, comp, per[t], uidpos, frames, fi, dyn)
        stats_text(fig, d, t, comp, per[t], stat_holder)

    if args.frame is not None or args.png:
        fi = args.frame if args.frame is not None else len(frames) // 2
        render(min(fi, len(frames) - 1))
        out = args.png or str(here / "preview.png")
        fig.savefig(out, facecolor=BG, dpi=args.dpi)
        print("wrote", out)
        return

    imgs = []
    for fi in range(len(frames)):
        render(fi)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(Image.fromarray(buf[:, :, :3].copy()))
        if fi % 40 == 0:
            print(f"frame {fi}/{len(frames)}")

    # seamless loop: brief hold, then a dim-dissolve back to the first frame. The
    # brightness dips through the middle of the dissolve so the two overlaid states
    # (esp. the changing stat digits) never ghost as double text -- it reads as a
    # gentle "settle and restart" rather than a hard jump.
    hold, xf = 2, 9
    tail = [imgs[-1]] * hold
    for k in range(1, xf + 1):
        a = k / (xf + 1)
        blended = Image.blend(imgs[-1], imgs[0], a)
        dim = 1.0 - 0.5 * float(np.sin(np.pi * a))     # 1 -> 0.5 (mid) -> 1
        tail.append(blended.point(lambda p, d=dim: int(p * d)))
    full = imgs + tail

    # shared adaptive palette (from the most colourful frame) -> no flicker
    def ncolors(im):
        c = im.getcolors(maxcolors=1 << 24)
        return len(c) if c else 0
    # No dithering: the scene is flat-colour glyphs on a flat dark surface, so an
    # undithered quantise stays crisp AND avoids the speckle that bloats the GIF.
    pal = max(imgs, key=ncolors).convert("P", palette=Image.ADAPTIVE, colors=256)
    pframes = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in full]
    dur = int(1000 / args.fps)
    durations = [dur] * len(imgs) + [int(dur * 1.4)] * hold + [int(dur * 0.9)] * xf
    assert len(durations) == len(pframes)
    pframes[0].save(args.out, save_all=True, append_images=pframes[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)
    print("wrote", args.out, f"({len(pframes)} frames, {os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
