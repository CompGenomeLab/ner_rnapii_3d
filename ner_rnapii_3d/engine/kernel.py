"""Numba 1D engine kernels (bit-identical to the pure-Python engine).

Phase 3: cohesin-only dynamics (no RNAPII, no lesions). The kernel reproduces
``lef_dynamics.{load_one,translocate}`` draw-for-draw using the ported MT19937
(:mod:`_mt19937`), so the same seed yields a byte-identical trajectory. RNAPII
and lesions are added in later phases.

Struct-of-arrays layout (one entry per cohesin slot):
    lpos/rpos      left/right leg site
    lctcf/rctcf    captured-at-CTCF flag
    lstalled/rstalled  stalled-this-tick flag (feeds unload_prob)
``occupied`` is the dense int8 grid (0 free, 1 cohesin), exactly as today.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .rng import nb_seed, nb_random, nb_randint

FREE = 0
COHESIN = 1
RNAPII_CELL = 2

STATE_PRE_INITIATION = 0
STATE_PAUSED = 1
STATE_ELONGATING = 2
STATE_TERMINATING = 3
STATE_STALLED = 4


def build_ctcf_arrays(args: dict, N: int):
    """Dense per-(side, site) CTCF capture/release probabilities.

    ``args["ctcfCapture"]`` / ``ctcfRelease`` are ``{-1: {pos: p}, 1: {pos: p}}``
    dicts; row 0 is the left leg (side -1), row 1 the right leg (side +1).
    """
    cap = np.zeros((2, N), dtype=np.float64)
    rel = np.zeros((2, N), dtype=np.float64)
    for row, side in ((0, -1), (1, 1)):
        for pos, p in (args.get("ctcfCapture", {}).get(side, {}) or {}).items():
            cap[row, int(pos)] = float(p)
        for pos, p in (args.get("ctcfRelease", {}).get(side, {}) or {}).items():
            rel[row, int(pos)] = float(p)
    return cap, rel


@njit(cache=True)
def _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
              slot, N, chain_length):
    """Place a cohesin into ``slot`` on a free same-chain adjacent pair.

    Mirrors ``lef_dynamics.load_one``: up to max(100, 2N) random attempts
    (``randint(N-1)``), then a deterministic left-to-right fallback scan.
    """
    limit = 2 * N
    if limit < 100:
        limit = 100
    for _ in range(limit):
        a = nb_randint(mt, 0, N - 1)
        if (a // chain_length) != ((a + 1) // chain_length):
            continue
        if occupied[a] == FREE and occupied[a + 1] == FREE:
            occupied[a] = COHESIN
            occupied[a + 1] = COHESIN
            lpos[slot] = a; rpos[slot] = a + 1
            lctcf[slot] = 0; rctcf[slot] = 0
            lstalled[slot] = 0; rstalled[slot] = 0
            return True
    for a in range(N - 1):
        if (a // chain_length) != ((a + 1) // chain_length):
            continue
        if occupied[a] == FREE and occupied[a + 1] == FREE:
            occupied[a] = COHESIN
            occupied[a + 1] = COHESIN
            lpos[slot] = a; rpos[slot] = a + 1
            lctcf[slot] = 0; rctcf[slot] = 0
            lstalled[slot] = 0; rstalled[slot] = 0
            return True
    raise RuntimeError("No same-chain adjacent empty sites available for cohesin loading")


@njit(cache=True)
def _translocate(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                 cap, rel, nlef, N, chain_length,
                 lifetime, lifetime_stalled, lifetime_ctcf):
    # Phase 1: unload + reload, REVERSE order (matches del/insert-at-i semantics).
    for i in range(nlef - 1, -1, -1):
        if lctcf[i] or rctcf[i]:
            prob = 1.0 / lifetime_ctcf
        elif lstalled[i] or rstalled[i]:
            prob = 1.0 / lifetime_stalled
        else:
            prob = 1.0 / lifetime
        if nb_random(mt) < prob:
            occupied[lpos[i]] = FREE
            occupied[rpos[i]] = FREE
            _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                      i, N, chain_length)

    # Phase 2: CTCF capture (always 2 draws) then release (draws only on captured legs).
    for i in range(nlef):
        if nb_random(mt) < cap[0, lpos[i]]:
            lctcf[i] = 1
        if nb_random(mt) < cap[1, rpos[i]]:
            rctcf[i] = 1
        if lctcf[i] or rctcf[i]:
            if lctcf[i] and nb_random(mt) < rel[0, lpos[i]]:
                lctcf[i] = 0
            if rctcf[i] and nb_random(mt) < rel[1, rpos[i]]:
                rctcf[i] = 0

    # Phase 3: per-leg translocation (no draws).
    for i in range(nlef):
        if not lctcf[i]:
            t = lpos[i] - 1
            if t < 0 or t >= N or (lpos[i] // chain_length) != (t // chain_length) or occupied[t] != FREE:
                lstalled[i] = 1
            else:
                lstalled[i] = 0
                occupied[lpos[i]] = FREE
                occupied[t] = COHESIN
                lpos[i] = t
        if not rctcf[i]:
            t = rpos[i] + 1
            if t < 0 or t >= N or (rpos[i] // chain_length) != (t // chain_length) or occupied[t] != FREE:
                rstalled[i] = 1
            else:
                rstalled[i] = 0
                occupied[rpos[i]] = FREE
                occupied[t] = COHESIN
                rpos[i] = t


@njit(cache=True)
def simulate_cohesin(seed, N, chain_length, nlef, cap, rel,
                     lifetime, lifetime_stalled, lifetime_ctcf, warmup, traj):
    """Run the cohesin-only engine; return positions ``(traj, nlef, 2)`` int32."""
    mt = nb_seed(seed)
    occupied = np.zeros(N, dtype=np.int8)
    lpos = np.zeros(nlef, dtype=np.int64); rpos = np.zeros(nlef, dtype=np.int64)
    lctcf = np.zeros(nlef, dtype=np.int8); rctcf = np.zeros(nlef, dtype=np.int8)
    lstalled = np.zeros(nlef, dtype=np.int8); rstalled = np.zeros(nlef, dtype=np.int8)

    for slot in range(nlef):
        _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                  slot, N, chain_length)

    for _ in range(warmup):
        _translocate(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                     cap, rel, nlef, N, chain_length, lifetime, lifetime_stalled, lifetime_ctcf)

    out = np.empty((traj, nlef, 2), dtype=np.int32)
    for t in range(traj):
        _translocate(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                     cap, rel, nlef, N, chain_length, lifetime, lifetime_stalled, lifetime_ctcf)
        for i in range(nlef):
            out[t, i, 0] = lpos[i]
            out[t, i, 1] = rpos[i]
    return out


# ===========================================================================
# Phase 4a: RNAPII-only kernel (0 cohesins).
#
# With no cohesins the cohesin<->RNAPII coupling is inert (RNAPII-RNAPII
# exclusion is handled by `occupied` alone, so the rnapii_by_pos index is never
# read in a behaviour-affecting way). This isolates and validates the full Pol II
# state machine: load -> PRE-INIT -> PAUSED -> ELONGATING -> step -> TES ->
# TERMINATING -> unload, plus the 3' termination window. The cohesin coupling is
# added in Phase 4b. Core path only: no enhancer gating, no lesions, no Tei
# restraint (all RNG-free or skipped here); per-gene premature termination and
# termination windows ARE handled. Bit-identical to load_rnapii +
# stateful_translocate_rnapii (tests/test_equivalence.py).
# ===========================================================================

# Gene SoA column order (one row per gene):
GCOL = ("tss", "tes", "direction", "load_prob", "initiation_prob",
        "pause_release_prob", "pause_offset", "elongation_step_prob",
        "termination_prob", "termination_window", "pause_term_prob")


def build_gene_soa(genes):
    """Pack a list of Gene objects into the per-column arrays the kernel reads."""
    n = len(genes)
    tss = np.empty(n, np.int64); tes = np.empty(n, np.int64)
    direction = np.empty(n, np.int64); load_prob = np.empty(n, np.float64)
    init_prob = np.empty(n, np.float64); release_prob = np.empty(n, np.float64)
    pause_offset = np.empty(n, np.int64); elong_prob = np.empty(n, np.float64)
    term_prob = np.empty(n, np.float64); term_window = np.empty(n, np.int64)
    pause_term_prob = np.empty(n, np.float64)
    for i, g in enumerate(genes):
        tss[i] = g.tss; tes[i] = g.tes; direction[i] = g.direction
        load_prob[i] = g.load_prob; init_prob[i] = g.initiation_prob
        release_prob[i] = g.pause_release_prob; pause_offset[i] = g.pause_offset
        elong_prob[i] = g.elongation_step_prob; term_prob[i] = g.termination_prob
        term_window[i] = g.termination_window; pause_term_prob[i] = g.pause_term_prob
    return (tss, tes, direction, load_prob, init_prob, release_prob,
            pause_offset, elong_prob, term_prob, term_window, pause_term_prob)


@njit(cache=True)
def _rn_try_step(mt, occupied, rn_pos, idx, tes, direction, N, chain_length):
    """RNAPII single +direction step, no-cohesin path. Returns True if advanced.

    Mirrors _try_single_step for the FREE / RNAPII_CELL / TES cases (the COHESIN
    branch is unreachable with 0 cohesins and draws no RNG, so it is omitted).
    """
    pos = rn_pos[idx]
    target = pos + direction
    if target < 0 or target >= N or (pos // chain_length) != (target // chain_length):
        return False
    if target == tes:
        if occupied[target] != FREE:
            return False
        if occupied[pos] == RNAPII_CELL:
            occupied[pos] = FREE
        occupied[target] = RNAPII_CELL
        rn_pos[idx] = target
        return True
    cell = occupied[target]
    if cell == RNAPII_CELL:
        return False
    # cell == FREE (COHESIN cannot occur with 0 cohesins)
    if occupied[pos] == RNAPII_CELL:
        occupied[pos] = FREE
    occupied[target] = RNAPII_CELL
    rn_pos[idx] = target
    return True


@njit(cache=True)
def _promoter_occluded(occupied, tss, direction, pause_off, occl):
    """Pause-initiation limit: new initiation is suppressed while the
    promoter-proximal region is occupied. The TSS itself must be free (existing
    rule, blocks on a Pol OR cohesin); additionally, when ``occl`` is on, a paused
    or pre-initiation Pol anywhere in the window [TSS .. TSS+dir*pause_off] blocks
    a new initiation (Gressel 2019 Nat Commun "pause-initiation limit"; the
    one-Pol-per-promoter-proximal-region rule of Bartman 2019). At a coarse lattice
    the real ~50 bp steric window is lost, so it is enforced explicitly here."""
    if occupied[tss] != FREE:
        return True
    if occl != 0:
        N = occupied.shape[0]
        for s in range(1, pause_off + 1):
            site = tss + direction * s
            if site < 0 or site >= N:
                break
            if occupied[site] == RNAPII_CELL:
                return True
    return False


@njit(cache=True)
def _load_rnapii(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid, n_rn, uid_next,
                 g_tss, g_dir, g_load_prob, g_pause_off, occl, n_genes):
    """One load pass: per gene draw random; spawn PRE-INITIATION Pol at a free TSS
    (subject to the promoter-proximal occlusion / pause-initiation limit)."""
    for g in range(n_genes):
        if nb_random(mt) >= g_load_prob[g]:
            continue
        tss = g_tss[g]
        if _promoter_occluded(occupied, tss, g_dir[g], g_pause_off[g], occl):
            continue
        rn_pos[n_rn] = tss
        rn_gene[n_rn] = g
        rn_state[n_rn] = STATE_PRE_INITIATION
        rn_uid[n_rn] = uid_next
        occupied[tss] = RNAPII_CELL
        n_rn += 1
        uid_next += 1
    return n_rn, uid_next


@njit(cache=True)
def _stateful_translocate(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid, n_rn,
                          g_tss, g_tes, g_dir, g_init, g_release, g_pause_off,
                          g_elong, g_term, g_window, g_pause_term,
                          pause_term_global, term_step_prob, stride, N, chain_length):
    """One RNAPII dynamics pass (core path, 0 cohesins). Returns new n_rn.

    Dead Pol II are marked and compacted out at the end, preserving list order
    (equivalent to the reversed-iteration `del rnapiis[idx]`).
    """
    dead = np.zeros(n_rn, dtype=np.int8)
    for idx in range(n_rn - 1, -1, -1):
        g = rn_gene[idx]
        tes = g_tes[g]
        direction = g_dir[g]
        past_tes = (rn_pos[idx] - tes) * direction
        if past_tes >= 0:
            rn_state[idx] = STATE_TERMINATING
            win = g_window[g]
            if win <= 0:
                if nb_random(mt) < g_term[g]:
                    if occupied[rn_pos[idx]] == RNAPII_CELL:
                        occupied[rn_pos[idx]] = FREE
                    dead[idx] = 1
                continue
            # 3' termination window: release at window end or stochastically.
            if past_tes >= win or nb_random(mt) < g_term[g]:
                if occupied[rn_pos[idx]] == RNAPII_CELL:
                    occupied[rn_pos[idx]] = FREE
                dead[idx] = 1
                continue
            if term_step_prob >= 1.0 or nb_random(mt) < term_step_prob:
                _rn_try_step(mt, occupied, rn_pos, idx, tes, direction, N, chain_length)
            continue

        state = rn_state[idx]
        if state == STATE_PRE_INITIATION:
            if nb_random(mt) < g_init[g]:
                for _ in range(g_pause_off[g]):
                    if not _rn_try_step(mt, occupied, rn_pos, idx, tes, direction, N, chain_length):
                        break
                rn_state[idx] = STATE_PAUSED
            continue

        if state == STATE_PAUSED:
            g_pt = g_pause_term[g] if g_pause_term[g] >= 0.0 else pause_term_global
            if g_pt > 0.0 and nb_random(mt) < g_pt:
                if occupied[rn_pos[idx]] == RNAPII_CELL:
                    occupied[rn_pos[idx]] = FREE
                dead[idx] = 1
                continue
            release_prob = g_release[g]
            if nb_random(mt) >= release_prob:
                continue
            rn_state[idx] = STATE_ELONGATING
            # fall through to elongation

        just_released = (state == STATE_PAUSED)
        stalled = False
        for i in range(stride):
            forced = just_released and i == 0
            if (not forced) and nb_random(mt) >= g_elong[g]:
                continue
            if not _rn_try_step(mt, occupied, rn_pos, idx, tes, direction, N, chain_length):
                stalled = True
                break
            if rn_pos[idx] == tes:
                break
        rn_state[idx] = STATE_STALLED if stalled else STATE_ELONGATING

    # compact out dead Pol, preserving order
    w = 0
    for idx in range(n_rn):
        if not dead[idx]:
            if w != idx:
                rn_pos[w] = rn_pos[idx]; rn_gene[w] = rn_gene[idx]
                rn_state[w] = rn_state[idx]; rn_uid[w] = rn_uid[idx]
            w += 1
    return w


@njit(cache=True)
def simulate_rnapii_only(seed, N, chain_length, cap, gene_soa,
                         pause_term_global, term_step_prob, stride, warmup, traj):
    """RNAPII-only engine (0 cohesins). Returns (positions, states, ids):
    positions (traj, cap, 2) [pos, gene_id]; states/ids (traj, cap); -1 padded."""
    (g_tss, g_tes, g_dir, g_load, g_init, g_release, g_pause_off,
     g_elong, g_term, g_window, g_pause_term) = gene_soa
    n_genes = g_tss.shape[0]
    mt = nb_seed(seed)
    occupied = np.zeros(N, dtype=np.int8)
    rn_pos = np.full(cap, -1, dtype=np.int64)
    rn_gene = np.full(cap, -1, dtype=np.int64)
    rn_state = np.full(cap, -1, dtype=np.int64)
    rn_uid = np.full(cap, -1, dtype=np.int64)
    n_rn = 0
    uid_next = 0

    for _ in range(warmup):
        n_rn, uid_next = _load_rnapii(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid,
                                      n_rn, uid_next, g_tss, g_dir, g_load, g_pause_off, 1, n_genes)
        n_rn = _stateful_translocate(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid, n_rn,
                                     g_tss, g_tes, g_dir, g_init, g_release, g_pause_off,
                                     g_elong, g_term, g_window, g_pause_term,
                                     pause_term_global, term_step_prob, stride, N, chain_length)

    pos_out = np.empty((traj, cap, 2), np.int32); pos_out[:] = -1
    st_out = np.empty((traj, cap), np.int32); st_out[:] = -1
    id_out = np.empty((traj, cap), np.int32); id_out[:] = -1
    for t in range(traj):
        n_rn, uid_next = _load_rnapii(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid,
                                      n_rn, uid_next, g_tss, g_dir, g_load, g_pause_off, 1, n_genes)
        n_rn = _stateful_translocate(mt, occupied, rn_pos, rn_gene, rn_state, rn_uid, n_rn,
                                     g_tss, g_tes, g_dir, g_init, g_release, g_pause_off,
                                     g_elong, g_term, g_window, g_pause_term,
                                     pause_term_global, term_step_prob, stride, N, chain_length)
        m = n_rn if n_rn < cap else cap
        for i in range(m):
            pos_out[t, i, 0] = rn_pos[i]
            pos_out[t, i, 1] = rn_gene[i]
            st_out[t, i] = rn_state[i]
            id_out[t, i] = rn_uid[i]
    return pos_out, st_out, id_out


# ===========================================================================
# Phase 4b: cohesin <-> RNAPII coupling (full engine minus lesions).
#
# Maintains two dense site->index maps that mirror the Python dicts:
#   rnapii_at[site] = rnapii population index (or -1)   <-> rnapii_by_pos
#   leg_at[site]    = leg id = slot*2 + side (0 left,1 right), or -1  <-> cohesin_leg_by_pos
# Cohesin slots are stable (reload-into-slot), so leg ids are stable. RNAPII
# indices are stable DURING a tick (compaction is deferred to the end of the
# state-machine pass), so the maps stay valid for the in-tick lookups; after
# compaction rnapii_at is rebuilt for the surviving Pol. Bit-identical to
# load_rnapii + stateful_translocate_rnapii + translocate_with_rnapii
# (tests/test_equivalence.py).
# ===========================================================================


@njit(cache=True)
def _load_rnapii_c(mt, occupied, rnapii_at, rn_pos, rn_gene, rn_state, rn_uid,
                   n_rn, uid_next, g_tss, g_dir, g_load_prob, g_pause_off, occl, n_genes,
                   ep_count, g_load_req, g_k, g_logic, g_syn, rn_lesion_stalled):
    for g in range(n_genes):
        load_prob = g_load_prob[g]
        if g_load_req[g]:                         # enhancer-gated loading
            factor = _enh_factor(ep_count[g], g_k[g], g_logic[g], g_syn[g])
            if factor <= 0.0:
                continue                          # gate closed: skip BEFORE the load draw
            load_prob = load_prob * factor
            if load_prob > 1.0:
                load_prob = 1.0
        if nb_random(mt) >= load_prob:
            continue
        tss = g_tss[g]
        # promoter-proximal occlusion (pause-initiation limit): suppress new
        # initiation while a Pol occupies the promoter-proximal window.
        if _promoter_occluded(occupied, tss, g_dir[g], g_pause_off[g], occl):
            continue
        rn_pos[n_rn] = tss; rn_gene[n_rn] = g; rn_state[n_rn] = STATE_PRE_INITIATION
        rn_uid[n_rn] = uid_next
        rn_lesion_stalled[n_rn] = 0
        occupied[tss] = RNAPII_CELL
        rnapii_at[tss] = n_rn
        n_rn += 1; uid_next += 1
    return n_rn, uid_next


@njit(cache=True)
def _try_step_c(mt, occupied, rnapii_at, leg_at, rn_pos, ridx, tes, direction, state,
                lpos, rpos, lpushed, rpushed, lrnst, rrnst,
                stall_prob, push_prob, headon_push_prob,
                rn_lesion_stalled, lesion_at, les_type, les_state, has_lesions,
                N, chain_length):
    """RNAPII single step WITH cohesin coupling (FREE/RNAPII/COHESIN/TES).

    ``state`` is the Pol's current state. _resolve_head_on only draws / can push
    for ELONGATING Pol; PRE-INIT / PAUSED / TERMINATING / STALLED Pol stall on a
    cohesin WITHOUT drawing (mirrors `if state != ELONGATING: return stall`).
    A lesion that blocks RNAPII (Type A in pre-recognition, or any in repair) stalls
    the Pol one step upstream and sets lesion_stalled (no RNG; lesion_blocks_rnapii).
    """
    pos = rn_pos[ridx]
    rn_lesion_stalled[ridx] = 0          # reset (matches r.lesion_stalled = False at top)
    target = pos + direction
    if target < 0 or target >= N or (pos // chain_length) != (target // chain_length):
        return False
    if has_lesions:
        lidx = lesion_at[target]
        if lidx >= 0 and (les_state[lidx] == LESION_REPAIR or les_type[lidx] == LESION_TYPE_A):
            rn_lesion_stalled[ridx] = 1
            return False
    if target == tes:
        if occupied[target] != FREE:
            return False
        rnapii_at[pos] = -1
        if occupied[pos] == RNAPII_CELL:
            occupied[pos] = FREE
        occupied[target] = RNAPII_CELL
        rn_pos[ridx] = target
        rnapii_at[target] = ridx
        return True

    cell = occupied[target]
    if cell == RNAPII_CELL:
        return False

    if cell == COHESIN:
        if rnapii_at[target] >= 0:          # masked Pol under a tunnelling cohesin
            return False
        legid = leg_at[target]
        # _resolve_head_on: non-elongating Pol stalls without drawing.
        if state != STATE_ELONGATING:
            return False
        if nb_random(mt) < stall_prob:
            return False
        if legid >= 0:
            leg_dir = -1 if (legid % 2) == 0 else 1
            head_on = (leg_dir == -direction)
        else:
            head_on = False
        push_p = headon_push_prob if head_on else push_prob
        if not (nb_random(mt) < push_p):
            return False
        behind = target + direction
        if behind < 0 or behind >= N or (target // chain_length) != (behind // chain_length) \
                or occupied[behind] != FREE:
            return False
        occupied[behind] = COHESIN
        occupied[target] = RNAPII_CELL
        rnapii_at[pos] = -1
        if occupied[pos] == RNAPII_CELL:
            occupied[pos] = FREE
        leg_at[target] = -1
        if legid >= 0:
            slot = legid // 2
            if (legid % 2) == 0:
                lpos[slot] = behind; lpushed[slot] = 1; lrnst[slot] = 1
            else:
                rpos[slot] = behind; rpushed[slot] = 1; rrnst[slot] = 1
            leg_at[behind] = legid
        rn_pos[ridx] = target
        rnapii_at[target] = ridx
        return True

    # cell == FREE
    if occupied[pos] == RNAPII_CELL:
        occupied[pos] = FREE
    rnapii_at[pos] = -1
    occupied[target] = RNAPII_CELL
    rnapii_at[target] = ridx
    rn_pos[ridx] = target
    return True


@njit(cache=True)
def _stateful_translocate_c(mt, occupied, rnapii_at, leg_at, rn_pos, rn_gene,
                            rn_state, rn_uid, n_rn, lpos, rpos, lpushed, rpushed,
                            lrnst, rrnst, g_tes, g_dir, g_init, g_release, g_pause_off,
                            g_elong, g_term, g_window, g_pause_term,
                            pause_term_global, term_step_prob, stride,
                            stall_prob, push_prob, headon_push_prob, N, chain_length,
                            ep_count, g_req_enh, g_k, g_logic, g_syn,
                            rn_lesion_stalled, lesion_at, les_type, les_state, has_lesions,
                            dead):
    for idx in range(n_rn):
        dead[idx] = 0
    for idx in range(n_rn - 1, -1, -1):
        g = rn_gene[idx]; tes = g_tes[g]; direction = g_dir[g]
        # _evict_if_lesion_repaired: a Pol stalled by a lesion that has since been
        # repaired (no longer at pos+dir) is EVICTED, not resumed. Lesions updated
        # before this pass, so the site is free now.
        if has_lesions and rn_lesion_stalled[idx]:
            nxt = rn_pos[idx] + direction
            in_les = (0 <= nxt < N) and (lesion_at[nxt] >= 0)
            if not in_les:
                rnapii_at[rn_pos[idx]] = -1
                if occupied[rn_pos[idx]] == RNAPII_CELL:
                    occupied[rn_pos[idx]] = FREE
                dead[idx] = 1
                continue
        past_tes = (rn_pos[idx] - tes) * direction
        if past_tes >= 0:
            rn_state[idx] = STATE_TERMINATING
            win = g_window[g]
            if win <= 0:
                if nb_random(mt) < g_term[g]:
                    rnapii_at[rn_pos[idx]] = -1
                    if occupied[rn_pos[idx]] == RNAPII_CELL:
                        occupied[rn_pos[idx]] = FREE
                    dead[idx] = 1
                continue
            if past_tes >= win or nb_random(mt) < g_term[g]:
                rnapii_at[rn_pos[idx]] = -1
                if occupied[rn_pos[idx]] == RNAPII_CELL:
                    occupied[rn_pos[idx]] = FREE
                dead[idx] = 1
                continue
            if term_step_prob >= 1.0 or nb_random(mt) < term_step_prob:
                _try_step_c(mt, occupied, rnapii_at, leg_at, rn_pos, idx, tes, direction, rn_state[idx],
                            lpos, rpos, lpushed, rpushed, lrnst, rrnst,
                            stall_prob, push_prob, headon_push_prob,
                            rn_lesion_stalled, lesion_at, les_type, les_state, has_lesions,
                            N, chain_length)
            continue

        state = rn_state[idx]
        if state == STATE_PRE_INITIATION:
            if nb_random(mt) < g_init[g]:
                for _ in range(g_pause_off[g]):
                    if not _try_step_c(mt, occupied, rnapii_at, leg_at, rn_pos, idx, tes, direction, rn_state[idx],
                                       lpos, rpos, lpushed, rpushed, lrnst, rrnst,
                                       stall_prob, push_prob, headon_push_prob,
                            rn_lesion_stalled, lesion_at, les_type, les_state, has_lesions,
                            N, chain_length):
                        break
                rn_state[idx] = STATE_PAUSED
            continue

        if state == STATE_PAUSED:
            g_pt = g_pause_term[g] if g_pause_term[g] >= 0.0 else pause_term_global
            if g_pt > 0.0 and nb_random(mt) < g_pt:
                rnapii_at[rn_pos[idx]] = -1
                if occupied[rn_pos[idx]] == RNAPII_CELL:
                    occupied[rn_pos[idx]] = FREE
                dead[idx] = 1
                continue
            release_prob = g_release[g]
            if g_req_enh[g]:                      # enhancer-gated pause release
                factor = _enh_factor(ep_count[g], g_k[g], g_logic[g], g_syn[g])
                if factor <= 0.0:
                    continue                      # gate closed: stay PAUSED, BEFORE release draw
                release_prob = release_prob * factor
                if release_prob > 1.0:
                    release_prob = 1.0
            if nb_random(mt) >= release_prob:
                continue
            rn_state[idx] = STATE_ELONGATING

        just_released = (state == STATE_PAUSED)
        stalled = False
        for i in range(stride):
            forced = just_released and i == 0
            if (not forced) and nb_random(mt) >= g_elong[g]:
                continue
            if not _try_step_c(mt, occupied, rnapii_at, leg_at, rn_pos, idx, tes, direction, rn_state[idx],
                               lpos, rpos, lpushed, rpushed, lrnst, rrnst,
                               stall_prob, push_prob, headon_push_prob,
                            rn_lesion_stalled, lesion_at, les_type, les_state, has_lesions,
                            N, chain_length):
                stalled = True
                break
            if rn_pos[idx] == tes:
                break
        rn_state[idx] = STATE_STALLED if stalled else STATE_ELONGATING

    # compact dead Pol, preserving order, and rebuild rnapii_at for survivors
    w = 0
    for idx in range(n_rn):
        if not dead[idx]:
            if w != idx:
                rn_pos[w] = rn_pos[idx]; rn_gene[w] = rn_gene[idx]
                rn_state[w] = rn_state[idx]; rn_uid[w] = rn_uid[idx]
                rn_lesion_stalled[w] = rn_lesion_stalled[idx]
            rnapii_at[rn_pos[w]] = w
            w += 1
    return w


@njit(cache=True)
def _vacate_c(occupied, rnapii_at, p):
    occupied[p] = RNAPII_CELL if rnapii_at[p] >= 0 else FREE


@njit(cache=True)
def _translocate_c(mt, occupied, rnapii_at, leg_at, lpos, rpos, lctcf, rctcf,
                   lstalled, rstalled, lrnst, rrnst, lpushed, rpushed,
                   rn_state, cap, rel, block_prob, nlef, N, chain_length,
                   lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                   lesion_at, les_type, les_state, lesion_block_prob, rnapii_enabled, has_lesions):
    # Phase 1: unload + reload (reverse).
    for i in range(nlef - 1, -1, -1):
        if lctcf[i] or rctcf[i]:
            prob = 1.0 / lifetime_ctcf
        elif lrnst[i] or rrnst[i]:
            prob = 1.0 / lifetime_rnstalled
        elif lstalled[i] or rstalled[i]:
            prob = 1.0 / lifetime_stalled
        else:
            prob = 1.0 / lifetime
        if nb_random(mt) < prob:
            leg_at[lpos[i]] = -1
            leg_at[rpos[i]] = -1
            _vacate_c(occupied, rnapii_at, lpos[i])
            _vacate_c(occupied, rnapii_at, rpos[i])
            _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                      i, N, chain_length)
            leg_at[lpos[i]] = i * 2
            leg_at[rpos[i]] = i * 2 + 1
            lrnst[i] = 0; rrnst[i] = 0; lpushed[i] = 0; rpushed[i] = 0

    # Phase 2: capture / release (2 capture draws; release on captured legs).
    for i in range(nlef):
        if nb_random(mt) < cap[0, lpos[i]]:
            lctcf[i] = 1
        if nb_random(mt) < cap[1, rpos[i]]:
            rctcf[i] = 1
        if lctcf[i] or rctcf[i]:
            if lctcf[i] and nb_random(mt) < rel[0, lpos[i]]:
                lctcf[i] = 0
            if rctcf[i] and nb_random(mt) < rel[1, rpos[i]]:
                rctcf[i] = 0

    # Phase 3: per-leg translocation (RNAPII-aware).
    for i in range(nlef):
        for side in range(2):              # 0 = left (dir -1), 1 = right (dir +1)
            if side == 0:
                if lctcf[i]:
                    continue
                pos = lpos[i]; d = -1
            else:
                if rctcf[i]:
                    continue
                pos = rpos[i]; d = 1
            target = pos + d
            if target < 0 or target >= N or (pos // chain_length) != (target // chain_length):
                if side == 0:
                    lstalled[i] = 1; lrnst[i] = 0
                else:
                    rstalled[i] = 1; rrnst[i] = 0
                continue
            if has_lesions:                  # lesion_stalls_cohesin + lesion_block_prob draw
                lidx = lesion_at[target]
                if lidx >= 0:
                    ls = les_state[lidx]; lt = les_type[lidx]
                    if ls == LESION_REPAIR:
                        stalls = True; evict = 0
                    elif lt == LESION_TYPE_A:
                        stalls = (not rnapii_enabled); evict = 1
                    else:
                        stalls = False; evict = 0
                    if stalls and nb_random(mt) < lesion_block_prob:
                        if side == 0:
                            lstalled[i] = 1; lrnst[i] = evict
                        else:
                            rstalled[i] = 1; rrnst[i] = evict
                        continue
            cell = occupied[target]
            if cell == FREE:
                leg_at[pos] = -1
                _vacate_c(occupied, rnapii_at, pos)
                occupied[target] = COHESIN
                if side == 0:
                    lpos[i] = target; lstalled[i] = 0; lpushed[i] = 0; lrnst[i] = 0
                else:
                    rpos[i] = target; rstalled[i] = 0; rpushed[i] = 0; rrnst[i] = 0
                leg_at[target] = i * 2 + side
                continue
            if cell == COHESIN:
                if side == 0:
                    lstalled[i] = 1; lrnst[i] = 0
                else:
                    rstalled[i] = 1; rrnst[i] = 0
                continue
            # cell == RNAPII_CELL
            ridx = rnapii_at[target]
            blocked = True
            if ridx >= 0:
                st = rn_state[ridx]
                if st < 0:
                    st = STATE_PRE_INITIATION
                if not (nb_random(mt) < block_prob[st]):
                    blocked = False
            if not blocked:
                leg_at[pos] = -1
                _vacate_c(occupied, rnapii_at, pos)
                occupied[target] = COHESIN
                if side == 0:
                    lpos[i] = target; lstalled[i] = 0; lrnst[i] = 0
                else:
                    rpos[i] = target; rstalled[i] = 0; rrnst[i] = 0
                leg_at[target] = i * 2 + side
            else:
                if side == 0:
                    lstalled[i] = 1; lrnst[i] = 1
                else:
                    rstalled[i] = 1; rrnst[i] = 1


def build_block_prob(args: dict) -> np.ndarray:
    """Per-state cohesin-block probabilities [PRE_INIT, PAUSED, ELONG, TERM, STALLED]."""
    base = float(args.get("rnapii_block_prob", 1.0))
    elong = float(args.get("rnapii_elongating_block_prob", base))
    paused = float(args.get("rnapii_paused_block_prob", base))
    pre_init = float(args.get("rnapii_pre_initiation_block_prob", 1.0))
    term = float(args.get("rnapii_terminating_block_prob", paused))
    return np.array([pre_init, paused, elong, term, elong], dtype=np.float64)


@njit(cache=True)
def simulate_coupled(seed, N, chain_length, nlef, cap, gene_soa, gene_enh, ctcf_cap, ctcf_rel,
                     block_prob, lifetime, lifetime_stalled, lifetime_rnstalled,
                     lifetime_ctcf, pause_term_global, term_step_prob, stride,
                     stall_prob, push_prob, headon_push_prob, tol, record_cap, warmup, traj):
    """Full coupled engine (cohesin + RNAPII, no lesions). Returns
    (coh_pos, rn_pos_out, rn_state_out, rn_uid_out). ``cap`` sizes the Pol
    population arrays (>= max live Pol; the Python list is unbounded); ``record_cap``
    is the recorded width (= rnapii_cap; lef.run truncates `rnapiis[:rnapii_cap]`)."""
    (g_tss, g_tes, g_dir, g_load, g_init, g_release, g_pause_off,
     g_elong, g_term, g_window, g_pause_term) = gene_soa
    (enh_pos, enh_off, g_req_enh, g_load_req, g_k, g_logic, g_syn) = gene_enh
    n_genes = g_tss.shape[0]
    ep_count = np.zeros(n_genes, np.int64)
    ep_left = np.empty(nlef, np.int64)
    ep_right = np.empty(nlef, np.int64)
    ep_prefix_max = np.empty(nlef, np.int64)
    mt = nb_seed(seed)
    occupied = np.zeros(N, dtype=np.int8)
    leg_at = np.full(N, -1, dtype=np.int64)
    rnapii_at = np.full(N, -1, dtype=np.int64)

    lpos = np.zeros(nlef, np.int64); rpos = np.zeros(nlef, np.int64)
    lctcf = np.zeros(nlef, np.int8); rctcf = np.zeros(nlef, np.int8)
    lstalled = np.zeros(nlef, np.int8); rstalled = np.zeros(nlef, np.int8)
    lrnst = np.zeros(nlef, np.int8); rrnst = np.zeros(nlef, np.int8)
    lpushed = np.zeros(nlef, np.int8); rpushed = np.zeros(nlef, np.int8)

    for slot in range(nlef):
        _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                  slot, N, chain_length)
    for i in range(nlef):
        leg_at[lpos[i]] = i * 2
        leg_at[rpos[i]] = i * 2 + 1

    rn_pos = np.full(cap, -1, np.int64); rn_gene = np.full(cap, -1, np.int64)
    rn_state = np.full(cap, -1, np.int64); rn_uid = np.full(cap, -1, np.int64)
    rn_lesion_stalled = np.zeros(cap, np.int8)
    rn_dead = np.empty(cap, np.int8)
    # no-lesion regime: dummy lesion arrays (has_lesions=False -> never read)
    _les_at = np.zeros(N, np.int64); _les_t = np.zeros(1, np.int64); _les_s = np.zeros(1, np.int64)
    n_rn = 0; uid_next = 0

    def _tick(n_rn, uid_next):
        # EP contacts computed FIRST each tick (lef.run order), from current cohesins.
        _compute_ep_contacts(lpos, rpos, nlef, enh_pos, enh_off, g_tss, n_genes,
                             tol, chain_length, ep_count,
                             ep_left, ep_right, ep_prefix_max)
        n_rn, uid_next = _load_rnapii_c(mt, occupied, rnapii_at, rn_pos, rn_gene,
                                        rn_state, rn_uid, n_rn, uid_next, g_tss, g_dir, g_load, g_pause_off, 1, n_genes,
                                        ep_count, g_load_req, g_k, g_logic, g_syn, rn_lesion_stalled)
        n_rn = _stateful_translocate_c(mt, occupied, rnapii_at, leg_at, rn_pos, rn_gene,
                                       rn_state, rn_uid, n_rn, lpos, rpos, lpushed, rpushed,
                                       lrnst, rrnst, g_tes, g_dir, g_init, g_release, g_pause_off,
                                       g_elong, g_term, g_window, g_pause_term,
                                       pause_term_global, term_step_prob, stride,
                                       stall_prob, push_prob, headon_push_prob, N, chain_length,
                                       ep_count, g_req_enh, g_k, g_logic, g_syn,
                                       rn_lesion_stalled, _les_at, _les_t, _les_s, False,
                                       rn_dead)
        _translocate_c(mt, occupied, rnapii_at, leg_at, lpos, rpos, lctcf, rctcf,
                       lstalled, rstalled, lrnst, rrnst, lpushed, rpushed,
                       rn_state, ctcf_cap, ctcf_rel, block_prob, nlef, N, chain_length,
                       lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                       _les_at, _les_t, _les_s, 1.0, True, False)
        return n_rn, uid_next

    for _ in range(warmup):
        n_rn, uid_next = _tick(n_rn, uid_next)

    coh_pos = np.empty((traj, nlef, 2), np.int32)
    rp = np.empty((traj, record_cap, 2), np.int32); rp[:] = -1
    rs = np.empty((traj, record_cap), np.int32); rs[:] = -1
    ri = np.empty((traj, record_cap), np.int32); ri[:] = -1
    for t in range(traj):
        n_rn, uid_next = _tick(n_rn, uid_next)
        for i in range(nlef):
            coh_pos[t, i, 0] = lpos[i]; coh_pos[t, i, 1] = rpos[i]
        m = n_rn if n_rn < record_cap else record_cap
        for i in range(m):
            rp[t, i, 0] = rn_pos[i]; rp[t, i, 1] = rn_gene[i]
            rs[t, i] = rn_state[i]; ri[t, i] = rn_uid[i]
    return coh_pos, rp, rs, ri


# ===========================================================================
# Phase 5a: lesions + cohesin (RNAPII off) -- the production config4 regime.
#
# Lesion population is held in INSERTION ORDER (les_site/les_type/les_state) so
# the `u = random(len(lesions))` state-machine draw aligns with the Python dict's
# values() order; lesion_at[site] mirrors the dict key set for the gating lookups.
# Update order per tick (matches _advance_one_step): lesions FIRST, then cohesin.
# Bit-identical to update_lesions + translocate_with_rnapii (rnapii off) in
# tests/test_equivalence.py.
# ===========================================================================

LESION_TYPE_A = 0
LESION_TYPE_B = 1
LESION_PRE = 0
LESION_REPAIR = 1


def build_lesion_state(args: dict):
    """Extract the static lesion fields the kernel needs from a prepared args dict
    (after ``precompute_lesion_fields``). Returns a tuple consumed by the driver."""
    N = int(args["N"])
    b_enabled = bool(args.get("lesion_type_b_enabled", True))
    if b_enabled:
        target = int(args.get("lesion_target", 0))
        p = args.get("lesion_site_p")
        force_type = -1
    else:
        target = int(args.get("lesion_target_a", 0))
        p = args.get("lesion_genebody_p")
        force_type = LESION_TYPE_A
    if p is not None:
        cdf = np.asarray(p, dtype=np.float64).cumsum()
        cdf /= cdf[-1]
        has_p = True
    else:
        cdf = np.zeros(N, dtype=np.float64)
        has_p = False
    rm = args.get("lesion_rate_mult")
    if rm is not None:
        rate_mult = np.asarray(rm, dtype=np.float64); has_rm = True
    else:
        rate_mult = np.ones(N, dtype=np.float64); has_rm = False
    gbm = args.get("gene_body_mask")
    gene_body_mask = (np.asarray(gbm, dtype=np.int8) if gbm is not None
                      else np.zeros(N, dtype=np.int8))
    return (target, has_p, cdf, has_rm, rate_mult,
            int(args.get("lesion_prerecognition_ticks", 1)),
            int(args.get("lesion_repair_ticks", 1)),
            gene_body_mask, float(args.get("lesion_type_a_prob", 0.5)), force_type)


@njit(cache=True)
def _update_lesions(mt, N, les_site, les_type, les_state, lesion_at, n_les,
                    target, has_p, cdf, has_rm, rate_mult,
                    prerecog_ticks, repair_ticks, gene_body_mask, type_a_prob, force_type,
                    random_scratch, dead, candidate_scratch):
    """One lesion tick: stochastic PRE->REPAIR->removed, then refill to target."""
    if n_les > 0:
        pre_rate = 1.0 / (prerecog_ticks if prerecog_ticks >= 1 else 1)
        rep_rate = 1.0 / (repair_ticks if repair_ticks >= 1 else 1)
        for i in range(n_les):
            random_scratch[i] = nb_random(mt)
            dead[i] = 0
        for i in range(n_les):
            m = rate_mult[les_site[i]] if has_rm else 1.0
            if les_state[i] == LESION_PRE:
                pr = pre_rate * m
                if pr > 1.0:
                    pr = 1.0
                if random_scratch[i] < pr:
                    les_state[i] = LESION_REPAIR
            else:
                rr = rep_rate * m
                if rr > 1.0:
                    rr = 1.0
                if random_scratch[i] < rr:
                    dead[i] = 1
        w = 0
        for i in range(n_les):
            if dead[i]:
                lesion_at[les_site[i]] = -1
            else:
                if w != i:
                    les_site[w] = les_site[i]; les_type[w] = les_type[i]; les_state[w] = les_state[i]
                lesion_at[les_site[w]] = w
                w += 1
        n_les = w

    if n_les < target:
        for _ in range(8):
            need = target - n_les
            if need <= 0:
                break
            head = need // 2
            if head < 4:
                head = 4
            k = need + head
            # numpy draws the WHOLE candidate batch first (choice/randint), THEN the
            # per-site _assign_type draws happen in the loop -- match that draw order.
            if has_p:
                for j in range(k):
                    s = np.searchsorted(cdf, nb_random(mt), side="right")
                    candidate_scratch[j] = s if s < N else N - 1
            else:
                for j in range(k):
                    candidate_scratch[j] = nb_randint(mt, 0, N)
            for j in range(k):
                site = candidate_scratch[j]
                if lesion_at[site] < 0:
                    if force_type >= 0:
                        lt = force_type
                    elif gene_body_mask[site] != 0:
                        lt = LESION_TYPE_A if nb_random(mt) < type_a_prob else LESION_TYPE_B
                    else:
                        lt = LESION_TYPE_B
                    les_site[n_les] = site; les_type[n_les] = lt; les_state[n_les] = LESION_PRE
                    lesion_at[site] = n_les
                    n_les += 1
                    if n_les >= target:
                        break
            if n_les >= target:
                break
    return n_les


@njit(cache=True)
def _translocate_les(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                     lrnst, rrnst, cap, rel, lesion_at, les_type, les_state,
                     lesion_block_prob, rnapii_enabled, targeted, nlef, N, chain_length,
                     lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf):
    """Cohesin translocate with lesion gating, RNAPII off (mirrors
    translocate_with_rnapii when no Pol is present)."""
    for i in range(nlef - 1, -1, -1):
        if lctcf[i] or rctcf[i]:
            prob = 1.0 / lifetime_ctcf
        elif lrnst[i] or rrnst[i]:
            prob = 1.0 / lifetime_rnstalled
        elif lstalled[i] or rstalled[i]:
            prob = 1.0 / lifetime_stalled
        else:
            prob = 1.0 / lifetime
        if nb_random(mt) < prob:
            occupied[lpos[i]] = FREE
            occupied[rpos[i]] = FREE
            if targeted:                 # load_targeted draws the check random (p=0 -> load_one)
                nb_random(mt)
            _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                      i, N, chain_length)
            lrnst[i] = 0; rrnst[i] = 0

    for i in range(nlef):
        if nb_random(mt) < cap[0, lpos[i]]:
            lctcf[i] = 1
        if nb_random(mt) < cap[1, rpos[i]]:
            rctcf[i] = 1
        if lctcf[i] or rctcf[i]:
            if lctcf[i] and nb_random(mt) < rel[0, lpos[i]]:
                lctcf[i] = 0
            if rctcf[i] and nb_random(mt) < rel[1, rpos[i]]:
                rctcf[i] = 0

    for i in range(nlef):
        for side in range(2):
            if side == 0:
                if lctcf[i]:
                    continue
                pos = lpos[i]; d = -1
            else:
                if rctcf[i]:
                    continue
                pos = rpos[i]; d = 1
            target = pos + d
            if target < 0 or target >= N or (pos // chain_length) != (target // chain_length):
                if side == 0:
                    lstalled[i] = 1; lrnst[i] = 0
                else:
                    rstalled[i] = 1; rrnst[i] = 0
                continue
            lidx = lesion_at[target]
            if lidx >= 0:
                ls = les_state[lidx]; lt = les_type[lidx]
                if ls == LESION_REPAIR:
                    stalls = True; evict = 0
                elif lt == LESION_TYPE_A:
                    stalls = (not rnapii_enabled); evict = 1
                else:
                    stalls = False; evict = 0
                if stalls and nb_random(mt) < lesion_block_prob:
                    if side == 0:
                        lstalled[i] = 1; lrnst[i] = evict
                    else:
                        rstalled[i] = 1; rrnst[i] = evict
                    continue
            cell = occupied[target]
            if cell == FREE:
                occupied[pos] = FREE
                occupied[target] = COHESIN
                if side == 0:
                    lpos[i] = target; lstalled[i] = 0; lrnst[i] = 0
                else:
                    rpos[i] = target; rstalled[i] = 0; rrnst[i] = 0
            else:
                if side == 0:
                    lstalled[i] = 1; lrnst[i] = 0
                else:
                    rstalled[i] = 1; rrnst[i] = 0


@njit(cache=True)
def simulate_cohesin_lesion(seed, N, chain_length, nlef, ctcf_cap, ctcf_rel, lesion_state,
                            record_cap, lesion_block_prob, rnapii_enabled, targeted, lifetime,
                            lifetime_stalled, lifetime_rnstalled, lifetime_ctcf, warmup, traj):
    """Cohesin + lesions (RNAPII off). Returns (coh_pos, les_pos, les_state); lesions
    recorded sorted-by-site, -1 padded to ``record_cap`` (= lesion_target, matching
    lef.run, which sizes the record array independent of the active Type-A target).
    ``targeted`` = config uses load_targeted with loading_sites present (draws a check
    random per load; only the targeted_load_prob=0 path -> load_one is supported)."""
    (target, has_p, cdf, has_rm, rate_mult, prerecog_ticks, repair_ticks,
     gene_body_mask, type_a_prob, force_type) = lesion_state
    mt = nb_seed(seed)
    occupied = np.zeros(N, dtype=np.int8)
    lpos = np.zeros(nlef, np.int64); rpos = np.zeros(nlef, np.int64)
    lctcf = np.zeros(nlef, np.int8); rctcf = np.zeros(nlef, np.int8)
    lstalled = np.zeros(nlef, np.int8); rstalled = np.zeros(nlef, np.int8)
    lrnst = np.zeros(nlef, np.int8); rrnst = np.zeros(nlef, np.int8)
    for slot in range(nlef):
        if targeted:
            nb_random(mt)
        _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                  slot, N, chain_length)

    cap_les = record_cap if record_cap > 0 else 1   # >= active target always
    les_site = np.full(cap_les, -1, np.int64)
    les_type = np.full(cap_les, -1, np.int64)
    les_state = np.full(cap_les, -1, np.int64)
    lesion_at = np.full(N, -1, np.int64)
    n_les = 0
    lesion_random = np.empty(cap_les, np.float64)
    lesion_dead = np.empty(cap_les, np.int8)
    candidate_cap = target + (target // 2 if target // 2 >= 4 else 4)
    lesion_candidates = np.empty(candidate_cap if candidate_cap > 0 else 1, np.int64)

    def _tick(n_les):
        n_les = _update_lesions(mt, N, les_site, les_type, les_state, lesion_at, n_les,
                                target, has_p, cdf, has_rm, rate_mult, prerecog_ticks,
                                repair_ticks, gene_body_mask, type_a_prob, force_type,
                                lesion_random, lesion_dead, lesion_candidates)
        _translocate_les(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                         lrnst, rrnst, ctcf_cap, ctcf_rel, lesion_at, les_type, les_state,
                         lesion_block_prob, rnapii_enabled, targeted, nlef, N, chain_length,
                         lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf)
        return n_les

    for _ in range(warmup):
        n_les = _tick(n_les)

    coh = np.empty((traj, nlef, 2), np.int32)
    lp = np.empty((traj, cap_les, 2), np.int32); lp[:] = -1   # [site, type]
    lsr = np.empty((traj, cap_les), np.int32); lsr[:] = -1      # state
    for t in range(traj):
        n_les = _tick(n_les)
        for i in range(nlef):
            coh[t, i, 0] = lpos[i]; coh[t, i, 1] = rpos[i]
        ssort = np.sort(les_site[:n_les])
        for i in range(n_les):
            s = ssort[i]
            k = lesion_at[s]
            lp[t, i, 0] = s; lp[t, i, 1] = les_type[k]; lsr[t, i] = les_state[k]
    return coh, lp, lsr


# ===========================================================================
# Phase 4c: enhancer gating (E-P contact) for the RNAPII cutover.
#
# Genes with requires_enhancer / load_requires_enhancer only initiate-load /
# release-from-pause when an enhancer is in E-P contact (a cohesin loop brackets
# the TSS<->enhancer pair within `tol`). compute_ep_contacts is RNG-free but the
# gate `continue`s BEFORE the load / release draw, so it changes draw order and
# must be reproduced exactly. enhancer_logic codes: 0 additive, 1 any, 2 all,
# 3 synergistic.
# ===========================================================================

_ENH_LOGIC = {"additive": 0, "any": 1, "all": 2, "synergistic": 3}


def build_gene_enh(genes):
    """CSR enhancer table + per-gene gating fields for the kernel."""
    n = len(genes)
    off = np.zeros(n + 1, np.int64)
    flat = []
    req = np.zeros(n, np.int8); load_req = np.zeros(n, np.int8)
    k = np.zeros(n, np.int64); logic = np.zeros(n, np.int8); syn = np.ones(n, np.float64)
    for i, g in enumerate(genes):
        enh = list(getattr(g, "enhancers", ()) or ())
        flat.extend(int(e) for e in enh)
        off[i + 1] = len(flat)
        req[i] = 1 if getattr(g, "requires_enhancer", False) else 0
        load_req[i] = 1 if getattr(g, "load_requires_enhancer", False) else 0
        k[i] = len(enh)
        logic[i] = _ENH_LOGIC.get(getattr(g, "enhancer_logic", "additive"), 0)
        syn[i] = float(getattr(g, "enhancer_synergy", 1.5))
    enh_pos = np.array(flat, np.int64) if flat else np.zeros(0, np.int64)
    return enh_pos, off, req, load_req, k, logic, syn


@njit(cache=True)
def _enh_factor(n_contacts, k, logic, synergy):
    """enhancer_factor: output multiplier from #enhancers in contact (0 = gate closed)."""
    if k == 0:
        return 1.0
    if logic == 2:                       # all
        return 1.0 if n_contacts >= k else 0.0
    if n_contacts <= 0:
        return 0.0
    if logic == 1:                       # any
        return 1.0
    if logic == 3:                       # synergistic
        return float(n_contacts) ** synergy
    return float(n_contacts if n_contacts < k else k)   # additive: min(n, k)


@njit(cache=True)
def _compute_ep_contacts(lpos, rpos, nlef, enh_pos, enh_off, gene_tss, n_genes,
                         tol, chain_length, ep_count,
                         loop_left, loop_right, prefix_max_right):
    """#enhancers of each gene currently bracketed by some same-chain cohesin loop.

    Byte-identical to :func:`plugins.rnapii.compute_ep_contacts`: sort loops by
    left endpoint and carry a chain-local running maximum of the right endpoint.
    For a query ``(lo, hi)``, a contact exists iff the prefix of loops with
    ``L <= lo`` has ``max(R) >= hi``. This replaces the old full LEF scan for
    every enhancer with one binary search, without changing RNG use or results.
    """
    for c in range(nlef):
        lc = lpos[c]
        rc = rpos[c]
        if lc <= rc:
            loop_left[c] = lc
            loop_right[c] = rc
        else:
            loop_left[c] = rc
            loop_right[c] = lc

    order = np.argsort(loop_left[:nlef])
    previous_chain = -1
    running_max = -1
    for j in range(nlef):
        c = order[j]
        chain = loop_left[c] // chain_length
        if chain != previous_chain:
            previous_chain = chain
            running_max = -1
        if loop_right[c] > running_max:
            running_max = loop_right[c]
        prefix_max_right[j] = running_max

    for g in range(n_genes):
        tss = gene_tss[g]
        gchain = tss // chain_length
        chain_lo = gchain * chain_length
        chain_hi = chain_lo + chain_length

        # Locate this chain's loop slice in the globally L-sorted order. Keeping
        # the bounds explicit also preserves the reference behaviour for unusual
        # configs whose enhancer coordinate is outside its promoter's chain.
        left = 0
        right = nlef
        while left < right:
            mid = (left + right) // 2
            if loop_left[order[mid]] < chain_lo:
                left = mid + 1
            else:
                right = mid
        chain_start = left
        left = chain_start
        right = nlef
        while left < right:
            mid = (left + right) // 2
            if loop_left[order[mid]] < chain_hi:
                left = mid + 1
            else:
                right = mid
        chain_end = left

        cnt = 0
        for e in range(enh_off[g], enh_off[g + 1]):
            enh = enh_pos[e]
            lo = (tss if tss < enh else enh) + tol
            hi = (tss if tss > enh else enh) - tol

            # bisect_right(loop_left, lo) within the promoter chain.
            left = chain_start
            right = chain_end
            while left < right:
                mid = (left + right) // 2
                if loop_left[order[mid]] <= lo:
                    left = mid + 1
                else:
                    right = mid
            if left > chain_start and prefix_max_right[left - 1] >= hi:
                cnt += 1
        ep_count[g] = cnt


# ===========================================================================
# Phase 6b: dispatcher -- run the right kernel for a (cfg, args) and return the
# recorded arrays, so lef.run can route eligible configs through Numba.
# ===========================================================================

_DEFAULT_LOAD = ("load_one", "load_targeted")
_DEFAULT_TRANSLOCATE = ("translocate", "translocate_with_rnapii")


def _pname(spec):
    if spec is None:
        return None
    target = getattr(spec, "target", None)        # PluginSpec.target = "module:attr"
    if target is None:
        target = spec if isinstance(spec, str) else getattr(spec, "__name__", str(spec))
    return target.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def numba_eligible(cfg, args):
    """(eligible, reason). Numba supports only the DEFAULT plugin set + the
    regimes validated bit-identical; anything else falls back to the Python engine."""
    p = cfg.plugins
    if _pname(p.load) not in _DEFAULT_LOAD:
        return False, f"custom load plugin {_pname(p.load)}"
    if _pname(p.unload_prob) != "unload_prob":
        return False, "custom unload_prob"
    if _pname(p.capture) != "capture" or _pname(p.release) != "release":
        return False, "custom capture/release"
    if _pname(p.translocate) not in _DEFAULT_TRANSLOCATE:
        return False, f"custom translocate {_pname(p.translocate)}"
    rnapii = p.rnapii_load is not None and p.rnapii_translocate is not None
    if rnapii:
        if _pname(p.rnapii_load) != "load_rnapii":
            return False, "custom rnapii_load"
        if _pname(p.rnapii_translocate) != "stateful_translocate_rnapii":
            return False, "rnapii_translocate not stateful (v1 unsupported)"
    lesion = p.lesion is not None
    if lesion and _pname(p.lesion) != "update_lesions":
        return False, "custom lesion plugin"
    if float(args.get("targeted_load_prob", 0.0)) > 0.0:
        return False, "targeted_load_prob>0 (targeted placement not ported)"
    if (args.get("loading_sites") or []):
        return False, "non-empty loading_sites (targeted-load draw not ported)"
    if float(args.get("rnapii_pause_cohesin_restraint", 1.0)) < 1.0:
        return False, "Tei pause restraint not ported"
    # No memory guard: the engine streams the recording to HDF5 in memory-bounded
    # blocks (see init_chunked / advance_chunked / block_len_for), so it handles
    # arbitrarily large configs without holding the full trajectory in RAM.
    return True, ""


# The production entry point is init_chunked + advance_chunked (driven by lef.run),
# which streams the recording to HDF5 in memory-bounded blocks. The monolithic
# simulate_* drivers below are the bit-identical reference kernels exercised by the
# kernel tests (tests/test_equivalence.py); the chunked engine reuses their exact
# per-tick functions, so the two are equivalent (tests/test_chunking.py).


@njit(cache=True)
def simulate_full(seed, N, chain_length, nlef, cap, gene_soa, gene_enh, ctcf_cap, ctcf_rel,
                  block_prob, lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                  pause_term_global, term_step_prob, stride, stall_prob, push_prob,
                  headon_push_prob, tol, record_cap,
                  les_target, les_has_p, les_cdf, les_has_rm, les_rate_mult, les_prerecog,
                  les_repair, les_gbmask, les_type_a_prob, les_force_type, lesion_block_prob,
                  record_les_cap, warmup, traj):
    """Full engine: cohesin + RNAPII + lesions (the combined regime). Returns
    (coh_pos, rn_pos, rn_state, rn_uid, les_pos[:,:,(site,type)], les_state)."""
    (g_tss, g_tes, g_dir, g_load, g_init, g_release, g_pause_off,
     g_elong, g_term, g_window, g_pause_term) = gene_soa
    (enh_pos, enh_off, g_req_enh, g_load_req, g_k, g_logic, g_syn) = gene_enh
    n_genes = g_tss.shape[0]
    ep_count = np.zeros(n_genes, np.int64)
    ep_left = np.empty(nlef, np.int64)
    ep_right = np.empty(nlef, np.int64)
    ep_prefix_max = np.empty(nlef, np.int64)
    mt = nb_seed(seed)
    occupied = np.zeros(N, dtype=np.int8)
    leg_at = np.full(N, -1, dtype=np.int64)
    rnapii_at = np.full(N, -1, dtype=np.int64)
    lesion_at = np.full(N, -1, dtype=np.int64)

    lpos = np.zeros(nlef, np.int64); rpos = np.zeros(nlef, np.int64)
    lctcf = np.zeros(nlef, np.int8); rctcf = np.zeros(nlef, np.int8)
    lstalled = np.zeros(nlef, np.int8); rstalled = np.zeros(nlef, np.int8)
    lrnst = np.zeros(nlef, np.int8); rrnst = np.zeros(nlef, np.int8)
    lpushed = np.zeros(nlef, np.int8); rpushed = np.zeros(nlef, np.int8)
    for slot in range(nlef):
        _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled, slot, N, chain_length)
    for i in range(nlef):
        leg_at[lpos[i]] = i * 2
        leg_at[rpos[i]] = i * 2 + 1

    rn_pos = np.full(cap, -1, np.int64); rn_gene = np.full(cap, -1, np.int64)
    rn_state = np.full(cap, -1, np.int64); rn_uid = np.full(cap, -1, np.int64)
    rn_lesion_stalled = np.zeros(cap, np.int8)
    rn_dead = np.empty(cap, np.int8)
    n_rn = 0; uid_next = 0

    lcap = record_les_cap if record_les_cap > 0 else 1
    les_site = np.full(lcap, -1, np.int64); les_typ = np.full(lcap, -1, np.int64)
    les_sta = np.full(lcap, -1, np.int64)
    n_les = 0
    lesion_random = np.empty(lcap, np.float64)
    lesion_dead = np.empty(lcap, np.int8)
    candidate_cap = les_target + (les_target // 2 if les_target // 2 >= 4 else 4)
    lesion_candidates = np.empty(candidate_cap if candidate_cap > 0 else 1, np.int64)

    def _tick(n_rn, uid_next, n_les):
        # Order matches _advance_one_step: lesions FIRST, then EP, load, rnapii, cohesin.
        n_les = _update_lesions(mt, N, les_site, les_typ, les_sta, lesion_at, n_les,
                                les_target, les_has_p, les_cdf, les_has_rm, les_rate_mult,
                                les_prerecog, les_repair, les_gbmask, les_type_a_prob,
                                les_force_type, lesion_random, lesion_dead,
                                lesion_candidates)
        _compute_ep_contacts(lpos, rpos, nlef, enh_pos, enh_off, g_tss, n_genes,
                             tol, chain_length, ep_count,
                             ep_left, ep_right, ep_prefix_max)
        n_rn, uid_next = _load_rnapii_c(mt, occupied, rnapii_at, rn_pos, rn_gene, rn_state, rn_uid,
                                        n_rn, uid_next, g_tss, g_dir, g_load, g_pause_off, 1, n_genes,
                                        ep_count, g_load_req, g_k, g_logic, g_syn, rn_lesion_stalled)
        n_rn = _stateful_translocate_c(mt, occupied, rnapii_at, leg_at, rn_pos, rn_gene,
                                       rn_state, rn_uid, n_rn, lpos, rpos, lpushed, rpushed,
                                       lrnst, rrnst, g_tes, g_dir, g_init, g_release, g_pause_off,
                                       g_elong, g_term, g_window, g_pause_term,
                                       pause_term_global, term_step_prob, stride,
                                       stall_prob, push_prob, headon_push_prob, N, chain_length,
                                       ep_count, g_req_enh, g_k, g_logic, g_syn,
                                       rn_lesion_stalled, lesion_at, les_typ, les_sta, True,
                                       rn_dead)
        _translocate_c(mt, occupied, rnapii_at, leg_at, lpos, rpos, lctcf, rctcf,
                       lstalled, rstalled, lrnst, rrnst, lpushed, rpushed,
                       rn_state, ctcf_cap, ctcf_rel, block_prob, nlef, N, chain_length,
                       lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                       lesion_at, les_typ, les_sta, lesion_block_prob, True, True)
        return n_rn, uid_next, n_les

    for _ in range(warmup):
        n_rn, uid_next, n_les = _tick(n_rn, uid_next, n_les)

    coh_pos = np.empty((traj, nlef, 2), np.int32)
    rp = np.empty((traj, record_cap, 2), np.int32); rp[:] = -1
    rs = np.empty((traj, record_cap), np.int32); rs[:] = -1
    ri = np.empty((traj, record_cap), np.int32); ri[:] = -1
    lp = np.empty((traj, lcap, 2), np.int32); lp[:] = -1
    lsr = np.empty((traj, lcap), np.int32); lsr[:] = -1
    for t in range(traj):
        n_rn, uid_next, n_les = _tick(n_rn, uid_next, n_les)
        for i in range(nlef):
            coh_pos[t, i, 0] = lpos[i]; coh_pos[t, i, 1] = rpos[i]
        m = n_rn if n_rn < record_cap else record_cap
        for i in range(m):
            rp[t, i, 0] = rn_pos[i]; rp[t, i, 1] = rn_gene[i]
            rs[t, i] = rn_state[i]; ri[t, i] = rn_uid[i]
        ssort = np.sort(les_site[:n_les])
        for i in range(n_les):
            s = ssort[i]; k = lesion_at[s]
            lp[t, i, 0] = s; lp[t, i, 1] = les_typ[k]; lsr[t, i] = les_sta[k]
    return coh_pos, rp, rs, ri, lp, lsr


# ===========================================================================
# Chunked engine: run the dynamics in BLOCKS, persisting all state across calls,
# so the recording never materialises the full (traj, rec_cap) array in RAM.
# Python (lef.run) drives the blocks and streams each to HDF5 -- like the legacy
# Python engine -- so the Numba engine handles arbitrarily large configs without
# OOM. Reuses the proven per-tick functions, so it is bit-identical to the
# monolithic simulate_* drivers (verified in tests/test_chunking.py).
# ===========================================================================


@njit(cache=True)
def _advance_block(mt, occupied, leg_at, rnapii_at, lesion_at,
                   lpos, rpos, lctcf, rctcf, lstalled, rstalled, lrnst, rrnst, lpushed, rpushed,
                   rn_pos, rn_gene, rn_state, rn_uid, rn_lesion_stalled, rn_dead,
                   les_site, les_typ, les_sta, lesion_random, lesion_dead,
                   lesion_candidates, ep_count, ep_left, ep_right,
                   ep_prefix_max, nstate,
                   g_tss, g_tes, g_dir, g_load, g_init, g_release, g_pause_off,
                   g_elong, g_term, g_window, g_pause_term,
                   enh_pos, enh_off, g_req_enh, g_load_req, g_k, g_logic, g_syn,
                   ctcf_cap, ctcf_rel, block_prob,
                   les_target, les_has_p, les_cdf, les_has_rm, les_rate_mult, les_prerecog,
                   les_repair, les_gbmask, les_type_a_prob, les_force_type, lesion_block_prob,
                   lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                   pause_term_global, term_step_prob, stride, stall_prob, push_prob, headon_push_prob, tol,
                   N, chain_length, nlef, n_genes, has_rnapii, has_lesions, occl,
                   block_len, do_record, record_cap, lcap,
                   coh_buf, rp_buf, rs_buf, ri_buf, lp_buf, ls_buf):
    """Run ``block_len`` ticks; ``nstate`` stores population state and block peak.
    If ``do_record``, fill the (block_len, ...) buffers. Per-tick order matches
    _advance_one_step: lesions, EP, load, rnapii, cohesin."""
    n_rn = nstate[0]; uid_next = nstate[1]; n_les = nstate[2]
    peak_rn = n_rn
    for t in range(block_len):
        if has_lesions:
            n_les = _update_lesions(mt, N, les_site, les_typ, les_sta, lesion_at, n_les,
                                    les_target, les_has_p, les_cdf, les_has_rm, les_rate_mult,
                                    les_prerecog, les_repair, les_gbmask, les_type_a_prob,
                                    les_force_type, lesion_random, lesion_dead,
                                    lesion_candidates)
        if has_rnapii:
            _compute_ep_contacts(lpos, rpos, nlef, enh_pos, enh_off, g_tss, n_genes,
                                 tol, chain_length, ep_count,
                                 ep_left, ep_right, ep_prefix_max)
            n_rn, uid_next = _load_rnapii_c(mt, occupied, rnapii_at, rn_pos, rn_gene, rn_state, rn_uid,
                                            n_rn, uid_next, g_tss, g_dir, g_load, g_pause_off, occl, n_genes,
                                            ep_count, g_load_req, g_k, g_logic, g_syn, rn_lesion_stalled)
            n_rn = _stateful_translocate_c(mt, occupied, rnapii_at, leg_at, rn_pos, rn_gene,
                                           rn_state, rn_uid, n_rn, lpos, rpos, lpushed, rpushed,
                                           lrnst, rrnst, g_tes, g_dir, g_init, g_release, g_pause_off,
                                           g_elong, g_term, g_window, g_pause_term,
                                           pause_term_global, term_step_prob, stride,
                                           stall_prob, push_prob, headon_push_prob, N, chain_length,
                                           ep_count, g_req_enh, g_k, g_logic, g_syn,
                                           rn_lesion_stalled, lesion_at, les_typ, les_sta, has_lesions,
                                           rn_dead)
            _translocate_c(mt, occupied, rnapii_at, leg_at, lpos, rpos, lctcf, rctcf,
                           lstalled, rstalled, lrnst, rrnst, lpushed, rpushed,
                           rn_state, ctcf_cap, ctcf_rel, block_prob, nlef, N, chain_length,
                           lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf,
                           lesion_at, les_typ, les_sta, lesion_block_prob, True, has_lesions)
        elif has_lesions:
            _translocate_les(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                             lrnst, rrnst, ctcf_cap, ctcf_rel, lesion_at, les_typ, les_sta,
                             lesion_block_prob, False, False, nlef, N, chain_length,
                             lifetime, lifetime_stalled, lifetime_rnstalled, lifetime_ctcf)
        else:
            _translocate(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                         ctcf_cap, ctcf_rel, nlef, N, chain_length,
                         lifetime, lifetime_stalled, lifetime_ctcf)

        if n_rn > peak_rn:
            peak_rn = n_rn

        if do_record:
            for i in range(nlef):
                coh_buf[t, i, 0] = lpos[i]; coh_buf[t, i, 1] = rpos[i]
            if has_rnapii:
                m = n_rn if n_rn < record_cap else record_cap
                for i in range(m):
                    rp_buf[t, i, 0] = rn_pos[i]; rp_buf[t, i, 1] = rn_gene[i]
                    rs_buf[t, i] = rn_state[i]; ri_buf[t, i] = rn_uid[i]
            if has_lesions:
                ssort = np.sort(les_site[:n_les])
                mm = n_les if n_les < lcap else lcap
                for i in range(mm):
                    s = ssort[i]; k = lesion_at[s]
                    lp_buf[t, i, 0] = s; lp_buf[t, i, 1] = les_typ[k]; ls_buf[t, i] = les_sta[k]
    nstate[0] = n_rn; nstate[1] = uid_next; nstate[2] = n_les; nstate[3] = peak_rn


@njit(cache=True)
def _init_cohesins(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled,
                   leg_at, nlef, N, chain_length):
    for slot in range(nlef):
        _load_one(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled, slot, N, chain_length)
    for i in range(nlef):
        leg_at[lpos[i]] = i * 2
        leg_at[rpos[i]] = i * 2 + 1


# Cap the in-RAM recording BLOCK buffer (block_len x record_cap) to bound memory;
# the kernel streams blocks to HDF5, so this only limits the transient buffer size.
_BLOCK_BYTES_MAX = 256_000_000


class ChunkedRun:
    """Holds all Numba-engine state across blocks. Built by :func:`init_chunked`;
    advanced by :func:`advance_chunked`. Reproduces the monolithic simulate_*
    drivers bit-for-bit (same seed, same per-tick functions, same RNG order)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def init_chunked(cfg, args):
    """Build the full Numba-engine state for ``cfg`` + topology ``args``."""
    N = int(args["N"]); chain_length = int(args.get("chain_length", N))
    nlef = int(cfg.num_lefs)
    seed = int(cfg.seed) if cfg.seed is not None else 0
    rnapii = cfg.plugins.rnapii_load is not None and cfg.plugins.rnapii_translocate is not None
    lesion = cfg.plugins.lesion is not None
    bp = int(getattr(cfg, "bp_per_site", 250))
    tol = int(args.get("ep_contact_tolerance", max(1, round(2000 / bp))))

    if nlef > (N // chain_length) * (chain_length // 2):
        raise ValueError(
            f"Cannot place {nlef} LEFs on {N // chain_length} chains of "
            f"length {chain_length}: too few free adjacent in-chain pairs"
        )
    if rnapii:
        if int(cfg.max_rnapii) <= 0:
            raise ValueError("max_rnapii must be > 0 when RNAPII is enabled")
        for i, gene in enumerate(args.get("genes", ())):
            sites = (("tss", int(gene.tss)), ("tes", int(gene.tes)))
            for label, site in sites:
                if not (0 <= site < N):
                    raise ValueError(
                        f"Gene {i} {label}={site} is outside the {N}-site lattice"
                    )
            if int(gene.tss) // chain_length != int(gene.tes) // chain_length:
                raise ValueError(
                    f"Gene {i} crosses a chain boundary: tss={gene.tss}, tes={gene.tes}"
                )
            for enhancer in gene.enhancers:
                if not (0 <= int(enhancer) < N):
                    raise ValueError(
                        f"Gene {i} enhancer={enhancer} is outside the {N}-site lattice"
                    )

    mt = nb_seed(seed)
    occupied = np.zeros(N, np.int8)
    leg_at = np.full(N, -1, np.int64); rnapii_at = np.full(N, -1, np.int64)
    lesion_at = np.full(N, -1, np.int64)
    lpos = np.zeros(nlef, np.int64); rpos = np.zeros(nlef, np.int64)
    lctcf = np.zeros(nlef, np.int8); rctcf = np.zeros(nlef, np.int8)
    lstalled = np.zeros(nlef, np.int8); rstalled = np.zeros(nlef, np.int8)
    lrnst = np.zeros(nlef, np.int8); rrnst = np.zeros(nlef, np.int8)
    lpushed = np.zeros(nlef, np.int8); rpushed = np.zeros(nlef, np.int8)
    _init_cohesins(mt, occupied, lpos, rpos, lctcf, rctcf, lstalled, rstalled, leg_at, nlef, N, chain_length)

    cap = N                                   # Pol population upper bound (sites)
    rn_pos = np.full(cap, -1, np.int64); rn_gene = np.full(cap, -1, np.int64)
    rn_state = np.full(cap, -1, np.int64); rn_uid = np.full(cap, -1, np.int64)
    rn_lesion_stalled = np.zeros(cap, np.int8)
    rn_dead = np.empty(cap, np.int8)

    g = build_gene_soa(args["genes"]) if rnapii else build_gene_soa([])
    e = build_gene_enh(args["genes"]) if rnapii else build_gene_enh([])
    n_genes = g[0].shape[0]
    ep_count = np.zeros(n_genes, np.int64)
    ep_left = np.empty(nlef, np.int64)
    ep_right = np.empty(nlef, np.int64)
    ep_prefix_max = np.empty(nlef, np.int64)
    ctcf_cap, ctcf_rel = build_ctcf_arrays(args, N)
    block_prob = build_block_prob(args) if rnapii else np.ones(5, np.float64)

    if lesion:
        ls = build_lesion_state(args)
        lcap = int(args.get("lesion_target", 0))
        active_target = int(ls[0])
        if not (0 <= active_target <= lcap <= N):
            raise ValueError(
                "Invalid lesion capacity: require 0 <= active target <= "
                f"lesion_target <= N, got {active_target}, {lcap}, {N}"
            )
    else:
        ls = (0, False, np.zeros(1), False, np.ones(1), 1, 1, np.zeros(1, np.int8), 0.5, -1)
        lcap = 0
    lcap_alloc = lcap if lcap > 0 else 1
    les_site = np.full(lcap_alloc, -1, np.int64); les_typ = np.full(lcap_alloc, -1, np.int64)
    les_sta = np.full(lcap_alloc, -1, np.int64)
    lesion_random = np.empty(lcap_alloc, np.float64)
    lesion_dead = np.empty(lcap_alloc, np.int8)
    lesion_target = int(ls[0])
    candidate_cap = lesion_target + (
        lesion_target // 2 if lesion_target // 2 >= 4 else 4
    )
    lesion_candidates = np.empty(candidate_cap if candidate_cap > 0 else 1, np.int64)

    record_cap = int(cfg.max_rnapii) * int(cfg.num_chains) if rnapii else 0

    return ChunkedRun(
        mt=mt, occupied=occupied, leg_at=leg_at, rnapii_at=rnapii_at, lesion_at=lesion_at,
        lpos=lpos, rpos=rpos, lctcf=lctcf, rctcf=rctcf, lstalled=lstalled, rstalled=rstalled,
        lrnst=lrnst, rrnst=rrnst, lpushed=lpushed, rpushed=rpushed,
        rn_pos=rn_pos, rn_gene=rn_gene, rn_state=rn_state, rn_uid=rn_uid,
        rn_lesion_stalled=rn_lesion_stalled, rn_dead=rn_dead,
        les_site=les_site, les_typ=les_typ, les_sta=les_sta,
        lesion_random=lesion_random, lesion_dead=lesion_dead,
        lesion_candidates=lesion_candidates, ep_count=ep_count,
        ep_left=ep_left, ep_right=ep_right, ep_prefix_max=ep_prefix_max,
        nstate=np.zeros(4, np.int64), g=g, e=e, ctcf_cap=ctcf_cap, ctcf_rel=ctcf_rel, block_prob=block_prob,
        ls=ls, lesion_block_prob=float(args.get("lesion_block_prob", 1.0)),
        lifetime=float(args["LIFETIME"]), lifetime_stalled=float(args["LIFETIME_STALLED"]),
        lifetime_rnstalled=float(args.get("LIFETIME_RNAPII_STALLED", args["LIFETIME_STALLED"])),
        lifetime_ctcf=float(args.get("LIFETIME_CTCF", args["LIFETIME"])),
        pause_term_global=float(args.get("rnapii_pause_term_prob", 0.0)),
        term_step_prob=float(args.get("rnapii_termination_step_prob", 1.0)),
        stride=int(args.get("rnapii_stride", 1)), stall_prob=float(args.get("rnapii_stall_prob", 0.0)),
        push_prob=float(args.get("rnapii_push_prob", 0.0)), headon_push_prob=float(args.get("rnapii_headon_push_prob", 0.0)),
        tol=tol, N=N, chain_length=chain_length, nlef=nlef, n_genes=n_genes,
        has_rnapii=rnapii, has_lesions=lesion, record_cap=record_cap, lcap=lcap,
        # pause-initiation limit: 1 = promoter-proximal occlusion ON (default).
        occl=1 if args.get("rnapii_promoter_occlusion", True) else 0)


def block_len_for(run, traj):
    """Memory-bounded block length so the per-block recording buffers fit in RAM.
    Accounts for ALL recorded streams -- cohesin, RNAPII, and lesions -- since each
    is sized to its steric/homeostatic ceiling (deliberately oversized so it can
    never truncate), and a large lesion population is as memory-heavy as RNAPII."""
    per_tick = run.nlef * 8                                   # cohesin: (nlef, 2) int32
    if run.has_rnapii and run.record_cap > 0:
        per_tick += run.record_cap * 16                      # rp(8) + rs(4) + ri(4)
    if run.has_lesions and run.lcap > 0:
        per_tick += run.lcap * 12                            # lp(8) + ls(4)
    return max(1, min(traj, _BLOCK_BYTES_MAX // max(1, per_tick)))


def advance_chunked(run, block_len, do_record, compact_recording=False):
    """Advance ``block_len`` ticks. If ``do_record`` return (coh, rp, rs, ri, lp, ls)
    block arrays (rp/.. = None when no RNAPII; lp/ls = None when no lesions).

    ``compact_recording`` narrows only the transient RNAPII buffers. Starting
    from the current population, at most one Pol per gene can load per tick, so
    ``n_rn + block_len * n_genes`` is a strict upper bound on every population
    recorded in this block. The default retains the full-width public API used
    by the equivalence harness; the HDF5 streaming path opts into compact buffers
    and supplies ``-1`` padding through the dataset fill value.
    """
    r = run
    if do_record:
        coh = np.empty((block_len, r.nlef, 2), np.int32)
        if r.has_rnapii:
            rc = r.record_cap
            if compact_recording:
                max_new = int(r.nstate[0]) + int(block_len) * int(r.n_genes)
                rc = min(rc, r.N, max_new)
                # Keep arrays two-dimensional even for a block in which no Pol
                # can exist; the on-disk dataset is likewise kept at width >= 1.
                rc = max(1, rc)
        else:
            rc = 1
        rp = np.empty((block_len, rc, 2), np.int32); rp[:] = -1
        rs = np.empty((block_len, rc), np.int32); rs[:] = -1
        ri = np.empty((block_len, rc), np.int32); ri[:] = -1
        lc = r.lcap if (r.has_lesions and r.lcap > 0) else 1
        lp = np.empty((block_len, lc, 2), np.int32); lp[:] = -1
        ls = np.empty((block_len, lc), np.int32); ls[:] = -1
    else:
        # no recording: pass minimal correctly-shaped dummies to the kernel
        coh = np.empty((1, r.nlef, 2), np.int32)
        rp = np.empty((1, 1, 2), np.int32); lp = np.empty((1, 1, 2), np.int32)
        rs = ri = ls = np.empty((1, 1), np.int32)
    _advance_block(
        r.mt, r.occupied, r.leg_at, r.rnapii_at, r.lesion_at,
        r.lpos, r.rpos, r.lctcf, r.rctcf, r.lstalled, r.rstalled, r.lrnst, r.rrnst, r.lpushed, r.rpushed,
        r.rn_pos, r.rn_gene, r.rn_state, r.rn_uid, r.rn_lesion_stalled, r.rn_dead,
        r.les_site, r.les_typ, r.les_sta, r.lesion_random, r.lesion_dead,
        r.lesion_candidates, r.ep_count, r.ep_left, r.ep_right,
        r.ep_prefix_max, r.nstate,
        *r.g, *r.e, r.ctcf_cap, r.ctcf_rel, r.block_prob, *r.ls, r.lesion_block_prob,
        r.lifetime, r.lifetime_stalled, r.lifetime_rnstalled, r.lifetime_ctcf,
        r.pause_term_global, r.term_step_prob, r.stride, r.stall_prob, r.push_prob, r.headon_push_prob, r.tol,
        r.N, r.chain_length, r.nlef, r.n_genes, r.has_rnapii, r.has_lesions, r.occl,
        block_len, do_record, rc if do_record else (r.record_cap if r.has_rnapii else 1),
        (r.lcap if r.has_lesions else 1),
        coh, rp, rs, ri, lp, ls)
    if not do_record:
        return None
    if r.has_rnapii and int(r.nstate[3]) > int(r.record_cap):
        raise RuntimeError(
            "RNAPII recording capacity exceeded: observed "
            f"{int(r.nstate[3])} live polymerases, but max_rnapii * num_chains "
            f"is {int(r.record_cap)}; increase lef.max_rnapii"
        )
    return (coh, rp if r.has_rnapii else None, rs if r.has_rnapii else None,
            ri if r.has_rnapii else None, lp if r.has_lesions else None, ls if r.has_lesions else None)
