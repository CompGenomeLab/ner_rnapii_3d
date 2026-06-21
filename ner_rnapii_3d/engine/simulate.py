"""Stage 1 driver: 1D LEF (+ optional RNAPII) dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import List

import h5py
import numpy as np

from ..config import LEFConfig, _validate_scale, resolve_plugin
from ..plugins.lef_dynamics import Cohesin
from ..plugins.rnapii import compute_ep_contacts
from .progress import ProgressMeter, log


def _live_width(buf: np.ndarray) -> int:
    """Leading column count that is ever live across a recording block.

    The kernel packs live entries into the low columns ``[0:n]`` (RNAPII position /
    lesion site ``>= 0``) and leaves the tail as ``-1`` padding. Returns the smallest
    ``W`` such that columns ``[W:]`` are ``-1`` for every tick in the block, so writing
    only ``buf[:, :W]`` and relying on the dataset ``fillvalue=-1`` for the untouched
    tail is byte-identical to writing the full ``record_cap`` width -- it just skips
    compressing the (often >99%) padding. ``buf`` is the ``(block, cap, 2)`` array whose
    column 0 holds the position/site. Because live entries are packed contiguously this
    equals ``(buf[:, :, 0] >= 0).sum(axis=1).max()`` (the old per-block ``max_live``),
    so it also serves as the resize tracker (one scan instead of two)."""
    if buf.shape[0] == 0 or buf.shape[1] == 0:
        return 0
    live_cols = np.flatnonzero((buf[:, :, 0] >= 0).any(axis=0))
    return int(live_cols[-1]) + 1 if live_cols.size else 0


def initial_state(cfg: LEFConfig, args: dict, load_fn) -> tuple[np.ndarray, List[Cohesin]]:
    """Build occupancy + cohesin list."""
    occupied = np.zeros(cfg.num_sites, dtype=np.int8)

    cohesins: List[Cohesin] = []
    for _ in range(cfg.num_lefs):
        load_fn(cohesins, occupied, args)

    # Initialise the cohesin-leg lookup for the RNAPII-aware translocate.
    leg_by_pos: dict = args.setdefault("cohesin_leg_by_pos", {})
    for coh in cohesins:
        leg_by_pos[coh.left.pos] = coh.left
        leg_by_pos[coh.right.pos] = coh.right

    return occupied, cohesins


def _advance_one_step(
    *,
    rnapii_enabled: bool,
    rnapiis: list,
    cohesins: List[Cohesin],
    occupied: np.ndarray,
    args: dict,
    rnapii_load_fn,
    rnapii_translocate_fn,
    translocate_fn,
    unload_fn,
    load_fn,
    capture_fn,
    release_fn,
    lesion_update_fn=None,
) -> None:
    """Advance one 1D dynamics tick for warmup or recording."""
    # Lesions update first: occurrence + repair before cohesin / RNAPII move,
    # so the new lesion landscape is what the polymers see this tick.
    if lesion_update_fn is not None:
        lesion_update_fn(args)

    if rnapii_enabled:
        if args.get("genes"):
            # E-P loop-containment slack: ~2 kb of bp slack, in sites. Default derived from the
            # lattice resolution (not a 1-site=1-kb literal) when topology didn't set it. The
            # value is fixed for the run, so memoise it on args (computed once, not every tick).
            tol = args.get("_ep_tol_cache")
            if tol is None:
                tol = int(args.get("ep_contact_tolerance",
                                   max(1, round(2000 / int(args.get("bp_per_site", 250))))))
                args["_ep_tol_cache"] = tol
            args["current_ep_contacts"] = compute_ep_contacts(
                cohesins,
                args["genes"],
                tolerance=tol,
                chain_length=args.get("chain_length"),
            )
        rnapii_load_fn(rnapiis, occupied, args)
        rnapii_translocate_fn(rnapiis, cohesins, occupied, args)

    translocate_fn(
        cohesins,
        occupied,
        args,
        unload_prob_fn=unload_fn,
        load_fn=load_fn,
        capture_fn=capture_fn,
        release_fn=release_fn,
    )


def run(cfg: LEFConfig) -> Path:
    """Simulate LEF (+ optional RNAPII) dynamics and write ``LEFPositions.h5``."""

    # ``run`` also accepts directly constructed LEFConfig objects, which have not
    # passed through load_lef_config's validation.
    _validate_scale(cfg)

    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))

    plugins = cfg.plugins

    # Only the topology plugin runs in Python (it builds ``args``); every per-tick
    # plugin (load / unload / capture / release / translocate / rnapii / lesion) is
    # executed inside the Numba kernel. numba_eligible() enforces that the configured
    # plugins match the validated default set the kernel reproduces.
    topology_fn = resolve_plugin(plugins.topology)

    rnapii_enabled = (
        plugins.rnapii_load is not None and plugins.rnapii_translocate is not None
    )
    lesion_enabled = plugins.lesion is not None

    args = topology_fn(cfg, **cfg.topology_kwargs)
    args["num_chains"] = cfg.num_chains
    # Lesion logic gates Type-A pre-recognition cohesin stalling on whether real
    # RNAPII is in the model (if so, the stalled Pol II does the blocking).
    args["rnapii_enabled"] = rnapii_enabled

    log.info(
        "[lef] 1D dynamics: N=%d sites, %d LEFs, rnapii=%s, lesions=%s | "
        "warmup=%d steps, trajectory=%d steps",
        cfg.num_sites, cfg.num_lefs, rnapii_enabled, lesion_enabled,
        max(0, int(cfg.warmup_steps)), cfg.trajectory_length,
    )

    # The Numba engine is the production tick loop -- bit-identical to the Python
    # plugins (validated in tests/test_numba_*), and faster. It streams the
    # recording to HDF5 in memory-bounded blocks (init_chunked / advance_chunked),
    # so it handles arbitrarily large configs. The Python plugin functions remain
    # the equivalence oracle (tests) and are imported by polymer/qc/compare; only
    # the per-tick orchestration moved into the kernel. Custom plugins (anything
    # outside the validated default set) are not supported -> a clear error.
    from .kernel import init_chunked, advance_chunked, block_len_for, numba_eligible

    ok, why = numba_eligible(cfg, args)
    if not ok:
        raise ValueError(
            f"This config is not supported by the Numba engine ({why}). The "
            f"loop-extrusion pipeline runs exclusively on the Numba engine; only "
            f"the validated default plugin set is supported."
        )
    run_state = init_chunked(cfg, args)

    warmup_steps = max(0, int(cfg.warmup_steps))
    if warmup_steps:
        warmup_meter = ProgressMeter(warmup_steps, "lef:warmup")
        wstep = max(1, min(warmup_steps, 20000))   # progress granularity only (no recording)
        done = 0
        while done < warmup_steps:
            bl = min(wstep, warmup_steps - done)
            advance_chunked(run_state, bl, do_record=False)
            done += bl
            warmup_meter.update(done)
        warmup_meter.done()

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_lefs = cfg.num_lefs
    traj_len = cfg.trajectory_length

    # max_rnapii is per-chain; total recording width scales with num_chains.
    rnapii_cap = cfg.max_rnapii * cfg.num_chains

    # The kernel records into per-block buffers whose time depth is memory-bounded
    # (block_len_for caps the total recorded-streams bytes per block); each block is
    # streamed straight to HDF5. So peak RAM is one block, never the full trajectory.
    block_len = block_len_for(run_state, traj_len)

    # Storage chunks: whole rows (full width, unsplit last axis) with a time depth
    # matching the write block. h5py's auto-chunking otherwise splits the columns
    # AND the size-2 last axis, so a script-side time-slice read (``ds[a:b]``) has
    # to touch many tiny chunks; whole-row chunks let it decompress whole rows
    # (~20% faster reads, e.g. gen_transcription_metrics). Aligning the time depth
    # to the write block also keeps each ``dset[start:end] = buffer`` write within
    # whole chunks. Data is byte-identical; only the on-disk layout changes.
    _blk = block_len

    def _tchunk(row_bytes: int, cap_bytes: int = 8 << 20) -> int:
        return max(1, min(traj_len, _blk, max(1, cap_bytes // max(1, row_bytes))))

    # The RNAPII datasets are sized to the steric ceiling (rnapii_cap) but only the
    # low columns are ever live (the recorder packs the population into [0:n]); the
    # rest is -1 padding. With whole-row chunks an analysis that needs only the live
    # columns still decompresses the full width. So chunk the RNAPII arrays by COLUMN
    # and resize the column dim down to the actual max live width once the run ends
    # (the dropped columns are all -1 -> lossless). That makes a narrow read ~90x
    # cheaper (e.g. gen_transcription_metrics) and shrinks the file, at ~8% more write.
    _RN_COL = 512

    def _wchunks(cap: int, cell_bytes: int):
        cc = min(cap, _RN_COL)
        td = max(1, min(traj_len, _blk, (8 << 20) // max(1, cc * cell_bytes)))
        return td, cc

    # HDF5 filter pipeline (lossless), applied to every recording dataset. gzip is
    # the default: the -1-padded, position-coherent arrays deflate ~5x smaller than
    # lzf at comparable read speed; shuffle adds ~15% for free. See LEFConfig.
    comp = (getattr(cfg, "compression", "gzip") or "none").lower()
    _filter: dict = {}
    if comp == "gzip":
        _filter["compression"] = "gzip"
        _filter["compression_opts"] = int(getattr(cfg, "compression_level", 4))
    elif comp == "lzf":
        _filter["compression"] = "lzf"
    elif comp != "none":
        raise ValueError(f"lef.compression must be 'gzip', 'lzf' or 'none', got {comp!r}")
    if comp != "none" and getattr(cfg, "shuffle", True):
        _filter["shuffle"] = True

    with h5py.File(out_path, "w") as fh:
        position_storage = dict(_filter)
        if traj_len > 0 and n_lefs > 0:
            position_storage["chunks"] = (_tchunk(n_lefs * 2 * 4), n_lefs, 2)
        dset = fh.create_dataset(
            "positions", shape=(traj_len, n_lefs, 2), dtype=np.int32,
            **position_storage,
        )
        dset_rnapii = None
        dset_states = None
        dset_ids = None
        if rnapii_enabled:
            rn_pos_storage = dict(_filter)
            rn_scalar_storage = dict(_filter)
            if traj_len > 0:
                rn_pos_storage["chunks"] = _wchunks(rnapii_cap, 8) + (2,)
                rn_scalar_storage["chunks"] = _wchunks(rnapii_cap, 1)
            dset_rnapii = fh.create_dataset(
                "rnapii_positions",
                shape=(traj_len, rnapii_cap, 2),
                maxshape=(traj_len, rnapii_cap, 2),
                dtype=np.int32,
                **rn_pos_storage,
                fillvalue=-1,
            )
            dset_states = fh.create_dataset(
                "rnapii_states",
                shape=(traj_len, rnapii_cap),
                maxshape=(traj_len, rnapii_cap),
                dtype=np.int8,
                **rn_scalar_storage,
                fillvalue=-1,
            )
            # Stable per-Pol identity: the live rnapii list is compacted on unload,
            # so a column index is NOT a fixed Pol II. Record each Pol's uid so
            # downstream can reconstruct per-Pol tracks across column shuffles.
            dset_ids = fh.create_dataset(
                "rnapii_ids",
                shape=(traj_len, rnapii_cap),
                maxshape=(traj_len, rnapii_cap),
                dtype=np.int32,
                **(
                    dict(_filter, chunks=_wchunks(rnapii_cap, 4))
                    if traj_len > 0 else _filter
                ),
                fillvalue=-1,
            )

        # Lesion recording capacity = the homeostatic target (N // spacing); the
        # population never exceeds it, so no truncation. Sites, types and states
        # are recorded as parallel -1-padded datasets (cf. RNAPII positions/states).
        lesion_cap = int(args.get("lesion_target", 0))
        lesion_recorded = lesion_enabled and lesion_cap > 0
        dset_lesions = dset_lesion_types = dset_lesion_states = None
        if lesion_recorded:
            lesion_i4_storage = dict(_filter)
            lesion_i1_storage = dict(_filter)
            if traj_len > 0:
                lesion_i4_storage["chunks"] = (_tchunk(lesion_cap * 4), lesion_cap)
                lesion_i1_storage["chunks"] = (_tchunk(lesion_cap), lesion_cap)
            dset_lesions = fh.create_dataset(
                "lesions", shape=(traj_len, lesion_cap), dtype=np.int32,
                fillvalue=-1, **lesion_i4_storage,
            )
            dset_lesion_types = fh.create_dataset(
                "lesion_types", shape=(traj_len, lesion_cap), dtype=np.int8,
                fillvalue=-1, **lesion_i1_storage,
            )
            dset_lesion_states = fh.create_dataset(
                "lesion_states", shape=(traj_len, lesion_cap), dtype=np.int8,
                fillvalue=-1, **lesion_i1_storage,
            )

        # Stream the trajectory in memory-bounded blocks: the kernel advances
        # ``block_len`` ticks and returns that block's recording buffers, which are
        # written straight to HDF5. Peak RAM is one block, so this scales to any
        # trajectory length without holding the full recording in memory.
        rec_meter = ProgressMeter(traj_len, "lef:record")
        start = 0
        max_live_rnapii = 0           # widest live RNAPII population seen (for resize)
        while start < traj_len:
            end = min(start + block_len, traj_len)
            coh_b, rp_b, rs_b, ri_b, lp_b, ls_b = advance_chunked(
                run_state, end - start, do_record=True, compact_recording=True
            )
            dset[start:end] = coh_b
            if dset_rnapii is not None:
                # Live Pols are packed into the low columns [0:n]; the tail is -1
                # padding (>99% of record_cap, which is the steric ceiling). Write
                # only the live columns and let the dataset fillvalue=-1 cover the
                # untouched tail: byte-identical to writing the full width (the cells
                # we skip would have been written as -1), but it skips gzip-ing the
                # padding -- the dominant record-stage cost. w doubles as the resize
                # tracker, so this is also the single live-width scan (was line ~287).
                w = _live_width(rp_b)
                if w:
                    dset_rnapii[start:end, :w] = rp_b[:, :w]
                    dset_states[start:end, :w] = rs_b[:, :w]
                    dset_ids[start:end, :w] = ri_b[:, :w]
                if w > max_live_rnapii:
                    max_live_rnapii = w
            if dset_lesions is not None:
                # lp_b is (block, lesion_cap, 2) = [site, type]; ls_b is [state]. Same
                # live-width narrowing as RNAPII (fillvalue=-1 tail); lesion datasets
                # are not resized, so they keep full lesion_cap width either way.
                wl = _live_width(lp_b)
                if wl:
                    dset_lesions[start:end, :wl] = lp_b[:, :wl, 0]
                    dset_lesion_types[start:end, :wl] = lp_b[:, :wl, 1]
                    dset_lesion_states[start:end, :wl] = ls_b[:, :wl]
            rec_meter.update(end)
            start = end

        rec_meter.done()

        # Drop the never-populated RNAPII columns (all -1, lossless): shrink the
        # column dim to the actual max live width so analyses read only live columns.
        if dset_rnapii is not None:
            w = max(1, max_live_rnapii)
            if w < rnapii_cap:
                dset_rnapii.resize(w, axis=1)
                dset_states.resize(w, axis=1)
                dset_ids.resize(w, axis=1)

        fh.attrs["N"] = cfg.num_sites
        fh.attrs["LEFNum"] = n_lefs
        fh.attrs["chain_length"] = cfg.chain_length
        fh.attrs["num_chains"] = cfg.num_chains
        # Self-describing lattice/time scale so every downstream reader (qc, compare,
        # gen_transcription_metrics, gen_cohesin_barrier_eval, ...) converts sites<->bp from the
        # trajectory itself instead of re-assuming 1 site = 1 kb.
        fh.attrs["bp_per_site"] = int(getattr(cfg, "bp_per_site", 250))
        fh.attrs["tick_seconds"] = float(cfg.tick_seconds)
        fh.attrs["lifetime"] = cfg.lifetime
        fh.attrs["lifetime_stalled"] = cfg.lifetime_stalled
        fh.attrs["separation"] = cfg.separation
        fh.attrs["warmup_steps"] = cfg.warmup_steps
        if cfg.seed is not None:
            fh.attrs["seed"] = int(cfg.seed)
        fh.attrs["rnapii_enabled"] = bool(rnapii_enabled)
        fh.attrs["lesion_enabled"] = bool(lesion_enabled)
        if lesion_recorded:
            fh.attrs["lesion_target"] = lesion_cap
            fh.attrs["lesion_spacing"] = int(args.get("lesion_spacing", 0))
            fh.attrs["lesion_type_a_prob"] = float(args.get("lesion_type_a_prob", 0.5))
            fh.attrs["lesion_prerecognition_ticks"] = int(args.get("lesion_prerecognition_ticks", 0))
            fh.attrs["lesion_repair_ticks"] = int(args.get("lesion_repair_ticks", 0))
            fh.attrs["lesion_tad_size_exponent"] = float(args.get("lesion_tad_size_exponent", 1.0))
            fh.attrs["lesion_tad_repair_exponent"] = float(args.get("lesion_tad_repair_exponent", 0.0))
            fh.attrs["lesion_type_b_enabled"] = bool(args.get("lesion_type_b_enabled", True))
            fh.attrs["lesion_target_a"] = int(args.get("lesion_target_a", 0))
            fh.attrs["lesion_block_prob"] = float(args.get("lesion_block_prob", 0.0))
        if rnapii_enabled:
            fh.attrs["max_rnapii"] = cfg.max_rnapii
            fh.attrs["rnapii_cap"] = rnapii_cap
            fh.attrs["rnapii_pause_term_prob"] = float(args.get("rnapii_pause_term_prob", 0.0))
            genes = args.get("genes", [])
            if genes:
                gene_arr = np.array(
                    [(g.gene_id, g.tss, g.tes, g.direction, g.load_prob,
                      (getattr(g, "gene_class", "") or "").encode("ascii", "ignore")[:16])
                     for g in genes],
                    dtype=[("gene_id", "i4"), ("tss", "i4"),
                           ("tes", "i4"), ("direction", "i4"),
                           ("load_prob", "f4"), ("gene_class", "S16")],
                )
                fh.create_dataset("genes", data=gene_arr)

        # Ground-truth CTCF anchor table (per-TAD oriented boundaries). Written
        # whenever the topology plugin exposes it (convergent topologies); absent
        # for uniform/symmetric topologies, so downstream analysis falls back to
        # config-derived anchors. Tiny -> no compression.
        anchors = args.get("ctcf_anchors")
        if anchors:
            anchor_arr = np.array(
                anchors,
                dtype=[
                    ("abs_position", "i4"),
                    ("chain_position", "i4"),
                    ("chain_index", "i4"),
                    ("side", "i1"),
                    ("strength", "f4"),
                    ("tad_index", "i4"),
                    ("edge", "i1"),
                ],
            )
            fh.create_dataset("ctcf_anchors", data=anchor_arr)
            fh.attrs["boundary_model"] = "per_tad_oriented"
            fh.attrs["ctcf_anchor_schema_version"] = 1
            fh.attrs["ctcf_anchors_per_chain"] = int(
                len(anchor_arr) // max(cfg.num_chains, 1)
            )

    return out_path
