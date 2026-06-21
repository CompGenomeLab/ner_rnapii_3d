"""HDF5 compression backend: the filter pipeline is driven by config, and is
lossless (the recorded data is byte-identical regardless of the filter)."""
import copy

import h5py
import numpy as np
import pytest

import ner_rnapii_3d as ner


def _run(cfg, out_dir, **over):
    cfg = copy.deepcopy(cfg)
    cfg.seed = 5
    cfg.warmup_steps = 10
    cfg.trajectory_length = 40
    cfg.output_path = str(out_dir / "LEFPositions.h5")
    for k, v in over.items():
        setattr(cfg, k, v)
    return ner.run(cfg)


def _filters(path):
    with h5py.File(path) as f:
        return {k: (f[k].compression, f[k].compression_opts, bool(f[k].shuffle))
                for k in f if isinstance(f[k], h5py.Dataset) and f[k].chunks}


def test_default_is_gzip_shuffle(cohesin_cfg, tmp_path):
    """Out of the box every recording dataset is gzip level 4 + shuffle."""
    p = _run(cohesin_cfg, tmp_path)
    for name, (comp, opts, shuf) in _filters(p).items():
        assert comp == "gzip", f"{name}: {comp}"
        assert opts == 4 and shuf is True, f"{name}: opts={opts} shuffle={shuf}"


def test_rnapii_datasets_compressed(rnapii_cfg, tmp_path):
    """The wide -1-padded RNAPII arrays (the disk hog) carry the filter too."""
    p = _run(rnapii_cfg, tmp_path)
    flt = _filters(p)
    for name in ("rnapii_positions", "rnapii_states", "rnapii_ids"):
        assert flt[name][0] == "gzip" and flt[name][2] is True


def test_compression_level_respected(cohesin_cfg, tmp_path):
    p = _run(cohesin_cfg, tmp_path, compression_level=9)
    assert all(opts == 9 for _, opts, _ in _filters(p).values())


def test_lzf_backend(cohesin_cfg, tmp_path):
    p = _run(cohesin_cfg, tmp_path, compression="lzf")
    for comp, _opts, _shuf in _filters(p).values():
        assert comp == "lzf"


def test_none_backend(cohesin_cfg, tmp_path):
    p = _run(cohesin_cfg, tmp_path, compression="none")
    for comp, _opts, shuf in _filters(p).values():
        assert comp is None and shuf is False


def test_invalid_backend_raises(cohesin_cfg, tmp_path):
    with pytest.raises(ValueError):
        _run(cohesin_cfg, tmp_path, compression="zstd-not-supported")


@pytest.mark.parametrize("backend", ["gzip", "lzf", "none"])
def test_lossless(rnapii_cfg, tmp_path, backend):
    """Compression must not change a single recorded value: positions and the
    RNAPII arrays are byte-identical to an uncompressed run of the same seed."""
    ref = _run(rnapii_cfg, tmp_path / "ref", compression="none")
    got = _run(rnapii_cfg, tmp_path / backend, compression=backend,
               **({"compression_level": 6} if backend == "gzip" else {}))
    with h5py.File(ref) as fr, h5py.File(got) as fg:
        assert set(fr) == set(fg)
        for k in fr:
            if isinstance(fr[k], h5py.Dataset):
                assert np.array_equal(fr[k][()], fg[k][()]), f"{k} changed under {backend}"


def test_gzip_smaller_than_lzf(rnapii_cfg, tmp_path):
    """On the padded RNAPII recording, gzip must beat lzf on disk."""
    import os
    lzf = os.path.getsize(_run(rnapii_cfg, tmp_path / "lzf", compression="lzf"))
    gz = os.path.getsize(_run(rnapii_cfg, tmp_path / "gz", compression="gzip"))
    assert gz < lzf, f"gzip {gz} !< lzf {lzf}"
