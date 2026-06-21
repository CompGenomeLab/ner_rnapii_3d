"""Bit-exact correctness of the MT19937 port (``ner_rnapii_3d.engine.rng``).

The engine owns its random stream so that the Python reference and the Numba
kernel are byte-identical given the same seed (see the engine docstring). That
contract rests entirely on this port reproducing NumPy's *legacy*
``numpy.random.RandomState`` (MT19937) draw-for-draw. This module pins that down:

  1. The reference :class:`~ner_rnapii_3d.engine.rng.MT19937` class reproduces
     ``np.random.RandomState`` BYTE-FOR-BYTE across several seeds for
     ``random_sample`` (scalar + vector), ``randint`` over small ranges and over
     ranges above ``2**32``, and ``choice`` with a probability vector ``p``.
  2. The ``@njit`` free-functions (``nb_seed`` / ``nb_next_u32`` / ``nb_next_u64``
     / ``nb_random`` / ``nb_randint``) produce the SAME stream as the reference
     class and as NumPy, driven through tiny njit wrappers.

All comparisons are EXACT integer / float equality, not approximate -- this is the
foundation of the engine's determinism.
"""
from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from ner_rnapii_3d.engine import rng


SEEDS = [0, 1, 7, 42, 123456, 4294967295]  # incl. seed == 2**32 - 1 (max uint32)

# A non-uniform probability vector for choice() (deliberately unnormalised so the
# port's internal ``cdf /= cdf[-1]`` is exercised against NumPy's own).
P_VEC = np.array([0.1, 0.2, 0.05, 0.4, 0.25], dtype=np.float64)
P_UNNORM = np.array([1.0, 3.0, 0.5, 2.0, 0.0, 4.5], dtype=np.float64)

# A range whose span exceeds 2**32, forcing randint down NumPy's uint64 path
# (two uint32 words per attempt) rather than the uint32 fast path.
BIG_LOW, BIG_HIGH = 0, (1 << 40) + 12345
# A range that fits in 32 bits but is wider than a byte (uint32 fast path).
MID_LOW, MID_HIGH = -5, 1_000_003


# --------------------------------------------------------------------------- #
# njit wrappers: drive the @njit free-functions from a tiny compiled loop so the
# stream is produced inside nopython mode (exactly as the kernel consumes it).
# Compiled once at import; cache=False keeps them independent of the engine cache.
# --------------------------------------------------------------------------- #
@njit(cache=False)
def _nb_draw_u32(mt, n):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = rng.nb_next_u32(mt)
    return out


@njit(cache=False)
def _nb_draw_u64(mt, n):
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = rng.nb_next_u64(mt)
    return out


@njit(cache=False)
def _nb_draw_random(mt, n):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rng.nb_random(mt)
    return out


@njit(cache=False)
def _nb_draw_randint(mt, low, high, n):
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = rng.nb_randint(mt, low, high)
    return out


# =========================================================================== #
# 1. Reference MT19937 class  ==  numpy.random.RandomState  (byte-for-byte)
# =========================================================================== #
@pytest.mark.parametrize("seed", SEEDS)
def test_random_sample_scalar_matches_numpy(seed):
    """``random_sample()`` scalar draws are bit-identical to legacy RandomState."""
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    drawn = [ref.random_sample() for _ in range(64)]
    expected = [rs.random_sample() for _ in range(64)]
    # exact float64 equality (no tolerance)
    assert drawn == expected
    # sanity: the draws are genuine [0,1) floats, not a constant
    assert all(0.0 <= x < 1.0 for x in drawn)
    assert len(set(drawn)) == len(drawn)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [1, 17, 624, 625])  # incl. an MT-block boundary (624)
def test_random_sample_vector_matches_numpy(seed, n):
    """Vectorised ``random_array(n)`` matches ``RandomState.random_sample(n)`` exactly,
    including across the 624-word Mersenne refill boundary."""
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = ref.random_array(n)
    expected = rs.random_sample(n)
    assert got.dtype == np.float64
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("low,high", [(0, 10), (3, 4), (MID_LOW, MID_HIGH)])
def test_randint_small_range_matches_numpy(seed, low, high):
    """``randint(low, high)`` over ranges that fit in 32 bits is bit-identical to
    legacy RandomState (default int64 dtype, half-open)."""
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = [ref.randint(low, high) for _ in range(50)]
    expected = [int(rs.randint(low, high)) for _ in range(50)]
    assert got == expected
    assert all(low <= v < high for v in got)


@pytest.mark.parametrize("seed", SEEDS)
def test_randint_above_2_pow_32_matches_numpy(seed):
    """``randint`` over a range wider than 2**32 routes through NumPy's uint64
    rejection path (two uint32 words per attempt) and stays bit-identical."""
    assert (BIG_HIGH - BIG_LOW - 1) > 0xFFFFFFFF  # really exercises the u64 path
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = [ref.randint(BIG_LOW, BIG_HIGH) for _ in range(50)]
    expected = [int(rs.randint(BIG_LOW, BIG_HIGH)) for _ in range(50)]
    assert got == expected
    assert all(BIG_LOW <= v < BIG_HIGH for v in got)


@pytest.mark.parametrize("seed", SEEDS)
def test_randint_size_vector_matches_numpy(seed):
    """Vectorised ``randint_size`` matches ``RandomState.randint(..., size=...)``."""
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = ref.randint_size(0, 97, 40)
    expected = rs.randint(0, 97, size=40)
    assert got.dtype == np.int64
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [P_VEC, P_UNNORM], ids=["normalised", "unnormalised"])
def test_choice_vector_with_p_matches_numpy(seed, p):
    """``choice(a, size, p)`` (replace=True) matches legacy RandomState exactly,
    including when ``p`` is not pre-normalised."""
    a = len(p)
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = ref.choice_p(a, 32, p)
    # NumPy's choice requires a normalised p, so normalise for the reference call
    expected = rs.choice(a, size=32, p=p / p.sum())
    assert got.dtype == np.int64
    assert np.array_equal(got, expected)
    assert got.min() >= 0 and got.max() < a


@pytest.mark.parametrize("seed", SEEDS)
def test_choice_scalar_with_p_matches_numpy(seed):
    """``choice(a, p=p)`` with ``size=None`` (scalar) matches legacy RandomState."""
    a = len(P_VEC)
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = [ref.choice_p_scalar(a, P_VEC) for _ in range(32)]
    expected = [int(rs.choice(a, p=P_VEC)) for _ in range(32)]
    assert got == expected
    assert all(0 <= v < a for v in got)


# =========================================================================== #
# 2. @njit free-functions  ==  reference class / numpy  (same stream)
# =========================================================================== #
@pytest.mark.parametrize("seed", SEEDS)
def test_nb_seed_state_matches_reference(seed):
    """``nb_seed`` builds the same 625-word state as the reference class seeding
    (624 MT words + the position index)."""
    mt = rng.nb_seed(seed)
    ref = rng.MT19937(seed)
    assert mt.dtype == np.uint32
    assert mt.shape == (625,)
    assert list(mt[:624].astype(np.uint64)) == [int(x) for x in ref.mt]
    assert int(mt[624]) == ref.mti  # both == 624 right after seeding


@pytest.mark.parametrize("seed", SEEDS)
def test_nb_next_u32_matches_reference_class(seed):
    """The njit 32-bit generator yields the same uint32 words as the reference class,
    spanning the first Mersenne refill (>624 draws)."""
    mt = rng.nb_seed(seed)
    ref = rng.MT19937(seed)
    got = _nb_draw_u32(mt, 700)
    expected = [ref._next_uint32() for _ in range(700)]
    assert [int(x) for x in got] == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_nb_next_u64_matches_reference_class(seed):
    """The njit 64-bit draw (hi uint32 first, then lo) matches the reference class."""
    mt = rng.nb_seed(seed)
    ref = rng.MT19937(seed)
    got = _nb_draw_u64(mt, 64)
    expected = [ref._next_uint64() for _ in range(64)]
    assert [int(x) for x in got] == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_nb_random_matches_numpy(seed):
    """The njit ``random_sample`` matches legacy RandomState float64 draws exactly."""
    mt = rng.nb_seed(seed)
    rs = np.random.RandomState(seed)
    got = _nb_draw_random(mt, 64)
    expected = rs.random_sample(64)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("low,high", [(0, 10), (MID_LOW, MID_HIGH)])
def test_nb_randint_small_range_matches_numpy(seed, low, high):
    """The njit ``randint`` over <=32-bit ranges matches legacy RandomState exactly."""
    mt = rng.nb_seed(seed)
    rs = np.random.RandomState(seed)
    got = _nb_draw_randint(mt, low, high, 50)
    expected = np.array([rs.randint(low, high) for _ in range(50)], dtype=np.int64)
    assert np.array_equal(got, expected)
    assert got.min() >= low and got.max() < high


@pytest.mark.parametrize("seed", SEEDS)
def test_nb_randint_above_2_pow_32_matches_numpy(seed):
    """The njit ``randint`` over a >2**32 range uses the uint64 path and stays
    bit-identical to legacy RandomState (and to the reference class)."""
    mt = rng.nb_seed(seed)
    ref = rng.MT19937(seed)
    rs = np.random.RandomState(seed)
    got = _nb_draw_randint(mt, BIG_LOW, BIG_HIGH, 50)
    expected = np.array([rs.randint(BIG_LOW, BIG_HIGH) for _ in range(50)], dtype=np.int64)
    ref_vals = np.array([ref.randint(BIG_LOW, BIG_HIGH) for _ in range(50)], dtype=np.int64)
    assert np.array_equal(got, expected)
    assert np.array_equal(got, ref_vals)


# =========================================================================== #
# 3. Cross-check: a mixed draw sequence keeps class and njit in lock-step.
# =========================================================================== #
@pytest.mark.parametrize("seed", [0, 42, 4294967295])
def test_njit_and_class_word_consumption_stay_aligned(seed):
    """A mixed sequence of float/int draws consumes the SAME number of underlying
    uint32 words in both forms, so the raw stream stays aligned afterwards.

    This guards the word-consumption contract (e.g. randint's u32-vs-u64 path
    choice): if either form burned a different number of words for the same draw,
    the trailing raw uint32 stream would diverge.
    """
    # reference class: do some distribution draws, then read raw words
    ref = rng.MT19937(seed)
    ref.random_sample()                  # 2 u32 words
    ref.randint(0, 10)                    # >=1 u32 word (fast path)
    ref.randint(BIG_LOW, BIG_HIGH)        # >=1 u64 = 2 u32 words (slow path)
    ref.random_array(3)                   # 6 u32 words
    ref_tail = [ref._next_uint32() for _ in range(16)]

    # njit form: replay the identical draw sequence, then read raw words
    mt = rng.nb_seed(seed)
    rng.nb_random(mt)
    rng.nb_randint(mt, 0, 10)
    rng.nb_randint(mt, BIG_LOW, BIG_HIGH)
    for _ in range(3):
        rng.nb_random(mt)
    nb_tail = [int(x) for x in _nb_draw_u32(mt, 16)]

    assert nb_tail == ref_tail
