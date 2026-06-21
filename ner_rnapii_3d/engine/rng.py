"""Bit-exact reimplementation of NumPy's legacy ``RandomState`` (MT19937).

The Numba engine must reproduce the EXACT random stream that the pure-Python
engine draws from ``numpy.random`` (the global legacy ``RandomState``), so the
two are bit-identical given the same seed. Numba's built-in RNG is a *separate*
state and is not contractually stream-compatible across versions, so instead we
port NumPy's algorithms here and own the stream ourselves.

This reference is written in plain Python (explicit 32/64-bit masking) and is
verified bit-for-bit against ``numpy.random.RandomState`` in
``tests/test_rng.py``. The njit kernel (later phase) reuses
these exact algorithms on a typed state array.

Encoded NumPy facts (legacy RandomState, numpy 1.26):
  * scalar uint32 seed  -> ``init_genrand`` (NOT init_by_array)
  * next_uint32         -> MT19937 generate + temper
  * next_uint64         -> (next_uint32() << 32) | next_uint32()   (hi drawn first)
  * random_sample()     -> (a>>5)*67108864.0 + (b>>6)) / 2**53     (a,b two uint32)
  * randint default dtype is np.int_ == int64 on LP64 -> _rand_int64:
        rng = high-low-1; mask = smallest 2^k-1 >= rng;
        reject next_uint64()&mask until <= rng;  return low + val
  * choice(a, size, p)  -> cdf=cumsum(p); cdf/=cdf[-1];
        idx = searchsorted(cdf, random_sample(size), 'right')
"""

from __future__ import annotations

import numpy as np

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF
_TWO53 = 9007199254740992.0   # 2**53


class MT19937:
    """NumPy-legacy-compatible Mersenne Twister.

    ``MT19937(seed)`` then ``random_sample``/``randint``/``choice`` reproduce
    ``np.random.RandomState(seed)`` draw-for-draw.
    """

    __slots__ = ("mt", "mti")

    def __init__(self, seed: int):
        self.mt = [0] * _N
        self.mti = _N + 1
        self.seed(seed)

    # -- seeding (scalar uint32 seed -> init_genrand) -----------------------
    def seed(self, s: int) -> None:
        s = int(s) & _MASK32
        mt = self.mt
        mt[0] = s
        for i in range(1, _N):
            prev = mt[i - 1]
            mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & _MASK32
        self.mti = _N

    # -- core 32-bit generator (mt19937_next) ------------------------------
    def _next_uint32(self) -> int:
        mt = self.mt
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & _MASK32

    def _next_uint64(self) -> int:
        hi = self._next_uint32()
        lo = self._next_uint32()
        return ((hi << 32) | lo) & _MASK64

    # -- distributions ------------------------------------------------------
    def random_sample(self) -> float:
        a = self._next_uint32() >> 5
        b = self._next_uint32() >> 6
        return (a * 67108864.0 + b) / _TWO53

    @staticmethod
    def _gen_mask(rng: int) -> int:
        mask = rng
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        mask |= mask >> 32
        return mask & _MASK64

    def _bounded(self, rng: int, mask: int) -> int:
        """Masked-rejection draw of a value in [0, rng].

        NumPy's int64 ``randint`` routes ranges that fit in 32 bits
        (``rng <= 0xFFFFFFFF``, i.e. every realistic lattice range) through the
        uint32 path -- one ``next_uint32`` per attempt -- and only uses uint64
        draws for larger ranges. Matching this word-consumption is what keeps the
        stream aligned with NumPy.
        """
        if rng <= _MASK32:
            while True:
                val = self._next_uint32() & mask
                if val <= rng:
                    return val
        while True:
            val = self._next_uint64() & mask
            if val <= rng:
                return val

    def randint(self, low: int, high: int) -> int:
        """``np.random.randint(low, high)`` (default int64 dtype), half-open."""
        rng = high - low - 1
        if rng < 0:
            raise ValueError("low >= high")
        if rng == 0:
            return low
        return low + self._bounded(rng, self._gen_mask(rng))

    def randint_size(self, low: int, high: int, size: int) -> np.ndarray:
        """Vectorized ``np.random.randint(low, high, size=size)`` (int64)."""
        out = np.empty(size, dtype=np.int64)
        rng = high - low - 1
        if rng < 0:
            raise ValueError("low >= high")
        if rng == 0:
            out[:] = low
            return out
        mask = self._gen_mask(rng)
        for i in range(size):
            out[i] = low + self._bounded(rng, mask)
        return out

    def random_array(self, size: int) -> np.ndarray:
        """Vectorized ``np.random.random_sample(size)``."""
        out = np.empty(size, dtype=np.float64)
        for i in range(size):
            out[i] = self.random_sample()
        return out

    def choice_p(self, a: int, size: int, p: np.ndarray) -> np.ndarray:
        """``np.random.choice(a, size=size, p=p)`` (replace=True)."""
        cdf = np.asarray(p, dtype=np.float64).cumsum()
        cdf /= cdf[-1]
        u = self.random_array(size)
        idx = cdf.searchsorted(u, side="right")
        return idx.astype(np.int64)

    def choice_p_scalar(self, a: int, p: np.ndarray) -> int:
        """``np.random.choice(a, p=p)`` with size=None (scalar)."""
        cdf = np.asarray(p, dtype=np.float64).cumsum()
        cdf /= cdf[-1]
        u = self.random_sample()
        return int(cdf.searchsorted(u, side="right"))


# ---------------------------------------------------------------------------
# njit-compatible form (same algorithms, typed state array).
#
# State is a uint32 array of length 625: ``mt[:624]`` are the MT words and
# ``mt[624]`` holds the position index (``mti``). The kernel carries this array
# and draws from it; uint32/uint64 arithmetic is explicit so njit's fixed-width
# wraparound matches NumPy's. Verified bit-identical to NumPy (and to the class
# above) in ``tests/test_rng.py``.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
except Exception:                                       # pragma: no cover
    def _njit(*a, **k):
        def wrap(f):
            return f
        return wrap if (a and callable(a[0]) is False) else (a[0] if a else wrap)

_U32 = np.uint32
_U64 = np.uint64


@_njit(cache=True)
def nb_seed(seed):
    mt = np.empty(625, dtype=np.uint32)
    mt[0] = np.uint32(seed & 0xFFFFFFFF)
    for i in range(1, 624):
        prev = mt[i - 1]
        mt[i] = np.uint32(1812433253) * (prev ^ (prev >> np.uint32(30))) + np.uint32(i)
    mt[624] = np.uint32(624)
    return mt


@_njit(cache=True)
def nb_next_u32(mt):
    mti = mt[624]
    if mti >= np.uint32(624):
        A = np.uint32(0x9908B0DF)
        U = np.uint32(0x80000000)
        L = np.uint32(0x7FFFFFFF)
        one = np.uint32(1)
        for kk in range(624 - 397):
            y = (mt[kk] & U) | (mt[kk + 1] & L)
            mt[kk] = mt[kk + 397] ^ (y >> one) ^ (A if (y & one) else np.uint32(0))
        for kk in range(624 - 397, 623):
            y = (mt[kk] & U) | (mt[kk + 1] & L)
            mt[kk] = mt[kk - 227] ^ (y >> one) ^ (A if (y & one) else np.uint32(0))
        y = (mt[623] & U) | (mt[0] & L)
        mt[623] = mt[396] ^ (y >> one) ^ (A if (y & one) else np.uint32(0))
        mti = np.uint32(0)
    y = mt[mti]
    mt[624] = mti + np.uint32(1)
    y ^= (y >> np.uint32(11))
    y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= (y >> np.uint32(18))
    return y


@_njit(cache=True)
def nb_next_u64(mt):
    hi = np.uint64(nb_next_u32(mt))
    lo = np.uint64(nb_next_u32(mt))
    return (hi << np.uint64(32)) | lo


@_njit(cache=True)
def nb_random(mt):
    a = np.float64(nb_next_u32(mt) >> np.uint32(5))
    b = np.float64(nb_next_u32(mt) >> np.uint32(6))
    return (a * 67108864.0 + b) / 9007199254740992.0


@_njit(cache=True)
def nb_gen_mask(rng):
    # rng is a Python/np int; build smallest 2^k-1 >= rng as uint64
    m = np.uint64(rng)
    m |= m >> np.uint64(1)
    m |= m >> np.uint64(2)
    m |= m >> np.uint64(4)
    m |= m >> np.uint64(8)
    m |= m >> np.uint64(16)
    m |= m >> np.uint64(32)
    return m


@_njit(cache=True)
def nb_randint(mt, low, high):
    """Bit-identical to ``np.random.randint(low, high)`` (int64), half-open."""
    rng = high - low - 1
    if rng == 0:
        return low
    mask = nb_gen_mask(rng)
    if rng <= 0xFFFFFFFF:
        m32 = np.uint32(mask)
        rng32 = np.uint32(rng)
        while True:
            val = nb_next_u32(mt) & m32
            if val <= rng32:
                return low + np.int64(val)
    rng64 = np.uint64(rng)
    while True:
        val64 = nb_next_u64(mt) & mask
        if val64 <= rng64:
            return low + np.int64(val64)
