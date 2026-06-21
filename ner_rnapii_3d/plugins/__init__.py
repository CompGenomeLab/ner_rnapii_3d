"""1D-engine plugins.

``topology`` is invoked in Python to build the simulation ``args`` (lattice,
CTCF anchors, genes, lesion fields). The remaining modules (``lef_dynamics``,
``rnapii``, ``lesions``) hold the data classes (Cohesin/Leg/Gene/RNAPII/Lesion)
and the reference per-tick functions; the production tick loop reimplements
their logic inside the Numba kernel, and these stay as the equivalence oracle.
Module names match the original pipeline so legacy config target strings resolve.
"""
