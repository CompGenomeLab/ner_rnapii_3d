"""ner_rnapii_3d -- 1D loop-extrusion + RNAPII + lesion simulation engine.

Given a YAML config, the engine simulates cohesin loop extrusion, RNAPII
transcription and DNA-lesion dynamics on a 1D lattice and writes a
``LEFPositions.h5`` trajectory. The dynamics run on a Numba kernel with an
embedded MT19937 port, so a given seed yields a byte-identical, version-robust
trajectory.
"""

from .config import (
    LEFConfig,
    LEFPlugins,
    PipelineConfig,
    PluginSpec,
    load_config,
    load_lef_config,
    resolve_plugin,
)
from .engine.simulate import run

__all__ = [
    "LEFConfig",
    "LEFPlugins",
    "PipelineConfig",
    "PluginSpec",
    "load_config",
    "load_lef_config",
    "resolve_plugin",
    "run",
]
