"""Typed configuration for the 1D loop-extrusion + transcription engine.

Configuration lives in a YAML file and is materialised into dataclasses. Every
"mechanic" hook is a :class:`PluginSpec` (a ``module:attr`` target plus optional
``kwargs``).

Only the ``lef:`` section of a config is consumed by this 1D engine; any other
top-level sections are ignored, so a config may carry extra sections without
affecting the simulation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

# Plugin targets are fully-qualified ``module:attr`` strings into this package.
_PKG = __name__.rsplit(".", 1)[0]                       # "ner_rnapii_3d"
_PLUGINS = f"{_PKG}.plugins"
DEFAULT_LEF_DYNAMICS = f"{_PLUGINS}.lef_dynamics"
DEFAULT_TOPOLOGY = f"{_PLUGINS}.topology"
DEFAULT_RNAPII = f"{_PLUGINS}.rnapii"
DEFAULT_LESIONS = f"{_PLUGINS}.lesions"


@dataclass
class PluginSpec:
    """Pointer to a callable plus the kwargs it should be invoked with."""

    target: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "PluginSpec":
        if obj is None:
            raise ValueError("PluginSpec value is required")
        if isinstance(obj, PluginSpec):
            return obj
        if isinstance(obj, str):
            return cls(target=obj)
        if isinstance(obj, Mapping):
            if "target" not in obj:
                raise ValueError("Plugin mapping must contain a 'target' key")
            return cls(target=str(obj["target"]), kwargs=dict(obj.get("kwargs", {})))
        raise TypeError(f"Cannot build PluginSpec from {type(obj).__name__}")


def resolve_plugin(spec: PluginSpec) -> Callable[..., Any]:
    """Import ``module:attr`` (or ``module.attr``) and return the callable."""

    target = spec.target
    if ":" in target:
        module_name, attr = target.split(":", 1)
    else:
        module_name, _, attr = target.rpartition(".")
        if not module_name:
            raise ValueError(f"Plugin target '{target}' must be 'module:attr'")
    module = importlib.import_module(module_name)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Plugin '{target}' not found in module '{module.__name__}'") from exc
    if not callable(obj):
        raise TypeError(f"Plugin '{target}' is not callable")
    return obj


@dataclass
class LEFPlugins:
    """Plugin slots that customise 1D mechanics."""

    topology: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_TOPOLOGY}:uniform_tad_topology")
    )
    load: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:load_one")
    )
    unload_prob: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:unload_prob")
    )
    capture: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:capture")
    )
    release: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:release")
    )
    translocate: PluginSpec = field(
        default_factory=lambda: PluginSpec(target=f"{DEFAULT_LEF_DYNAMICS}:translocate")
    )
    # Optional RNAPII slots. When both are set, the engine also drives
    # transcription dynamics each step (and translocate should be
    # ``translocate_with_rnapii`` so cohesin honours RNAPII presence).
    rnapii_load: Optional[PluginSpec] = None
    rnapii_translocate: Optional[PluginSpec] = None
    # Optional lesion (DNA-damage) slot. When set, the engine updates lesions each
    # step; cohesin and RNAPII then honour lesion sites as barriers.
    lesion: Optional[PluginSpec] = None


@dataclass
class LEFConfig:
    """1D loop-extrusion factor (+ RNAPII + lesion) dynamics."""

    # Lattice resolution: bp represented by ONE lattice site. The single source of
    # truth for the bp<->site scale; all genomic distances here are in SITES.
    bp_per_site: int = 250
    chain_length: int = 4000
    num_chains: int = 10
    separation: int = 800            # lattice sites between LEFs (= separation_bp / bp_per_site)
    lifetime: int = 200              # average lifetime (ticks)
    lifetime_stalled: int = 200      # lifetime when stalled (not at CTCF)
    lifetime_ctcf: Optional[int] = None  # lifetime when captured at CTCF; None -> lifetime
    tick_seconds: float = 2.0        # biological seconds per 1D tick
    warmup_steps: int = 0            # dynamics steps to discard before saving
    trajectory_length: int = 100000  # number of LEF dynamics steps to save
    chunk_size: int = 50             # (retained for config compatibility; recording is
                                     #  streamed in memory-bounded blocks by the engine)
    output_path: str = "trajectory/LEFPositions.h5"
    seed: Optional[int] = None       # RNG seed for reproducible 1D dynamics

    # HDF5 compression for the recorded trajectory. The recording arrays are mostly
    # ``-1`` padding (RNAPII/lesion capacity is the steric/homeostatic ceiling) plus
    # spatially-coherent positions, so DEFLATE compresses them ~5x better than lzf
    # with comparable read speed; ``shuffle`` adds ~15% for free. gzip is also a
    # standard HDF5 filter (portable), unlike the h5py-only lzf. Lossless -> the data
    # is byte-identical regardless of these settings.
    compression: str = "gzip"        # "gzip" | "lzf" | "none"
    compression_level: int = 4       # gzip/DEFLATE level 1-9 (4 ~= the size/speed knee)
    shuffle: bool = True             # byte-shuffle filter (helps gzip on int data)

    # Topology kwargs: forwarded to the topology plugin.
    topology_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Maximum concurrent RNAPII count PER CHAIN; the HDF5 ``rnapii_positions``
    # dataset is padded to ``max_rnapii * num_chains``.
    max_rnapii: int = 64

    plugins: LEFPlugins = field(default_factory=LEFPlugins)

    @property
    def num_sites(self) -> int:
        return self.chain_length * self.num_chains

    @property
    def num_lefs(self) -> int:
        return self.num_sites // self.separation

    @property
    def bp_per_chain(self) -> int:
        """Physical chain length in bp (chain_length sites * bp_per_site)."""
        return self.chain_length * self.bp_per_site


# ---------------------------------------------------------------------------
# YAML <-> dataclass plumbing
# ---------------------------------------------------------------------------


def _from_dict(cls, data: Any) -> Any:
    """Recursively build dataclasses from plain mappings."""
    if data is None:
        return cls()
    if cls is PluginSpec:
        return PluginSpec.from_obj(data)
    if not is_dataclass(cls):
        return data
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs: Dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for key, value in data.items():
        if key not in type_hints:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = _resolve_type(cls, key)
        kwargs[key] = _coerce(field_type, value)
    return cls(**kwargs)


def _resolve_type(cls, key: str):
    hints = get_type_hints(cls)
    return hints.get(key)


def _coerce(field_type, value):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if field_type is PluginSpec or (
        origin is None and isinstance(field_type, type) and issubclass(field_type, PluginSpec)
    ):
        return PluginSpec.from_obj(value)
    if origin is Union:  # Optional[X]
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
    if is_dataclass(field_type):
        return _from_dict(field_type, value)
    if origin in (list, List):
        inner = args[0] if args else Any
        return [_coerce(inner, v) for v in value]
    if origin in (dict, Dict):
        return dict(value)
    return value


def load_lef_config(
    path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> LEFConfig:
    """Load the ``lef:`` section of a config YAML into a :class:`LEFConfig`.

    Any other top-level sections are ignored. When ``output_path`` is given it is
    treated as the run output directory and ``output_path``/LEFPositions.h5 is used.
    """
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    lef_raw = raw.get("lef", raw if "bp_per_site" in raw or "plugins" in raw else {})
    cfg = _from_dict(LEFConfig, lef_raw)
    _validate_scale(cfg)
    if output_path is not None:
        cfg.output_path = str(Path(output_path) / "LEFPositions.h5")
    return cfg


def _validate_scale(cfg: LEFConfig) -> LEFConfig:
    bp = int(cfg.bp_per_site)
    if bp <= 0:
        raise ValueError(f"lef.bp_per_site must be > 0, got {bp}")
    if int(cfg.chain_length) <= 0:
        raise ValueError(f"lef.chain_length must be > 0, got {cfg.chain_length}")
    if int(cfg.num_chains) <= 0:
        raise ValueError(f"lef.num_chains must be > 0, got {cfg.num_chains}")
    if int(cfg.separation) <= 0:
        raise ValueError(f"lef.separation must be > 0, got {cfg.separation}")
    if cfg.tick_seconds is None or cfg.tick_seconds <= 0:
        raise ValueError(f"lef.tick_seconds must be > 0, got {cfg.tick_seconds}")
    for name in ("lifetime", "lifetime_stalled"):
        value = getattr(cfg, name)
        if value is None or value <= 0:
            raise ValueError(f"lef.{name} must be > 0, got {value}")
    if cfg.lifetime_ctcf is not None and cfg.lifetime_ctcf <= 0:
        raise ValueError(
            f"lef.lifetime_ctcf must be > 0 or null, got {cfg.lifetime_ctcf}"
        )
    if int(cfg.warmup_steps) < 0:
        raise ValueError(f"lef.warmup_steps must be >= 0, got {cfg.warmup_steps}")
    if int(cfg.trajectory_length) < 0:
        raise ValueError(
            f"lef.trajectory_length must be >= 0, got {cfg.trajectory_length}"
        )
    if int(cfg.max_rnapii) < 0:
        raise ValueError(f"lef.max_rnapii must be >= 0, got {cfg.max_rnapii}")

    # Every initially loaded LEF needs a distinct adjacent in-chain pair. When
    # this capacity is exceeded, _load_one returns False and the old caller left
    # that slot at position zero, silently corrupting occupancy and recordings.
    pair_capacity = int(cfg.num_chains) * (int(cfg.chain_length) // 2)
    if cfg.num_lefs > pair_capacity:
        raise ValueError(
            "lef.separation requests "
            f"{cfg.num_lefs} LEFs, but {cfg.num_chains} chains of length "
            f"{cfg.chain_length} can hold at most {pair_capacity} adjacent pairs"
        )

    has_load = cfg.plugins.rnapii_load is not None
    has_move = cfg.plugins.rnapii_translocate is not None
    if has_load != has_move:
        raise ValueError(
            "lef.plugins.rnapii_load and rnapii_translocate must either both be set "
            "or both be null"
        )
    if has_load and int(cfg.max_rnapii) == 0:
        raise ValueError("lef.max_rnapii must be > 0 when RNAPII is enabled")
    return cfg


@dataclass
class PipelineConfig:
    """Thin view exposing ``.lef`` (a :class:`LEFConfig`). Provided so the analysis
    and sweep scripts can use the ``load_config(...).lef`` access pattern."""

    lef: LEFConfig


def load_config(
    path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> PipelineConfig:
    """Loader returning a :class:`PipelineConfig` whose ``.lef`` is the parsed
    :class:`LEFConfig` (the ``.lef`` accessor used by the scripts)."""
    return PipelineConfig(lef=load_lef_config(path, output_path))
