"""Config parsing + plugin resolution tests for ``ner_rnapii_3d.config``.

These exercise the YAML <-> dataclass plumbing and the plugin pointer/resolution
machinery directly (no simulation), asserting concrete expected values:

* ``load_lef_config`` consumes ONLY the ``lef:`` section and ignores other
  top-level sections; unknown keys *inside* ``lef`` are rejected.
* ``output_path`` arg rewrites ``LEFConfig.output_path`` to ``<dir>/LEFPositions.h5``.
* ``PluginSpec.from_obj`` accepts a ``'module:attr'`` string and a
  ``{target, kwargs}`` mapping, is idempotent on a ``PluginSpec``, and rejects
  ``None`` / bad types / a mapping without ``target``.
* ``resolve_plugin`` imports a REAL native target and returns the callable for
  both ``module:attr`` and ``module.attr`` forms, and raises on a missing attr,
  a non-callable target, and a bare (module-less) target.
* ``_validate_scale`` rejects ``bp_per_site <= 0`` and ``tick_seconds <= 0``.
* ``load_config(path).lef`` is a ``LEFConfig``.
* ``LEFConfig`` properties ``num_sites`` / ``num_lefs`` / ``bp_per_chain`` compute.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ner_rnapii_3d.config import (
    LEFConfig,
    PipelineConfig,
    PluginSpec,
    load_config,
    load_lef_config,
    resolve_plugin,
)

# A real, importable, callable target living in the package -- the default
# cohesin loader. Used to prove resolve_plugin returns a live callable.
NATIVE_MODULE = "ner_rnapii_3d.plugins.lef_dynamics"
NATIVE_ATTR = "load_one"
# A real module-level attribute that exists but is NOT callable (a string const).
NON_CALLABLE_TARGET = "ner_rnapii_3d.config:_PKG"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- #
# load_lef_config: section selection
# --------------------------------------------------------------------------- #
def test_load_lef_config_reads_only_lef_section(tmp_path):
    """Only ``lef:`` is consumed; sibling ``polymer:``/``foo:`` sections are ignored."""
    p = _write(
        tmp_path,
        "with_junk.yaml",
        """
lef:
  bp_per_site: 100
  chain_length: 500
  num_chains: 3
  separation: 250
polymer:
  monomer_count: 99999
  nested: {error_radius: 1.0}
foo:
  - 1
  - 2
  - 3
""",
    )
    cfg = load_lef_config(p)
    assert isinstance(cfg, LEFConfig)
    # Values come straight from the lef: block ...
    assert cfg.bp_per_site == 100
    assert cfg.chain_length == 500
    assert cfg.num_chains == 3
    assert cfg.separation == 250
    # ... and the junk sections leave no trace (e.g. defaults preserved).
    assert cfg.lifetime == LEFConfig.lifetime  # untouched class default
    assert cfg.max_rnapii == LEFConfig.max_rnapii


def test_unknown_key_inside_lef_raises(tmp_path):
    """An unrecognised key *inside* ``lef:`` is a hard error (KeyError)."""
    p = _write(
        tmp_path,
        "bad_key.yaml",
        "lef:\n  bp_per_site: 100\n  not_a_real_field: 5\n",
    )
    with pytest.raises(KeyError, match="not_a_real_field"):
        load_lef_config(p)


# --------------------------------------------------------------------------- #
# output_path arg
# --------------------------------------------------------------------------- #
def test_output_path_arg_sets_lefpositions_h5(tmp_path):
    """The ``output_path`` arg is treated as a run dir -> ``<dir>/LEFPositions.h5``."""
    p = _write(tmp_path, "ok.yaml", "lef:\n  bp_per_site: 100\n")
    run_dir = tmp_path / "run42"
    cfg = load_lef_config(p, output_path=str(run_dir))
    assert cfg.output_path == str(run_dir / "LEFPositions.h5")
    assert Path(cfg.output_path).name == "LEFPositions.h5"


def test_output_path_defaults_when_not_given(tmp_path):
    """Without the arg, ``output_path`` keeps the dataclass default."""
    p = _write(tmp_path, "ok2.yaml", "lef:\n  bp_per_site: 100\n")
    cfg = load_lef_config(p)
    assert cfg.output_path == LEFConfig.output_path  # "trajectory/LEFPositions.h5"


# --------------------------------------------------------------------------- #
# PluginSpec.from_obj
# --------------------------------------------------------------------------- #
def test_pluginspec_from_string():
    """A ``'module:attr'`` string becomes a target with empty kwargs."""
    spec = PluginSpec.from_obj("pkg.mod:func")
    assert isinstance(spec, PluginSpec)
    assert spec.target == "pkg.mod:func"
    assert spec.kwargs == {}


def test_pluginspec_from_mapping_with_kwargs():
    """A ``{target, kwargs}`` mapping carries both through verbatim."""
    spec = PluginSpec.from_obj({"target": "pkg.mod:func", "kwargs": {"a": 1, "b": 2}})
    assert spec.target == "pkg.mod:func"
    assert spec.kwargs == {"a": 1, "b": 2}


def test_pluginspec_from_mapping_without_kwargs_defaults_empty():
    """A mapping omitting ``kwargs`` yields an empty (independent) kwargs dict."""
    spec = PluginSpec.from_obj({"target": "pkg.mod:func"})
    assert spec.target == "pkg.mod:func"
    assert spec.kwargs == {}


def test_pluginspec_from_obj_is_idempotent():
    """Passing an existing PluginSpec returns it unchanged (same object)."""
    original = PluginSpec(target="x:y", kwargs={"k": 1})
    assert PluginSpec.from_obj(original) is original


@pytest.mark.parametrize(
    "bad, exc, match",
    [
        (None, ValueError, "required"),
        (123, TypeError, "int"),
        (3.5, TypeError, "float"),
        ([1, 2], TypeError, "list"),
        ({"kwargs": {}}, ValueError, "'target'"),
    ],
)
def test_pluginspec_from_obj_rejects_bad_input(bad, exc, match):
    """None, scalars/lists, and target-less mappings are all rejected."""
    with pytest.raises(exc, match=match):
        PluginSpec.from_obj(bad)


# --------------------------------------------------------------------------- #
# resolve_plugin
# --------------------------------------------------------------------------- #
def test_resolve_plugin_returns_real_callable_colon_form():
    """``module:attr`` resolves to the live native callable."""
    fn = resolve_plugin(PluginSpec(target=f"{NATIVE_MODULE}:{NATIVE_ATTR}"))
    assert callable(fn)
    assert fn.__name__ == NATIVE_ATTR
    # It really is the package's load_one, not some other object.
    import ner_rnapii_3d.plugins.lef_dynamics as ld

    assert fn is ld.load_one


def test_resolve_plugin_dotted_form_matches_colon_form():
    """``module.attr`` resolves to the SAME object as ``module:attr``."""
    colon = resolve_plugin(PluginSpec(target=f"{NATIVE_MODULE}:{NATIVE_ATTR}"))
    dotted = resolve_plugin(PluginSpec(target=f"{NATIVE_MODULE}.{NATIVE_ATTR}"))
    assert dotted is colon


def test_resolve_plugin_missing_attr_raises_importerror():
    """A real module but a nonexistent attribute -> ImportError naming the target."""
    target = f"{NATIVE_MODULE}:no_such_function"
    with pytest.raises(ImportError, match="no_such_function"):
        resolve_plugin(PluginSpec(target=target))


def test_resolve_plugin_non_callable_raises_typeerror():
    """A target pointing at a non-callable module attribute -> TypeError."""
    import ner_rnapii_3d.config as cfg_mod

    # Guard the premise: the target must exist and genuinely be non-callable.
    assert not callable(cfg_mod._PKG)
    with pytest.raises(TypeError, match="not callable"):
        resolve_plugin(PluginSpec(target=NON_CALLABLE_TARGET))


def test_resolve_plugin_bare_target_raises_valueerror():
    """A bare name with no module part is rejected before import."""
    with pytest.raises(ValueError, match="must be 'module:attr'"):
        resolve_plugin(PluginSpec(target="justaname"))


# --------------------------------------------------------------------------- #
# _validate_scale (via load_lef_config)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_bp", [0, -1, -250])
def test_bp_per_site_must_be_positive(tmp_path, bad_bp):
    """``bp_per_site <= 0`` is rejected with a descriptive ValueError."""
    p = _write(tmp_path, f"bp_{bad_bp}.yaml", f"lef:\n  bp_per_site: {bad_bp}\n")
    with pytest.raises(ValueError, match="bp_per_site must be > 0"):
        load_lef_config(p)


@pytest.mark.parametrize("bad_tick", ["0", "-1.0", "-0.5"])
def test_tick_seconds_must_be_positive(tmp_path, bad_tick):
    """``tick_seconds <= 0`` is rejected with a descriptive ValueError."""
    p = _write(tmp_path, f"tick_{bad_tick}.yaml", f"lef:\n  tick_seconds: {bad_tick}\n")
    with pytest.raises(ValueError, match="tick_seconds must be > 0"):
        load_lef_config(p)


def test_positive_scale_passes_validation(tmp_path):
    """A config with positive scale loads cleanly and keeps the given values."""
    p = _write(
        tmp_path,
        "scale_ok.yaml",
        "lef:\n  bp_per_site: 50\n  tick_seconds: 0.25\n",
    )
    cfg = load_lef_config(p)
    assert cfg.bp_per_site == 50
    assert cfg.tick_seconds == 0.25


# --------------------------------------------------------------------------- #
# load_config (.lef accessor)
# --------------------------------------------------------------------------- #
def test_load_config_returns_pipeline_with_lef(tmp_path):
    """``load_config`` returns a PipelineConfig whose ``.lef`` is a LEFConfig."""
    p = _write(tmp_path, "pipe.yaml", "lef:\n  bp_per_site: 100\n  chain_length: 600\n")
    pipe = load_config(p)
    assert isinstance(pipe, PipelineConfig)
    assert isinstance(pipe.lef, LEFConfig)
    assert pipe.lef.chain_length == 600


def test_load_config_threads_output_path(tmp_path):
    """``load_config``'s ``output_path`` is forwarded onto the inner LEFConfig."""
    p = _write(tmp_path, "pipe2.yaml", "lef:\n  bp_per_site: 100\n")
    run_dir = tmp_path / "out"
    pipe = load_config(p, output_path=str(run_dir))
    assert pipe.lef.output_path == str(run_dir / "LEFPositions.h5")


# --------------------------------------------------------------------------- #
# LEFConfig derived properties
# --------------------------------------------------------------------------- #
def test_lefconfig_derived_properties_explicit():
    """num_sites / num_lefs / bp_per_chain follow from the primitive fields."""
    cfg = LEFConfig(
        bp_per_site=100,
        chain_length=500,
        num_chains=3,
        separation=250,
    )
    assert cfg.num_sites == 500 * 3          # chain_length * num_chains == 1500
    assert cfg.num_lefs == 1500 // 250       # num_sites // separation == 6
    assert cfg.bp_per_chain == 500 * 100     # chain_length * bp_per_site == 50000


def test_lefconfig_num_lefs_floor_divides():
    """num_lefs is an integer floor division (drops the remainder)."""
    cfg = LEFConfig(chain_length=10, num_chains=1, separation=3)
    # num_sites == 10; 10 // 3 == 3 (not 3.33).
    assert cfg.num_sites == 10
    assert cfg.num_lefs == 3
    assert isinstance(cfg.num_lefs, int)


def test_lefconfig_defaults_match_documented_scale():
    """The shipped defaults give the documented 40k-site / 50-LEF / 1Mbp-chain scale."""
    cfg = LEFConfig()
    assert cfg.num_sites == 4000 * 10        # 40000
    assert cfg.num_lefs == 40000 // 800      # 50
    assert cfg.bp_per_chain == 4000 * 250    # 1_000_000


# --------------------------------------------------------------------------- #
# round-trip: plugins section parses into typed PluginSpecs
# --------------------------------------------------------------------------- #
def test_plugins_section_parses_into_pluginspecs(tmp_path):
    """A nested ``plugins:`` block yields typed PluginSpecs (string + mapping forms)."""
    p = _write(
        tmp_path,
        "plugins.yaml",
        """
lef:
  bp_per_site: 100
  plugins:
    topology:
      target: ner_rnapii_3d.plugins.topology:uniform_tad_topology
      kwargs:
        some_kw: 7
    load: ner_rnapii_3d.plugins.lef_dynamics:load_one
""",
    )
    cfg = load_lef_config(p)
    # mapping form -> target + kwargs
    assert isinstance(cfg.plugins.topology, PluginSpec)
    assert cfg.plugins.topology.target == "ner_rnapii_3d.plugins.topology:uniform_tad_topology"
    assert cfg.plugins.topology.kwargs == {"some_kw": 7}
    # string form -> target, empty kwargs
    assert isinstance(cfg.plugins.load, PluginSpec)
    assert cfg.plugins.load.target == "ner_rnapii_3d.plugins.lef_dynamics:load_one"
    assert cfg.plugins.load.kwargs == {}
    # optional slots stay None when absent
    assert cfg.plugins.rnapii_load is None
    assert cfg.plugins.lesion is None
    # the parsed default topology really resolves to a live callable
    assert callable(resolve_plugin(cfg.plugins.topology))
