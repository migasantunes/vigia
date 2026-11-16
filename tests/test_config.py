from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from vigia.classical.motion import MotionConfig
from vigia.config import (
    ConfigError,
    apply_override,
    deep_merge,
    load_config,
    parse_value,
    resolved_config_hash,
)
from vigia.eval.alerting import AlertConfig
from vigia.models.single_frame import SingleFrameConfig
from vigia.models.temporal import TemporalConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- composition -------------------------------------------------------------------------


def test_defaults_merge_in_order_and_file_wins(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"x": 1, "y": 1, "z": 1})
    _write(tmp_path / "b.yaml", {"y": 2, "z": 2})
    top = _write(tmp_path / "top.yaml", {"defaults": ["a.yaml", "b.yaml"], "z": 3})
    assert load_config(top) == {"x": 1, "y": 2, "z": 3}


def test_defaults_are_relative_to_the_including_file(tmp_path: Path) -> None:
    _write(tmp_path / "base" / "leaf.yaml", {"leaf": True})
    _write(tmp_path / "base" / "mid.yaml", {"defaults": ["leaf.yaml"], "mid": True})
    top = _write(tmp_path / "exp" / "top.yaml", {"defaults": ["../base/mid.yaml"]})
    assert load_config(top) == {"leaf": True, "mid": True}


def test_deep_merge_of_mappings(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"train": {"lr": 0.1, "epochs": 5, "opt": {"name": "sgd"}}})
    top = _write(
        tmp_path / "top.yaml",
        {"defaults": ["a.yaml"], "train": {"lr": 0.01, "opt": {"momentum": 0.9}}},
    )
    assert load_config(top) == {
        "train": {"lr": 0.01, "epochs": 5, "opt": {"name": "sgd", "momentum": 0.9}}
    }


def test_lists_are_replaced_not_concatenated(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"keys": ["camera_id", "station_id"]})
    top = _write(tmp_path / "top.yaml", {"defaults": ["a.yaml"], "keys": ["station_id"]})
    assert load_config(top) == {"keys": ["station_id"]}


def test_mapping_replaced_by_scalar_and_back() -> None:
    assert deep_merge({"a": {"b": 1}}, {"a": 2}) == {"a": 2}
    assert deep_merge({"a": 2}, {"a": {"b": 1}}) == {"a": {"b": 1}}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": [1]}}
    update = {"a": {"c": 2}}
    merged = deep_merge(base, update)
    merged["a"]["b"].append(99)
    assert base == {"a": {"b": [1]}}
    assert update == {"a": {"c": 2}}


def test_empty_file_and_null_defaults(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    top = _write(tmp_path / "top.yaml", {"defaults": None, "a": 1})
    assert load_config(tmp_path / "empty.yaml") == {}
    assert load_config(top) == {"a": 1}


def test_include_cycle_is_detected(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"defaults": ["b.yaml"]})
    _write(tmp_path / "b.yaml", {"defaults": ["a.yaml"]})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_self_include_is_a_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"defaults": ["a.yaml"]})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_diamond_include_is_not_a_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "shared.yaml", {"s": 1})
    _write(tmp_path / "l.yaml", {"defaults": ["shared.yaml"], "l": 1})
    _write(tmp_path / "r.yaml", {"defaults": ["shared.yaml"], "r": 1})
    top = _write(tmp_path / "top.yaml", {"defaults": ["l.yaml", "r.yaml"]})
    assert load_config(top) == {"s": 1, "l": 1, "r": 1}


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    top = _write(tmp_path / "top.yaml", {"defaults": ["nope.yaml"]})
    with pytest.raises(ConfigError, match="not found"):
        load_config(top)


@pytest.mark.parametrize("bad", [[1, 2], "just a string", 3])
def test_non_mapping_document(tmp_path: Path, bad: Any) -> None:
    path = _write(tmp_path / "bad.yaml", bad)
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


@pytest.mark.parametrize("bad", ["a.yaml", [1], {"a": "b.yaml"}])
def test_malformed_defaults(tmp_path: Path, bad: Any) -> None:
    path = _write(tmp_path / "top.yaml", {"defaults": bad})
    with pytest.raises(ConfigError, match="defaults"):
        load_config(path)


# --- overrides ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2", 2),
        ("-3", -3),
        ("0.001", 0.001),
        ("1e-3", 1e-3),
        ("1.5E+2", 150.0),
        ("true", True),
        ("false", False),
        ("null", None),
        ("~", None),
        ("resnet34", "resnet34"),
        ("'2'", "2"),
        ("[1, 2]", [1, 2]),
        ("", None),
    ],
)
def test_parse_value(text: str, expected: Any) -> None:
    value = parse_value(text)
    assert value == expected
    assert type(value) is type(expected)


def test_overrides_are_applied_after_composition(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"train": {"lr": 0.1}, "eval": {"policy": {"k": 1, "n": 3}}})
    top = _write(tmp_path / "top.yaml", {"defaults": ["a.yaml"]})
    cfg = load_config(top, ["train.lr=0.001", "eval.policy.k=2", "name=run1"])
    assert cfg == {"train": {"lr": 0.001}, "eval": {"policy": {"k": 2, "n": 3}}, "name": "run1"}


def test_later_override_wins() -> None:
    assert apply_override(apply_override({}, "a=1"), "a=2") == {"a": 2}


def test_override_value_may_contain_equals() -> None:
    assert apply_override({}, "note=a=b") == {"note": "a=b"}


def test_override_creates_intermediate_mappings() -> None:
    assert apply_override({"x": 1}, "a.b.c=true") == {"x": 1, "a": {"b": {"c": True}}}


def test_override_does_not_mutate_input() -> None:
    cfg = {"a": {"b": 1}}
    apply_override(cfg, "a.b=2")
    assert cfg == {"a": {"b": 1}}


def test_override_can_replace_a_mapping_leaf_with_a_scalar() -> None:
    assert apply_override({"a": {"b": 1}}, "a=3") == {"a": 3}


@pytest.mark.parametrize(
    ("config", "override", "match"),
    [
        ({"train": {"lr": 0.1}}, "train.lr.x=1", "not a mapping"),
        ({"keys": [1, 2]}, "keys.0=5", "not a mapping"),
        ({"a": None}, "a.b=1", "not a mapping"),
        ({}, "no_equals_sign", "key=value"),
        ({}, "=1", "empty key"),
        ({}, "a..b=1", "empty key"),
        ({}, "a.=1", "empty key"),
    ],
)
def test_override_errors(config: dict[str, Any], override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_override(config, override)


# --- hash --------------------------------------------------------------------------------


def test_hash_is_stable_under_key_order() -> None:
    a = {"train": {"lr": 0.1, "epochs": 5}, "model": {"kind": "x"}}
    b = {"model": {"kind": "x"}, "train": {"epochs": 5, "lr": 0.1}}
    assert resolved_config_hash(a) == resolved_config_hash(b)
    assert len(resolved_config_hash(a)) == 12
    int(resolved_config_hash(a), 16)


def test_hash_changes_with_content() -> None:
    assert resolved_config_hash({"a": 1}) != resolved_config_hash({"a": 2})
    assert resolved_config_hash({"a": 1}) != resolved_config_hash({"a": 1.5})


def test_hash_known_value() -> None:
    # sha256(b'{"a":1}')
    assert resolved_config_hash({"a": 1}) == "015abd7f5cc5"


# --- the repository's own configs ---------------------------------------------------------

ALL_YAML = sorted(CONFIGS.rglob("*.yaml"))
EXPERIMENTS = sorted((CONFIGS / "experiment").glob("*.yaml"))


def test_configs_exist() -> None:
    assert ALL_YAML
    assert {p.name for p in EXPERIMENTS} >= {"p1_single_frame.yaml", "p1_temporal.yaml"}


@pytest.mark.parametrize("path", ALL_YAML, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_every_config_parses(path: Path) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, dict)
    assert cfg


def _model_kwargs(model: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in model.items() if k != "kind"}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_experiments_compose_and_match_the_dataclasses(path: Path) -> None:
    cfg = load_config(path)
    assert "defaults" not in cfg
    for section in ("data", "model", "eval", "train", "experiment"):
        assert section in cfg, section
    model_cls = {"single_frame": SingleFrameConfig, "temporal": TemporalConfig}
    cls = model_cls[cfg["model"]["kind"]]
    kwargs = _model_kwargs(cfg["model"])
    assert set(kwargs) == {f.name for f in dataclasses.fields(cls)}
    cls(**kwargs)
    AlertConfig(**cfg["eval"]["policy"])
    assert cfg["eval"]["max_gap_s"] == 180
    assert isinstance(cfg["train"]["seed"], int)


def test_experiment_override_through_cli_syntax() -> None:
    cfg = load_config(CONFIGS / "experiment" / "p1_temporal.yaml", ["model.context_frames=4"])
    assert TemporalConfig(**_model_kwargs(cfg["model"])).context_frames == 4


def test_motion_config_defaults_are_valid() -> None:
    MotionConfig()
