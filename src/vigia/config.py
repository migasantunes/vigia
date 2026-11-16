"""YAML config loading with ``defaults`` composition and dotted CLI overrides.

A config file is a YAML mapping. An optional top-level ``defaults`` key lists other YAML
files, with paths relative to the including file; they are loaded (recursively, with their
own ``defaults``) and merged in order, and the including file's own keys are merged last.
Merging is deep for mappings; any other value, lists included, is replaced whole by the
later one. The ``defaults`` key itself never appears in the result.

Overrides are ``dotted.key=value`` strings applied after composition. The value is parsed
with ``yaml.safe_load`` (see ``parse_value``), so ``3`` is an int, ``0.001`` and ``1e-3``
floats, ``true`` a bool, ``null`` None, ``[1, 2]`` a list, and anything else a string.
Inside YAML files, write floats with a dot (``0.001`` or ``1.0e-3``): PyYAML reads a bare
``1e-3`` in a file as a string.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_KEY = "defaults"


class ConfigError(ValueError):
    """A config file or override is malformed."""


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new mapping: ``update`` merged into ``base``.

    Keys present in both whose values are both mappings are merged recursively; otherwise
    the value from ``update`` wins (lists are replaced, not concatenated). Inputs are not
    modified, and the result shares no mutable objects with them.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in update.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_SCIENTIFIC = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def parse_value(text: str) -> Any:
    """Parse an override value with ``yaml.safe_load``.

    One exception: PyYAML follows YAML 1.1 and reads ``1e-3`` (no dot in the mantissa)
    as a string; such scientific-notation literals are returned as floats instead.
    """
    parsed = yaml.safe_load(text)
    if isinstance(parsed, str) and _SCIENTIFIC.fullmatch(parsed.strip()):
        return float(parsed)
    return parsed


def _load_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(p) for p in (*stack, resolved))
        raise ConfigError(f"include cycle: {chain}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {resolved}") from None
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{resolved}: top level must be a mapping, got {type(data).__name__}")

    includes = data.pop(DEFAULTS_KEY, [])
    if includes is None:
        includes = []
    if not isinstance(includes, list) or not all(isinstance(i, str) for i in includes):
        raise ConfigError(f"{resolved}: '{DEFAULTS_KEY}' must be a list of relative paths")

    merged: dict[str, Any] = {}
    for include in includes:
        child = _load_file(resolved.parent / include, (*stack, resolved))
        merged = deep_merge(merged, child)
    return deep_merge(merged, data)


def apply_override(config: Mapping[str, Any], override: str) -> dict[str, Any]:
    """Return a copy of ``config`` with one ``dotted.key=value`` override applied.

    Missing intermediate mappings are created. Raises ``ConfigError`` (a ValueError) if the
    string has no ``=``, the key is empty or has an empty segment, or an intermediate key
    holds a value that is not a mapping.
    """
    key, sep, raw_value = override.partition("=")
    key = key.strip()
    if not sep:
        raise ConfigError(f"override must look like key=value, got {override!r}")
    parts = key.split(".")
    if not key or any(not p for p in parts):
        raise ConfigError(f"override has an empty key segment: {override!r}")
    value = parse_value(raw_value)

    result = copy.deepcopy(dict(config))
    node: dict[str, Any] = result
    for depth, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None and part not in node:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            where = ".".join(parts[: depth + 1])
            raise ConfigError(
                f"cannot override {key!r}: {where!r} is a {type(child).__name__}, not a mapping"
            )
        node = child
    node[parts[-1]] = value
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a YAML config, compose its ``defaults``, then apply ``overrides`` in order.

    Raises:
        ConfigError: (a ValueError) on a missing file, a non-mapping document, a malformed
            ``defaults`` list, an include cycle, or a malformed override.
    """
    config = _load_file(Path(path), ())
    for override in overrides:
        config = apply_override(config, override)
    return config


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    """First 12 hex characters of the SHA-256 of the config as canonical JSON.

    Canonical JSON = sorted keys, no whitespace, non-ASCII kept as UTF-8, so the hash does
    not depend on key order. Values must be JSON-serialisable (anything ``yaml.safe_load``
    produces except dates; dates and other objects are rendered with ``str``).
    """
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
