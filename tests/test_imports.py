"""Every module under src/vigia imports with only the core dependencies installed.

The subprocess check also holds when the heavy extras ARE installed: it asserts that
importing the whole package never loads a heavy library, so heavy imports really are
deferred to call time.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import subprocess
import sys

import pytest

import vigia

HEAVY = (
    "cv2",
    "torch",
    "torchvision",
    "timm",
    "albumentations",
    "onnx",
    "onnxruntime",
    "fastapi",
    "uvicorn",
    "mlflow",
    "PIL",
)


def _module_names() -> list[str]:
    names = ["vigia"]
    names += [m.name for m in pkgutil.walk_packages(vigia.__path__, prefix="vigia.")]
    return sorted(names)


MODULES = _module_names()


def test_walk_finds_the_package_layout() -> None:
    expected = {
        "vigia.cli",
        "vigia.config",
        "vigia.models.base",
        "vigia.models.single_frame",
        "vigia.models.temporal",
        "vigia.classical.motion",
        "vigia.data.transforms",
        "vigia.data.figlib",
        "vigia.data.pyrosdis",
        "vigia.train.loop",
        "vigia.serve.export",
        "vigia.serve.runtime",
        "vigia.serve.service",
        "vigia.serve.replay",
    }
    assert expected <= set(MODULES)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_importing_everything_loads_no_heavy_library() -> None:
    code = (
        "import importlib, json, sys\n"
        f"for name in {MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        f"heavy = {HEAVY!r}\n"
        "loaded = sorted({m.split('.')[0] for m in sys.modules} & set(heavy))\n"
        "print(json.dumps(loaded))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], f"heavy libraries loaded at import time: {loaded}"
