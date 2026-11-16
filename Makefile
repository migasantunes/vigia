PYTHON ?= python3
MARKERS := not vision and not train and not serve and not data

.PHONY: install lint format typecheck test test-all check

install:
	uv pip install -e ".[dev]" || $(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy

test:
	$(PYTHON) -m pytest -m "$(MARKERS)"

test-all:
	$(PYTHON) -m pytest

check: lint typecheck test
