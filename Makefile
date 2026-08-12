# Makefile for b3_secfem
#
# Convenience targets for development.
#
# IMPORTANT: The full test suite requires fenicsx / dolfinx. We use a
# pre-built micromamba environment (`b3secfem`) where dolfinx, petsc4py,
# and numpy are mutually ABI-compatible (numpy 2.x + petsc 3.25 + dolfinx 0.10).
#
# Typical first-time setup:
#   micromamba create -n b3secfem -c conda-forge fenics-dolfinx petsc4py
#   make install
#   make test
#
# The previous `uv venv --system-site-packages` path is kept under
# `install-venv` for users without micromamba, but is fragile: pip-installed
# numpy in the venv can shadow the system numpy that apt's petsc4py was
# compiled against, breaking imports.

MAMBA_ENV ?= b3secfem
MAMBA_RUN ?= micromamba run -n $(MAMBA_ENV)

.PHONY: help install test test-pure smoke lint format check clean install-venv

PURE_TESTS := tests/test_materials.py tests/test_rotation3d.py tests/test_config.py \
	tests/test_public_api.py tests/test_cli.py tests/test_common.py

help:
	@echo "b3_secfem developer targets:"
	@echo ""
	@echo "  make install      - editable-install b3_mat, airfoilmesh, b3_secfem[dev] into the '$(MAMBA_ENV)' env (run once)"
	@echo "  make test         - run the full test suite (requires prior 'make install')"
	@echo "  make test-pure    - run only the pure-Python tests (no dolfinx needed)"
	@echo "  make smoke        - alias of test-pure (CI / pre-commit gate without fenicsx)"
	@echo "  make lint         - run ruff linter"
	@echo "  make format       - run ruff formatter"
	@echo "  make check        - lint + full test suite + unit-load recovery smoke test"
	@echo "  make clean        - remove caches and the legacy uv venv"
	@echo "  make install-venv - legacy: build a uv venv with --system-site-packages (fragile, see header)"

install:
	@echo "==> Installing local b3_mat + airfoilmesh (siblings) + b3_secfem[dev] into '$(MAMBA_ENV)'"
	$(MAMBA_RUN) pip install -e ../b3_mat -e ../b3_af -e ".[dev]"

test:
	$(MAMBA_RUN) python -m pytest -q --tb=short

# Pure-Python tests (no dolfinx required) — runnable in any env with the deps.
test-pure:
	$(MAMBA_RUN) python -m pytest -q --tb=short $(PURE_TESTS)

smoke: test-pure

lint:
	$(MAMBA_RUN) ruff check src tests

format:
	$(MAMBA_RUN) ruff format src tests

# Pre-commit gate: lint, test suite, and the airfoil unit-load recovery
# example whose final algebraic identity ||R @ Gamma - I||_inf must be at
# machine precision (asserts the recover_unit_load_strains formula).
check: lint test
	$(MAMBA_RUN) python examples/airfoil_unit_load_stress.py

clean:
	rm -rf .venv .ruff_cache .pytest_cache __pycache__ */__pycache__
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Legacy uv-venv path (system-site-packages + apt-installed fenicsx). Kept
# because the original setup worked on machines without micromamba, but is
# vulnerable to numpy ABI shadowing — see comment at the top.
# ---------------------------------------------------------------------------
VENV := .venv
SYSTEM_PYTHON ?= /usr/bin/python3

$(VENV)/pyvenv.cfg:
	uv venv --python $(SYSTEM_PYTHON) --system-site-packages $(VENV)

install-venv: $(VENV)/pyvenv.cfg
	uv pip install -e ../b3_mat -e ../b3_af -e ".[dev]"
