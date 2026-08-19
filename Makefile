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

.PHONY: help install test test-pure lint format pre-commit check clean install-venv docs-figures docs

help:
	@echo "b3_secfem developer targets:"
	@echo ""
	@echo "  make install      - editable-install b3_mat, airfoilmesh, b3_secfem[dev] into the '$(MAMBA_ENV)' env (run once)"
	@echo "  make test         - run the full test suite (requires prior 'make install')"
	@echo "  make test-pure    - pure-Python tests only (no dolfinx; matches CI)"
	@echo "  make lint         - ruff check"
	@echo "  make format       - ruff format"
	@echo "  make pre-commit   - pre-commit run --all-files (ruff + basic hooks)"
	@echo "  make check        - lint + full test suite + unit-load recovery smoke test"
	@echo "  make clean        - remove caches and the legacy uv venv"
	@echo "  make install-venv - legacy: build a uv venv with --system-site-packages (fragile, see header)"
	@echo "  make docs-figures - regenerate public/figures from rotation3d"
	@echo "  make docs         - serve DocKB (dockb, PORT=3777)"

docs-figures:
	$(MAMBA_RUN) python docs/scripts/gen_frames.py
	$(MAMBA_RUN) python docs/scripts/gen_recovery.py

docs:
	dockb $(or $(PORT),3777)

install:
	@echo "==> Installing local b3_mat + airfoilmesh (siblings) + b3_secfem[dev] into '$(MAMBA_ENV)'"
	$(MAMBA_RUN) pip install -e ../b3_mat -e ../b3_af -e ".[dev]"
	$(MAMBA_RUN) pre-commit install || true

test:
	$(MAMBA_RUN) python -m pytest -q --tb=short

# Pure-Python tests (no dolfinx required) — runnable in any env with the deps.
test-pure:
	$(MAMBA_RUN) python -m pytest -q --tb=short tests/test_materials.py tests/test_rotation3d.py tests/test_adapters.py tests/test_config.py

lint:
	$(MAMBA_RUN) ruff check src tests

format:
	$(MAMBA_RUN) ruff format src tests

pre-commit:
	$(MAMBA_RUN) pre-commit run --all-files

# Local gate: lint, test suite, and airfoil unit-load recovery smoke.
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
