"""Env prep and warm-up skip path — no FEM install required."""

from __future__ import annotations

import os
import sys

from b3_secfem import prepare_env, warm_up
from b3_secfem.warm import _THREAD_VARS


def test_prepare_env_setdefault_does_not_clobber(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    prepare_env(threads=1)
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_prepare_env_sets_unset_thread_vars(monkeypatch):
    for var in _THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    prepare_env(threads=2)
    for var in _THREAD_VARS:
        assert os.environ[var] == "2"


def test_prepare_env_cache_home_sets_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    prepare_env(cache_home=tmp_path)
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)


def test_warm_up_without_dolfinx_returns_zero(monkeypatch):
    """Force the skip branch even if dolfinx is installed (warm.py:80-83)."""
    monkeypatch.setitem(sys.modules, "dolfinx", None)
    assert warm_up() == 0.0
