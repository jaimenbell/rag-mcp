"""Console entry point: `rag-mcp` (pip-installed) / `python -m rag_mcp`.

The built wheel previously installed fine but had no way to actually run the
server: no [project.scripts], no rag_mcp/__main__.py, and the operational
run_server.py (repo root) is correctly excluded from the package. These tests
pin the contract for the fix:

  * pyproject.toml declares a console script pointing at a real callable.
  * `python -m rag_mcp` resolves and its --help works with zero config.
  * With no RAG_MCP_* env configured, the entry point fails LOUD -- a clear,
    actionable message on stderr and a non-zero exit -- never a raw traceback,
    and never bakes in a machine-specific default path (this repo was
    secret-scrubbed; config must come from env/args only).
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _clean_env():
    """A subprocess env with no RAG_MCP_* vars set, regardless of the host shell."""
    import os

    env = dict(os.environ)
    for key in list(env):
        if key.startswith("RAG_MCP_"):
            del env[key]
    return env


class TestPyprojectDeclaresConsoleScript:
    def test_rag_mcp_script_declared(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        assert "rag-mcp" in scripts

    def test_script_target_resolves_to_a_real_callable(self):
        """The module:attr target setuptools would wire as the console script
        actually imports and is callable -- not a typo'd path."""
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        target = pyproject["project"]["scripts"]["rag-mcp"]
        module_path, _, attr = target.partition(":")
        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr)
        assert callable(fn)


class TestMainModule:
    def test_dunder_main_module_exists(self):
        assert (REPO_ROOT / "rag_mcp" / "__main__.py").exists()

    def test_module_exposes_main_callable(self):
        mod = importlib.import_module("rag_mcp.__main__")
        assert callable(mod.main)

    def test_python_dash_m_rag_mcp_help_exits_zero(self):
        """`python -m rag_mcp --help` must work with ZERO config -- proves the
        module is runnable at all before any config concern applies."""
        result = subprocess.run(
            [sys.executable, "-m", "rag_mcp", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_clean_env(),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage" in result.stdout.lower()


class TestMissingConfigFailsLoud:
    def test_main_returns_nonzero_without_config(self, monkeypatch, capsys):
        from rag_mcp.__main__ import main

        monkeypatch.delenv("RAG_MCP_CORPUS_ROOT", raising=False)
        monkeypatch.delenv("RAG_MCP_DB_PATH", raising=False)

        rc = main([])

        assert rc != 0
        captured = capsys.readouterr()
        assert "RAG_MCP_CORPUS_ROOT" in captured.err
        assert "Traceback" not in captured.err

    def test_subprocess_missing_config_fails_loud_not_traceback(self):
        """Real end-to-end: spawn `python -m rag_mcp` with no RAG_MCP_* env at
        all. Must exit non-zero with a clear stderr message, never hang
        waiting on stdio and never print a raw Python traceback."""
        result = subprocess.run(
            [sys.executable, "-m", "rag_mcp"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_clean_env(),
            timeout=15,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "RAG_MCP_CORPUS_ROOT" in result.stderr


class TestNoHardcodedMachinePath:
    def test_entrypoint_source_has_no_hardcoded_path(self):
        """Regression guard: the entry point must resolve config from env only.
        A hardcoded default corpus/db path would silently work on the author's
        machine and break -- or worse, leak a local path -- on anyone else's."""
        source = (REPO_ROOT / "rag_mcp" / "__main__.py").read_text(encoding="utf-8")
        for needle in ("C:\\Users", "/home/", "/Users/", "jaime"):
            assert needle not in source, f"hardcoded path fragment {needle!r} found"
