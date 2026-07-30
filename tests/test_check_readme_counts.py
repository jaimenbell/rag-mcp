"""Tests for scripts/check_readme_counts.py (the CI count-verification gate).

The gate compares this README's test-count claim -- the ``# N passed``
comment in the Tests section code fence -- against the live pytest run's
summary line. TDD'd with fixture READMEs covering match / drift / missing.
Exercised via subprocess so the CLI contract (exit codes) is what's tested:
  0 = claim matches the live run
  1 = drift (claim disagrees with the live run)
  2 = missing (no claim found in README, or no summary in pytest output)
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_readme_counts.py"


def _load_gate_module():
    """Import scripts/check_readme_counts.py directly (it's not a package)."""
    spec = importlib.util.spec_from_file_location("check_readme_counts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PYTEST_OUTPUT_OK = ".....................\n53 passed in 3.00s\n"

README_MATCH = """\
# rag-mcp

## Tests
```bash
python -m pytest        # 53 passed
```
"""

README_DRIFT = """\
# rag-mcp

## Tests
```bash
python -m pytest        # 51 passed
```
"""

README_MISSING = """\
# rag-mcp

No test counts are claimed anywhere in this document.
"""


def run_gate(tmp_path, readme_text, pytest_output_text):
    readme = tmp_path / "README.md"
    readme.write_text(readme_text, encoding="utf-8")
    out_file = tmp_path / "pytest-output.txt"
    out_file.write_text(pytest_output_text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(out_file), "--readme", str(readme)],
        capture_output=True,
        text=True,
    )


class TestGateExitCodes:
    def test_match_exits_zero(self, tmp_path):
        result = run_gate(tmp_path, README_MATCH, PYTEST_OUTPUT_OK)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_drift_exits_one(self, tmp_path):
        result = run_gate(tmp_path, README_DRIFT, PYTEST_OUTPUT_OK)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "DRIFT" in result.stdout

    def test_missing_claim_exits_two(self, tmp_path):
        result = run_gate(tmp_path, README_MISSING, PYTEST_OUTPUT_OK)
        assert result.returncode == 2, result.stdout + result.stderr

    def test_missing_summary_exits_two(self, tmp_path):
        result = run_gate(tmp_path, README_MATCH, "no summary line here\n")
        assert result.returncode == 2, result.stdout + result.stderr


class TestParsing:
    def test_summary_with_skips_still_parses_passed(self, tmp_path):
        result = run_gate(tmp_path, README_MATCH, "53 passed, 2 skipped in 4.10s\n")
        assert result.returncode == 0

    def test_failed_run_summary_is_drift_not_match(self, tmp_path):
        # "1 failed, 52 passed" must compare 52 (not 53) against the claim.
        result = run_gate(tmp_path, README_MATCH, "1 failed, 52 passed in 3.50s\n")
        assert result.returncode == 1

    def test_last_summary_line_wins(self, tmp_path):
        output = "quoted docs say 5 passed in 1.00s\n53 passed in 3.00s\n"
        result = run_gate(tmp_path, README_MATCH, output)
        assert result.returncode == 0


class TestSummaryLineShapes:
    """A real pytest summary can carry warning/xfailed/xpassed/deselected/error
    fields alongside 'passed'. The regex must tolerate any of those while
    still anchoring on 'N passed ... in <time>s' and capturing the passed
    count specifically (not some other field's count)."""

    def _parse(self, module, text):
        return module.parse_pytest_summary(text)

    def test_passed_with_warning(self):
        module = _load_gate_module()
        assert self._parse(module, "115 passed, 1 warning in 16.65s\n") == 115

    def test_passed_alone(self):
        module = _load_gate_module()
        assert self._parse(module, "115 passed in 12.3s\n") == 115

    def test_passed_with_skipped(self):
        module = _load_gate_module()
        assert self._parse(module, "110 passed, 6 skipped in 5.0s\n") == 110

    def test_passed_with_skipped_and_warnings(self):
        module = _load_gate_module()
        assert self._parse(module, "100 passed, 2 skipped, 3 warnings in 9.9s\n") == 100

    def test_passed_with_xfailed_and_deselected(self):
        module = _load_gate_module()
        text = "85 passed, 2 xfailed, 1 deselected in 8.2s\n"
        assert self._parse(module, text) == 85

    def test_passed_with_xpassed(self):
        module = _load_gate_module()
        assert self._parse(module, "90 passed, 1 xpassed in 7.1s\n") == 90

    def test_passed_with_error(self):
        module = _load_gate_module()
        assert self._parse(module, "60 passed, 1 error in 4.4s\n") == 60

    def test_failed_before_passed_still_captures_passed(self):
        module = _load_gate_module()
        text = "3 failed, 110 passed, 2 errors in 5.0s\n"
        assert self._parse(module, text) == 110

    def test_gate_end_to_end_with_warning_summary(self, tmp_path):
        # The exact shape that broke the gate on 2026-07-30: a StoreGuardSkipped
        # warning turns "115 passed in 16.65s" into "115 passed, 1 warning in
        # 16.65s", and the old regex only tolerated an optional skipped clause.
        readme = """\
# rag-mcp

## Tests
```bash
python -m pytest        # 115 passed
```
"""
        result = run_gate(tmp_path, readme, "115 passed, 1 warning in 16.65s\n")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_last_summary_line_wins_with_warning_variant(self, tmp_path):
        output = (
            "quoted docs say 5 passed, 1 warning in 1.00s\n"
            "53 passed, 2 warnings in 3.00s\n"
        )
        result = run_gate(tmp_path, README_MATCH, output)
        assert result.returncode == 0

    def test_does_not_broaden_to_arbitrary_prose(self):
        # Sanity check: the fix must not turn this into a regex that matches
        # any old sentence containing "passed" and "in Ns" -- the captured
        # number must still be the one directly attached to "passed".
        module = _load_gate_module()
        text = "we passed the review in 2.00s, then 40 failed in 9.9s\n"
        assert self._parse(module, text) is None
