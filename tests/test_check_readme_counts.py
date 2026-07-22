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

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_readme_counts.py"

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
