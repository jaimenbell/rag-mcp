"""CI count-verification gate: README test-count claim vs the live pytest run.

Stdlib-only. Parses this repo's actual README count phrasing -- the Tests
section code fence:

    python -m pytest        # 53 passed

and compares the claimed count against the final summary line of a captured
``pytest`` output file (``[N failed, ]<N> passed[, M skipped] in <T>s``).

Usage:
    python scripts/check_readme_counts.py pytest-output.txt [--readme README.md]

Exit codes:
    0  claim matches the live run
    1  drift: the claim disagrees with the live run
    2  missing: no claim found in the README, or no summary in the output
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# "python -m pytest        # 53 passed" -- the comment is the claim.
CLAIM_RE = re.compile(r"pytest\s*#\s*(\d+)\s+passed")
# A real pytest summary can carry other comma-separated outcome fields
# (warning(s), skipped, xfailed, xpassed, deselected, error(s), ...) in any
# order/combination around "passed" -- tolerate any number of them while
# still anchoring on the passed count itself and requiring the line end in
# "in <time>s".
SUMMARY_RE = re.compile(r"(\d+) passed(?:, \d+ \w+)* in [\d.]+s")


def parse_pytest_summary(text):
    """Return the passed count from the LAST pytest summary line, else None."""
    matches = SUMMARY_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def find_claims(readme_text):
    """Return list of claimed passed counts in the README."""
    return [int(m.group(1)) for m in CLAIM_RE.finditer(readme_text)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytest_output", help="file containing pytest output")
    parser.add_argument(
        "--readme",
        default=str(REPO_ROOT / "README.md"),
        help="README to check (default: repo README.md)",
    )
    args = parser.parse_args(argv)

    output_text = Path(args.pytest_output).read_text(encoding="utf-8")
    readme_text = Path(args.readme).read_text(encoding="utf-8")

    actual_passed = parse_pytest_summary(output_text)
    if actual_passed is None:
        print("MISSING: no pytest summary line found in %s" % args.pytest_output)
        return 2

    claims = find_claims(readme_text)
    if not claims:
        print("MISSING: no test-count claim found in %s" % args.readme)
        return 2

    print(
        "live run: %d passed -- checking %d README claim(s)"
        % (actual_passed, len(claims))
    )
    drift = False
    for claimed in claims:
        if claimed != actual_passed:
            drift = True
            print(
                "DRIFT: README claims %d passed, live run has %d"
                % (claimed, actual_passed)
            )
        else:
            print("OK: README claim of %d passed matches the live run" % claimed)

    if drift:
        print("FAIL: README count claim has drifted from the live suite.")
        return 1
    print("OK: README count claim matches the live suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
