"""CI count-verification gate: README test-count claim vs the live pytest run.

Stdlib-only. Parses this repo's actual README count phrasing -- the Tests
section code fence:

    python -m pytest        # 53 passed

and compares the claimed count against the final summary line of a captured
``pytest`` output file (``[N failed, ]<N> passed[, M skipped] in <T>s``).

PLATFORM-CONDITIONAL SKIPS (root-caused 2026-09-15). This repo has exactly one
platform-conditional test: tests/test_reindex_handle_release.py's
WINDOWS_ONLY guard (os.name != "nt"), which is a deliberate, correct skip --
the hazard it covers is Windows-specific. The README's claim is authored from
a full local run (author's machine is Windows, so that test executes and
passes), but Linux CI always skips it, so CI's live "passed" count is
permanently one less than the README's claim even when nothing has drifted.
Comparing "passed" alone therefore made CI un-satisfiable on Linux forever
(confirmed: CI red on the 08-21 run for a DIFFERENT reason -- a genuinely
stale claim, 149 vs an already-155-test suite -- and red again on the 09-03
HEAD commit purely from this platform gap: claim 246, Linux live "245 passed,
1 skipped").

The fix: the claim represents the TOTAL collected count (passed + skipped),
not a platform-specific pass count, so the gate compares
``live passed + live skipped`` against the claim. A failure or error is still
an automatic drift regardless of whether the totals happen to line up --
this gate never goes silent on a red suite.

Usage:
    python scripts/check_readme_counts.py pytest-output.txt [--readme README.md]

Exit codes:
    0  claim matches the live run's total collected count, and it's not red
    1  drift: the claim disagrees with the live totals, or the run is red
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

# A real pytest summary line is a comma-separated list of "<N> <outcome>"
# clauses ending in "in <time>s" -- outcome fields (failed, passed, skipped,
# warning(s), xfailed, xpassed, deselected, error(s), ...) can appear in any
# order/combination. Anchor on a literal "<N> passed" clause (required --
# this is what keeps the regex from matching unrelated prose, per
# test_does_not_broaden_to_arbitrary_prose) but capture the WHOLE clause list
# around it, not just the passed count, so failed/error/skipped can also be
# read back out of the same matched line.
_KNOWN_FIELD = r"(?:failed|passed|skipped|warnings?|xfailed|xpassed|deselected|errors?)"
SUMMARY_RE = re.compile(
    r"(?:\d+ " + _KNOWN_FIELD + r", )*\d+ passed(?:, \d+ \w+)* in [\d.]+s"
)


def _last_summary_line(text):
    """Return the LAST full pytest summary line (the whole clause list), or
    None if no line anchored on '<N> passed ... in <T>s' is found."""
    matches = SUMMARY_RE.findall(text)
    return matches[-1] if matches else None


def _field_count(line, name):
    """Return the integer count for one outcome field in a summary line, or
    0 if that field is absent (pytest omits a field entirely when it's 0)."""
    m = re.search(r"(\d+) " + name + r"\b", line)
    return int(m.group(1)) if m else 0


def parse_pytest_summary(text):
    """Return the passed count from the LAST pytest summary line, else None."""
    line = _last_summary_line(text)
    if line is None:
        return None
    return _field_count(line, "passed")


def parse_pytest_totals(text):
    """Return a dict of {passed, skipped, failed, errors} from the LAST
    pytest summary line, or None if no summary line is found. Fields not
    present in the line count as 0 (pytest omits zero-valued fields)."""
    line = _last_summary_line(text)
    if line is None:
        return None
    return {
        "passed": _field_count(line, "passed"),
        "skipped": _field_count(line, "skipped"),
        "failed": _field_count(line, "failed"),
        "errors": _field_count(line, "errors?"),
    }


def find_claims(readme_text):
    """Return list of claimed total-collected counts in the README."""
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

    totals = parse_pytest_totals(output_text)
    if totals is None:
        print("MISSING: no pytest summary line found in %s" % args.pytest_output)
        return 2

    claims = find_claims(readme_text)
    if not claims:
        print("MISSING: no test-count claim found in %s" % args.readme)
        return 2

    actual_total = totals["passed"] + totals["skipped"]
    is_red = totals["failed"] > 0 or totals["errors"] > 0

    print(
        "live run: %d passed, %d skipped, %d failed, %d errors "
        "(%d total collected) -- checking %d README claim(s)"
        % (
            totals["passed"],
            totals["skipped"],
            totals["failed"],
            totals["errors"],
            actual_total,
            len(claims),
        )
    )
    if totals["skipped"] > 0:
        print(
            "NOTE: %d test(s) skipped -- if these are platform-conditional "
            "guards (e.g. WINDOWS_ONLY in test_reindex_handle_release.py), "
            "that is expected on this platform and is folded into the total "
            "collected count below, not treated as drift. Run with -rs to "
            "see each skip's reason." % totals["skipped"]
        )

    drift = False
    if is_red:
        drift = True
        print(
            "DRIFT: live run is RED (%d failed, %d errors) -- a gate never "
            "goes silent on a failing suite, regardless of whether the "
            "totals happen to match a README claim."
            % (totals["failed"], totals["errors"])
        )

    for claimed in claims:
        if claimed != actual_total:
            drift = True
            print(
                "DRIFT: README claims %d passed, live run has %d total "
                "collected (%d passed + %d skipped)"
                % (claimed, actual_total, totals["passed"], totals["skipped"])
            )
        else:
            print(
                "OK: README claim of %d passed matches the live run's %d "
                "total collected" % (claimed, actual_total)
            )

    if drift:
        print("FAIL: README count claim has drifted from the live suite.")
        return 1
    print("OK: README count claim matches the live suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
