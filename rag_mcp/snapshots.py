"""Consecutive-snapshot chunk de-duplication for dated daily-report series.

The defect this exists for
--------------------------
The corpus contains machine-generated DAILY SNAPSHOT series -- one file per day,
same template, e.g. ``Bots/fleet-health-2026-07-23.md``. When the thing being
reported does not move, the day's file repeats yesterday's paragraphs verbatim.

Measured 2026-07-30 on the live vault: the query
``options-bot 23 percent drawdown recovery honest loss`` returned 9 of its 10
top slots as BYTE-IDENTICAL copies of one ``## RED Bots`` block, because
options-bot sat RED at an unchanged 23.00% for weeks. The document that actually
explained the root cause was pushed to rank 16. Across 2026-07-17..07-31 that
block takes only FIVE distinct values in FIFTEEN files.

Blanket-excluding the series was rejected: these files are the only per-date
citable evidence for "what did fleet health say on date X".

What this module does instead
-----------------------------
Within a series, in date order, a chunk is SUPPRESSED only when the immediately
preceding snapshot in that series contained a byte-identical chunk under the same
heading. The FIRST occurrence is always embedded and keeps its own date as its
``source``; the later repeats are not embedded, and instead extend the surviving
chunk's ``repeat_dates`` metadata. So per-date citability survives: a hit says
"this was the state on 2026-07-23, and identically on 07-24..07-27".

Because the comparison is against the PREVIOUS snapshot only (not a global
content set), an A -> B -> A sequence keeps BOTH A instances. A value that
returns after changing is a new fact, not a repeat.

SCOPE (stated here, with the pattern, on purpose)
-------------------------------------------------
This applies ONLY to files that satisfy ALL of:
  1. filename ends in ``-YYYY-MM-DD`` before the extension (:data:`SERIES_RE`);
  2. at least :data:`MIN_SERIES_LENGTH` such files share the same directory AND
     the same stem -- one dated file is a one-off report, not a series;
  3. the repeat is byte-identical, under an identical heading, and adjacent in
     the series.
Anything else -- ordinary notes, leading-date filenames, a lone dated file, a
near-identical-but-changed block -- is untouched. Measured against the live
vault this selects 13 series / ~4.8k chunks of a ~50.7k-chunk corpus, and
suppresses ~846 of them; it fires on zero non-snapshot files.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# Trailing ISO date, e.g. "fleet-health-2026-07-23". Leading-date names
# ("2026-05-12 CPI Day Playbook") are deliberately NOT matched: those are
# hand-written notes, not a generated series.
SERIES_RE = re.compile(r"^(?P<stem>.+)-(?P<date>\d{4}-\d{2}-\d{2})$")

# A stem needs at least this many dated files in one directory to count as a
# series. Two is a coincidence; three is a cadence.
MIN_SERIES_LENGTH = 3

# Above this many repeat dates, store a compact range instead of the full list
# so one long-unchanged block cannot bloat a metadata row without bound.
MAX_REPEAT_DATES_LISTED = 60


def parse_series(rel: str) -> tuple[str, str] | None:
    """``("Bots/fleet-health", "2026-07-23")`` for a dated snapshot, else None.

    *rel* is a corpus-relative POSIX path. The series key includes the directory
    so two same-named series in different folders never merge.
    """
    head, _, name = rel.rpartition("/")
    stem, dot, _ext = name.rpartition(".")
    if not dot:  # no extension at all
        stem = name
    m = SERIES_RE.match(stem)
    if not m:
        return None
    key = m.group("stem")
    return (f"{head}/{key}" if head else key, m.group("date"))


def group_series(
    rels: Iterable[str], *, min_len: int = MIN_SERIES_LENGTH
) -> dict[str, list[str]]:
    """Group dated files into series, dropping anything shorter than *min_len*.

    Each series' file list is returned in date order (ties broken by path), which
    is the order "consecutive" is defined against.
    """
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in rels:
        parsed = parse_series(rel)
        if parsed is None:
            continue
        key, date = parsed
        buckets[key].append((date, rel))
    out: dict[str, list[str]] = {}
    for key, dated in buckets.items():
        if len(dated) < min_len:
            continue
        out[key] = [rel for _date, rel in sorted(dated)]
    return out


def chunk_key(heading: str | None, text: str) -> str:
    """Identity of a chunk's CONTENT: same heading + byte-identical body.

    Deliberately excludes ``chunk_index`` -- an edit earlier in the file shifts
    every later index, and a block that merely moved is still the same block.
    """
    h = hashlib.sha256()
    h.update((heading or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class SnapshotPlan:
    """Which chunks of which snapshot files to embed, and what they stand for."""

    # rel -> chunk indices to embed. A rel present here with all indices is a
    # series file that simply had nothing to collapse.
    kept: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # rel -> {surviving chunk index -> dates of later snapshots that repeated it}
    repeats: dict[str, dict[int, tuple[str, ...]]] = field(default_factory=dict)
    # rel -> the file's own snapshot date
    dates: dict[str, str] = field(default_factory=dict)
    suppressed: int = 0

    def is_snapshot(self, rel: str) -> bool:
        return rel in self.kept

    def repeat_dates(self, rel: str, index: int) -> tuple[str, ...]:
        return self.repeats.get(rel, {}).get(index, ())

    def span_signature(self, rel: str) -> str:
        """Stable fingerprint of this file's surviving-chunk spans.

        Lets ingest tell whether an UNCHANGED file's metadata still describes the
        world -- a survivor's span grows every day the repeat continues, and that
        growth must reach the index even though the file itself never changed.
        """
        per = self.repeats.get(rel)
        if not per:
            return ""
        parts = [f"{i}:{','.join(per[i])}" for i in sorted(per)]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


EMPTY_PLAN = SnapshotPlan()


def build_plan(
    series: Mapping[str, Sequence[str]],
    keys_by_rel: Mapping[str, Sequence[str]],
) -> SnapshotPlan:
    """Decide, per series file, which chunk indices to embed.

    Args:
        series: series key -> file rels IN DATE ORDER (see :func:`group_series`).
        keys_by_rel: rel -> that file's chunk keys, in chunk order. A rel missing
            here (unreadable / empty / no chunks) breaks the chain: the next file
            in the series is compared against nothing and keeps everything, which
            is the safe direction.
    """
    plan = SnapshotPlan()
    for rels in series.values():
        # key -> queue of (owner_rel, owner_index) carried from the previous
        # snapshot. A suppressed chunk carries its ORIGINAL owner forward, so a
        # run of N identical days all point at the first day, not at day N-1.
        carried: dict[str, list[tuple[str, int]]] = {}
        for rel in rels:
            keys = keys_by_rel.get(rel)
            if keys is None:
                carried = {}
                continue
            parsed = parse_series(rel)
            date = parsed[1] if parsed else ""
            plan.dates[rel] = date
            available = {k: list(v) for k, v in carried.items()}
            owners: dict[str, list[tuple[str, int]]] = defaultdict(list)
            kept: list[int] = []
            for index, key in enumerate(keys):
                pool = available.get(key)
                if pool:
                    owner_rel, owner_index = pool.pop(0)
                    plan.repeats.setdefault(owner_rel, {}).setdefault(owner_index, [])
                    plan.repeats[owner_rel][owner_index].append(date)
                    owners[key].append((owner_rel, owner_index))
                    plan.suppressed += 1
                else:
                    # Either genuinely new, or a within-file duplicate beyond the
                    # count yesterday had -- keep it rather than over-collapse.
                    kept.append(index)
                    owners[key].append((rel, index))
            plan.kept[rel] = tuple(kept)
            carried = owners
    # Freeze the date lists so a plan cannot be mutated after it is handed out.
    plan.repeats = {
        rel: {i: tuple(d) for i, d in per.items()} for rel, per in plan.repeats.items()
    }
    return plan


def format_repeat_dates(dates: Sequence[str]) -> str:
    """Metadata-safe rendering of the dates a surviving chunk also stood for."""
    if not dates:
        return ""
    if len(dates) <= MAX_REPEAT_DATES_LISTED:
        return ",".join(dates)
    return f"{dates[0]}..{dates[-1]} ({len(dates)} snapshots)"
