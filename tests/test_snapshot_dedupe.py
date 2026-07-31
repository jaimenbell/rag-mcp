"""Consecutive-snapshot de-duplication: repeats collapse, changes survive, dates hold.

The defect being pinned (measured on the live vault 2026-07-30): the query
"options-bot 23 percent drawdown recovery honest loss" returned 9 of its 10 top
slots as byte-identical copies of one `## RED Bots` block, repeated across ~13
consecutive daily snapshot files, burying the document that explained the root
cause at rank 16.

The failure mode these tests exist to prevent is OVER-collapsing: losing the
ability to answer "what did this say on date X" is the exact reason blanket
exclusion of the series was rejected. So every collapse assertion here is paired
with a citability assertion.

CLOCK DISCIPLINE: every date in this file is derived from the hardcoded
`FIRST_DAY` below. Nothing reads the wall clock -- a suite whose corpus depends
on today's date passes until the day it does not.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rag_mcp.ingest import ingest
from rag_mcp.search import search_knowledge
from rag_mcp.snapshots import (
    MIN_SERIES_LENGTH,
    build_plan,
    chunk_key,
    format_repeat_dates,
    group_series,
    parse_series,
)
from rag_mcp.store import HashEmbedder, VectorStore

# Frozen, never datetime.now(). The corpus these tests build must be identical
# on every run and on every machine.
FIRST_DAY = datetime(2026, 7, 17)

RED_A = "- **options-bot**: cumulative_drawdown[RED]: max 30d drawdown 23.00% >= cap 12.0%"
RED_B = "- **options-bot**: cumulative_drawdown[RED]: max 30d drawdown 31.50% >= cap 12.0%"
STABLE_NOTE = "Fleet posture unchanged. No new alerts were raised in this cycle."


def day(offset: int) -> str:
    """ISO date `offset` days after the frozen FIRST_DAY."""
    return (FIRST_DAY + timedelta(days=offset)).strftime("%Y-%m-%d")


def snapshot_text(date: str, red_block: str, note: str = STABLE_NOTE) -> str:
    """A fleet-health-shaped daily snapshot.

    The frontmatter carries the date, so every day always has at least one
    genuinely unique chunk -- exactly like the real files.
    """
    return (
        f"---\ndate: {date}\ngenerated_at: {date}T23:51:52+00:00\n---\n\n"
        f"# Fleet Health - {date}\n\n"
        f"## RED Bots\n\n{red_block}\n\n"
        f"## Notes\n\n{note}\n"
    )


def write_series(root, reds, *, stem="fleet-health", subdir="Bots"):
    """Write one snapshot per entry in `reds`, on consecutive frozen dates."""
    folder = root / subdir if subdir else root
    folder.mkdir(parents=True, exist_ok=True)
    rels = []
    for offset, red in enumerate(reds):
        date = day(offset)
        name = f"{stem}-{date}.md"
        (folder / name).write_text(snapshot_text(date, red), encoding="utf-8")
        rels.append(f"{subdir}/{name}" if subdir else name)
    return rels


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "vault"
    d.mkdir()
    return d


def _store(tmp_path, embedder=None, name="dedupe.chroma"):
    return VectorStore(
        path=str(tmp_path / name),
        collection_name="knowledge",
        embedder=embedder if embedder is not None else HashEmbedder(),
    )


def red_chunks(store):
    """Every stored chunk that came from a `## RED Bots` section."""
    return [m for m in store.all_metadatas() if m.get("heading") == "RED Bots"]


class CountingEmbedder(HashEmbedder):
    def __init__(self, dim: int = 256) -> None:
        super().__init__(dim=dim)
        self.texts_embedded = 0

    def __call__(self, texts):
        self.texts_embedded += len(texts)
        return super().__call__(list(texts))


# ---------------------------------------------------------------------------
# The three behaviors the fix must pin
# ---------------------------------------------------------------------------

class TestIdenticalConsecutiveDaysCollapse:
    def test_five_identical_days_leave_one_red_chunk(self, corpus, tmp_path):
        """THE defect. Five snapshots, one unchanging status line -> one chunk."""
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        report = ingest(corpus, store)

        surviving = red_chunks(store)
        assert len(surviving) == 1, (
            f"expected the repeated RED block to collapse to a single chunk, got "
            f"{len(surviving)} copies from {[m['source'] for m in surviving]}"
        )
        # Two repeated sections (RED Bots + Notes) x four repeat days. The
        # frontmatter chunk carries the date and so never repeats.
        assert report.chunks_deduped == 8

    def test_a_repeated_prose_section_collapses_too(self, corpus, tmp_path):
        """Not special-cased to one heading: any verbatim repeat collapses."""
        write_series(corpus, [RED_A] * 4)
        store = _store(tmp_path)
        ingest(corpus, store)
        notes = [m for m in store.all_metadatas() if m.get("heading") == "Notes"]
        assert len(notes) == 1

    def test_dedupe_can_be_turned_off(self, corpus, tmp_path):
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        report = ingest(corpus, store, dedupe_snapshots=False)
        assert len(red_chunks(store)) == 5
        assert report.chunks_deduped == 0


class TestChangedDaysArePreserved:
    def test_the_day_the_number_moved_is_kept(self, corpus, tmp_path):
        """A/A/A/B/B -> the B block survives as its own chunk on its own date."""
        write_series(corpus, [RED_A, RED_A, RED_A, RED_B, RED_B])
        store = _store(tmp_path)
        ingest(corpus, store)

        surviving = red_chunks(store)
        assert len(surviving) == 2
        by_source = {m["source"]: m for m in surviving}
        assert f"Bots/fleet-health-{day(0)}.md" in by_source
        assert f"Bots/fleet-health-{day(3)}.md" in by_source, (
            "the day the drawdown changed was collapsed away -- a changed day is "
            "new information and must never be suppressed"
        )
        docs = [h["document"] for h in store.query("drawdown 31.50 cap", k=10)]
        assert any("31.50%" in d for d in docs)

    def test_a_value_that_returns_after_changing_is_not_collapsed(self, corpus, tmp_path):
        """A/B/A: the second A is a NEW fact (it came back), not a repeat."""
        write_series(corpus, [RED_A, RED_B, RED_A])
        store = _store(tmp_path)
        ingest(corpus, store)
        sources = sorted(m["source"] for m in red_chunks(store))
        assert sources == [
            f"Bots/fleet-health-{day(0)}.md",
            f"Bots/fleet-health-{day(1)}.md",
            f"Bots/fleet-health-{day(2)}.md",
        ]

    def test_ordinary_notes_are_never_touched(self, corpus, tmp_path):
        """SCOPE: two files with identical bodies and no dated-series filename."""
        (corpus / "alpha.md").write_text("# A\n\n" + STABLE_NOTE + "\n", encoding="utf-8")
        (corpus / "beta.md").write_text("# B\n\n" + STABLE_NOTE + "\n", encoding="utf-8")
        store = _store(tmp_path)
        report = ingest(corpus, store)
        assert report.chunks_deduped == 0
        assert {m["source"] for m in store.all_metadatas()} == {"alpha.md", "beta.md"}


class TestFirstOccurrenceStaysAttributableToItsDate:
    def test_survivor_is_the_earliest_day_and_carries_the_span(self, corpus, tmp_path):
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        ingest(corpus, store)

        (meta,) = red_chunks(store)
        assert meta["source"] == f"Bots/fleet-health-{day(0)}.md", (
            "the surviving copy must be the FIRST occurrence, so its citation "
            "points at the date the state actually began"
        )
        assert meta["snapshot_date"] == day(0)
        # Losing this is the failure the whole design exists to avoid.
        assert meta["repeat_count"] == 5
        assert meta["repeat_dates"] == ",".join(day(i) for i in range(1, 5))

    def test_per_date_question_is_still_answerable_through_search(self, corpus, tmp_path):
        """A middle day has no chunk of its own -- the citation must say so."""
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        ingest(corpus, store)

        res = search_knowledge(
            "options-bot cumulative drawdown 23.00", k=5, store=store, corpus_root=corpus
        )
        hits = [r for r in res["results"] if "23.00%" in r["text"]]
        assert hits, "the surviving status block must still be retrievable"
        cit = hits[0]["citation"]
        assert cit["snapshot_date"] == day(0)
        assert day(3) in cit["also_unchanged_on"], (
            "a query about day 3 has no way to learn the day-0 chunk covers it"
        )
        assert cit["snapshots_covered"] == 5

    def test_every_snapshot_file_keeps_its_own_date_bearing_chunk(self, corpus, tmp_path):
        """No day disappears from the index entirely."""
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        ingest(corpus, store)
        sources = {m["source"] for m in store.all_metadatas()}
        for i in range(5):
            assert f"Bots/fleet-health-{day(i)}.md" in sources

    def test_citation_shape_is_unchanged_for_ordinary_notes(self, corpus, tmp_path):
        (corpus / "cats.md").write_text("# Cats\n\nA cat purrs.\n", encoding="utf-8")
        store = _store(tmp_path)
        ingest(corpus, store)
        res = search_knowledge("cat purrs", k=2, store=store, corpus_root=corpus)
        assert set(res["results"][0]["citation"]) == {"source", "heading", "chunk_index"}


# ---------------------------------------------------------------------------
# Interaction with the incremental manifest (the seam this extends)
# ---------------------------------------------------------------------------

class TestIncrementalInteraction:
    def test_a_new_day_extends_the_survivors_span_without_re_embedding(
        self, corpus, tmp_path
    ):
        """The span grows on a day the survivor's own file did not change."""
        write_series(corpus, [RED_A] * 3)
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus, store)

        (corpus / "Bots" / f"fleet-health-{day(3)}.md").write_text(
            snapshot_text(day(3), RED_A), encoding="utf-8"
        )
        embedder.texts_embedded = 0
        ingest(corpus, store)

        (meta,) = red_chunks(store)
        assert meta["repeat_count"] == 4
        assert day(3) in meta["repeat_dates"]
        # Only the new day's own unique (frontmatter) chunk should be embedded --
        # never the unchanged survivor whose metadata just moved.
        assert embedder.texts_embedded <= 2, (
            f"re-embedded {embedder.texts_embedded} chunks to record a metadata "
            "change; the manifest saving is being paid back out"
        )

    def test_existing_duplicates_are_pruned_without_any_re_embedding(
        self, corpus, tmp_path
    ):
        """Deploy path: an index built WITHOUT dedupe must shed its duplicates on
        the next tick, and must not pay a full re-embed to do it."""
        write_series(corpus, [RED_A] * 5)
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus, store, dedupe_snapshots=False)
        assert len(red_chunks(store)) == 5
        before = store.count()

        embedder.texts_embedded = 0
        report = ingest(corpus, store)

        assert len(red_chunks(store)) == 1
        assert store.count() < before
        assert report.chunks_deleted >= 4
        assert embedder.texts_embedded == 0, (
            "pruning duplicates must be a delete, not a rebuild"
        )

    def test_turning_dedupe_off_restores_the_suppressed_chunks(self, corpus, tmp_path):
        """The reverse must work too, or the flag is a one-way door: the content
        hash alone would happily skip these files forever."""
        write_series(corpus, [RED_A] * 5)
        store = _store(tmp_path)
        ingest(corpus, store)
        assert len(red_chunks(store)) == 1

        ingest(corpus, store, dedupe_snapshots=False)
        assert len(red_chunks(store)) == 5

    def test_quiet_run_after_dedupe_is_a_no_op(self, corpus, tmp_path):
        write_series(corpus, [RED_A] * 5)
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus, store)
        count = store.count()

        embedder.texts_embedded = 0
        report = ingest(corpus, store)
        assert store.count() == count
        assert embedder.texts_embedded == 0
        assert report.chunks_added == 0
        assert report.chunks_deleted == 0


# ---------------------------------------------------------------------------
# Scope: what counts as a series at all
# ---------------------------------------------------------------------------

class TestSeriesDetectionScope:
    def test_trailing_date_is_a_series_leading_date_is_not(self):
        assert parse_series("Bots/fleet-health-2026-07-23.md") == (
            "Bots/fleet-health",
            "2026-07-23",
        )
        assert parse_series("Bots/2026-05-12 CPI Day Playbook.md") is None
        assert parse_series("Bots/notes.md") is None

    def test_same_stem_in_different_folders_are_different_series(self):
        rels = [f"A/x-{day(i)}.md" for i in range(3)] + [
            f"B/x-{day(i)}.md" for i in range(3)
        ]
        assert set(group_series(rels)) == {"A/x", "B/x"}

    def test_a_short_run_is_a_one_off_not_a_series(self):
        rels = [f"R/one-off-{day(i)}.md" for i in range(MIN_SERIES_LENGTH - 1)]
        assert group_series(rels) == {}

    def test_two_identical_one_off_reports_are_not_collapsed(self, corpus, tmp_path):
        """Below the series threshold, nothing is touched -- a coincidence is not
        a cadence."""
        write_series(corpus, [RED_A] * (MIN_SERIES_LENGTH - 1), stem="one-off")
        store = _store(tmp_path)
        report = ingest(corpus, store)
        assert report.chunks_deduped == 0
        assert len(red_chunks(store)) == MIN_SERIES_LENGTH - 1


# ---------------------------------------------------------------------------
# Pure planning logic
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_a_run_of_repeats_all_point_at_the_first_day(self):
        rels = [f"S/f-{day(i)}.md" for i in range(4)]
        keys = {rel: ["k1", "k2"] for rel in rels}
        plan = build_plan({"S/f": rels}, keys)
        assert plan.kept[rels[0]] == (0, 1)
        assert all(plan.kept[r] == () for r in rels[1:])
        # Chained through day 1 and 2, yet still attributed to day 0.
        assert plan.repeats[rels[0]][0] == (day(1), day(2), day(3))
        assert plan.suppressed == 6

    def test_a_within_file_duplicate_beyond_yesterdays_count_is_kept(self):
        """Multiset, not set: two copies today vs one yesterday keeps one."""
        rels = [f"S/f-{day(i)}.md" for i in range(3)]
        keys = {rels[0]: ["k1"], rels[1]: ["k1", "k1"], rels[2]: ["k1", "k1"]}
        plan = build_plan({"S/f": rels}, keys)
        assert plan.kept[rels[1]] == (1,)
        assert plan.kept[rels[2]] == ()

    def test_an_unreadable_file_breaks_the_chain_safely(self):
        """A gap must under-collapse, never suppress against stale content."""
        rels = [f"S/f-{day(i)}.md" for i in range(3)]
        keys = {rels[0]: ["k1"], rels[2]: ["k1"]}  # middle file missing
        plan = build_plan({"S/f": rels}, keys)
        assert plan.kept[rels[2]] == (0,)
        assert plan.suppressed == 0

    def test_chunk_key_separates_heading_from_body(self):
        assert chunk_key("A", "B") != chunk_key("AB", "")
        assert chunk_key("A", "B") == chunk_key("A", "B")
        assert chunk_key(None, "B") == chunk_key("", "B")

    def test_long_spans_are_summarized_not_dumped(self):
        dates = [day(i) for i in range(200)]
        rendered = format_repeat_dates(dates)
        assert rendered.startswith(dates[0])
        assert "200 snapshots" in rendered
        assert len(rendered) < 80
        assert format_repeat_dates([]) == ""
