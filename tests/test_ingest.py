"""Phase 1 - ingest pipeline: population, metadata, idempotency, fail-soft loading."""
from __future__ import annotations

import pytest

from rag_mcp.ingest import EXCLUDE_PREFIXES, ingest, iter_corpus_files, load_file


def test_iter_corpus_files_only_markdown(corpus):
    names = {p.name for p in iter_corpus_files(corpus)}
    assert "cats.md" in names
    assert "dogs.md" in names
    assert "notes.txt" not in names  # non-markdown ignored


def test_load_file_skips_garbled(corpus):
    # Garbled (invalid UTF-8) returns None rather than raising.
    assert load_file(corpus / "garbled.md") is None


def test_ingest_populates_store(corpus, store):
    report = ingest(corpus, store)
    assert report.chunks_added > 0
    assert store.count() == report.chunks_added


def test_ingest_skips_empty_and_garbled(corpus, store):
    report = ingest(corpus, store)
    # cats.md + dogs.md ingest; empty.md + garbled.md skipped; notes.txt never seen.
    assert report.files_ingested == 2
    assert report.files_skipped == 2


def test_metadata_captured(corpus, store):
    ingest(corpus, store)
    metas = store.all_metadatas()
    assert metas, "expected metadata rows"
    sources = {m["source"] for m in metas}
    assert "cats.md" in sources
    # dogs.md has a '## Behavior' subheading that must be captured.
    headings = {m.get("heading") for m in metas}
    assert "Behavior" in headings
    for m in metas:
        assert "source" in m and "heading" in m and "chunk_index" in m


def test_idempotent_reingest(corpus, store):
    first = ingest(corpus, store)
    count_after_first = store.count()
    second = ingest(corpus, store)
    # Re-ingesting the same corpus must not duplicate.
    assert store.count() == count_after_first
    assert second.chunks_added == first.chunks_added


def test_relevant_chunk_retrievable(corpus, store):
    ingest(corpus, store)
    hits = store.query("loyal dog that barks", k=1)
    assert hits
    assert hits[0]["metadata"]["source"] == "dogs.md"


# ---------------------------------------------------------------------------
# exclude_prefixes — new tests (KB issue #7)
# ---------------------------------------------------------------------------

@pytest.fixture
def backup_corpus(tmp_path):
    """Corpus with a live skill and a backup copy under the default-excluded prefix."""
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "solana.md").write_text(
        "# Solana\n\nLive skill content.", encoding="utf-8"
    )
    backup = tmp_path / "infrastructure" / "claude-config-backup" / "skills"
    backup.mkdir(parents=True)
    (backup / "solana.md").write_text(
        "# Solana\n\nStale backup content.", encoding="utf-8"
    )
    return tmp_path


def test_iter_corpus_files_excludes_default_prefix(backup_corpus):
    """Default EXCLUDE_PREFIXES hides the claude-config-backup subtree."""
    paths = iter_corpus_files(backup_corpus)
    posix_paths = {p.relative_to(backup_corpus).as_posix() for p in paths}
    assert not any("claude-config-backup" in p for p in posix_paths)


def test_iter_corpus_files_non_excluded_sibling_included(backup_corpus):
    """Files outside the excluded prefix are still returned."""
    paths = iter_corpus_files(backup_corpus)
    posix_paths = {p.relative_to(backup_corpus).as_posix() for p in paths}
    assert "skills/solana.md" in posix_paths


def test_iter_corpus_files_empty_exclude_returns_all_md(backup_corpus):
    """Passing exclude_prefixes=() bypasses all exclusions — all markdown returned."""
    paths = iter_corpus_files(backup_corpus, exclude_prefixes=())
    posix_paths = {p.relative_to(backup_corpus).as_posix() for p in paths}
    assert "skills/solana.md" in posix_paths
    assert any("claude-config-backup" in p for p in posix_paths)


def test_ingest_exclude_prefix_skips_subtree(backup_corpus, store):
    """ingest() respects default exclude_prefixes: backup subtree never ingested."""
    report = ingest(backup_corpus, store)
    assert report.files_ingested == 1  # only the live skill
    assert report.files_seen == 1      # backup never enters the loop


# ---------------------------------------------------------------------------
# doc_class metadata — frontmatter classification (RM-ragmcp-docclass slice 1)
# ---------------------------------------------------------------------------
# FIRES: a handoff-shaped fixture (via `type:` or `tags:` frontmatter, or the
#        filename-fallback for the real vault session mirrors) -> "handoff".
# SILENT: a research-note fixture (frontmatter present but not handoff-shaped),
#         a no-frontmatter fixture, and an invalid-frontmatter fixture all land
#         on "note" without raising.


def test_metadata_doc_class_handoff_via_type_field(tmp_path, store):
    (tmp_path / "note.md").write_text(
        "---\ntype: handoff\ntitle: Session Handoff\n---\n"
        "# Handoff\n\nSome content here about the session state.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_metadata_doc_class_handoff_via_tags_field(tmp_path, store):
    (tmp_path / "note.md").write_text(
        "---\ntags: [routine, handoff, eod]\n---\n"
        "# Routine\n\nBody text describing the routine.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_metadata_doc_class_tags_field_case_insensitive(tmp_path, store):
    (tmp_path / "note.md").write_text(
        "---\ntags: [Routine, HANDOFF]\n---\n# Routine\n\nBody text here.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_metadata_doc_class_research_note_frontmatter_is_note(tmp_path, store):
    (tmp_path / "research.md").write_text(
        "---\ntype: research\ntitle: A Research Note\ntags: [research]\n---\n"
        "# Research\n\nFindings are documented here in detail.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


def test_metadata_doc_class_no_frontmatter_is_note_no_crash(tmp_path, store):
    (tmp_path / "plain.md").write_text(
        "# Plain note\n\nNo frontmatter at all, just ordinary prose content.\n",
        encoding="utf-8",
    )
    report = ingest(tmp_path, store)
    assert report.files_ingested == 1
    metas = store.all_metadatas()
    assert all(m["doc_class"] == "note" for m in metas)


def test_metadata_doc_class_invalid_frontmatter_no_crash(tmp_path, store):
    # Malformed YAML inside the delimiters must not raise -- frontmatter is an
    # optional Obsidian convention, not a hard contract.
    (tmp_path / "broken.md").write_text(
        "---\n: not valid yaml: [\n---\n# Broken\n\nBody text follows here.\n",
        encoding="utf-8",
    )
    report = ingest(tmp_path, store)
    assert report.files_ingested == 1
    metas = store.all_metadatas()
    assert all(m["doc_class"] == "note" for m in metas)


def test_metadata_doc_class_every_chunk_of_doc_shares_class(tmp_path, store):
    # A doc long enough to span multiple chunks must carry the SAME doc_class
    # on every chunk -- classification is per-document, not per-chunk.
    body = "\n\n".join(f"Paragraph {i} of the handoff body text repeated." for i in range(40))
    (tmp_path / "note.md").write_text(
        f"---\ntype: handoff\n---\n# Handoff\n\n{body}\n", encoding="utf-8"
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "note.md"]
    assert len(metas) > 1, "fixture must produce >1 chunk to prove per-chunk parity"
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_metadata_doc_class_mirror_filename_fallback_no_frontmatter(tmp_path, store):
    # Mirrors the real vault shape: context/RESUME.md and context/ACTIVE.md ship
    # with NO frontmatter at all (verified live 2026-09-03), so frontmatter alone
    # would never tag them "handoff" -- the filename fallback exists for this.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "RESUME.md").write_text(
        "START IN PLAN MODE\n\nSome session bookkeeping text goes here.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "context/RESUME.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_metadata_doc_class_mirror_fallback_scoped_to_context_dir(tmp_path, store):
    # SILENT/scope proof: same basename, but NOT under a `context/` parent --
    # must stay "note", proving the fallback isn't a bare filename match anywhere
    # in the corpus (mirrors research_precheck.py's own _is_handoff_mirror scope).
    (tmp_path / "handoff.md").write_text(
        "Not a session mirror -- just an unrelated file with this name.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)
