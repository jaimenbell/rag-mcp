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


def test_load_file_strips_bom(tmp_path):
    # RM-fixafter-ragmcp slice 6: a BOM-prefixed file must decode with the BOM
    # stripped -- a leading U+FEFF would otherwise break _parse_frontmatter's
    # `text.startswith("---")` check.
    p = tmp_path / "bom.md"
    p.write_bytes("---\ntype: handoff\n---\n# Handoff\n\nBody.\n".encode("utf-8-sig"))
    text = load_file(p)
    assert text is not None
    assert text.startswith("---"), "BOM was not stripped"
    assert "﻿" not in text


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


def test_metadata_doc_class_handoff_via_type_field_bom_prefixed(tmp_path, store):
    # RM-fixafter-ragmcp slice 6: a BOM-prefixed file with `type: handoff`
    # frontmatter must still classify "handoff" end-to-end through ingest().
    (tmp_path / "bom-note.md").write_bytes(
        (
            "---\ntype: handoff\ntitle: Session Handoff\n---\n"
            "# Handoff\n\nSome content here about the session state.\n"
        ).encode("utf-8-sig")
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


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
    # must stay "note", proving the fallback isn't a bare filename match
    # anywhere in the corpus.
    (tmp_path / "handoff.md").write_text(
        "Not a session mirror -- just an unrelated file with this name.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = store.all_metadatas()
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


# ---------------------------------------------------------------------------
# handoff-mirror config seam (RM-fixafter-ragmcp slice 4) -- the default
# "context" dir + basename convention was hard-coded; a different corpus must
# be able to override it via ingest()'s params without editing this module.
# Defaults must keep today's behavior unchanged (proven by every test above,
# none of which pass the new params).
# ---------------------------------------------------------------------------


def test_ingest_custom_handoff_mirror_dir(tmp_path, store):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "handoff.md").write_text(
        "No frontmatter, but this dir is not 'context' -- default would say note.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store, handoff_mirror_dir="sessions")
    metas = [m for m in store.all_metadatas() if m["source"] == "sessions/handoff.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_ingest_custom_handoff_mirror_dir_default_context_unaffected(tmp_path, store):
    # Overriding the dir means the DEFAULT "context" dir no longer qualifies.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.md").write_text("Body text under context/, dir overridden away.\n", encoding="utf-8")
    ingest(tmp_path, store, handoff_mirror_dir="sessions")
    metas = [m for m in store.all_metadatas() if m["source"] == "context/handoff.md"]
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


def test_ingest_custom_handoff_mirror_basenames_replaces_at_this_layer(tmp_path, store):
    # ingest()'s own param REPLACES, exactly like exclude_prefixes= does --
    # AUGMENT-the-defaults is a CLI-layer behavior (see test_cli.py's
    # --handoff-mirror-basename tests), not something ingest() does itself.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "status.md").write_text(
        "No frontmatter, custom basename, no 'handoff' substring either.\n",
        encoding="utf-8",
    )
    (ctx / "RESUME.md").write_text(
        "A default mirror name -- must NOT match once basenames is replaced.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store, handoff_mirror_basenames={"status.md"})
    metas = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
    assert metas["context/status.md"] == "handoff"
    assert metas["context/RESUME.md"] == "note"


# ---------------------------------------------------------------------------
# matcher correctness (RM-fixafter-ragmcp slice 5) -- the "handoff" filename
# check must be an anchored TOKEN match, not a bare substring, so a title
# that merely CONTAINS the word does not false-positive, while a real
# dated/versioned mirror name still matches. Also: the directory-name
# comparison must be case-insensitive.
# ---------------------------------------------------------------------------


def test_doc_class_dir_comparison_is_case_insensitive(tmp_path, store):
    ctx = tmp_path / "Context"
    ctx.mkdir()
    (ctx / "ACTIVE.md").write_text("No frontmatter; parent dir is 'Context', not 'context'.\n", encoding="utf-8")
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "Context/ACTIVE.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_handoff_substring_title_is_note_not_handoff(tmp_path, store):
    # FIRES the anti-false-positive: a spec doc ABOUT the handoff skill, not a
    # session handoff record. Free text after "handoff-" fits none of the
    # anchor's allowed suffix shapes (digit-led, <=3-char code, -vN, or a
    # trailing date), so this must stay "note".
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff-skill-redesign-spec.md").write_text(
        "No frontmatter. This is a spec document about redesigning the handoff skill.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/handoff-skill-redesign-spec.md"
    ]
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


def test_doc_class_handoff_versioned_mirror_name_is_handoff(tmp_path, store):
    # A real dated/versioned per-thread handoff mirror: the "-alphahive-
    # world-v5" suffix ends in a version tag, one of the anchor's allowed
    # shapes -- must still match.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff-alphahive-world-v5.md").write_text(
        "No frontmatter. Per-thread session handoff bookkeeping content.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/handoff-alphahive-world-v5.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_trailing_handoff_token_is_handoff(tmp_path, store):
    # Trailing-word form: the token is the LAST word before .md, not the first.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "morning-dispatch-handoff.md").write_text(
        "No frontmatter. Real vault mirrors sometimes name the token trailing.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/morning-dispatch-handoff.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_ingest_default_handoff_mirror_params_unchanged(tmp_path, store):
    # No override passed -> identical to calling ingest() with no knowledge of
    # these params at all (this is the whole "defaults keep today's behaviour"
    # contract).
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "ACTIVE.md").write_text("Default mirror name, no override passed.\n", encoding="utf-8")
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "context/ACTIVE.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)
