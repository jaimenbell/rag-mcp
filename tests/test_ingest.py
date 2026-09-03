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
    #
    # RM-fixafter2 slice 2 note: "RESUME.md" is no longer a usable negative
    # fixture here. Overriding handoff_mirror_basenames REPLACES the
    # basename SET, but "resume" is now also a leading-stem regex match
    # (see _HANDOFF_LEADING_RE) independent of that set -- the two signals
    # are OR'd, so a bare stem name can't be un-matched by the basenames
    # override alone. "custom-status.md" (not a stem word at all) is the
    # fixture that actually exercises the override/replace semantic.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "status.md").write_text(
        "No frontmatter, custom basename, no 'handoff' substring either.\n",
        encoding="utf-8",
    )
    (ctx / "custom-status.md").write_text(
        "Not a default basename and not a stem word -- must stay 'note' once "
        "the basename set is replaced without it.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store, handoff_mirror_basenames={"status.md"})
    metas = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
    assert metas["context/status.md"] == "handoff"
    assert metas["context/custom-status.md"] == "note"


def test_ingest_bare_stem_basename_matches_via_regex_even_when_basenames_overridden(
    tmp_path, store
):
    # Documents the interaction above from the other direction: "RESUME.md"
    # keeps classifying "handoff" purely via the leading-stem regex, even
    # though it is no longer present in the (fully replaced) basenames set.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "RESUME.md").write_text("A bare stem basename.\n", encoding="utf-8")
    ingest(tmp_path, store, handoff_mirror_basenames={"status.md"})
    metas = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
    assert metas["context/RESUME.md"] == "handoff"


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


def test_doc_class_handoff_leading_token_free_trailing_text_is_handoff(tmp_path, store):
    # RM-fixafter2 slice 2 grammar change: the leading-stem match no longer
    # constrains what follows the separator to a narrow shape (digit-led,
    # <=3-char code, -vN, trailing date) -- ANY text after "handoff-"/"-_ "
    # now qualifies, because the narrow-shape design was itself the source of
    # the false NEGATIVES this fix-after exists to close (see
    # test_doc_class_handoff_leading_stem_free_text_variants below). This
    # widens the net back over titles like "handoff-skill-redesign-spec.md"
    # -- accepted: doc_class is a convenience filter scoped to `context/`,
    # not a security boundary, and simple-and-correct-on-the-real-corpus beats
    # a narrow shape-language that still missed real mirrors.
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
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_handoff_versioned_mirror_name_is_handoff(tmp_path, store):
    # A real dated/versioned per-thread handoff mirror -- must still match
    # under the new leading-stem-only grammar (first token is "handoff").
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff-projectx-world-v5.md").write_text(
        "No frontmatter. Per-thread session handoff bookkeeping content.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/handoff-projectx-world-v5.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_handoff_digit_led_trailing_text_is_handoff(tmp_path, store):
    # Finding #5 control: the old `_HANDOFF_TOKEN_SUFFIX` had an "unbounded"
    # bug where a digit-led suffix let anything through -- subsumed by the
    # slice-2 grammar rewrite (leading-stem match accepts any trailing text
    # by design now, see the test above), so this is simply the same
    # behavior for a different trailing shape, not a special case anymore.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff-3d-printing-guide.md").write_text(
        "No frontmatter. Unrelated content about 3D printing.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/handoff-3d-printing-guide.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_trailing_handoff_token_is_note_not_handoff(tmp_path, store):
    # RM-fixafter2 slice 2: the TRAILING form ("word-handoff.md") is DROPPED.
    # It was the source of the false positive this fix-after reports
    # (context/notes-on-handoff.md -> handoff); the grammar is leading-stem
    # only now, so a token in trailing position never matches.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "morning-dispatch-handoff.md").write_text(
        "No frontmatter. The token is trailing, not leading -- must be 'note' now.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas()
        if m["source"] == "context/morning-dispatch-handoff.md"
    ]
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


# ---------------------------------------------------------------------------
# leading-stem grammar rewrite (RM-fixafter2 slice 2, findings #3/#4/#5/#7).
#
# GRAMMAR (one sentence): a basename (extension-stripped, any of
# MARKDOWN_EXTS) classifies "handoff" iff its FIRST '-'/'_'-delimited token
# -- optionally after a YYYY-MM-DD date prefix -- case-insensitively equals
# one of the mirror stems {handoff, active, resume}, with no constraint on
# what (if anything) follows; a stem appearing anywhere else in the name
# (trailing, embedded, substring-only) does not count.
#
# FIRES: a real vault mirror whose fallback previously missed it because the
#        stem wasn't "handoff", or the trailing shape was too narrow --
#        RESUME-world-v6.md, handoff-projectx.md, ACTIVE-projectx.md,
#        resume-2026-09-03.md, RESUME-world-v7.md.
# SILENT: "handoff" present but NOT as the first token --
#        critique-of-the-handoff.md, research-handoff.md, _handoff.md.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename",
    [
        "RESUME-world-v6.md",
        "handoff-projectx.md",
        "ACTIVE-projectx.md",
        "resume-2026-09-03.md",
        "RESUME-world-v7.md",
    ],
)
def test_doc_class_handoff_leading_stem_free_text_variants(tmp_path, store, basename):
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / basename).write_text(
        "No frontmatter. Real vault session-bookkeeping mirror content.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == f"context/{basename}"]
    assert metas, f"{basename} was not ingested"
    assert all(m["doc_class"] == "handoff" for m in metas)


@pytest.mark.parametrize(
    "basename",
    [
        "critique-of-the-handoff.md",
        "research-handoff.md",
        "_handoff.md",
    ],
)
def test_doc_class_handoff_stem_not_leading_is_note(tmp_path, store, basename):
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / basename).write_text(
        "No frontmatter. The word 'handoff' is not the first token here.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == f"context/{basename}"]
    assert metas, f"{basename} was not ingested"
    assert all(m["doc_class"] == "note" for m in metas)


def test_doc_class_stem_immediately_followed_by_text_no_separator_is_note(
    tmp_path, store
):
    # Review follow-up: the grammar comment above _HANDOFF_LEADING_RE claimed
    # "no constraint on what follows the stem", which overstated it -- the
    # regex requires trailing text to start with its own "-"/"_" separator
    # (or be absent) because that is what makes "handoff" a bounded TOKEN
    # rather than a prefix. "handoff2026.md" has no separator, so
    # "handoff2026" is one token, not the stem "handoff" -- must stay "note".
    # (Comment corrected to match; this proves the actual code, not just the
    # corrected wording.)
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff2026.md").write_text(
        "No frontmatter. The stem is immediately followed by digits, no separator.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "context/handoff2026.md"]
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


def test_doc_class_dot_separator_after_stem_is_handoff(tmp_path, store):
    # Finding #9 (RM-fixafter3): the post-stem separator class was narrowed
    # from `[-_.]` to `[-_]` (RM-fixafter2 slice 2 rewrite comment above,
    # ~L93-99) with no test covering the "." case -- "handoff.v2.md" silently
    # flipped from "handoff" to "note". "." is a deliberate separator choice
    # (a versioned mirror name like "handoff.v2.md" is a realistic filename
    # shape), not an oversight; restore it and pin it with a test.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.v2.md").write_text(
        "No frontmatter. Dot-separated version suffix after the stem.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "context/handoff.v2.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_doc_class_markdown_extension_leading_stem_is_handoff(tmp_path, store):
    # Finding #7: the matcher must build its extension alternation from
    # MARKDOWN_EXTS (".md", ".markdown"), not hard-require ".md" -- a
    # ".markdown" mirror file was previously invisible to the fallback.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.markdown").write_text(
        "No frontmatter. A .markdown mirror, not .md.\n", encoding="utf-8"
    )
    ingest(tmp_path, store)
    metas = [m for m in store.all_metadatas() if m["source"] == "context/handoff.markdown"]
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


# ---------------------------------------------------------------------------
# handoff_mirror_dir empty-string guard (RM-fixafter2 slice 1) -- an empty
# string makes `path.parent.name.lower() == handoff_mirror_dir.lower()` true
# for every ROOT-level file (PurePosixPath("root.md").parent.name == ""),
# silently widening the doc_class fallback from "context" to the whole
# corpus. ingest() must fail loud instead of silently doing that.
# ---------------------------------------------------------------------------


def test_ingest_rejects_empty_handoff_mirror_dir(tmp_path, store):
    with pytest.raises(ValueError):
        ingest(tmp_path, store, handoff_mirror_dir="")


def test_ingest_rejects_whitespace_only_handoff_mirror_dir(tmp_path, store):
    with pytest.raises(ValueError):
        ingest(tmp_path, store, handoff_mirror_dir="   ")


# ---------------------------------------------------------------------------
# handoff_mirror_dir shape guard (RM-fixafter3, finding #4) -- `path.parent
# .name` (what `_doc_class` compares against) is ALWAYS a bare component for
# a real file's relative path: pathlib strips slashes. A caller-supplied
# handoff_mirror_dir that isn't ALSO bare -- "context/" (trailing slash),
# "/context" (leading slash) -- can therefore never equal it, which
# previously silently disabled the whole filename fallback for every file
# in the corpus instead of raising loudly like the empty-string case above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_dir", ["context/", "/context"])
def test_ingest_rejects_non_bare_handoff_mirror_dir(tmp_path, store, bad_dir):
    with pytest.raises(ValueError):
        ingest(tmp_path, store, handoff_mirror_dir=bad_dir)


@pytest.mark.parametrize("bad_dir", ["context/", "/context", ""])
def test_doc_class_malformed_handoff_mirror_dir_never_matches(bad_dir):
    # Belt-and-braces: _doc_class's OWN guard, exercised directly (bypassing
    # ingest()'s outer raise) -- a malformed dir degrades to "no match"
    # (never "note" widened to match everything, never a crash), which is
    # the safe direction for a filter that is a convenience, not a security
    # boundary.
    from rag_mcp.ingest import _doc_class

    assert (
        _doc_class(
            "context/handoff.md",
            "No frontmatter.\n",
            handoff_mirror_dir=bad_dir,
        )
        == "note"
    )


# ---------------------------------------------------------------------------
# handoff_mirror_dir whitespace normalization (RM-fixafter3, finding #2) --
# `_classifier_config_fingerprint` normalizes with .strip().lower() (feeding
# the metav that decides whether a metadata-only refresh fires), while
# `_doc_class`'s own directory comparison only .lower()'d it. A padded value
# like " context " therefore produced the SAME metav as "context" (the
# fingerprint strips) while classifying every real "context/" file
# DIFFERENTLY (doc_class compared unstripped) -- a silent classification
# drift with no metadata-refresh trigger to catch it. ingest() now
# normalizes (.strip()) once at its own boundary, so every downstream use
# (fingerprint AND _doc_class) sees the identical value.
# ---------------------------------------------------------------------------


def test_ingest_handoff_mirror_dir_whitespace_normalized_consistently(tmp_path, store):
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.md").write_text("No frontmatter, real mirror file.\n", encoding="utf-8")

    ingest(tmp_path, store, handoff_mirror_dir=" context ")
    metas = [m for m in store.all_metadatas() if m["source"] == "context/handoff.md"]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas), (
        "a whitespace-padded handoff_mirror_dir must still match the real "
        "'context' directory, exactly like the unpadded value does"
    )


# ---------------------------------------------------------------------------
# handoff_mirror_stems opt-out (RM-fixafter3, finding #6) -- {"handoff",
# "active", "resume"} match as a free leading token independently of
# handoff_mirror_basenames, with no way to opt a corpus out of "active"/
# "resume" specifically (context/active-clients-2026.md -> "handoff" even
# though it's an ordinary business file, not a session mirror).
# handoff_mirror_stems= now lets a caller narrow the stem set; the default
# is unchanged (matches every test above that doesn't pass it).
# ---------------------------------------------------------------------------


def test_ingest_default_stems_still_match_active_leading_token(tmp_path, store):
    # Pins the DEFAULT (unchanged) so the narrowing test below is a genuine
    # opt-out, not a behavior change.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "active-clients-2026.md").write_text(
        "Not a session mirror -- an ordinary business file that happens to "
        "start with 'active-'.\n",
        encoding="utf-8",
    )
    ingest(tmp_path, store)
    metas = [
        m for m in store.all_metadatas() if m["source"] == "context/active-clients-2026.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)


def test_ingest_handoff_mirror_stems_narrowed_opts_out_active(tmp_path, store):
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "active-clients-2026.md").write_text(
        "Not a session mirror -- an ordinary business file.\n", encoding="utf-8"
    )
    ingest(tmp_path, store, handoff_mirror_stems=("handoff",))
    metas = [
        m for m in store.all_metadatas() if m["source"] == "context/active-clients-2026.md"
    ]
    assert metas
    assert all(m["doc_class"] == "note" for m in metas)


def test_ingest_handoff_mirror_stems_narrowed_keeps_handoff_matching(tmp_path, store):
    # SILENT/scope proof: narrowing stems to just "handoff" must not also
    # break "handoff" itself.
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff-projectx.md").write_text(
        "No frontmatter. Real session-bookkeeping mirror content.\n", encoding="utf-8"
    )
    ingest(tmp_path, store, handoff_mirror_stems=("handoff",))
    metas = [
        m for m in store.all_metadatas() if m["source"] == "context/handoff-projectx.md"
    ]
    assert metas
    assert all(m["doc_class"] == "handoff" for m in metas)
