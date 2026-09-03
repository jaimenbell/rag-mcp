"""Incremental ingest: only changed files get re-embedded, stale chunks get pruned.

The point of these tests is not just "the numbers in the report look right" --
it is that the EXPENSIVE work (embedding) genuinely does not happen for
unchanged files. A counting embedder proves that directly, because a report
counter could be correct while the work still ran.

Every trust-failure mode must degrade to a FULL re-embed, never to a wrong skip:
a wrong skip leaves a note permanently stale in the index with no symptom.
"""
from __future__ import annotations

import json

import pytest

from rag_mcp.ingest import (
    CURRENT_METADATA_VERSION,
    DEFAULT_HANDOFF_MIRROR_BASENAMES,
    DEFAULT_HANDOFF_MIRROR_DIR,
    DEFAULT_HANDOFF_MIRROR_STEMS,
    _classifier_config_fingerprint,
    _effective_metadata_version,
    ingest,
)
from rag_mcp.manifest import (
    MANIFEST_FILENAME,
    IngestManifest,
    RunIdentity,
    content_hash,
    load_manifest,
)
from rag_mcp.store import HashEmbedder, VectorStore

# Expected metav for a default-config ingest() call -- the metadata backfill
# tests compare against this instead of the bare CURRENT_METADATA_VERSION
# now that metav also folds in the classifier config fingerprint (finding #2).
_DEFAULT_METAV = _effective_metadata_version(
    DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES
)


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that records how many texts it was asked to embed."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__(dim=dim)
        self.embed_calls = 0
        self.texts_embedded = 0

    def __call__(self, texts):
        self.embed_calls += 1
        self.texts_embedded += len(texts)
        return super().__call__(list(texts))


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "dogs.md").write_text(
        "# Dogs\n\n## Behavior\n\nDogs are loyal pack animals. A dog barks.\n",
        encoding="utf-8",
    )
    (d / "cats.md").write_text(
        "# Cats\n\nCats are small carnivorous mammals. A cat purrs.\n",
        encoding="utf-8",
    )
    return d


def _store(tmp_path, embedder=None, name="inc.chroma", collection="knowledge"):
    return VectorStore(
        path=str(tmp_path / name),
        collection_name=collection,
        embedder=embedder if embedder is not None else HashEmbedder(),
    )


class TestSkipsUnchangedFiles:
    def test_second_run_reports_everything_unchanged(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        first = ingest(corpus_dir, store)
        assert first.files_ingested == 2
        assert first.files_unchanged == 0

        second = ingest(corpus_dir, store)
        assert second.files_unchanged == 2
        assert second.files_ingested == 0
        assert second.chunks_added == 0
        assert second.incremental is True

    def test_second_run_does_no_embedding_work_at_all(self, corpus_dir, tmp_path):
        """THE load-bearing assertion: unchanged files cost zero embeddings."""
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)
        assert embedder.texts_embedded > 0, "first run should embed"

        embedder.embed_calls = 0
        embedder.texts_embedded = 0
        ingest(corpus_dir, store)
        assert embedder.texts_embedded == 0, (
            "unchanged files were re-embedded -- the incremental skip is not working"
        )

    def test_store_contents_survive_an_incremental_noop_run(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        before = store.count()
        ingest(corpus_dir, store)
        assert store.count() == before

        hits = store.query("dog barks", k=3)
        assert any(h["metadata"]["source"] == "dogs.md" for h in hits)


class TestChangeDetection:
    def test_edited_file_is_re_embedded_and_others_are_not(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)

        (corpus_dir / "dogs.md").write_text(
            "# Dogs\n\n## Behavior\n\nDogs howl at the moon on quiet nights.\n",
            encoding="utf-8",
        )
        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)

        assert report.files_ingested == 1
        assert report.files_unchanged == 1
        assert embedder.texts_embedded > 0

        hits = store.query("howl moon", k=3)
        assert any("howl" in h["document"] for h in hits)

    def test_new_file_is_picked_up(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        (corpus_dir / "birds.md").write_text(
            "# Birds\n\nBirds are feathered vertebrates. A bird sings.\n",
            encoding="utf-8",
        )
        report = ingest(corpus_dir, store)
        assert report.files_ingested == 1
        assert report.files_unchanged == 2
        hits = store.query("bird sings", k=3)
        assert any(h["metadata"]["source"] == "birds.md" for h in hits)

    def test_rewriting_identical_content_is_still_a_skip(self, corpus_dir, tmp_path):
        """mtime moves, content does not -> must NOT re-embed (hash, not mtime)."""
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)

        original = (corpus_dir / "dogs.md").read_text(encoding="utf-8")
        (corpus_dir / "dogs.md").write_text(original, encoding="utf-8")

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)
        assert report.files_unchanged == 2
        assert embedder.texts_embedded == 0


class TestPruning:
    """Upsert alone never removed anything -- that is why a weekly rebuild existed."""

    def test_deleted_file_has_its_chunks_pruned(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        assert any(m["source"] == "cats.md" for m in store.all_metadatas())

        (corpus_dir / "cats.md").unlink()
        report = ingest(corpus_dir, store)

        assert report.chunks_deleted > 0
        sources = {m["source"] for m in store.all_metadatas()}
        assert "cats.md" not in sources, "chunks of a deleted note still in the index"
        assert "dogs.md" in sources

    def test_shrunk_file_drops_its_trailing_chunks(self, tmp_path):
        corpus = tmp_path / "c"
        corpus.mkdir()
        long_doc = "\n\n".join(
            f"## Section {i}\n\n" + ("filler sentence about topic %d. " % i) * 60
            for i in range(8)
        )
        (corpus / "big.md").write_text("# Big\n\n" + long_doc, encoding="utf-8")

        store = _store(tmp_path)
        first = ingest(corpus, store)
        assert first.chunks_added > 2, "need a genuinely multi-chunk file"
        before = store.count()

        (corpus / "big.md").write_text("# Big\n\nNow very short.\n", encoding="utf-8")
        report = ingest(corpus, store)

        assert report.chunks_deleted > 0
        assert store.count() < before
        # No orphaned chunk may still carry the old filler text.
        docs = [h["document"] for h in store.query("filler sentence topic", k=10)]
        assert not any("filler sentence" in d for d in docs), (
            "stale trailing chunks survived the shrink"
        )

    def test_renamed_file_leaves_no_orphan(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        (corpus_dir / "dogs.md").rename(corpus_dir / "canines.md")
        ingest(corpus_dir, store)
        sources = {m["source"] for m in store.all_metadatas()}
        assert sources == {"cats.md", "canines.md"}


class TestTrustFailuresDegradeToFullRebuild:
    """Every one of these must re-embed everything, never skip incorrectly."""

    def _full_reembed_happened(self, embedder) -> bool:
        return embedder.texts_embedded > 0

    def test_changing_the_embedder_identity_forces_re_embed(self, corpus_dir, tmp_path):
        """The 2026-07-02 MiniLM -> bge cutover class of bug: a manifest written
        by a different embedder must never authorize a skip."""
        store1 = _store(tmp_path, CountingEmbedder(dim=256), name="a.chroma")
        ingest(corpus_dir, store1)

        # Same store dir is off-limits (dim guard), so use a fresh dir but copy
        # the manifest across -- simulating a manifest that outlived its embedder.
        manifest_src = (tmp_path / "a.chroma" / MANIFEST_FILENAME).read_text("utf-8")
        store2_dir = tmp_path / "b.chroma"
        embedder2 = CountingEmbedder(dim=128)
        store2 = _store(tmp_path, embedder2, name="b.chroma")
        (store2_dir / MANIFEST_FILENAME).write_text(manifest_src, encoding="utf-8")

        embedder2.texts_embedded = 0
        report = ingest(corpus_dir, store2)
        assert report.files_unchanged == 0
        assert self._full_reembed_happened(embedder2)

    def test_changed_chunking_params_force_re_embed(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store, max_chars=200, overlap=20)
        assert report.files_unchanged == 0
        assert self._full_reembed_happened(embedder)

    def test_corrupt_manifest_is_ignored(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)
        (tmp_path / "inc.chroma" / MANIFEST_FILENAME).write_text("{not json", "utf-8")

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)
        assert report.files_unchanged == 0
        assert self._full_reembed_happened(embedder)

    def test_missing_manifest_is_a_full_run(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)
        (tmp_path / "inc.chroma" / MANIFEST_FILENAME).unlink()

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)
        assert report.files_unchanged == 0
        assert self._full_reembed_happened(embedder)

    def test_populated_manifest_against_empty_store_re_embeds(self, corpus_dir, tmp_path):
        """Desync guard: trusting the manifest here would leave the index empty
        forever, with a green 'nothing changed' report every run."""
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)

        ids = [m["source"] + "::" + str(m["chunk_index"]) for m in store.all_metadatas()]
        store.delete(ids=ids)
        assert store.count() == 0

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)
        assert report.files_unchanged == 0
        assert store.count() > 0
        assert self._full_reembed_happened(embedder)

    def test_incremental_false_forces_full_re_embed(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store, incremental=False)
        assert report.files_unchanged == 0
        assert report.files_ingested == 2
        assert self._full_reembed_happened(embedder)

    def test_full_run_still_leaves_a_usable_manifest(self, corpus_dir, tmp_path):
        """A --full/--clean rebuild must WRITE a manifest even though it does not
        READ one -- otherwise the next scheduled tick is a full re-embed again,
        and the expensive path becomes permanent."""
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store, incremental=False)
        assert (tmp_path / "inc.chroma" / MANIFEST_FILENAME).is_file()

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)
        assert report.files_unchanged == 2
        assert embedder.texts_embedded == 0

    def test_in_memory_store_still_works(self, corpus_dir):
        """path=None has nowhere to persist a manifest; must not crash."""
        store = VectorStore(path=None, collection_name="knowledge", embedder=HashEmbedder())
        report = ingest(corpus_dir, store)
        assert report.files_ingested == 2
        assert report.incremental is False


class TestManifestFile:
    def test_manifest_is_written_inside_the_store_dir(self, corpus_dir, tmp_path):
        """Co-located so --clean's rmtree drops both together -- they can never
        outlive each other and desync."""
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        assert (tmp_path / "inc.chroma" / MANIFEST_FILENAME).is_file()

    def test_manifest_records_hash_and_chunk_count_per_file(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        raw = json.loads((tmp_path / "inc.chroma" / MANIFEST_FILENAME).read_text("utf-8"))
        assert set(raw["files"]) == {"dogs.md", "cats.md"}
        expected = content_hash((corpus_dir / "dogs.md").read_text(encoding="utf-8"))
        assert raw["files"]["dogs.md"]["sha256"] == expected
        assert raw["files"]["dogs.md"]["chunks"] >= 1

    def test_manifest_leaves_no_temp_files_behind(self, corpus_dir, tmp_path):
        store = _store(tmp_path)
        ingest(corpus_dir, store)
        leftovers = list((tmp_path / "inc.chroma").glob("*.tmp"))
        assert leftovers == []

    def test_identity_mismatch_returns_empty_manifest(self, tmp_path):
        ident = RunIdentity("HashEmbedder", 256, "knowledge", 1000, 100)
        other = RunIdentity("BgeEmbedder", 1024, "knowledge", 1000, 100)
        m = IngestManifest(ident)
        m.record("a.md", "deadbeef", 3)
        m.save(tmp_path)

        assert load_manifest(tmp_path, ident).files
        assert load_manifest(tmp_path, other).files == {}

    def test_content_hash_ignores_nothing_but_is_stable(self):
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")


# ---------------------------------------------------------------------------
# metadata backfill (RM-fixafter-ragmcp slice 1) -- a `_doc_class`/`_metadata`
# rule change must reach ALREADY-EMBEDDED, content-unchanged chunks on the
# next incremental run, not just newly (re)embedded ones.
#
# FIRES: a file whose manifest record predates CURRENT_METADATA_VERSION (the
#        real-world shape: a pre-doc_class-feature manifest has no `metav` at
#        all, which reads as 0) gets its stored chunks' metadata refreshed --
#        WITHOUT any re-embedding, which is the entire point of doing this as
#        a metadata-only update rather than forcing a full re-embed.
# SILENT: a file whose manifest record already carries the current
#         `metav` triggers zero `update_metadatas` calls on the next run.
# ---------------------------------------------------------------------------


class TestMetadataBackfill:
    def test_stale_metadata_version_backfills_without_reembedding(self, corpus_dir, tmp_path):
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)
        ingest(corpus_dir, store)  # writes metav=CURRENT_METADATA_VERSION already

        # Simulate a PRE-feature manifest + store: strip `metav` from the
        # on-disk manifest and blank `doc_class` in the store's metadata --
        # mirrors the live shape the code reviewer measured (99.89% of chunks
        # missing doc_class after an ordinary incremental run).
        manifest_path = tmp_path / "inc.chroma" / MANIFEST_FILENAME
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rec in raw["files"].values():
            rec.pop("metav", None)
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")

        metas = store.all_metadatas()
        ids = [f"{m['source']}::{m['chunk_index']}" for m in metas]
        blanked = [{**m, "doc_class": ""} for m in metas]
        store.update_metadatas(ids=ids, metadatas=blanked)
        assert all(m["doc_class"] == "" for m in store.all_metadatas())

        embedder.texts_embedded = 0
        report = ingest(corpus_dir, store)

        assert embedder.texts_embedded == 0, (
            "backfilling stale metadata must not re-embed -- that defeats the "
            "entire incremental path"
        )
        assert report.chunks_added == 0
        assert report.chunks_metadata_refreshed == len(ids), (
            "chunks_metadata_refreshed must count the backfill so an operator "
            "can see one happened from the run summary alone"
        )
        assert report.chunks_metadata_missing == 0, (
            "SILENT control: a fully-synced store must report zero missing"
        )
        assert all(m["doc_class"] == "note" for m in store.all_metadatas()), (
            "doc_class was not backfilled onto already-embedded chunks"
        )

        raw_after = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert all(
            rec.get("metav") == _DEFAULT_METAV for rec in raw_after["files"].values()
        ), "manifest must record the new metadata version after a backfill"

    def test_current_metadata_version_triggers_zero_updates(
        self, corpus_dir, tmp_path, monkeypatch
    ):
        store = _store(tmp_path)
        ingest(corpus_dir, store)  # already at CURRENT_METADATA_VERSION

        calls: list[int] = []
        original = store.update_metadatas

        def spy(*, ids, metadatas):
            calls.append(len(ids))
            return original(ids=ids, metadatas=metadatas)

        monkeypatch.setattr(store, "update_metadatas", spy)

        report = ingest(corpus_dir, store)

        assert calls == [], "already-current metadata must not be re-written"
        assert report.files_unchanged == 2
        assert report.chunks_metadata_refreshed == 0, (
            "SILENT control: nothing changed, so the counter must read zero"
        )

    def test_desynced_id_missing_from_store_rolls_back_metav_and_is_counted(
        self, corpus_dir, tmp_path
    ):
        # Finding #12: update_metadatas() silently ignores an id the store
        # doesn't have (Chroma warns, doesn't raise). Simulate a manifest/
        # store desync -- the manifest still claims dogs.md::0 is stored, but
        # it isn't (e.g. an out-of-band delete, or a crash between a prior
        # add() and manifest.save()) -- and prove the fix does NOT silently
        # stamp that file current.
        store = _store(tmp_path)
        ingest(corpus_dir, store)  # writes metav=CURRENT_METADATA_VERSION

        manifest_path = tmp_path / "inc.chroma" / MANIFEST_FILENAME
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["files"]["dogs.md"].pop("metav", None)  # force meta_stale for dogs.md only
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")

        store.delete(ids=["dogs.md::0"])  # the desync: manifest still believes it's there

        report = ingest(corpus_dir, store)

        assert report.chunks_metadata_missing >= 1, (
            "the missing id must be counted, not silently absorbed"
        )
        raw_after = json.loads(manifest_path.read_text(encoding="utf-8"))
        dogs_metav = raw_after["files"]["dogs.md"].get("metav")
        assert dogs_metav != _DEFAULT_METAV, (
            "a file whose refresh could not be verified present must NOT be "
            "stamped current -- the next incremental run must retry it, not "
            "trust a permanently-missing doc_class behind an exit-0 run"
        )
        # SILENT half in the same test: cats.md was never touched, so its
        # refresh (if any) fully lands and its metav DOES get stamped.
        cats_metav = raw_after["files"]["cats.md"].get("metav")
        assert cats_metav == _DEFAULT_METAV

    def test_missing_chunk_is_re_added_not_stranded_forever(self, corpus_dir, tmp_path):
        # Finding #1 (RM-fixafter3): the finding-#12 rollback above only pops
        # `metav`, which forces meta_stale again -- but leaves the manifest's
        # `kept`/`chunks` bookkeeping still claiming the missing id is stored.
        # `prev_ids` is built from that bookkeeping, so `to_add` never
        # includes the missing chunk and every subsequent run repeats the
        # same no-op metadata-refresh attempt forever: dogs.md::0 never
        # reappears and chunks_metadata_missing never returns to 0.
        #
        # FIRES: dogs.md::0 deleted out-of-band + meta_stale forced -> ONE
        #        more incremental run re-embeds it and it exists again.
        # SILENT: a fully in-sync store needs zero re-adds on a repeat run
        #        (already proven by TestSkipsUnchangedFiles above).
        store = _store(tmp_path)
        ingest(corpus_dir, store)  # writes metav=CURRENT_METADATA_VERSION

        manifest_path = tmp_path / "inc.chroma" / MANIFEST_FILENAME
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["files"]["dogs.md"].pop("metav", None)  # force meta_stale for dogs.md only
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")

        store.delete(ids=["dogs.md::0"])  # out-of-band desync

        report1 = ingest(corpus_dir, store)  # detects the gap
        assert report1.chunks_metadata_missing >= 1
        assert store.existing_ids(["dogs.md::0"]) == set(), (
            "sanity: the chunk is genuinely absent after run 1"
        )

        report2 = ingest(corpus_dir, store)  # must re-embed the missing chunk
        assert report2.chunks_added >= 1, (
            "the missing chunk must be re-added on a subsequent run instead "
            "of being stranded behind an infinite no-op metadata-refresh retry"
        )
        assert store.existing_ids(["dogs.md::0"]) == {"dogs.md::0"}, (
            "dogs.md::0 must be back in the store"
        )
        assert report2.chunks_metadata_missing == 0, (
            "once genuinely re-added, this run has nothing left to report missing"
        )

        raw_final = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert raw_final["files"]["dogs.md"].get("metav") == _DEFAULT_METAV, (
            "once genuinely re-added, the file's metav must be stamped current again"
        )


class TestClassifierConfigMetavFingerprint:
    """RM-fixafter2 slice 5, finding #2 -- meta_stale keyed only off
    CURRENT_METADATA_VERSION means changing handoff_mirror_dir/
    handoff_mirror_basenames against an already-embedded store is a silent
    no-op: doc_class in the store stays whatever the OLD config produced
    until an operator remembers to --full/--clean rebuild. Folding a
    fingerprint of the classifier config into the per-file metav makes a
    config change itself a metadata-stale trigger.

    FIRES: run 1 (default config) -> "note"; run 2, same store, with
           handoff_mirror_dir="sessions" -> "handoff", with NO re-embed.
    SILENT: same config run twice -> zero update_metadatas calls.
    """

    def test_config_change_triggers_metadata_only_backfill(self, tmp_path):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        (sessions / "handoff.md").write_text(
            "No frontmatter. Default config (dir='context') says 'note'.\n",
            encoding="utf-8",
        )
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)

        ingest(tmp_path, store)  # default handoff_mirror_dir="context"
        metas = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
        assert metas["sessions/handoff.md"] == "note"

        embedder.texts_embedded = 0
        report = ingest(tmp_path, store, handoff_mirror_dir="sessions")

        assert embedder.texts_embedded == 0, (
            "a classifier config change must backfill via metadata refresh, "
            "never a re-embed"
        )
        assert report.chunks_metadata_refreshed > 0
        metas_after = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
        assert metas_after["sessions/handoff.md"] == "handoff"

    def test_unchanged_config_triggers_zero_refresh(self, tmp_path, monkeypatch):
        ctx = tmp_path / "context"
        ctx.mkdir()
        (ctx / "handoff.md").write_text("No frontmatter.\n", encoding="utf-8")
        store = _store(tmp_path)
        ingest(tmp_path, store)

        calls: list[int] = []
        original = store.update_metadatas

        def spy(*, ids, metadatas):
            calls.append(len(ids))
            return original(ids=ids, metadatas=metadatas)

        monkeypatch.setattr(store, "update_metadatas", spy)

        report = ingest(tmp_path, store)  # identical config, identical content

        assert calls == [], "SILENT control: unchanged config must not re-write metadata"
        assert report.chunks_metadata_refreshed == 0


class TestGrammarConstantsFoldedIntoFingerprint:
    """RM-fixafter3, finding #3 -- the fingerprint hashed only
    handoff_mirror_dir/handoff_mirror_basenames; a GRAMMAR edit (the stems
    tuple, the leading-stem regex skeleton, or MARKDOWN_EXTS) was a silent
    no-op on a populated store. _classifier_config_fingerprint now also
    hashes the fully-built leading-stem regex PATTERN STRING for the
    effective handoff_mirror_stems -- that one string already embeds the
    stems tuple, the date-prefix/separator skeleton, AND the extension
    alternation, so an edit to any of the three changes the fingerprint.

    FIRES (runtime, ingest()-level): run 1 (default stems) -> "handoff" for
        a file only "active" catches; run 2, same store, narrower
        handoff_mirror_stems -> reclassified via metadata-only refresh, NO
        re-embed.
    FIRES (source-edit simulation, unit-level): a monkeypatched
        _HANDOFF_LEADING_RE (standing in for a future grammar edit shipped
        in this module) changes the fingerprint.
    SILENT: same stems run twice -> zero update_metadatas calls; same
        fingerprint inputs called twice -> identical fingerprint.
    """

    def test_stems_change_triggers_metadata_only_backfill(self, tmp_path):
        ctx = tmp_path / "context"
        ctx.mkdir()
        (ctx / "active-clients-2026.md").write_text(
            "Not a session mirror -- an ordinary business file.\n", encoding="utf-8"
        )
        embedder = CountingEmbedder()
        store = _store(tmp_path, embedder)

        ingest(tmp_path, store)  # default stems -> "handoff"
        metas = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
        assert metas["context/active-clients-2026.md"] == "handoff"

        embedder.texts_embedded = 0
        report = ingest(tmp_path, store, handoff_mirror_stems=("handoff",))

        assert embedder.texts_embedded == 0, (
            "a grammar (stems) change must backfill via metadata refresh, "
            "never a re-embed"
        )
        assert report.chunks_metadata_refreshed > 0
        metas_after = {m["source"]: m["doc_class"] for m in store.all_metadatas()}
        assert metas_after["context/active-clients-2026.md"] == "note"

    def test_unchanged_stems_triggers_zero_refresh(self, tmp_path, monkeypatch):
        ctx = tmp_path / "context"
        ctx.mkdir()
        (ctx / "active-clients-2026.md").write_text(
            "Not a session mirror.\n", encoding="utf-8"
        )
        store = _store(tmp_path)
        ingest(tmp_path, store)

        calls: list[int] = []
        original = store.update_metadatas

        def spy(*, ids, metadatas):
            calls.append(len(ids))
            return original(ids=ids, metadatas=metadatas)

        monkeypatch.setattr(store, "update_metadatas", spy)

        report = ingest(tmp_path, store)  # identical config, identical content

        assert calls == [], "SILENT control: unchanged stems must not re-write metadata"
        assert report.chunks_metadata_refreshed == 0

    def test_fingerprint_differs_for_narrowed_stems(self):
        default_fp = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES, DEFAULT_HANDOFF_MIRROR_STEMS
        )
        narrowed_fp = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES, ("handoff",)
        )
        assert narrowed_fp != default_fp

    def test_fingerprint_identical_for_same_stems_called_twice(self):
        # SILENT control for the unit-level check above.
        a = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES, DEFAULT_HANDOFF_MIRROR_STEMS
        )
        b = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES, DEFAULT_HANDOFF_MIRROR_STEMS
        )
        assert a == b

    def test_fingerprint_changes_if_leading_regex_pattern_edited(self, monkeypatch):
        # Ties the fingerprint to _HANDOFF_LEADING_RE's actual pattern text
        # (which embeds stems, the date-prefix/separator skeleton, AND
        # MARKDOWN_EXTS all in one string) -- simulates a future grammar
        # edit shipped directly in the module (not via the stems parameter)
        # to prove it can't land as a silent no-op on a populated store.
        import re as re_module

        from rag_mcp import ingest as ingest_mod

        base = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES
        )
        monkeypatch.setattr(
            ingest_mod,
            "_HANDOFF_LEADING_RE",
            re_module.compile(
                r"^(?:handoff|active|resume)(?:[-_.].*)?(?:\.md|\.markdown|\.mdx)$",
                re_module.IGNORECASE,
            ),
        )
        changed = _classifier_config_fingerprint(
            DEFAULT_HANDOFF_MIRROR_DIR, DEFAULT_HANDOFF_MIRROR_BASENAMES
        )
        assert changed != base, (
            "a grammar edit to the leading-stem regex pattern (stems, date "
            "prefix, separator class, or extension alternation) must change "
            "the fingerprint, or a populated store silently keeps stale "
            "doc_class forever after the edit ships"
        )
