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

from rag_mcp.ingest import ingest
from rag_mcp.manifest import (
    MANIFEST_FILENAME,
    IngestManifest,
    RunIdentity,
    content_hash,
    load_manifest,
)
from rag_mcp.store import HashEmbedder, VectorStore


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
