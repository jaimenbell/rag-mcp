"""Phase 2 - search_knowledge reliability proof: citations, auth-scope, fail-soft, bounds."""
from __future__ import annotations

import pytest

from rag_mcp.ingest import ingest
from rag_mcp.search import MAX_K, MAX_QUERY_LEN, search_knowledge


@pytest.fixture
def populated(corpus, store):
    ingest(corpus, store)
    return store, corpus  # corpus dir is the auth-scope root


def test_relevant_result_with_citation(populated):
    store, root = populated
    res = search_knowledge("loyal dog that barks at owner", k=3, store=store, corpus_root=root)
    assert res["ok"] is True
    assert res["results"]
    top = res["results"][0]
    assert top["citation"]["source"] == "dogs.md"
    assert "dog" in top["text"].lower()


def test_citation_fields_present(populated):
    store, root = populated
    res = search_knowledge("cats purr", k=2, store=store, corpus_root=root)
    for r in res["results"]:
        cit = r["citation"]
        assert set(cit) == {"source", "heading", "chunk_index"}
        assert cit["source"]


def test_invalid_query_rejected(populated):
    store, root = populated
    for bad in ("", "   ", None):
        res = search_knowledge(bad, k=3, store=store, corpus_root=root)
        assert res["ok"] is False
        assert res["error"]["type"] == "invalid_query"


def test_oversized_query_rejected(populated):
    store, root = populated
    over = "a" * (MAX_QUERY_LEN + 1)
    res = search_knowledge(over, k=3, store=store, corpus_root=root)
    assert res["ok"] is False
    assert res["error"]["type"] == "invalid_query"
    assert str(MAX_QUERY_LEN) in res["error"]["message"]


def test_query_at_max_length_allowed(populated):
    store, root = populated
    exactly = "dog " * (MAX_QUERY_LEN // 4)
    exactly = exactly[:MAX_QUERY_LEN]
    res = search_knowledge(exactly, k=3, store=store, corpus_root=root)
    assert res["ok"] is True


def test_empty_store_structured_error(store, tmp_path):
    # store fixture is empty (nothing ingested).
    res = search_knowledge("anything", k=3, store=store, corpus_root=tmp_path)
    assert res["ok"] is False
    assert res["error"]["type"] == "empty_store"
    assert res["results"] == []


def test_store_unreachable_returns_structured_error(tmp_path):
    class DownStore:
        def count(self):
            raise ConnectionError("backend down")

        def query(self, *a, **k):
            raise ConnectionError("backend down")

    res = search_knowledge("q", k=3, store=DownStore(), corpus_root=tmp_path)
    assert res["ok"] is False
    assert res["error"]["type"] == "store_unreachable"
    # Crucially, it returned a dict instead of raising.


def test_out_of_corpus_source_refused(store, tmp_path, embedder):
    # Inject a chunk whose source escapes the corpus root.
    store.add(
        ids=["evil::0"],
        documents=["secret password leaked from outside the corpus"],
        metadatas=[{"source": "../../secrets.md", "heading": "", "chunk_index": 0}],
    )
    res = search_knowledge("secret password leaked", k=5, store=store, corpus_root=tmp_path)
    assert res["ok"] is True
    sources = [r["citation"]["source"] for r in res["results"]]
    assert "../../secrets.md" not in sources
    assert res["dropped_out_of_scope"] >= 1


def test_k_bounds_respected(populated):
    store, root = populated
    # Over-large k clamps to MAX_K.
    res_hi = search_knowledge("animal", k=1000, store=store, corpus_root=root)
    assert res_hi["k"] == MAX_K
    assert len(res_hi["results"]) <= MAX_K
    # Non-positive k clamps up to 1.
    res_lo = search_knowledge("animal", k=0, store=store, corpus_root=root)
    assert res_lo["k"] == 1
    assert len(res_lo["results"]) <= 1


# ---------------------------------------------------------------------------
# doc_class filter (RM-ragmcp-docclass slice 3)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_corpus(tmp_path):
    """One plain note + one handoff-shaped mirror, same body text so a query
    for that text would otherwise return both -- isolates the filter's effect.
    """
    (tmp_path / "note.md").write_text(
        "# Note\n\nA loyal dog barks at the mail carrier every single day.\n",
        encoding="utf-8",
    )
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "handoff.md").write_text(
        "---\ntype: handoff\n---\n"
        "# Handoff\n\nA loyal dog barks at the mail carrier every single day.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_doc_class_filter_restricts_to_matching_class(mixed_corpus, store):
    # FIRES: doc_class="note" excludes the handoff-classed mirror.
    ingest(mixed_corpus, store)
    res = search_knowledge(
        "loyal dog barks", k=5, store=store, corpus_root=mixed_corpus, doc_class="note"
    )
    assert res["ok"] is True
    assert res["results"]
    assert all(r["citation"]["source"] == "note.md" for r in res["results"])


def test_doc_class_none_returns_all_classes(mixed_corpus, store):
    # SILENT: omitting doc_class is unchanged -- both classes come back.
    ingest(mixed_corpus, store)
    res = search_knowledge("loyal dog barks", k=5, store=store, corpus_root=mixed_corpus)
    sources = {r["citation"]["source"] for r in res["results"]}
    assert sources == {"note.md", "context/handoff.md"}


def test_doc_class_filter_field_absent_from_store_returns_empty_not_error(populated):
    # The shared `populated` fixture's corpus predates doc_class entirely in
    # this test's own conftest sense (ingest always sets it now, but a filter
    # on a value nothing carries must still fail soft to an empty, ok result).
    store, root = populated
    res = search_knowledge(
        "loyal dog", k=5, store=store, corpus_root=root, doc_class="archive"
    )
    assert res["ok"] is True
    assert res["results"] == []
