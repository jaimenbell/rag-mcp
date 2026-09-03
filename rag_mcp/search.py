"""search_knowledge - the retrieval entrypoint behind the MCP tool.

Reliability properties (the whole pitch), all proven by tests:

  * CITED      - every result carries source path + heading + chunk index.
  * AUTH-SCOPED - results are confined to files under `corpus_root`; any hit whose
                  source path escapes the root (absolute path, '..' traversal) is
                  dropped. The tool cannot surface anything outside its corpus.
  * FAIL-SOFT   - a down/empty store, a bad query, or any internal error returns a
                  STRUCTURED error object. This function never raises; it cannot crash
                  the agent that calls it.
  * BOUNDED     - k is clamped to [1, MAX_K]; query length is capped at MAX_QUERY_LEN
                  chars before it ever reaches the embedder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_K = 20
MAX_QUERY_LEN = 4000  # generous for a real search query; caps embedder input cost

# Closed set of doc_class values search_knowledge accepts. Deliberately closed
# (not "any string the store happens to contain") so a typo -- "notes" instead
# of "note", or wrong-case "Note" -- fails LOUD as invalid_doc_class instead of
# silently returning an empty, ok:true result that reads as "no matches" rather
# than "your filter was wrong". Case-sensitive; matches rag_mcp.ingest._doc_class
# exactly. Extend here (and in server.py's tool schema enum) when _doc_class
# grows a new classification.
ALLOWED_DOC_CLASSES = frozenset({"note", "handoff"})


def _within_root(root: Path, source: str) -> bool:
    """True iff `source` (relative to root) stays inside root (no escape)."""
    if not source:
        return False
    try:
        resolved = (root / source).resolve()
        resolved.relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _error(kind: str, message: str, *, k: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": {"type": kind, "message": message},
        "results": [],
    }
    if k is not None:
        payload["k"] = k
    return payload


def search_knowledge(
    query: str,
    k: int = 5,
    *,
    store: Any,
    corpus_root: Path | str,
    doc_class: str | None = None,
) -> dict[str, Any]:
    """Embed the query, retrieve top-k chunks, return text + citations.

    ``doc_class``, when given, restricts results to chunks whose ingest-time
    metadata carries that exact ``doc_class`` (see rag_mcp.ingest._doc_class --
    "note" vs "handoff" today; the full set is ``ALLOWED_DOC_CLASSES``).
    ``None`` (the default) applies no filter and is byte-identical to the
    pre-filter behavior. Matching is CASE-SENSITIVE and against a CLOSED set:
    a value that is not a string, or a string not in ``ALLOWED_DOC_CLASSES``
    (wrong case, a typo, or an empty string), returns an ``invalid_doc_class``
    error rather than silently matching nothing -- a typo in a filter value
    should never look identical to "no results". A syntactically valid
    ``doc_class`` that simply has no matches in this store (e.g. a store whose
    chunks predate the field, or predate a newly added class) still fails soft
    to an empty, ``ok: true`` result -- that is a real, expected "nothing
    matched", not a caller error.

    Always returns a dict; never raises. See module docstring for the contract.
    """
    if not isinstance(query, str) or not query.strip():
        return _error("invalid_query", "query must be a non-empty string")

    if len(query) > MAX_QUERY_LEN:
        return _error(
            "invalid_query",
            f"query exceeds max length of {MAX_QUERY_LEN} characters "
            f"(got {len(query)}); shorten the query",
        )

    if doc_class is not None and (
        not isinstance(doc_class, str) or doc_class not in ALLOWED_DOC_CLASSES
    ):
        return _error(
            "invalid_doc_class",
            f"doc_class must be one of {sorted(ALLOWED_DOC_CLASSES)} "
            f"(case-sensitive) or omitted; got {doc_class!r}",
        )

    # Bound k.
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    effective_k = max(1, min(k, MAX_K))

    root = Path(corpus_root)

    # Fail-soft: a down store must not crash the agent.
    try:
        if store.count() == 0:
            return _error("empty_store", "knowledge store is empty; run ingest first", k=effective_k)
    except Exception as exc:  # noqa: BLE001 - any backend failure -> structured error
        return _error("store_unreachable", f"could not reach the knowledge store: {exc}", k=effective_k)

    # doc_class is already validated above: either None (no filter) or a
    # member of ALLOWED_DOC_CLASSES.
    where = {"doc_class": doc_class} if doc_class is not None else None

    try:
        hits = store.query(query, k=effective_k, where=where)
    except Exception as exc:  # noqa: BLE001
        return _error("store_unreachable", f"query failed: {exc}", k=effective_k)

    results: list[dict[str, Any]] = []
    dropped_out_of_scope = 0
    for hit in hits:
        meta = hit.get("metadata") or {}
        source = str(meta.get("source", ""))
        # AUTH-SCOPE enforcement: refuse anything outside the configured corpus root.
        if not _within_root(root, source):
            dropped_out_of_scope += 1
            continue
        distance = hit.get("distance")
        score = None
        if isinstance(distance, (int, float)):
            score = round(max(0.0, 1.0 - float(distance)), 4)
        citation: dict[str, Any] = {
            "source": source,
            "heading": meta.get("heading", "") or None,
            "chunk_index": meta.get("chunk_index"),
        }
        # Snapshot-series chunks only. Ingest collapses a daily report block that
        # is byte-identical to the previous day's, keeping the FIRST occurrence.
        # These fields are what keeps that lossless for citation: the surviving
        # chunk states its own date and the later dates it also stood for, so
        # "what did this say on <date>" is still answerable. Absent for ordinary
        # notes -- the citation shape does not change for them.
        if meta.get("snapshot_date"):
            citation["snapshot_date"] = meta["snapshot_date"]
            repeat_dates = str(meta.get("repeat_dates") or "")
            if repeat_dates:
                citation["also_unchanged_on"] = repeat_dates
                citation["snapshots_covered"] = meta.get("repeat_count")
        results.append(
            {
                "text": hit.get("document", ""),
                "citation": citation,
                "score": score,
            }
        )

    return {
        "ok": True,
        "query": query,
        "k": effective_k,
        "count": len(results),
        "dropped_out_of_scope": dropped_out_of_scope,
        "results": results,
    }
