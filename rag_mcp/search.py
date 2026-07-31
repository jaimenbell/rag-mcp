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
) -> dict[str, Any]:
    """Embed the query, retrieve top-k chunks, return text + citations.

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

    try:
        hits = store.query(query, k=effective_k)
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
