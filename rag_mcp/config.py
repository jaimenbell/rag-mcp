"""Runtime configuration for rag-mcp, resolved from env vars.

  RAG_MCP_CORPUS_ROOT  - absolute path to the corpus the tool may read (auth scope).
  RAG_MCP_DB_PATH      - absolute path to the on-disk Chroma store.
  RAG_MCP_COLLECTION   - collection name (default 'knowledge').
  RAG_MCP_EMBEDDER     - 'default' (real ONNX MiniLM) or 'hash' (offline test embedder).

The corpus root doubles as the security boundary: retrieval results are confined to
files under it, and the resolved root must exist as a directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .store import DefaultEmbedder, Embedder, HashEmbedder, VectorStore


@dataclass(frozen=True)
class Config:
    corpus_root: Path
    db_path: Path
    collection: str = "knowledge"
    embedder_name: str = "default"

    @classmethod
    def from_env(cls) -> "Config":
        root = os.environ.get("RAG_MCP_CORPUS_ROOT")
        db = os.environ.get("RAG_MCP_DB_PATH")
        if not root:
            raise ValueError("RAG_MCP_CORPUS_ROOT is not set")
        if not db:
            raise ValueError("RAG_MCP_DB_PATH is not set")
        return cls(
            corpus_root=Path(root).resolve(),
            db_path=Path(db).resolve(),
            collection=os.environ.get("RAG_MCP_COLLECTION", "knowledge"),
            embedder_name=os.environ.get("RAG_MCP_EMBEDDER", "default"),
        )

    def make_embedder(self) -> Embedder:
        if self.embedder_name == "hash":
            return HashEmbedder()
        return DefaultEmbedder()

    def open_store(self) -> VectorStore:
        return VectorStore(
            path=str(self.db_path),
            collection_name=self.collection,
            embedder=self.make_embedder(),
        )
