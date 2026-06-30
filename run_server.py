#!/usr/bin/env python3
"""Entrypoint for the rag-mcp MCP server.

Run as a top-level script (e.g. `python run_server.py`) so the repo root lands on
sys.path[0] and the `rag_mcp` package imports cleanly. This is the path referenced
by mcp.yaml's runtime.script, matching how the mcp-factory hub spawns servers.
"""
from __future__ import annotations

import asyncio

from rag_mcp.server import _main

if __name__ == "__main__":
    asyncio.run(_main())
