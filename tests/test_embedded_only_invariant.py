"""Pin the embedded-only invariant that is rag-mcp's entire CVE mitigation.

chromadb's Python FastAPI server carries a pre-auth code-injection flaw
(CVE-2026-45829, CVSS 10.0, CWE-94/502; GHSA-f4j7-r4q5-qw2c) affecting
1.0.0 through at least 1.5.9. There is no patched release -- 1.5.9 is the
newest version published and it is itself in the affected range, so pinning
forward is not a remediation and neither is upgrading.

What actually protects us is architectural: rag-mcp only ever runs chromadb
embedded and in-process, over an stdio MCP transport, and never starts the
FastAPI server that carries the flaw. The vulnerable module is installed in
site-packages; it is simply never imported. That invariant held by convention
and by nothing else until this file existed -- a single well-meaning change to
`HttpClient` would silently re-open a critical remote attack surface without
failing anything.

SCOPE: the `rag_mcp/` package source only. Deliberately NOT `tests/` -- this
file necessarily contains every forbidden token below, and a scan that
included its own tests would either match itself or need a carve-out that
would rot. Test code cannot start a production server, so package-only is
the correct and sufficient boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "rag_mcp"

# Client constructors that keep chromadb in-process. Anything else is a
# deployment-shape change and must be a deliberate, reviewed decision.
_ALLOWED_CLIENTS = {"EphemeralClient", "PersistentClient"}

# Substrings that indicate a networked chroma deployment. Kept as plain text
# so the test catches a string, an import, or a config value alike.
_FORBIDDEN_MARKERS = (
    "HttpClient",
    "AsyncHttpClient",
    "chromadb.server",
    "chroma_server_host",
    "chroma_server_http_port",
    "CHROMA_SERVER",
)


def _package_sources() -> list[Path]:
    sources = sorted(_PACKAGE_ROOT.rglob("*.py"))
    assert sources, f"no sources found under {_PACKAGE_ROOT} -- scan would pass vacuously"
    return sources


@pytest.mark.parametrize("marker", _FORBIDDEN_MARKERS)
def test_no_networked_chroma_marker_in_package(marker: str) -> None:
    """No source file may reference a client/server chroma deployment."""
    offenders = [
        f"{path.relative_to(_PACKAGE_ROOT.parent)}:{lineno}"
        for path in _package_sources()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if marker in line
    ]
    assert not offenders, (
        f"{marker!r} found in {offenders}. rag-mcp must run chromadb embedded "
        f"only -- the FastAPI server path carries CVE-2026-45829 (no fix "
        f"exists). If this deployment shape is changing on purpose, that is a "
        f"security decision, not a test failure to silence."
    )


def test_only_embedded_chromadb_clients_are_constructed() -> None:
    """Every `chromadb.<X>Client(...)` call must be an embedded constructor."""
    constructed: set[str] = set()
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr.endswith("Client"):
                constructed.add(func.attr)
            elif isinstance(func, ast.Name) and func.id.endswith("Client"):
                constructed.add(func.id)

    assert constructed, "no chromadb client construction found -- the scan is not observing anything"
    unexpected = constructed - _ALLOWED_CLIENTS
    assert not unexpected, (
        f"non-embedded chromadb client(s) constructed: {sorted(unexpected)}. "
        f"Allowed: {sorted(_ALLOWED_CLIENTS)}."
    )
