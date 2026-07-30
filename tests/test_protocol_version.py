"""Protocol-revision guards: catch a SILENT DOWNGRADE off 2026-07-28.

Why this file exists
--------------------
`mcp` 2.0.0 is the first SDK release implementing protocol revision 2026-07-28.
Nothing in the server code names a version -- it is inherited entirely from the
pinned SDK. So a dependency rollback (or a resolver picking an older wheel)
would silently drop this server back to an older revision with a green suite
and no other symptom. These tests make that loud.

The 2026-07-28 revision is NOT reachable via the `initialize` handshake. It is a
"modern" revision using a stateless per-request `_meta` envelope, discovered via
`server/discover`. The era is chosen per-connection by the CLIENT's first frame
(mcp/server/runner.py::serve_dual_era_loop), so a server can serve both:

  * a legacy client opening with `initialize`  -> caps at LATEST_HANDSHAKE_VERSION
    (2025-11-25), and that is correct behavior, not a downgrade;
  * a modern client stamping the envelope      -> 2026-07-28.

Both paths are asserted below, so we can tell a genuine regression apart from a
legacy client simply being legacy.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from mcp import types
from mcp.client.client import Client
from mcp_types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    KNOWN_PROTOCOL_VERSIONS,
    LATEST_HANDSHAKE_VERSION,
    LATEST_PROTOCOL_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)

from rag_mcp import server as srv

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_REVISION = "2026-07-28"


class TestSdkImplementsTargetRevision:
    """The installed SDK actually speaks 2026-07-28."""

    def test_latest_protocol_version_is_target(self):
        assert LATEST_PROTOCOL_VERSION == TARGET_REVISION, (
            f"SDK's newest protocol revision is {LATEST_PROTOCOL_VERSION!r}, "
            f"expected {TARGET_REVISION!r} -- the mcp pin was probably rolled back."
        )

    def test_target_is_a_modern_envelope_revision(self):
        """Pins the architectural fact the rest of this file depends on."""
        assert TARGET_REVISION in MODERN_PROTOCOL_VERSIONS
        assert TARGET_REVISION not in HANDSHAKE_PROTOCOL_VERSIONS
        assert TARGET_REVISION in KNOWN_PROTOCOL_VERSIONS

    def test_handshake_era_still_caps_where_we_think(self):
        """If the SDK ever makes 2026-07-28 handshake-reachable, this fails and
        the comments/docs in this repo need revisiting."""
        assert LATEST_HANDSHAKE_VERSION == "2025-11-25"


class TestPinIsExactAndCurrent:
    """This repo pins exact by policy; the pin is what delivers the revision."""

    def _mcp_pin(self) -> str:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        deps = pyproject["project"]["dependencies"]
        pins = [d for d in deps if d.replace(" ", "").startswith("mcp==")]
        assert len(pins) == 1, f"expected exactly one exact mcp pin, got {pins!r}"
        return pins[0].replace(" ", "")

    def test_mcp_pin_is_exact(self):
        pin = self._mcp_pin()
        assert pin.startswith("mcp=="), f"mcp must be pinned exactly, got {pin!r}"

    def test_pinned_version_is_at_least_2_0_0(self):
        """<2 cannot implement 2026-07-28 (the SDK's own README says keep a <2
        bound until migrated -- this repo HAS migrated, so the floor is 2.0.0)."""
        pin = self._mcp_pin()
        version = pin.split("==", 1)[1]
        major = int(version.split(".", 1)[0])
        assert major >= 2, f"mcp {version} predates the 2026-07-28 revision"

    def test_requirements_txt_agrees_with_pyproject(self):
        """Two files pin mcp; drift between them is how a rollback sneaks in."""
        req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        lines = [
            ln.strip().replace(" ", "")
            for ln in req.splitlines()
            if ln.strip().replace(" ", "").startswith("mcp==")
        ]
        assert len(lines) == 1, f"expected one mcp pin in requirements.txt, got {lines!r}"
        assert lines[0] == self._mcp_pin(), (
            f"requirements.txt pins {lines[0]!r} but pyproject pins {self._mcp_pin()!r}"
        )

    def test_installed_version_matches_the_pin(self):
        from importlib.metadata import version as installed_version

        pin = self._mcp_pin().split("==", 1)[1]
        assert installed_version("mcp") == pin, (
            "installed mcp differs from the pyproject pin -- the environment drifted"
        )


class TestServerAdvertisesTargetRevision:
    """server/discover is how a modern client learns what we speak."""

    def test_discover_advertises_modern_versions(self):
        result = _run(_discover())
        assert TARGET_REVISION in result.supported_versions


class TestNegotiatedVersionEndToEnd:
    """Drive the real server object through a real client, per era."""

    def test_modern_client_negotiates_target_revision(self):
        """THE guard: a 2026-07-28 client actually gets 2026-07-28."""
        negotiated = _run(_negotiate(mode=TARGET_REVISION))
        assert negotiated == TARGET_REVISION

    def test_auto_mode_probes_up_to_target_revision(self):
        """'auto' probes server/discover first; against this server it must land
        on the modern revision, NOT silently fall back to the handshake."""
        negotiated = _run(_negotiate(mode="auto"))
        assert negotiated == TARGET_REVISION, (
            f"auto-mode negotiated {negotiated!r} -- server/discover probe likely "
            "failed and it fell back to the legacy initialize handshake."
        )

    def test_legacy_client_still_works_and_caps_at_handshake_latest(self):
        """Back-compat: an old client is served, at the handshake ceiling.
        This is expected, not a downgrade -- it documents the boundary."""
        negotiated = _run(_negotiate(mode="legacy"))
        assert negotiated == LATEST_HANDSHAKE_VERSION

    def test_tool_call_works_on_a_modern_connection(self):
        """A version number alone is not proof the connection is usable."""
        names = _run(_list_tool_names(mode=TARGET_REVISION))
        assert "search_knowledge" in names


# --------------------------------------------------------------------------
# helpers -- Client accepts an in-process Server and dispatches without
# JSON-RPC framing, so these are hermetic: no subprocess, no store, no model.
# (Real-transport coverage lives in test_server_stdio.py.)
# --------------------------------------------------------------------------


def _run(coro):
    import anyio

    return anyio.run(lambda: coro)


async def _negotiate(mode: str) -> str:
    async with Client(srv.server, mode=mode) as client:
        return client.protocol_version


async def _list_tool_names(mode: str) -> set[str]:
    async with Client(srv.server, mode=mode) as client:
        result = await client.list_tools()
        return {t.name for t in result.tools}


async def _discover() -> types.DiscoverResult:
    async with Client(srv.server, mode="auto") as client:
        return await client.session.discover()
