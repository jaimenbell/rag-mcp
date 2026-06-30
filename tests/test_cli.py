"""CLI integration tests: --exclude flag parsing and default-exclude behaviour.

All tests run offline (--embedder hash) — no ONNX model loaded, no network.

Design note: CLI --exclude flags AUGMENT the built-in EXCLUDE_PREFIXES defaults;
they do not replace them. So passing --exclude foo/ means the store will skip
both infrastructure/claude-config-backup/ (default) AND foo/.

argparse note: --embedder is a global flag and must appear BEFORE the subcommand
name in argv (e.g. ["--embedder", "hash", "ingest", ...]).
"""
from __future__ import annotations

import json

import pytest

from rag_mcp.cli import main


@pytest.fixture
def backup_corpus(tmp_path):
    """Corpus with a live skill and a backup copy in the default-excluded subtree."""
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "solana.md").write_text(
        "# Solana\n\nLive version.", encoding="utf-8"
    )
    backup = tmp_path / "infrastructure" / "claude-config-backup" / "skills"
    backup.mkdir(parents=True)
    (backup / "solana.md").write_text(
        "# Solana\n\nStale backup.", encoding="utf-8"
    )
    return tmp_path


def test_cli_ingest_default_excludes_backup(backup_corpus, tmp_path, capsys):
    """No --exclude flag: built-in default silently excludes the backup subtree."""
    db = tmp_path / "store.chroma"
    rc = main(["--embedder", "hash", "ingest", str(backup_corpus), "--db", str(db)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files_ingested"] == 1  # only live skill counted
    assert report["files_seen"] == 1      # backup never enters the loop


def test_cli_ingest_extra_exclude_augments_defaults(backup_corpus, tmp_path, capsys):
    """--exclude skills/ is merged with the defaults; live file also excluded."""
    db = tmp_path / "store2.chroma"
    rc = main(
        [
            "--embedder", "hash",
            "ingest", str(backup_corpus),
            "--db", str(db),
            "--exclude", "skills/",
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    # skills/solana.md excluded by flag; backup excluded by default → nothing in.
    assert report["files_ingested"] == 0
    assert report["files_seen"] == 0


def test_cli_ingest_exclude_nonmatching_leaves_live_intact(backup_corpus, tmp_path, capsys):
    """--exclude on a non-existent prefix keeps live files; defaults still block backup."""
    db = tmp_path / "store3.chroma"
    rc = main(
        [
            "--embedder", "hash",
            "ingest", str(backup_corpus),
            "--db", str(db),
            "--exclude", "nonexistent/",
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files_ingested"] == 1
