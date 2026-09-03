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


# ---------------------------------------------------------------------------
# --doc-class (RM-fixafter-ragmcp slice 2) -- the search_knowledge doc_class
# filter was never reachable from the CLI, so a caller that shells out to
# `python -m rag_mcp.cli ... query ...` (rather than importing search_knowledge
# directly) could not use it.
#
# FIRES: --doc-class note excludes a handoff-classified fixture from results.
# SILENT: omitting the flag includes it (byte-identical to pre-feature).
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_class_corpus(tmp_path):
    (tmp_path / "handoff.md").write_text(
        "---\ntype: handoff\n---\n# Handoff\n\nSession bookkeeping about the dog project.\n",
        encoding="utf-8",
    )
    (tmp_path / "note.md").write_text(
        "# Note\n\nA plain research note about the dog project.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_cli_query_doc_class_filters_out_excluded_class(mixed_class_corpus, tmp_path, capsys):
    db = tmp_path / "store.chroma"
    rc = main(
        ["--embedder", "hash", "ingest", str(mixed_class_corpus), "--db", str(db)]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "--embedder", "hash",
            "query", "dog project",
            "--db", str(db),
            "--corpus", str(mixed_class_corpus),
            "--doc-class", "note",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    sources = {r["citation"]["source"] for r in result["results"]}
    assert "handoff.md" not in sources
    assert "note.md" in sources


def test_cli_ingest_handoff_mirror_dir_flag(tmp_path, capsys):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "handoff.md").write_text(
        "No frontmatter; dir not 'context' so default would classify note.\n",
        encoding="utf-8",
    )
    db = tmp_path / "store4.chroma"
    rc = main(
        [
            "--embedder", "hash",
            "ingest", str(tmp_path), "--db", str(db),
            "--handoff-mirror-dir", "sessions",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    rc = main(
        [
            "--embedder", "hash",
            "query", "no frontmatter",
            "--db", str(db), "--corpus", str(tmp_path), "--doc-class", "handoff",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    sources = {r["citation"]["source"] for r in result["results"]}
    assert "sessions/handoff.md" in sources


def test_cli_ingest_handoff_mirror_basename_flag_augments(tmp_path, capsys):
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "status.md").write_text("No frontmatter, custom basename only.\n", encoding="utf-8")
    (ctx / "RESUME.md").write_text("Default mirror name, must still match.\n", encoding="utf-8")
    db = tmp_path / "store5.chroma"
    rc = main(
        [
            "--embedder", "hash",
            "ingest", str(tmp_path), "--db", str(db),
            "--handoff-mirror-basename", "status.md",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    rc = main(
        [
            "--embedder", "hash",
            "query", "custom basename mirror name",
            "-k", "5",
            "--db", str(db), "--corpus", str(tmp_path), "--doc-class", "handoff",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    sources = {r["citation"]["source"] for r in result["results"]}
    assert "context/status.md" in sources
    assert "context/RESUME.md" in sources


def test_cli_query_without_doc_class_includes_all(mixed_class_corpus, tmp_path, capsys):
    db = tmp_path / "store2.chroma"
    rc = main(
        ["--embedder", "hash", "ingest", str(mixed_class_corpus), "--db", str(db)]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "--embedder", "hash",
            "query", "dog project",
            "--db", str(db),
            "--corpus", str(mixed_class_corpus),
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    sources = {r["citation"]["source"] for r in result["results"]}
    assert "handoff.md" in sources
    assert "note.md" in sources
