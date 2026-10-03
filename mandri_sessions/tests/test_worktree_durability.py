import pytest
from mandri.sessions import worktree_transaction


def test_journal_write_replaces_existing_contents(tmp_path):
    path = tmp_path / "journal.json"
    worktree_transaction.durable_write(path, b"old state")
    worktree_transaction.durable_write(path, b"new state")
    assert path.read_bytes() == b"new state"
    assert not path.with_suffix(".json.tmp").exists()


def test_failed_file_sync_preserves_previous_journal(tmp_path, monkeypatch):
    path = tmp_path / "journal.json"
    worktree_transaction.durable_write(path, b"recoverable state")

    def failed_sync(descriptor):
        raise OSError("sync failed")

    monkeypatch.setattr(worktree_transaction.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="sync failed"):
        worktree_transaction.durable_write(path, b"incomplete state")
    assert path.read_bytes() == b"recoverable state"
