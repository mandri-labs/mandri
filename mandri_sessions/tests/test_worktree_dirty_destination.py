import json
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.sessions import worktree_transaction
from mandri.sessions.worktree_projection import index_path

from mandri_sessions.tests.test_worktree_integration import integrate
from mandri_sessions.tests.test_worktrees import create, git, repository, service

__all__ = ["repository"]


async def dirty_session(tmp_path, repository):
    text = "".join(f"line {number}\n" for number in range(15))
    (repository / "file.txt").write_text(text)
    (repository / ".gitignore").write_text("*.local\n")
    git(repository, "add", "file.txt", ".gitignore")
    git(repository, "commit", "-m", "Base")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    source = Path(worktree.path)
    (source / "file.txt").write_text(text.replace("line 2\n", "incoming\n"))
    (source / "second.txt").write_text("incoming second\n")
    (repository / "file.txt").write_text(text.replace("line 8\n", "staged\n"))
    (repository / "staged.txt").write_text("staged only\n")
    git(repository, "add", "file.txt", "staged.txt")
    (repository / "file.txt").write_text(
        text.replace("line 8\n", "staged\n").replace("line 12\n", "unstaged\n")
    )
    (repository / "staged.txt").write_text("unstaged layer\n")
    (repository / "untracked.txt").write_text("untracked\n")
    (repository / "secret.local").write_bytes(b"ignored\x00binary\n")
    return sessions, session, worktree


@pytest.mark.parametrize("strategy", ["squash", "merge"])
async def test_dirty_destination_keeps_each_local_layer_out_of_commit(
    tmp_path, repository, strategy
):
    sessions, session, worktree = await dirty_session(tmp_path, repository)
    source_index = index_path(worktree.path).read_bytes()
    target_index = index_path(str(repository)).read_bytes()
    preview = await sessions.worktrees.preview(session.id, None, strategy)
    assert preview.target == "main"
    assert Path(preview.target_path) == repository
    assert preview.target_dirty
    assert not preview.target_conflicts
    assert index_path(str(repository)).read_bytes() == target_index
    assert index_path(worktree.path).read_bytes() == source_index
    updated = await sessions.worktrees.integrate(
        session.id, "main", strategy, preview.token, "Feature"
    )
    assert "incoming" in git(repository, "show", "HEAD:file.txt")
    assert "staged" not in git(repository, "show", "HEAD:file.txt")
    assert "unstaged" not in git(repository, "show", "HEAD:file.txt")
    assert "+staged" in git(repository, "diff", "--cached", "--", "file.txt")
    assert "+unstaged" in git(repository, "diff", "--", "file.txt")
    assert git(repository, "show", ":staged.txt") == "staged only"
    assert (repository / "staged.txt").read_text() == "unstaged layer\n"
    assert (repository / "untracked.txt").read_text() == "untracked\n"
    assert (repository / "secret.local").read_bytes() == b"ignored\x00binary\n"
    assert "staged.txt" not in git(repository, "ls-tree", "--name-only", "HEAD")
    assert "?? untracked.txt" in git(repository, "status", "--porcelain")
    assert index_path(worktree.path).read_bytes() == source_index
    assert updated.pending_integration is None
    assert not (worktree_transaction.directory(worktree) / "journal.json").exists()


@pytest.mark.parametrize("staged", [False, True])
async def test_local_destination_conflict_is_separate_and_non_destructive(
    tmp_path, repository, staged
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("incoming\n")
    (repository / "file.txt").write_text("local\n")
    if staged:
        git(repository, "add", "file.txt")
    head, index = git(repository, "rev-parse", "HEAD"), index_path(str(repository)).read_bytes()
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    assert preview.conflicts == []
    assert preview.target_conflicts == ["file.txt"]
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(session.id, "main", "squash", preview.token, "Feature")
    assert error.value.code == "worktree_target_conflicts"
    assert git(repository, "rev-parse", "HEAD") == head
    assert index_path(str(repository)).read_bytes() == index
    assert (repository / "file.txt").read_text() == "local\n"


@pytest.mark.parametrize("boundary", ["before_index", "before_ref", "after_ref"])
async def test_interrupted_transaction_recovers_local_layers(
    tmp_path, repository, monkeypatch, boundary
):
    sessions, session, _worktree = await dirty_session(tmp_path, repository)
    head = git(repository, "rev-parse", "HEAD")
    index = index_path(str(repository)).read_bytes()
    content = (repository / "file.txt").read_bytes()
    if boundary == "before_index":
        original = worktree_transaction.install_index

        def fail_once(*args):
            monkeypatch.setattr(worktree_transaction, "install_index", original)
            raise OSError("index install interrupted")

        monkeypatch.setattr(worktree_transaction, "install_index", fail_once)
    elif boundary == "before_ref":
        original = worktree_transaction.RefTransaction.commit

        def fail_once(self):
            monkeypatch.setattr(worktree_transaction.RefTransaction, "commit", original)
            raise OSError("ref commit interrupted")

        monkeypatch.setattr(worktree_transaction.RefTransaction, "commit", fail_once)
    else:
        original = sessions.worktrees._recover_integration

        async def fail_once(*args):
            monkeypatch.setattr(sessions.worktrees, "_recover_integration", original)
            raise OSError("metadata save interrupted")

        monkeypatch.setattr(sessions.worktrees, "_recover_integration", fail_once)
    with pytest.raises(OSError):
        await integrate(sessions, session, "main")
    await sessions.worktrees.preview(session.id, "main", "squash")
    updated = await sessions.worktrees.get(session.id)
    assert updated.pending_integration is None
    assert git(repository, "show", ":staged.txt") == "staged only"
    assert (repository / "staged.txt").read_text() == "unstaged layer\n"
    assert (repository / "untracked.txt").read_text() == "untracked\n"
    assert (repository / "secret.local").read_bytes() == b"ignored\x00binary\n"
    if boundary == "after_ref":
        assert git(repository, "rev-parse", "HEAD") != head
        assert "+staged" in git(repository, "diff", "--cached", "--", "file.txt")
        assert "+unstaged" in git(repository, "diff", "--", "file.txt")
    else:
        assert git(repository, "rev-parse", "HEAD") == head
        assert index_path(str(repository)).read_bytes() == index
        assert (repository / "file.txt").read_bytes() == content


async def test_partial_checkout_recovers_only_owned_paths(tmp_path, repository, monkeypatch):
    sessions, session, _worktree = await dirty_session(tmp_path, repository)
    before = (repository / "file.txt").read_bytes()
    original = worktree_transaction.transition

    def interrupted(path, old, new):
        monkeypatch.setattr(worktree_transaction, "transition", original)
        text = (repository / "file.txt").read_text().replace("line 2\n", "incoming\n")
        (repository / "file.txt").write_text(text)
        (repository / "new-local.txt").write_text("external unrelated\n")
        raise OSError("partial checkout")

    monkeypatch.setattr(worktree_transaction, "transition", interrupted)
    with pytest.raises(OSError):
        await integrate(sessions, session, "main")
    assert (repository / "file.txt").read_bytes() == before
    assert (repository / "new-local.txt").read_text() == "external unrelated\n"
    assert not (repository / "second.txt").exists()
    assert (await sessions.worktrees.get(session.id)).pending_integration is None


async def test_external_edit_retains_journal_and_preview_can_retry_recovery(
    tmp_path, repository, monkeypatch
):
    sessions, session, worktree = await dirty_session(tmp_path, repository)
    before = (repository / "file.txt").read_bytes()
    original = worktree_transaction.transition

    def interrupted(path, old, new):
        monkeypatch.setattr(worktree_transaction, "transition", original)
        (repository / "file.txt").write_text("new external edit\n")
        raise OSError("checkout interrupted")

    monkeypatch.setattr(worktree_transaction, "transition", interrupted)
    with pytest.raises(ProtectionError) as error:
        await integrate(sessions, session, "main")
    assert error.value.code == "worktree_recovery_pending"
    journal = worktree_transaction.directory(worktree) / "journal.json"
    assert journal.exists()
    refs = json.loads(journal.read_text())["refs"]
    assert all(git(repository, "rev-parse", ref) for ref in refs)
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.preview(session.id, "main", "squash")
    assert error.value.code == "worktree_recovery_pending"
    assert (repository / "file.txt").read_text() == "new external edit\n"
    (repository / "file.txt").write_bytes(before)
    await sessions.worktrees.preview(session.id, "main", "squash")
    assert not journal.exists()
    assert (await sessions.worktrees.get(session.id)).pending_integration is None


async def test_locked_default_destination_returns_branch_choices_without_mutation(
    tmp_path, repository
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("incoming\n")
    git(repository, "branch", "other")
    index = index_path(str(repository)).read_bytes()
    lock = index_path(str(repository)).with_name("index.lock")
    lock.write_bytes(b"other Git operation")
    preview = await sessions.worktrees.preview(session.id, None, "squash")
    assert preview.target == "main"
    assert preview.target_error == "worktree_git_busy"
    assert "other" in preview.branches
    assert not preview.target_conflicts
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(session.id, "main", "squash", preview.token, "Feature")
    assert error.value.code == "worktree_git_busy"
    assert index_path(str(repository)).read_bytes() == index
    assert lock.read_bytes() == b"other Git operation"
    alternate = await sessions.worktrees.preview(session.id, "other", "squash")
    assert alternate.target_error is None
    await sessions.worktrees.integrate(session.id, "other", "squash", alternate.token, "Feature")
    assert git(repository, "show", "other:file.txt") == "incoming"
    assert (repository / "file.txt").read_text() == "initial\n"
    assert lock.read_bytes() == b"other Git operation"


@pytest.mark.parametrize("roundtrip", [False, True])
async def test_git_filters_must_roundtrip_affected_destination_files(
    tmp_path, repository, roundtrip
):
    git(repository, "config", "filter.normalize.clean", "tr A-Z a-z")
    git(repository, "config", "filter.normalize.smudge", "tr a-z A-Z")
    (repository / ".gitattributes").write_text("file.txt filter=normalize\n")
    git(repository, "add", ".gitattributes")
    git(repository, "commit", "-m", "Filter configuration")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("incoming\n")
    (repository / "file.txt").write_text("INITIAL\n" if roundtrip else "initial\n")
    before = (repository / "file.txt").read_bytes()
    head, index = git(repository, "rev-parse", "HEAD"), index_path(str(repository)).read_bytes()
    preview = await sessions.worktrees.preview(session.id, None, "squash")
    if roundtrip:
        assert preview.target_error is None
        await sessions.worktrees.integrate(session.id, "main", "squash", preview.token, "Feature")
        assert (repository / "file.txt").read_bytes() == b"INCOMING\n"
        assert git(repository, "show", "HEAD:file.txt") == "incoming"
    else:
        assert preview.target_error == "worktree_target_unsupported"
        with pytest.raises(ProtectionError) as error:
            await sessions.worktrees.integrate(
                session.id, "main", "squash", preview.token, "Feature"
            )
        assert error.value.code == "worktree_target_unsupported"
        assert (repository / "file.txt").read_bytes() == before
        assert git(repository, "rev-parse", "HEAD") == head
        assert index_path(str(repository)).read_bytes() == index
