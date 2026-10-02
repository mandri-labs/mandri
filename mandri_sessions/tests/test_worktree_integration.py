import dataclasses
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.sessions import worktree_integration

from mandri_sessions.tests.test_worktrees import create, git, repository, service

__all__ = ["repository"]


async def integrate(sessions, session, target, strategy="squash"):
    preview = await sessions.worktrees.preview(session.id, target, strategy)
    return await sessions.worktrees.integrate(
        session.id, target, strategy, preview.token, "Feature"
    )


async def test_squash_review_integrate_clean_keeps_session(tmp_path, repository):
    git(repository, "branch", "-m", "trunk")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    path = Path(worktree.path)
    (path / "file.txt").write_text("changed\n")
    (path / "new.txt").write_text("new\n")
    preview = await sessions.worktrees.preview(session.id, None, "squash")
    assert preview.target == "trunk"
    assert preview.files == ["file.txt", "new.txt"]
    assert "+changed" in preview.diff
    assert (repository / "file.txt").read_text() == "initial\n"
    updated = await integrate(sessions, session, "trunk")
    assert updated.integrated_commit == git(repository, "rev-parse", "trunk")
    assert git(repository, "rev-list", "--count", "trunk") == "2"
    assert (repository / "new.txt").read_text() == "new\n"
    assert not git(repository, "status", "--porcelain")
    assert (path / "file.txt").read_text() == "changed\n"
    closed = await sessions.worktrees.finish(session.id)
    assert closed.state == "closed"
    assert not path.exists()
    assert not git(repository, "branch", "--list", "feature")
    assert not git(repository, "for-each-ref", "refs/mandri")
    assert (await sessions.get_session(session.id)).worktree.state == "closed"
    with pytest.raises(ProtectionError, match="cleaned"):
        await sessions.worktrees.validate(session.id)
    await sessions.delete_session(session.id)


@pytest.mark.parametrize("strategy", ["squash", "merge"])
@pytest.mark.parametrize("after_integration", [False, True])
async def test_cleanup_requires_explicit_ignored_file_discard(
    tmp_path, repository, strategy, after_integration
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    path = Path(worktree.path)
    (path / "file.txt").write_text("changed\n")
    cache = path / ".pytest_cache"
    if after_integration:
        await integrate(sessions, session, "main", strategy)
    cache.mkdir()
    (cache / ".gitignore").write_text("*\n")
    (cache / "nodeids").write_text("cached tests\n")
    if not after_integration:
        await integrate(sessions, session, "main", strategy)
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.finish(session.id)
    assert error.value.code == "worktree_ignored_files"
    assert (cache / "nodeids").read_text() == "cached tests\n"
    assert (await sessions.worktrees.get(session.id)).state == "ready"
    with pytest.raises(ProtectionError):
        await sessions.delete_session(session.id)
    closed = await sessions.worktrees.finish(session.id, discard_ignored=True)
    assert closed.state == "closed"
    assert not path.exists()
    assert not git(repository, "branch", "--list", "feature")
    assert (repository / "file.txt").read_text() == "changed\n"


@pytest.mark.parametrize("change", ["tracked", "untracked", "commit", "destination"])
async def test_ignored_file_discard_preserves_unintegrated_work(tmp_path, repository, change):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("integrated\n")
    await integrate(sessions, session, "main")
    cache = path / ".pytest_cache"
    cache.mkdir()
    (cache / ".gitignore").write_text("*\n")
    (cache / "nodeids").write_text("keep\n")
    if change == "untracked":
        (path / "new.txt").write_text("not integrated\n")
    elif change == "destination":
        git(repository, "update-ref", "refs/heads/main", worktree.base_commit)
    else:
        (path / "file.txt").write_text("not integrated\n")
        if change == "commit":
            git(path, "commit", "-am", "New work")
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.finish(session.id, discard_ignored=True)
    assert error.value.code == "worktree_has_changes"
    assert (cache / "nodeids").read_text() == "keep\n"
    assert (await sessions.worktrees.get(session.id)).state == "ready"
    assert git(repository, "branch", "--list", worktree.branch)


async def test_remote_default_and_ambiguous_local_branches(tmp_path, repository):
    sessions = await service(tmp_path)
    git(repository, "branch", "trunk")
    _session, worktree = await create(sessions, repository)
    worktree = dataclasses.replace(worktree, base_ref=worktree.base_commit)
    assert worktree_integration.branches(worktree)[1] is None
    git(repository, "remote", "add", "origin", str(tmp_path / "remote.git"))
    git(repository, "update-ref", "refs/remotes/origin/trunk", "HEAD")
    git(repository, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk")
    assert worktree_integration.branches(worktree)[1] == "trunk"
    git(repository, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/missing")
    assert worktree_integration.branches(worktree)[1] is None


@pytest.mark.parametrize("change", ["source", "target", "index"])
async def test_stale_review_cannot_integrate(tmp_path, repository, change):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("feature\n")
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    if change == "source":
        (path / "extra").write_text("extra")
    elif change == "index":
        git(path, "add", "file.txt")
    else:
        (repository / "other").write_text("target")
        git(repository, "add", "other")
        git(repository, "commit", "-m", "Target advance")
    target = git(repository, "rev-parse", "main")
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(session.id, "main", "squash", preview.token, "Feature")
    assert error.value.code == "worktree_preview_changed"
    assert git(repository, "rev-parse", "main") == target


async def test_dirty_target_and_conflicts_leave_target_untouched(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("feature\n")
    (repository / "untracked").write_text("keep")
    (path / "untracked").write_text("source")
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    assert preview.target_dirty
    with pytest.raises(ProtectionError) as error:
        await integrate(sessions, session, "main")
    assert error.value.code == "worktree_target_conflicts"
    assert (repository / "untracked").read_text() == "keep"
    (repository / "untracked").unlink()
    (path / "untracked").unlink()
    (repository / "file.txt").write_text("target\n")
    git(repository, "commit", "-am", "Target change")
    head = git(repository, "rev-parse", "HEAD")
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    assert preview.conflicts == ["file.txt"]
    with pytest.raises(ProtectionError) as error:
        await integrate(sessions, session, "main")
    assert error.value.code == "worktree_conflicts"
    await sessions.worktrees.resolve(session.id, "main", "squash", preview.token)
    assert git(repository, "rev-parse", "HEAD") == head
    assert git(path, "ls-files", "--unmerged")
    (path / "file.txt").write_text("resolved\n")
    git(path, "add", "file.txt")
    git(path, "commit", "-m", "Resolve target changes")
    await integrate(sessions, session, "main")
    assert (repository / "file.txt").read_text() == "resolved\n"


async def test_continue_then_integrate_again_and_preserve_new_changes(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("first\n")
    await integrate(sessions, session, "main")
    (path / "file.txt").write_text("second\n")
    with pytest.raises(ProtectionError):
        await sessions.worktrees.finish(session.id)
    with pytest.raises(ProtectionError):
        await sessions.delete_session(session.id)
    await integrate(sessions, session, "main")
    assert (repository / "file.txt").read_text() == "second\n"
    await sessions.delete_session(session.id)
    assert not path.exists()


async def test_merge_preserves_commits_and_updates_unchecked_branch(tmp_path, repository):
    git(repository, "branch", "destination")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("feature\n")
    git(path, "commit", "-am", "First feature commit")
    source = git(path, "rev-parse", "HEAD")
    (path / "new.txt").write_text("pending\n")
    await integrate(sessions, session, "destination", "merge")
    git(repository, "merge-base", "--is-ancestor", source, "destination")
    assert (repository / "file.txt").read_text() == "initial\n"
    assert git(repository, "show", "destination:new.txt") == "pending"
    await sessions.worktrees.finish(session.id)


async def test_recover_integration_after_target_update(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("feature\n")
    plan = await sessions.worktrees.preview(session.id, "main", "squash")
    commit = worktree_integration.create_commit(worktree, plan, "Feature")
    pending = {
        "integrated_target": "main",
        "integrated_commit": commit,
        "integrated_head": plan.source_head,
        "integrated_tree": plan.source_tree,
        "integrated_index": git(Path(worktree.path), "write-tree"),
    }
    await sessions.worktrees._save(
        session.id, dataclasses.replace(worktree, pending_integration=pending)
    )
    worktree_integration.apply(worktree, plan, commit)
    await sessions.worktrees.recover()
    updated = await sessions.worktrees.get(session.id)
    assert updated.integrated_commit == commit
    assert updated.pending_integration is None
    await sessions.worktrees.finish(session.id)


@pytest.mark.parametrize("discard_ignored", [False, True])
async def test_cleanup_preserves_staged_only_content(tmp_path, repository, discard_ignored):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("staged\n")
    git(path, "add", "file.txt")
    (path / "file.txt").write_text("final\n")
    await integrate(sessions, session, "main")
    with pytest.raises(ProtectionError):
        await sessions.worktrees.finish(session.id, discard_ignored=discard_ignored)
    assert git(path, "show", ":file.txt") == "staged"


async def test_includes_forced_staged_ignored_files_and_preserves_index(tmp_path, repository):
    (repository / ".gitignore").write_text("*.generated\n")
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-m", "Ignore generated files")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "checked.generated").write_text("keep\n")
    git(path, "add", "-f", "checked.generated")
    index = git(path, "write-tree")
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    assert "checked.generated" in preview.files
    assert git(path, "write-tree") == index
    await integrate(sessions, session, "main")
    assert (repository / "checked.generated").read_text() == "keep\n"
    await sessions.worktrees.finish(session.id)


async def test_restart_after_checkout_removal_finishes_cleanup(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("feature\n")
    updated = await integrate(sessions, session, "main")
    await sessions.worktrees._save(session.id, dataclasses.replace(updated, state="closing"))
    git(repository, "worktree", "remove", "--force", worktree.path)
    await sessions.worktrees.recover()
    assert (await sessions.get_session(session.id)).worktree.state == "closed"
    assert not git(repository, "branch", "--list", worktree.branch)
    assert not git(repository, "for-each-ref", "refs/mandri")


async def test_restart_removal_rechecks_changes_instead_of_discarding(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("feature\n")
    updated = await integrate(sessions, session, "main")
    await sessions.worktrees._save(session.id, dataclasses.replace(updated, state="removing"))
    (path / "new.txt").write_text("not integrated\n")
    await sessions.worktrees.recover()
    assert (path / "new.txt").read_text() == "not integrated\n"
    assert (await sessions.get_session(session.id)).worktree.state == "removing"


async def test_failed_destination_write_preserves_previous_integration(
    tmp_path, repository, monkeypatch
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (path / "file.txt").write_text("first\n")
    previous = await integrate(sessions, session, "main")
    (path / "file.txt").write_text("second\n")

    def fail(*args):
        raise ProtectionError("worktree_unavailable", "Synthetic write failure")

    monkeypatch.setattr(worktree_integration, "apply", fail)
    with pytest.raises(ProtectionError, match="Synthetic"):
        await integrate(sessions, session, "main")
    updated = await sessions.worktrees.get(session.id)
    assert updated.integrated_commit == previous.integrated_commit
    assert updated.pending_integration is None
    assert (repository / "file.txt").read_text() == "first\n"
    assert (path / "file.txt").read_text() == "second\n"


async def test_cleanup_recovery_preserves_snapshot_if_destination_was_rewritten(
    tmp_path, repository
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "file.txt").write_text("feature\n")
    updated = await integrate(sessions, session, "main")
    await sessions.worktrees._save(session.id, dataclasses.replace(updated, state="closing"))
    git(repository, "worktree", "remove", "--force", worktree.path)
    git(repository, "branch", "destination", "main")
    git(repository, "switch", "destination")
    git(repository, "update-ref", "refs/heads/main", worktree.base_commit)
    await sessions.worktrees.recover()
    assert (await sessions.get_session(session.id)).worktree.state == "closing"
    assert git(repository, "branch", "--list", worktree.branch)
    assert git(repository, "show", f"{worktree_integration.ref(worktree)}:file.txt") == "feature"


async def test_target_ignored_file_is_never_overwritten(tmp_path, repository):
    (repository / ".gitignore").write_text("*.local\n")
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-m", "Ignore local files")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    path = Path(worktree.path)
    (repository / "data.local").write_text("precious\n")
    (path / "data.local").write_text("feature\n")
    git(path, "add", "-f", "data.local")
    with pytest.raises(ProtectionError):
        await integrate(sessions, session, "main")
    assert (repository / "data.local").read_text() == "precious\n"
    assert (await sessions.worktrees.get(session.id)).integrated_commit is None


async def test_initial_branch_config_is_not_a_repository_default(tmp_path, repository):
    git(repository, "config", "init.defaultBranch", "main")
    git(repository, "branch", "develop")
    sessions = await service(tmp_path)
    _session, worktree = await create(sessions, repository)
    worktree = dataclasses.replace(worktree, base_ref=worktree.base_commit)
    assert worktree_integration.branches(worktree)[1] is None


async def test_default_follows_local_branch_tracking_remote_head(tmp_path, repository):
    git(repository, "remote", "add", "origin", str(tmp_path / "remote.git"))
    git(repository, "update-ref", "refs/remotes/origin/trunk", "HEAD")
    git(repository, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk")
    git(repository, "branch", "local-default")
    git(repository, "branch", "--set-upstream-to", "origin/trunk", "local-default")
    sessions = await service(tmp_path)
    _session, worktree = await create(sessions, repository)
    assert worktree_integration.branches(worktree)[1] == "main"
    worktree = dataclasses.replace(worktree, base_ref=worktree.base_commit)
    assert worktree_integration.branches(worktree)[1] == "local-default"
