import asyncio
import subprocess
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, ProjectPath
from mandri.core.types.execution import ProtectionError
from mandri.sessions import worktree_git
from mandri.sessions.service import SessionsService

from mandri_sessions.tests.substitutes import FakeDatabase, FakeEngine


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.email", "test@example.test")
    git(path, "config", "user.name", "Test")
    git(path, "config", "core.autocrlf", "false")
    git(path, "config", "commit.gpgsign", "false")
    (path / "file.txt").write_text("initial\n")
    git(path, "add", "file.txt")
    git(path, "commit", "-m", "Initial")
    return path


async def service(tmp_path):
    return SessionsService(
        await FakeDatabase.create(), FakeEngine(), worktrees_dir=tmp_path / "worktrees"
    )


async def create(sessions, repository, name=None):
    session = await sessions.create_session(
        HarnessKind.CLAUDE, project_path=ProjectPath(str(repository))
    )
    worktree = await sessions.worktrees.prepare(session.id, str(repository), name)
    return session, worktree


async def test_create_rename_and_delete_keep_repository_clean(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    assert len(worktree.id.split("-")) == 3
    assert not Path(worktree.path).is_relative_to(repository)
    assert git(Path(worktree.path), "branch", "--show-current") == worktree.branch
    (Path(worktree.path) / "file.txt").write_text("edited\n")
    renamed = await sessions.worktrees.rename(session.id, "feature/new-ui")
    assert renamed.path == worktree.path
    assert git(Path(renamed.path), "branch", "--show-current") == "feature/new-ui"
    assert (Path(renamed.path) / "file.txt").read_text() == "edited\n"
    assert (repository / "file.txt").read_text() == "initial\n"
    with pytest.raises(ProtectionError) as error:
        await sessions.delete_session(session.id)
    assert error.value.code == "worktree_has_changes"
    assert (await sessions.get_session(session.id)).worktree.id == "feature/new-ui"
    await sessions.delete_session(session.id, discard_worktree=True)
    assert not Path(worktree.path).exists()
    assert git(repository, "branch", "--list") == "* main"
    assert len(git(repository, "worktree", "list", "--porcelain").split("worktree ")) == 2
    assert not git(repository, "status", "--porcelain")


async def test_collisions_regenerate_and_do_not_touch_existing_branches(
    tmp_path, repository, monkeypatch
):
    sessions = await service(tmp_path)
    git(repository, "branch", "taken")
    names = iter(["taken", "calm-jade-otter", "bright-amber-fox"])
    monkeypatch.setattr(worktree_git, "random_id", lambda: next(names))
    session, worktree = await create(sessions, repository, "taken")
    assert worktree.id == "calm-jade-otter"
    renamed = await sessions.worktrees.rename(session.id, "taken")
    assert renamed.id == "bright-amber-fox"
    await sessions.delete_session(session.id)
    assert git(repository, "branch", "--list", "taken") == "taken"


async def test_committed_work_requires_merge_or_explicit_discard(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    path = Path(worktree.path)
    (path / "file.txt").write_text("committed\n")
    git(path, "commit", "-am", "Feature")
    with pytest.raises(ProtectionError, match="Integrate"):
        await sessions.delete_session(session.id)
    git(repository, "merge", "--ff-only", worktree.branch)
    await sessions.delete_session(session.id)
    assert not path.exists()
    assert git(repository, "branch", "--list") == "* main"


@pytest.mark.parametrize(
    "name", ["../escape", "--force", "bad name", "a.lock", "a/../b", "a//b", "/abs", "trailing/"]
)
async def test_invalid_names_do_not_create_worktrees(tmp_path, repository, name):
    sessions = await service(tmp_path)
    with pytest.raises(ProtectionError) as error:
        await create(sessions, repository, name)
    assert error.value.code == "worktree_invalid_id"
    assert not (tmp_path / "worktrees").exists()


async def test_concurrent_same_name_creates_distinct_worktrees(tmp_path, repository):
    sessions = await service(tmp_path)
    created = await asyncio.gather(*(create(sessions, repository, "same") for _ in range(3)))
    assert len({item.id for _, item in created}) == 3
    for session, _ in created:
        await sessions.delete_session(session.id)


async def test_reopen_preserves_worktree_and_missing_directory_blocks_resume(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    restarted = SessionsService(sessions._db, FakeEngine(), worktrees_dir=tmp_path / "worktrees")
    await restarted.worktrees.recover()
    await restarted.worktrees.validate(session.id)
    assert (await restarted.get_session(session.id)).worktree == worktree
    git(repository, "worktree", "remove", worktree.path)
    with pytest.raises(ProtectionError) as error:
        await restarted.worktrees.validate(session.id)
    assert error.value.code == "worktree_missing"
    await restarted.delete_session(session.id)


async def test_ignored_files_are_preserved_without_explicit_discard(tmp_path, repository):
    (repository / ".gitignore").write_text("local.secret\n")
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-m", "Ignore local files")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository)
    (Path(worktree.path) / "local.secret").write_text("keep")
    with pytest.raises(ProtectionError) as error:
        await sessions.delete_session(session.id)
    assert error.value.code == "worktree_has_changes"
    await sessions.delete_session(session.id, discard_worktree=True)


async def test_prefix_collisions_are_retried(tmp_path, repository):
    sessions = await service(tmp_path)
    git(repository, "branch", "feature/search")
    first, one = await create(sessions, repository, "feature")
    second, two = await create(sessions, repository, "feature/search/nested")
    assert one.id != "feature" and two.id != "feature/search/nested"
    await sessions.delete_session(first.id)
    await sessions.delete_session(second.id)
    assert git(repository, "branch", "--list", "feature/search") == "feature/search"


async def test_interrupted_rename_is_reconciled_after_restart(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "old-name")
    await sessions._db.execute(
        "UPDATE session SET worktree = json_set(worktree, '$.pending_id', 'new-name') WHERE id = ?",
        (str(session.id),),
    )
    git(repository, "branch", "-m", "old-name", "new-name")
    await sessions.worktrees.recover()
    recovered = await sessions.get_session(session.id)
    assert recovered.worktree.id == "new-name"
    assert recovered.worktree.path == worktree.path
    await sessions.delete_session(session.id)
    assert git(repository, "branch", "--list") == "* main"


async def test_interrupted_removal_finishes_after_restart(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "old-name")
    await sessions._db.execute(
        "UPDATE session SET worktree = json_set(worktree, '$.state', 'removing') WHERE id = ?",
        (str(session.id),),
    )
    git(repository, "worktree", "remove", worktree.path)
    await sessions.worktrees.recover()
    assert await sessions.list_sessions() == []
    assert git(repository, "branch", "--list") == "* main"


async def test_detached_checkout_does_not_lose_its_managed_branch_on_refused_delete(
    tmp_path, repository
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    (Path(worktree.path) / "file.txt").write_text("unmerged\n")
    git(worktree.path, "commit", "-am", "Unmerged")
    git(worktree.path, "checkout", "--detach", "main")
    with pytest.raises(ProtectionError):
        await sessions.delete_session(session.id)
    assert Path(worktree.path).is_dir()
    await sessions.delete_session(session.id, discard_worktree=True)


async def test_subdirectory_worktree_uses_matching_relative_directory(tmp_path, repository):
    sub = repository / "src"
    sub.mkdir()
    (sub / "code.py").write_text("pass\n")
    git(repository, "add", "src")
    git(repository, "commit", "-m", "Source")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, sub)
    assert (await sessions.get_session(session.id)).project_path == str(Path(worktree.path) / "src")
    assert (Path(worktree.path) / "src" / "code.py").exists()
    await sessions.delete_session(session.id)
