import os
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.sessions.worktree_projection import index_path

from mandri_sessions.tests.test_worktrees import create, git, repository, service

__all__ = ["repository"]


async def test_review_preserves_both_indexes_byte_for_byte(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    source = Path(worktree.path)
    (source / "file.txt").write_text("feature\n")
    local = repository / "local.txt"
    local.write_text("staged\n")
    git(repository, "add", "local.txt")
    local.write_text("unstaged\n")
    source_index = index_path(str(source)).read_bytes()
    target_index = index_path(str(repository)).read_bytes()
    first = await sessions.worktrees.preview(session.id, "main", "squash")
    second = await sessions.worktrees.preview(session.id, "main", "squash")
    assert first.token == second.token
    assert first.target_dirty
    assert not first.target_conflicts
    assert index_path(str(source)).read_bytes() == source_index
    assert index_path(str(repository)).read_bytes() == target_index
    assert git(repository, "show", ":local.txt") == "staged"
    assert local.read_text() == "unstaged\n"


@pytest.mark.parametrize("kind", ["skip", "assume", "split", "intent", "sparse"])
async def test_special_destination_index_is_reviewable_but_never_changed(
    tmp_path, repository, kind
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    git(repository, "branch", "alternate")
    (Path(worktree.path) / "file.txt").write_text("feature\n")
    if kind == "skip":
        git(repository, "update-index", "--skip-worktree", "file.txt")
    elif kind == "assume":
        git(repository, "update-index", "--assume-unchanged", "file.txt")
    elif kind == "split":
        git(repository, "update-index", "--split-index")
    elif kind == "intent":
        (repository / "local.txt").write_text("local\n")
        git(repository, "add", "-N", "local.txt")
    else:
        git(repository, "config", "core.sparseCheckout", "true")
    head = git(repository, "rev-parse", "HEAD")
    before = index_path(str(repository)).read_bytes()
    review = await sessions.worktrees.preview(session.id, "main", "squash")
    assert review.branches == ["alternate", "main"]
    assert Path(review.target_path) == repository
    assert review.target_error == "worktree_target_unsupported"
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(session.id, "main", "squash", review.token, "Feature")
    assert error.value.code == "worktree_target_unsupported"
    assert git(repository, "rev-parse", "HEAD") == head
    assert index_path(str(repository)).read_bytes() == before
    alternate = await sessions.worktrees.preview(session.id, "alternate", "squash")
    assert alternate.target_error is None
    result = await sessions.worktrees.integrate(
        session.id, "alternate", "squash", alternate.token, "Feature"
    )
    assert git(repository, "show", "alternate:file.txt") == "feature"
    assert result.integrated_target == "alternate"
    assert (repository / "file.txt").read_text() == "initial\n"
    assert index_path(str(repository)).read_bytes() == before


@pytest.mark.parametrize("kind", ["unstaged", "untracked", "ignored", "checkout"])
async def test_destination_changes_invalidate_review_without_touching_any_work(
    tmp_path, repository, kind
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    source = Path(worktree.path)
    (source / "file.txt").write_text("feature\n")
    if kind == "ignored":
        (repository / ".git/info/exclude").write_text("cache.bin\n")
        (source / "cache.bin").write_bytes(b"incoming")
        git(source, "add", "-f", "cache.bin")
    if kind == "checkout":
        git(repository, "branch", "alternate")
    review = await sessions.worktrees.preview(session.id, "main", "squash")
    if kind == "unstaged":
        (repository / "file.txt").write_text("local\n")
    elif kind == "untracked":
        (repository / "local.txt").write_text("local\n")
    elif kind == "ignored":
        (repository / "cache.bin").write_bytes(b"ignored local content")
    else:
        git(repository, "checkout", "alternate")
    head = git(repository, "rev-parse", "main")
    index = index_path(str(repository)).read_bytes()
    local_files = {path.name: path.read_bytes() for path in repository.iterdir() if path.is_file()}
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(session.id, "main", "squash", review.token, "Feature")
    assert error.value.code == "worktree_preview_changed"
    assert git(repository, "rev-parse", "main") == head
    assert index_path(str(repository)).read_bytes() == index
    assert local_files == {
        path.name: path.read_bytes() for path in repository.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("kind", ["executable", "symlink"])
@pytest.mark.parametrize("strategy", ["squash", "merge"])
async def test_file_modes_and_links_integrate_without_committing_local_content(
    tmp_path, repository, kind, strategy
):
    if kind == "executable" and os.name == "nt":
        pytest.skip("Windows filesystems do not expose POSIX executable mode changes")
    git(repository, "config", "core.filemode", "true")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "feature")
    source = Path(worktree.path)
    if kind == "executable":
        (source / "file.txt").chmod(0o755)
    else:
        (source / "alias").symlink_to("file.txt")
    (repository / "file.txt").write_text("local pending\n")
    review = await sessions.worktrees.preview(session.id, "main", strategy)
    assert review.target_dirty
    assert not review.target_conflicts
    updated = await sessions.worktrees.integrate(
        session.id, "main", strategy, review.token, "Feature"
    )
    assert updated.integrated_commit == git(repository, "rev-parse", "main")
    assert git(repository, "show", "HEAD:file.txt") == "initial"
    assert git(repository, "show", ":file.txt") == "initial"
    assert (repository / "file.txt").read_text() == "local pending\n"
    if kind == "executable":
        assert (repository / "file.txt").stat().st_mode & 0o111
        assert git(repository, "ls-tree", "HEAD", "file.txt").startswith("100755")
    else:
        assert (repository / "alias").is_symlink()
        assert os.readlink(repository / "alias") == "file.txt"
        assert git(repository, "ls-tree", "HEAD", "alias").startswith("120000")
