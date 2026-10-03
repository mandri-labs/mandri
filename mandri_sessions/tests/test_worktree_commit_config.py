import dataclasses
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.sessions import worktree_integration, worktree_transaction
from mandri.sessions.worktree_projection import index_path

from mandri_sessions.tests.test_worktrees import create, git, repository, service

__all__ = ["repository"]


@pytest.fixture
def signing_key(tmp_path):
    key = tmp_path / "signing key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    return key


def project_config(path, key, *, signed=True, name="Project", email="project@example.test"):
    git(path, "config", "--worktree", "user.name", name)
    git(path, "config", "--worktree", "user.email", email)
    git(path, "config", "--worktree", "gpg.format", "ssh")
    git(path, "config", "--worktree", "user.signingkey", str(key))
    git(path, "config", "--worktree", "commit.gpgsign", str(signed).lower())


def verified_signature(repository, commit, key, principal, tmp_path):
    allowed = tmp_path / "allowed signers"
    allowed.write_text(principal + " " + key.with_suffix(".pub").read_text())
    git(repository, "-c", f"gpg.ssh.allowedSignersFile={allowed}", "verify-commit", commit)


def state(repository, source):
    return {
        "refs": git(repository, "for-each-ref", "--format=%(refname) %(objectname)"),
        "target_index": index_path(str(repository)).read_bytes(),
        "source_index": index_path(str(source)).read_bytes(),
        "target_file": (repository / "file.txt").read_bytes(),
        "source_file": (source / "file.txt").read_bytes(),
    }


@pytest.mark.parametrize("strategy", ["squash", "merge"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("target", ["main", "destination"])
async def test_commits_use_original_project_config_over_agent_worktree_config(
    tmp_path, repository, signing_key, strategy, signed, target
):
    git(repository, "branch", "destination")
    git(repository, "config", "extensions.worktreeConfig", "true")
    project_config(repository, signing_key, signed=signed)
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    source = Path(worktree.path)
    project_config(
        source,
        tmp_path / "missing agent key",
        signed=not signed,
        name="Agent workspace",
        email="agent@example.test",
    )
    (source / "file.txt").write_text("incoming\n")
    assert git(source, "config", "user.email") == "agent@example.test"
    assert git(source, "config", "--bool", "commit.gpgsign") == str(not signed).lower()
    source_index = index_path(worktree.path).read_bytes()
    preview = await sessions.worktrees.preview(session.id, target, strategy)
    updated = await sessions.worktrees.integrate(
        session.id, target, strategy, preview.token, "Project change"
    )
    commits = [updated.integrated_commit]
    if strategy == "merge":
        commits.append(git(repository, "rev-parse", f"{updated.integrated_commit}^2"))
    for commit in commits:
        assert git(repository, "show", "-s", "--format=%an <%ae>%n%cn <%ce>", commit) == (
            "Project <project@example.test>\nProject <project@example.test>"
        )
        assert ("\ngpgsig " in git(repository, "cat-file", "-p", commit)) == signed
        if signed:
            verified_signature(repository, commit, signing_key, "project@example.test", tmp_path)
    assert git(source, "rev-parse", "HEAD") == preview.source_head
    assert index_path(worktree.path).read_bytes() == source_index
    assert git(repository, "show", f"{target}:file.txt") == "incoming"


@pytest.mark.parametrize("strategy", ["squash", "merge"])
async def test_origin_can_itself_be_a_linked_project_worktree(
    tmp_path, repository, signing_key, strategy
):
    git(repository, "config", "extensions.worktreeConfig", "true")
    git(repository, "config", "commit.gpgsign", "false")
    origin = tmp_path / "project checkout"
    git(repository, "worktree", "add", "-b", "project", str(origin))
    project_config(origin, signing_key)
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, origin, "agent")
    assert worktree.repository == str(repository)
    assert worktree.source_path == str(origin)
    (Path(worktree.path) / "file.txt").write_text("incoming\n")
    preview = await sessions.worktrees.preview(session.id, "main", strategy)
    updated = await sessions.worktrees.integrate(
        session.id, "main", strategy, preview.token, "Linked project change"
    )
    assert git(repository, "show", "-s", "--format=%ae", updated.integrated_commit) == (
        "project@example.test"
    )
    verified_signature(
        repository, updated.integrated_commit, signing_key, "project@example.test", tmp_path
    )


@pytest.mark.parametrize("strategy", ["squash", "merge"])
@pytest.mark.parametrize("failure", ["missing_key", "invalid_policy"])
async def test_signing_failure_leaves_source_destination_and_metadata_unchanged(
    tmp_path, repository, strategy, failure
):
    git(repository, "config", "extensions.worktreeConfig", "true")
    project_config(repository, tmp_path / "missing key")
    if failure == "invalid_policy":
        git(repository, "config", "--worktree", "commit.gpgsign", "invalid")
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    source = Path(worktree.path)
    (source / "file.txt").write_text("incoming\n")
    (repository / "local.txt").write_text("staged local\n")
    git(repository, "add", "local.txt")
    (repository / "local.txt").write_text("unstaged local\n")
    preview = await sessions.worktrees.preview(session.id, "main", strategy)
    before = state(repository, source)
    with pytest.raises(ProtectionError) as error:
        await sessions.worktrees.integrate(
            session.id, "main", strategy, preview.token, "Must be signed"
        )
    assert error.value.code == "worktree_unavailable"
    assert state(repository, source) == before
    assert git(repository, "show", ":local.txt") == "staged local"
    assert (repository / "local.txt").read_text() == "unstaged local\n"
    updated = await sessions.worktrees.get(session.id)
    assert updated.pending_integration is None
    assert updated.integrated_commit is None
    assert not (worktree_transaction.directory(worktree) / "journal.json").exists()


async def test_branch_conditional_config_is_evaluated_in_original_project(tmp_path, repository):
    included = tmp_path / "project config"
    git(tmp_path, "config", "--file", str(included), "user.name", "Conditional project")
    git(tmp_path, "config", "--file", str(included), "user.email", "conditional@example.test")
    git(tmp_path, "config", "--file", str(included), "commit.gpgsign", "false")
    git(repository, "config", "includeIf.onbranch:main.path", str(included))
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    (Path(worktree.path) / "file.txt").write_text("incoming\n")
    assert git(repository, "config", "user.email") == "conditional@example.test"
    assert git(Path(worktree.path), "config", "user.email") == "test@example.test"
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    updated = await sessions.worktrees.integrate(
        session.id, "main", "squash", preview.token, "Conditional project change"
    )
    assert git(repository, "show", "-s", "--format=%ae", updated.integrated_commit) == (
        "conditional@example.test"
    )


async def test_replaced_origin_cannot_supply_an_unrelated_repository_config(tmp_path, repository):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    source = Path(worktree.path)
    (source / "file.txt").write_text("incoming\n")
    other = tmp_path / "other project"
    other.mkdir()
    git(other, "init", "-b", "main")
    changed = dataclasses.replace(worktree, source_path=str(other))
    await sessions.worktrees._save(session.id, changed)
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    before = state(repository, source)
    with pytest.raises(ProtectionError, match="original project"):
        await sessions.worktrees.integrate(
            session.id, "main", "squash", preview.token, "Wrong context"
        )
    assert state(repository, source) == before
    assert (await sessions.worktrees.get(session.id)).pending_integration is None


async def test_conditional_policies_of_two_projects_remain_distinct(
    tmp_path, repository, signing_key
):
    other = tmp_path / "second project"
    other.mkdir()
    git(other, "init", "-b", "main")
    git(other, "config", "user.name", "Initial")
    git(other, "config", "user.email", "initial@example.test")
    (other / "file.txt").write_text("initial\n")
    git(other, "add", "file.txt")
    git(other, "commit", "-m", "Initial")
    sessions = await service(tmp_path)
    for number, project in enumerate([repository, other]):
        signed = number == 0
        email = f"project{number}@example.test"
        config = tmp_path / f"project{number}.gitconfig"
        for key, value in [
            ("user.name", f"Project {number}"),
            ("user.email", email),
            ("commit.gpgsign", str(signed).lower()),
            ("gpg.format", "ssh"),
            ("user.signingkey", str(signing_key) if signed else str(tmp_path / "missing key")),
        ]:
            git(tmp_path, "config", "--file", str(config), key, value)
        git(project, "config", f"includeIf.gitdir:{project}/.path", str(config))
        session, worktree = await create(sessions, project, f"agent{number}")
        source = Path(worktree.path)
        assert not source.is_relative_to(project)
        assert git(source, "config", "user.email") == email
        (source / "file.txt").write_text("incoming\n")
        preview = await sessions.worktrees.preview(session.id, "main", "squash")
        updated = await sessions.worktrees.integrate(
            session.id, "main", "squash", preview.token, "Conditional project change"
        )
        commit = updated.integrated_commit
        assert git(project, "show", "-s", "--format=%ae", commit) == email
        assert ("\ngpgsig " in git(project, "cat-file", "-p", commit)) == signed
        if signed:
            verified_signature(project, commit, signing_key, email, tmp_path)


@pytest.mark.parametrize("available_key", [False, True])
async def test_resolution_snapshot_uses_project_signature_before_mutating_source(
    tmp_path, repository, signing_key, available_key
):
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    source = Path(worktree.path)
    (source / "file.txt").write_text("incoming\n")
    (repository / "file.txt").write_text("destination\n")
    git(repository, "commit", "-am", "Destination change")
    git(repository, "config", "extensions.worktreeConfig", "true")
    project_config(repository, signing_key if available_key else tmp_path / "missing key")
    project_config(source, tmp_path / "missing agent key", signed=False, email="agent@example.test")
    preview = await sessions.worktrees.preview(session.id, "main", "squash")
    assert preview.conflicts == ["file.txt"]
    before = state(repository, source)
    if available_key:
        await sessions.worktrees.resolve(session.id, "main", "squash", preview.token)
        commit = git(source, "rev-parse", "HEAD")
        assert git(source, "show", "-s", "--format=%ae", commit) == "project@example.test"
        assert git(source, "show", f"{commit}:file.txt") == "incoming"
        verified_signature(repository, commit, signing_key, "project@example.test", tmp_path)
        assert git(source, "ls-files", "--unmerged")
        assert git(repository, "rev-parse", "HEAD") == preview.target_head
    else:
        with pytest.raises(ProtectionError):
            worktree_integration.prepare_resolution(worktree, preview)
        assert state(repository, source) == before
        assert not git(source, "ls-files", "--unmerged")


async def test_final_merge_signature_failure_does_not_publish_partial_integration(
    tmp_path, repository, signing_key
):
    git(repository, "config", "extensions.worktreeConfig", "true")
    project_config(repository, signing_key)
    calls = tmp_path / "signer calls"
    program = tmp_path / "signer"
    executable = shutil.which("ssh-keygen")
    assert executable is not None
    marker = shlex.quote(str(calls))
    program.write_text(
        "#!/bin/sh\n"
        f"if [ -f {marker} ]; then\n"
        f"  printf '2\\n' > {marker}\n"
        "  printf 'Signature refused\\n' >&2\n"
        "  exit 1\n"
        "fi\n"
        f"printf '1\\n' > {marker}\n"
        f'exec {shlex.quote(executable)} "$@"\n'
    )
    program.chmod(0o700)
    git(repository, "config", "--worktree", "gpg.ssh.program", str(program))
    sessions = await service(tmp_path)
    session, worktree = await create(sessions, repository, "agent")
    source = Path(worktree.path)
    (source / "file.txt").write_text("incoming\n")
    preview = await sessions.worktrees.preview(session.id, "main", "merge")
    assert not calls.exists()
    before = state(repository, source)
    with pytest.raises(ProtectionError, match="Signature refused"):
        await sessions.worktrees.integrate(
            session.id, "main", "merge", preview.token, "Both commits must be signed"
        )
    assert calls.read_text() == "2\n"
    assert state(repository, source) == before
    updated = await sessions.worktrees.get(session.id)
    assert updated.pending_integration is None
    assert updated.integrated_commit is None
