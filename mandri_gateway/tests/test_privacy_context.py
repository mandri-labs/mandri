import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_context import git_values, local_text, metadata_read_flags, seed_context
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture(autouse=True)
def synthetic_process(monkeypatch, tmp_path):
    monkeypatch.setattr("mandri.gateway.privacy_context.getpass.getuser", lambda: "synthetic-user")
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.socket.gethostname", lambda: "synthetic-host"
    )
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.Path.home", lambda: tmp_path / "SyntheticHome"
    )
    config = tmp_path / "global-git-config"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "Project"
    git = root / ".git"
    git.mkdir(parents=True)
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    with (git / "config").open("a") as config:
        config.write(
            '[remote "origin"]\nurl = git@github.com:private-customer/private-repo.git\n'
            '[user]\nname = "Synthetic Author"\nemail = author@private.example\n'
            "[include]\npath = /unrelated/profile/config\n"
            "[credential]\nhelper = !never execute any process\n"
        )
    return root


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_selected_local_context_is_seeded_without_following_includes(repository):
    scope = SurrogateScope("context")
    engine = SurrogateEngine(scope)
    seed_context(engine, repository)
    originals = {item.original for item in scope.mappings}
    assert {
        "synthetic-user",
        "synthetic-host",
        "Synthetic Author",
        "author@private.example",
        "private-customer",
        "private-repo",
    } <= originals
    assert not any("unrelated" in item or "never execute" in item for item in originals)
    result = engine.protect_text("Synthetic Author uses synthetic-host for private-customer")
    assert "Synthetic Author" not in result
    assert "synthetic-host" not in result
    assert "private-customer" not in result
    assert (
        engine.restore_text(result) == "Synthetic Author uses synthetic-host for private-customer"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_worktree_common_config_and_selected_override_are_bounded(repository, tmp_path):
    root = tmp_path / "SelectedWorktree"
    for args in (
        ["-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture"],
        ["worktree", "add", "--detach", str(root)],
        ["config", "extensions.worktreeConfig", "true"],
    ):
        subprocess.run(["git", "-C", str(repository), *args], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "--worktree", "user.email", "selected@private.example"],
        capture_output=True,
        check=True,
    )
    directory = Path((root / ".git").read_text().strip().removeprefix("gitdir: "))
    selected = git_values(root)
    assert selected == {
        "origin": "git@github.com:private-customer/private-repo.git",
        "author": "Synthetic Author",
        "email": "selected@private.example",
    }
    (directory / "commondir").write_text("../../../unrelated\n")
    with pytest.raises(ProtectionError):
        git_values(root)


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_invalid_worktree_metadata_is_rejected_by_git(repository, tmp_path):
    root = tmp_path / "SelectedWorktree"
    root.mkdir()
    directory = repository / ".git" / "worktrees" / "selected"
    directory.mkdir(parents=True)
    (root / ".git").write_text(f"gitdir: {directory}\n")
    (directory / "gitdir").write_text(str(tmp_path / "different" / ".git"))
    (directory / "commondir").write_text("../..\n")
    with pytest.raises(ProtectionError):
        git_values(root)


@pytest.mark.parametrize("kind", ["oversize", "directory", "invalid_utf8"])
@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_metadata_reader_rejects_unbounded_or_nonregular_inputs(tmp_path, kind):
    selected = tmp_path / "config"
    if kind == "oversize":
        selected.write_bytes(b"x" * 65_537)
    elif kind == "directory":
        selected.mkdir()
    else:
        selected.write_bytes(b"\xff")
    with pytest.raises(ProtectionError):
        local_text(selected)


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_missing_repository_has_only_local_process_seeds(tmp_path):
    scope = SurrogateScope("empty")
    seed_context(SurrogateEngine(scope), tmp_path)
    assert {item.original for item in scope.mappings} == {"synthetic-user", "synthetic-host"}


@pytest.mark.parametrize("flag", ["O_NONBLOCK", "O_CLOEXEC"])
@pytest.mark.skipif(sys.platform == "win32", reason="Checks POSIX open flags")
def test_missing_secure_metadata_flags_fail_before_any_file_read(monkeypatch, tmp_path, flag):
    monkeypatch.delattr(f"mandri.gateway.privacy_context.os.{flag}", raising=False)
    reader = Mock(side_effect=AssertionError("unsupported metadata must not be read"))
    monkeypatch.setattr("mandri.gateway.privacy_context.os.open", reader)
    with pytest.raises(ProtectionError) as failure:
        local_text(tmp_path / "missing", optional=True)
    assert failure.value.code == "privacy_platform_unsupported"
    reader.assert_not_called()


@pytest.mark.skipif(sys.platform == "win32", reason="Checks POSIX open flags")
async def test_scope_readiness_reports_platform_before_available_keyring(monkeypatch, tmp_path):
    repository = Mock()
    repository.readiness = AsyncMock()
    scopes = PrivacyScopes(repository, PrivacySettings())
    monkeypatch.delattr("mandri.gateway.privacy_context.os.O_NONBLOCK", raising=False)
    with pytest.raises(ProtectionError) as failure:
        await scopes.readiness()
    assert failure.value.code == "privacy_platform_unsupported"
    repository.readiness.assert_not_awaited()
    with pytest.raises(ProtectionError) as creation:
        await scopes.create(str(tmp_path))
    assert creation.value.code == "privacy_platform_unsupported"


@pytest.mark.skipif(sys.platform == "win32", reason="Checks POSIX open flags")
def test_secure_metadata_flags_require_the_complete_available_mask(monkeypatch):
    for name, value in (("O_NONBLOCK", 512), ("O_CLOEXEC", 1024)):
        monkeypatch.setattr(f"mandri.gateway.privacy_context.os.{name}", value, raising=False)
    assert metadata_read_flags() & (512 | 1024) == 512 | 1024


def test_windows_metadata_reader_rejects_replaced_file(monkeypatch, tmp_path):
    selected = tmp_path / "config"
    selected.write_text("[user]\nname = Original\n")
    other = tmp_path / "other"
    other.write_text("[user]\nname = Other\n")
    monkeypatch.setattr("mandri.gateway.privacy_context._WINDOWS", True)
    monkeypatch.setattr("mandri.gateway.privacy_context.os.O_BINARY", 0, raising=False)
    monkeypatch.setattr("mandri.gateway.privacy_context.os.O_NOINHERIT", 0, raising=False)
    opened = os.open
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.os.open", lambda path, flags: opened(other, flags)
    )
    with pytest.raises(ProtectionError):
        local_text(selected)


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
def test_git_directory_indirection_uses_native_git_resolution(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    other = tmp_path / "unrelated"
    other.mkdir()
    subprocess.run(["git", "init", str(other)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(other), "config", "user.name", "Indirect Author"],
        capture_output=True,
        check=True,
    )
    (root / ".git").write_text(f"gitdir: {other / '.git'}")
    assert git_values(root) == {"author": "Indirect Author"}
    (root / ".git").write_text(f"gitdir: {other / 'missing'}")
    with pytest.raises(ProtectionError):
        git_values(root)
