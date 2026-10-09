import os
import subprocess
from unittest.mock import Mock

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_git import git_values
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture
def git_environment(monkeypatch, tmp_path):
    global_config = tmp_path / "global"
    global_config.write_text("[user]\nname=Global Author\nemail=global@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key in list(os.environ):
        if (
            key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
            or key in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_DIR", "GIT_WORK_TREE"}
        ):
            monkeypatch.delenv(key, raising=False)
    return global_config


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path, git_environment):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "main")
    return root


@pytest.mark.parametrize("condition", ["gitdir", "gitdir/i", "onbranch"])
def test_effective_git_identity_matches_git_in_each_directory(
    repository, git_environment, condition
):
    included = git_environment.parent / "included"
    included.write_text("[user]\nname=Selected Author\nemail=selected@example.com\n")
    pattern = "main" if condition == "onbranch" else str(repository).replace("\\", "/") + "/"
    if condition == "gitdir/i":
        pattern = pattern.upper()
    with git_environment.open("a") as stream:
        stream.write(f'[includeIf "{condition}:{pattern}"]\npath=included\n')
    values = git_values(repository)
    assert values["author"] == git(repository, "config", "--get", "user.name") == "Selected Author"
    assert values["email"] == git(repository, "config", "--get", "user.email")
    engine = SurrogateEngine(SurrogateScope("git-identity"))
    engine.register(values["author"], "identity")
    assert "Selected Author" not in engine.protect_text("Observed: Selected Author")


def test_worktree_config_overrides_common_config(repository, tmp_path):
    git(repository, "config", "user.name", "Common Author")
    git(repository, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture")
    worktree = tmp_path / "selected-worktree"
    git(repository, "worktree", "add", "--detach", str(worktree))
    git(repository, "config", "extensions.worktreeConfig", "true")
    git(worktree, "config", "--worktree", "user.name", "Worktree Author")
    assert git_values(repository)["author"] == "Common Author"
    assert (
        git_values(worktree)["author"]
        == git(worktree, "config", "--get", "user.name")
        == "Worktree Author"
    )


def test_nested_directory_global_override_and_multiline_values(
    repository, git_environment, monkeypatch
):
    nested = repository / "src" / "nested"
    nested.mkdir(parents=True)
    git(repository, "config", "user.name", "First\nSecond")
    assert git_values(nested)["author"] == "First\nSecond"
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Environment Author")
    assert git_values(nested)["author"] == "Environment Author"


@pytest.mark.parametrize("kind", ["timeout", "missing", "bad-status", "oversize", "invalid-utf8"])
def test_git_resolution_failures_fail_closed_without_private_diagnostics(
    monkeypatch, tmp_path, kind
):
    def run(command, **kwargs):
        assert command[-1] == r"^(user\.(name|email)|remote\.origin\.url)$"
        assert kwargs["cwd"] == tmp_path
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, 5, output=b"PRIVATE")
        if kind == "missing":
            raise FileNotFoundError("PRIVATE")
        if kind == "oversize":
            kwargs["stdout"].write(b"X" * 65_537)
        elif kind == "invalid-utf8":
            kwargs["stdout"].write(b"user.name\n\xff\0")
        return Mock(returncode=128 if kind == "bad-status" else 0)

    monkeypatch.setattr("mandri.gateway.privacy_git.subprocess.run", run)
    with pytest.raises(ProtectionError) as failure:
        git_values(tmp_path)
    assert failure.value.code == "privacy_context_unavailable"
    assert "PRIVATE" not in str(failure.value)


def test_no_git_identity_is_a_valid_empty_context(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mandri.gateway.privacy_git.subprocess.run", lambda *args, **kwargs: Mock(returncode=1)
    )
    assert git_values(tmp_path) == {}
