import asyncio
from types import SimpleNamespace

import pytest
from mandri.runtime.docker_backend import DockerBackend
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.docker_workspace import workspace_root, write_context
from mandri.runtime.errors.docker import DockerExecutionError


def test_resume_rejects_a_replaced_workspace_at_the_identical_path(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    (root / "original.txt").write_text("preserved")
    backend = DockerBackend(DockerConfig("image", tmp_path / "states"))
    context = backend.context("session", root, resume=False, image_id="sha256:fixed")
    write_context(tmp_path / "states/session", context)
    root.rename(tmp_path / "saved")
    root.mkdir()
    with pytest.raises(DockerExecutionError, match="workspace identity changed"):
        backend.context("session", root, resume=True, image_id="sha256:fixed")
    assert (tmp_path / "saved/original.txt").read_text() == "preserved"
    assert not list(root.iterdir())


@pytest.mark.parametrize("line", [b"", b"MANDRI_WORKSPACE_READY 1:other\n", b"native output\n"])
async def test_backend_requires_exact_pre_exec_mount_witness(tmp_path, line):
    backend = DockerBackend(DockerConfig("image", tmp_path))
    stdout = asyncio.StreamReader()
    stdout.feed_data(line)
    stdout.feed_eof()
    attached = SimpleNamespace(process=SimpleNamespace(stdout=stdout))
    with pytest.raises(DockerExecutionError) as failure:
        await backend._await_workspace(attached, "1:2")
    assert failure.value.reason == "workspace_identity_changed"


async def test_mount_witness_is_consumed_without_discarding_native_output(tmp_path):
    backend = DockerBackend(DockerConfig("image", tmp_path))
    stdout = asyncio.StreamReader()
    stdout.feed_data(b'MANDRI_WORKSPACE_READY 1:2\n{"id":1,"result":{}}\n')
    await backend._await_workspace(SimpleNamespace(process=SimpleNamespace(stdout=stdout)), "1:2")
    assert await stdout.readline() == b'{"id":1,"result":{}}\n'


def test_git_metadata_symlink_is_rejected_without_opening_target(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(
        "mandri.runtime.docker_git.os.open", lambda *args: pytest.fail("Opened symlink")
    )
    with pytest.raises(DockerExecutionError):
        workspace_root(root)


def test_git_metadata_oversize_entry_is_rejected_before_read(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").write_bytes(b"x" * 4097)
    monkeypatch.setattr(
        "mandri.runtime.docker_git.os.open", lambda *args: pytest.fail("Opened oversize entry")
    )
    with pytest.raises(DockerExecutionError):
        workspace_root(root)
