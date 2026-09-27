import base64
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.ids import HarnessKind, ProjectPath, SessionId
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.core.protocol.registry import SessionPromptParams
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.prompt import UserPrompt
from mandri.runtime.attachments import AttachmentError, AttachmentStore
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_files import open_session_file

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2WQAAAAASUVORK5CYII="
)


def session(tmp_path, *, backend=ExecutionBackend.HOST, harness=HarnessKind.CLAUDE):
    workspace = tmp_path / "workspace"
    cwd = workspace / "nested"
    cwd.mkdir(parents=True, exist_ok=True)
    state = tmp_path / "state" / "session"
    state.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        id=SessionId("session"),
        harness=harness,
        project_path=ProjectPath(str(cwd)),
        execution_backend=backend,
        privacy_mode=PrivacyMode.NONE,
        execution_context=json.dumps(
            {
                "version": "1",
                "workspace_root": str(workspace),
                "native_state_root": str(state),
                "container_root": "/workspace",
                "native_home": "/home/worker",
            }
        )
        if backend is ExecutionBackend.DOCKER
        else None,
    )


async def chunks(data):
    for offset in range(0, len(data), 7):
        yield data[offset : offset + 7]


@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize("privacy", list(PrivacyMode))
async def test_upload_prepare_and_reopen_in_execution_context(tmp_path, backend, privacy):
    record = session(tmp_path, backend=backend)
    record.privacy_mode = privacy
    store = AttachmentStore(tmp_path / "uploads")
    attachment = await store.upload(record, "a capture.png", chunks(PNG))
    prepared = store.prepare(record, "", [attachment.id])
    assert prepared.attachments[0].data == PNG
    assert "a%20capture.png" in prepared.text
    stream, name, size = open_session_file(record, store, prepared.attachments[0].path)
    with stream:
        assert stream.read() == PNG

    with pytest.raises(AttachmentError, match="unavailable"):
        open_session_file(record, store, "missing.txt")
    assert (name, size) == ("a capture.png", len(PNG))
    reopened = AttachmentStore(tmp_path / "uploads")
    assert reopened.prepare(record, "", [attachment.id]) == prepared
    assert not list(Path(record.project_path).iterdir())
    if backend is ExecutionBackend.DOCKER:
        assert prepared.attachments[0].path.startswith("/home/worker/mandri-attachments/")
    else:
        assert Path(prepared.attachments[0].path).is_relative_to(store.root)


@pytest.mark.parametrize("backend", list(ExecutionBackend))
async def test_relative_files_use_session_cwd_not_daemon_cwd(tmp_path, monkeypatch, backend):
    record = session(tmp_path, backend=backend)
    store = AttachmentStore(tmp_path / "uploads")
    (Path(record.project_path) / "result.md").write_text("workspace result")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "result.md").write_text("wrong file")
    stream, _, _ = open_session_file(record, store, "result.md")
    with stream:
        assert stream.read() == b"workspace result"
    if backend is ExecutionBackend.DOCKER:
        stream, _, _ = open_session_file(record, store, "/workspace/nested/result.md")
        with stream:
            assert stream.read() == b"workspace result"


async def test_reject_cross_session_and_outside_workspace(tmp_path):
    record = session(tmp_path)
    store = AttachmentStore(tmp_path / "uploads")
    item = await store.upload(record, "notes.txt", chunks(b"notes"))
    with pytest.raises(AttachmentError):
        store.read("another-session", item.id)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    (Path(record.project_path) / "escape").symlink_to(secret)
    for path in [str(secret), "../../secret.txt", "escape"]:
        with pytest.raises(AttachmentError):
            open_session_file(record, store, path)
    for identity in ["../secret", "a" * 32]:
        with pytest.raises(AttachmentError):
            store.prepare(record, "", [identity])


async def test_privacy_change_preserves_image_delivery_and_harness_support_is_checked(tmp_path):
    record = session(tmp_path)
    store = AttachmentStore(tmp_path / "uploads")
    item = await store.upload(record, "image.png", chunks(PNG))
    record.privacy_mode = PrivacyMode.SURROGATE
    assert store.prepare(record, "", [item.id]).attachments[0].data == PNG
    record.harness = HarnessKind.AGY
    with pytest.raises(AttachmentError, match="harness"):
        await store.upload(record, "file.txt", chunks(b"text"))


async def test_failed_upload_removes_partial_files(tmp_path, monkeypatch):
    record = session(tmp_path)
    store = AttachmentStore(tmp_path / "uploads")
    monkeypatch.setattr("mandri.runtime.attachments.MAX_FILE_BYTES", 10)
    with pytest.raises(AttachmentError, match="limit"):
        await store.upload(record, "large.txt", chunks(b"x" * 11))
    assert not list(store.directory(str(record.id)).iterdir())
    with pytest.raises(AttachmentError):
        await store.upload(record, "false.png", chunks(b"not an image"))


@pytest.mark.parametrize("privacy", list(PrivacyMode))
async def test_runtime_delivers_prepared_image_without_stopping_session(tmp_path, privacy):
    record = session(tmp_path)
    record.privacy_mode = privacy
    runtime = RuntimeService(
        {},
        sessions=SimpleNamespace(get_session=AsyncMock(return_value=record)),
        attachments_dir=tmp_path / "uploads",
    )
    runtime.registry.mark_live("session")
    control = SimpleNamespace(send_prompt=AsyncMock(return_value=PromptOutcome(PromptState.QUEUED)))
    runtime._session_state("session").control = control
    runtime.stop_session = AsyncMock()
    item = await runtime.attachments.upload(record, "image.png", chunks(PNG))
    result = await runtime.send_session_prompt("session", "", [item.id])
    assert result.state is PromptState.QUEUED
    prepared = control.send_prompt.call_args.args[0]
    assert isinstance(prepared, UserPrompt)
    assert prepared.attachments[0].data == PNG
    runtime.stop_session.assert_not_called()


def test_image_only_public_contract_and_empty_rejection():
    assert (
        SessionPromptParams(
            session_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", attachments=["a" * 32]
        ).content
        == ""
    )
    with pytest.raises(ValueError):
        SessionPromptParams(session_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", content=" ")


@pytest.mark.parametrize("privacy", list(PrivacyMode))
async def test_native_adapters_preserve_image_content_and_codex_steering(tmp_path, privacy):
    record = session(tmp_path)
    record.privacy_mode = privacy
    store = AttachmentStore(tmp_path / "uploads")
    item = await store.upload(record, "image.png", chunks(PNG))
    prepared = store.prepare(record, "caption", [item.id])
    claude = ClaudeControlAdapter(stdout_pump=None, stderr_pump=None, stdin=None)
    claude._ensure_pump = lambda: None
    claude._send = AsyncMock()
    await claude.send_prompt(prepared)
    content = claude._send.call_args.args[0]["message"]["content"]
    assert base64.b64decode(content[1]["source"]["data"]) == PNG
    assert content[0]["text"] == prepared.text
    codex = CodexControlAdapter(stdout_pump=None, stdin=None)
    codex._thread_id = "thread"
    codex._call = AsyncMock(return_value={"result": {"turn": {"id": "turn"}}})
    await codex.send_prompt(prepared)
    await codex.send_prompt(prepared)
    calls = codex._call.call_args_list
    assert [call.args[0] for call in calls] == ["turn/start", "turn/steer"]
    assert calls[1].args[1]["expectedTurnId"] == "turn"
    for call in calls:
        image = call.args[1]["input"][1]
        assert image["type"] == "localImage"
        assert Path(image["path"]).read_bytes() == PNG
    opencode = OpencodeControlAdapter("http://unused.invalid", "session")
    opencode._request = AsyncMock(return_value=httpx.Response(204))
    try:
        await opencode.send_prompt(prepared)
        method, path, body = opencode._request.call_args.args
        assert (method, path) == ("POST", "/session/session/prompt_async")
        assert base64.b64decode(body["parts"][1]["url"].split(",", 1)[1]) == PNG
    finally:
        await opencode.aclose()


@pytest.mark.parametrize(
    "name,data",
    [
        ("broken.jpg", b"\xff\xd8\xff\xe0\x00\x10broken"),
        ("huge.png", PNG[:16] + (100_000).to_bytes(4) * 2 + PNG[24:]),
        ("empty.png", PNG[:16] + bytes(8) + PNG[24:]),
        ("unsupported.gif", b"GIF89a"),
    ],
)
async def test_invalid_images_are_rejected_and_cleaned_up(tmp_path, name, data):
    record = session(tmp_path)
    store = AttachmentStore(tmp_path / "uploads")
    with pytest.raises(AttachmentError):
        await store.upload(record, name, chunks(data))
    assert not list(store.directory(str(record.id)).iterdir())


async def test_repeated_image_preparation_reuses_files_open_on_windows(tmp_path, monkeypatch):
    record = session(tmp_path, harness=HarnessKind.CODEX)
    store = AttachmentStore(tmp_path / "uploads")
    items = [
        await store.upload(record, f"Capture écran {index}.png", chunks(PNG)) for index in range(3)
    ]
    identities = [item.id for item in items]
    prepared = store.prepare(record, "Three images", identities)
    paths = [Path(item.path) for item in prepared.attachments]
    streams = [path.open("rb") for path in paths]
    replace = os.replace

    def windows_replace(source, destination):
        if Path(destination) in paths:
            error = PermissionError(errno.EACCES, "The file is open by another process")
            error.winerror = 32
            raise error
        return replace(source, destination)

    monkeypatch.setattr("mandri.runtime.attachments.sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr("mandri.runtime.attachments.os.replace", windows_replace)
    try:
        assert store.prepare(record, "Three images", identities) == prepared
        assert [stream.read() for stream in streams] == [PNG, PNG, PNG]
    finally:
        for stream in streams:
            stream.close()


async def test_host_attachment_storage_does_not_require_mounted_project(tmp_path):
    record = session(tmp_path)
    Path(record.project_path).rmdir()
    store = AttachmentStore(tmp_path / "uploads")
    item = await store.upload(record, "image.png", chunks(PNG))
    prepared = store.prepare(record, "", [item.id])
    stream, _, _ = open_session_file(record, store, prepared.attachments[0].path)
    with stream:
        assert stream.read() == PNG


async def test_prepare_repairs_modified_materialized_bytes_and_rejects_symlink(tmp_path):
    record = session(tmp_path)
    store = AttachmentStore(tmp_path / "uploads")
    item = await store.upload(record, "notes.txt", chunks(b"original"))
    original = store.prepare(record, "", [item.id])
    path = Path(original.attachments[0].path)
    path.chmod(0o644)
    path.write_bytes(b"modified")
    assert store.prepare(record, "", [item.id]) == original
    assert path.read_bytes() == b"original"
    path.chmod(0o644)
    path.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    path.symlink_to(outside)
    with pytest.raises(AttachmentError, match="symbolic link"):
        store.prepare(record, "", [item.id])
