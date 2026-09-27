import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.api.actions import build_action_registry
from mandri.api.deps import runtime_service, sessions_service
from mandri.core.ids import HarnessKind, ProjectPath, SessionId
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import RequestFrame
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.runtime.attachments import AttachmentStorageError, AttachmentStore
from mandri.runtime.commands import CommandService

SESSION_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.mark.parametrize("privacy", list(PrivacyMode))
@pytest.mark.parametrize(
    "name,content",
    [
        ("notes.md", b"# A file"),
        (
            "image.png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2WQAAAAASUVORK5CYII="
            ),
        ),
    ],
)
def test_upload_and_session_file_route_contract(make_client, tmp_path, privacy, name, content):
    workspace = tmp_path / "project"
    workspace.mkdir()
    record = SimpleNamespace(
        id=SessionId(SESSION_ID),
        harness=HarnessKind.CODEX,
        project_path=ProjectPath(str(workspace)),
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=privacy,
        execution_context=None,
    )
    store = AttachmentStore(tmp_path / "uploads")
    runtime = SimpleNamespace(attachments=store)
    sessions = SimpleNamespace(get_session=AsyncMock(return_value=record))
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(
        f"/v1/sessions/{SESSION_ID}/attachments",
        params={"name": name},
        content=content,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["size"] == len(content)
    prepared = store.prepare(record, "", [uploaded["id"]])
    assert prepared.text == uploaded["reference"]
    download = client.get(
        f"/v1/sessions/{SESSION_ID}/files", params={"path": prepared.attachments[0].path}
    )
    assert download.status_code == 200
    assert download.content == content
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["content-disposition"].startswith("attachment;")
    outside = tmp_path / "secret"
    outside.write_text("private")
    response = client.get(f"/v1/sessions/{SESSION_ID}/files", params={"path": str(outside)})
    assert response.status_code == 404


@pytest.mark.parametrize("stage", ["upload", "materialize"])
def test_storage_io_failure_has_specific_retryable_error(
    make_client, tmp_path, monkeypatch, caplog, stage
):
    record = SimpleNamespace(
        id=SessionId(SESSION_ID),
        harness=HarnessKind.CODEX,
        project_path=ProjectPath(str(tmp_path)),
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.NONE,
        execution_context=None,
    )
    store = AttachmentStore(tmp_path / "uploads")
    if stage == "upload":
        store.root.write_text("unavailable directory")
    else:
        monkeypatch.setattr(store, "materialize", Mock(side_effect=PermissionError(13, "locked")))
    runtime = SimpleNamespace(attachments=store)
    sessions = SimpleNamespace(get_session=AsyncMock(return_value=record))
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(
        f"/v1/sessions/{SESSION_ID}/attachments",
        params={"name": "notes.txt"},
        content=b"notes",
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "attachment_storage_unavailable"
    assert SESSION_ID in caplog.text
    assert "errno=" in caplog.text


async def test_prompt_storage_failure_preserves_operation_identity_and_specific_error():
    runtime = SimpleNamespace(
        commands=Mock(spec=CommandService),
        send_session_prompt=AsyncMock(
            side_effect=AttachmentStorageError("Session file storage is unavailable")
        ),
    )
    response = await build_action_registry(runtime).handle(
        RequestFrame(
            type="request",
            op_id="attachment-operation",
            action="session.prompt",
            params={"session_id": SESSION_ID, "content": "image", "attachments": ["a" * 32]},
        )
    )
    assert response.op_id == "attachment-operation"
    assert not response.ok
    assert response.error.code is ProtocolErrorCode.ATTACHMENT_STORAGE_UNAVAILABLE
