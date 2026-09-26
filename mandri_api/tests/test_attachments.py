import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import runtime_service, sessions_service
from mandri.core.ids import HarnessKind, ProjectPath, SessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.runtime.attachments import AttachmentStore

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
