import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.actions import _make_history_handler
from mandri.api.deps import runtime_service, sessions_service
from mandri.api.routers.transcript_records import RecordResponse
from mandri.core.protocol.registry import SessionHistoryParams
from mandri.runtime.control.errors import ControlTransportError
from mandri.sessions.transcripts.errors import HarnessStoreUnavailableError, TranscriptStoreError
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.record_download import RecordDownload
from starlette.requests import ClientDisconnect

SESSION = "00000000-0000-0000-0000-000000000001"


async def test_history_survives_status_store_failure():
    service = SimpleNamespace(
        history=AsyncMock(
            return_value=SimpleNamespace(entries=['"message"'], next_token=None, has_more=False)
        ),
        external_status=AsyncMock(side_effect=TranscriptStoreError("unavailable")),
    )
    result = await _make_history_handler(service)(SessionHistoryParams(session_id=SESSION))
    assert result["entries"] == ['"message"']
    assert result["external_busy"] is None
    assert result["turn_active"] is None


async def test_history_exposes_large_record_preview(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "CommandExecution",
                        "command": ["echo", "hello"],
                        "stdout": "output\n" * 200000,
                    },
                },
            }
        )
        + "\n"
    )
    service = SimpleNamespace(
        history=AsyncMock(return_value=recent_jsonl(path, None, 10)),
        external_status=AsyncMock(return_value=(False, False)),
    )
    result = await _make_history_handler(service)(SessionHistoryParams(session_id=SESSION))
    marker = json.loads(result["entries"][0])
    assert marker["event_kind"] == "command"
    assert marker["preview"].startswith("echo hello\n\noutput\n")
    assert marker["preview_truncated"] is True
    assert len(marker["preview"]) <= 16384


def test_download_endpoint_streams_the_original_record(make_client):
    handle = io.BytesIO(b'{"type":"compacted"}\n')
    service = SimpleNamespace(history_record=AsyncMock(return_value=RecordDownload(handle, 21)))
    with make_client({sessions_service: lambda: service}) as client:
        response = client.get(
            f"/v1/sessions/{SESSION}/history/record", params={"reference": "token"}
        )
    assert response.content == b'{"type":"compacted"}\n'
    assert response.headers["content-disposition"].startswith("attachment;")
    assert handle.closed
    service.history_record.assert_awaited_once_with(SESSION, "token")


async def test_download_closes_file_after_client_disconnect():
    handle = io.BytesIO(b"x" * 131072)
    response = RecordResponse(RecordDownload(handle, 131072))

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("disconnected")

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
    assert handle.closed


def test_resume_with_unknown_activity_returns_a_structured_error(make_client):
    runtime = SimpleNamespace(
        resume_session=AsyncMock(
            side_effect=HarnessStoreUnavailableError("external session activity is unknown")
        )
    )
    service = SimpleNamespace(get_session=AsyncMock())
    with make_client(
        {runtime_service: lambda: runtime, sessions_service: lambda: service}
    ) as client:
        response = client.post(f"/v1/sessions/{SESSION}/resume", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "harness_store_unavailable"


def test_resume_transport_failure_is_not_reported_as_unavailable_history(make_client):
    runtime = SimpleNamespace(
        resume_session=AsyncMock(side_effect=ControlTransportError("Harness initialization failed"))
    )
    service = SimpleNamespace(get_session=AsyncMock())
    with make_client(
        {runtime_service: lambda: runtime, sessions_service: lambda: service}
    ) as client:
        response = client.post(f"/v1/sessions/{SESSION}/resume", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "control_delivery_failed"
    assert response.json()["error"]["message"] == "Harness initialization failed"
