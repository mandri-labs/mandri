"""Resume REST permission validation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import runtime_service, sessions_service
from mandri.config.errors import ConfigError

SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"


@pytest.mark.parametrize(
    "body,expected", [(None, None), ({}, None), ({"mode": "acceptEdits"}, "acceptEdits")]
)
def test_resume_accepts_optional_permission_mode(make_client, body, expected):
    runtime = SimpleNamespace(
        resume_session=AsyncMock(
            return_value=SimpleNamespace(
                id=SESSION_ID,
                harness="claude",
                route_id="route",
                project_path="/workspace",
                mode=expected,
            )
        )
    )
    sessions = SimpleNamespace(get_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(f"/v1/sessions/{SESSION_ID}/resume", json=body)
    assert response.status_code == 200
    runtime.resume_session.assert_awaited_once_with(SESSION_ID, mode=expected)
    assert response.json()["mode"] == expected
    assert response.json()["harness"] == "claude"


def test_resume_rejects_harness_change(make_client):
    runtime = SimpleNamespace(resume_session=AsyncMock())
    sessions = SimpleNamespace(get_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(f"/v1/sessions/{SESSION_ID}/resume", json={"harness": "codex"})
    assert response.status_code == 422
    runtime.resume_session.assert_not_called()


def test_resume_invalid_permission_returns_validation_error(make_client):
    runtime = SimpleNamespace(resume_session=AsyncMock(side_effect=ConfigError("invalid mode")))
    sessions = SimpleNamespace(get_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(f"/v1/sessions/{SESSION_ID}/resume", json={"mode": "invalid"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"
