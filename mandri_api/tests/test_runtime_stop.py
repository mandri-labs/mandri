from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.api.deps import runtime_service, sessions_service

SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"


def test_stop_forces_termination_without_relaunching_native_harness(make_client):
    runtime = SimpleNamespace(stop_session=AsyncMock())
    sessions = SimpleNamespace(get_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(f"/v1/sessions/{SESSION_ID}/stop")
    assert response.status_code == 204
    runtime.stop_session.assert_awaited_once_with(SESSION_ID, force=True, restore_native=False)
