from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mandri.api.deps import runtime_service
from mandri.api.errors import register_error_handlers
from mandri.api.routers.sessions import router
from mandri.core.types.availability import SessionActivityState, SessionAvailability, SessionOwner
from mandri.sessions.errors import SessionRunningError

SESSION_ID = "00000000-0000-0000-0000-000000000001"


def test_availability_endpoint_and_explicit_release_confirmation():
    availability = SessionAvailability(
        SessionOwner.EXTERNAL, SessionActivityState.IDLE, reason="external_release_unsupported"
    )
    runtime = SimpleNamespace(
        session_availability=AsyncMock(return_value=availability),
        release_session=AsyncMock(side_effect=SessionRunningError("changing")),
    )
    app = FastAPI()
    app.include_router(router)
    register_error_handlers(app)
    app.dependency_overrides[runtime_service] = lambda: runtime
    with TestClient(app) as client:
        response = client.get(f"/sessions/{SESSION_ID}/availability")
        assert response.status_code == 200
        assert response.json() == {
            "owner": "external",
            "activity": "idle",
            "can_resume": False,
            "can_release": False,
            "can_restore": False,
            "reason": "external_release_unsupported",
        }
        response = client.post(f"/sessions/{SESSION_ID}/release", json={"confirmed": True})
        assert response.status_code == 409
        runtime.release_session.assert_awaited_once_with(SESSION_ID, confirmed=True)
