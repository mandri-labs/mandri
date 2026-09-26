from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import gateway_wiring, sessions_service
from mandri.core.types.execution import PrivacyMode, ProtectionError

SESSION = "e7c3b419-c7dc-4892-8d7f-cd0868489668"


@pytest.mark.parametrize("mode", [PrivacyMode.NONE, PrivacyMode.SURROGATE])
def test_session_registry_is_masked_and_never_cacheable(make_client, mode):
    session = SimpleNamespace(privacy_mode=mode, privacy_scope_id="scope")
    service = SimpleNamespace(get_session=AsyncMock(return_value=session))
    entries = [{"kind": "email", "redacted": "cam****@****invalid"}]
    scopes = SimpleNamespace(inventory=AsyncMock(return_value=(7, entries)))
    wiring = SimpleNamespace(privacy=SimpleNamespace(scopes=scopes))
    client = make_client({sessions_service: lambda: service, gateway_wiring: lambda: wiring})
    response = client.get(f"/v1/sessions/{SESSION}/privacy")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    if mode is PrivacyMode.SURROGATE:
        assert response.json() == {"revision": 7, "entries": entries}
        scopes.inventory.assert_awaited_once_with("scope")
    else:
        assert response.json() == {"revision": 0, "entries": []}
        scopes.inventory.assert_not_awaited()


def test_unavailable_registry_is_not_reported_as_empty(make_client):
    session = SimpleNamespace(privacy_mode=PrivacyMode.SURROGATE, privacy_scope_id="scope")
    service = SimpleNamespace(get_session=AsyncMock(return_value=session))
    scopes = SimpleNamespace(
        inventory=AsyncMock(
            side_effect=ProtectionError("privacy_key_unavailable", "Privacy key is unavailable")
        )
    )
    wiring = SimpleNamespace(privacy=SimpleNamespace(scopes=scopes))
    client = make_client({sessions_service: lambda: service, gateway_wiring: lambda: wiring})
    response = client.get(f"/v1/sessions/{SESSION}/privacy")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "privacy_key_unavailable"
