import json

import pytest


@pytest.fixture
def agy_oauth_token():
    return json.dumps(
        {
            "auth_method": "consumer",
            "token": {
                "access_token": "synthetic-access",
                "refresh_token": "synthetic-refresh",
                "token_type": "Bearer",
                "expiry": "2099-01-01T00:00:00Z",
            },
        }
    )


@pytest.fixture
def agy_native_credentials(monkeypatch, agy_oauth_token):
    monkeypatch.setattr("mandri.runtime.agy_auth._stored_credential", lambda: agy_oauth_token)
