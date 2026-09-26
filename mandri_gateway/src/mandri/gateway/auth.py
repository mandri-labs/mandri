"""Per-child gateway token issuance and verification."""

import hashlib
import hmac
import secrets
from collections.abc import Mapping

_PREFIX = "mandri"
_SECRET_BYTES = 32


def child_token_from_headers(headers: Mapping[str, str]) -> str | None:
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key
    google_key = headers.get("x-goog-api-key")
    if google_key:
        return google_key
    authorization = headers.get("Authorization")
    if authorization is not None and authorization.startswith("Bearer "):
        return authorization[len("Bearer ") :]
    return None


class ChildTokenAuth:
    """Stateless HMAC tokens bound to a single gateway route."""

    def __init__(self) -> None:
        self._secret = secrets.token_bytes(_SECRET_BYTES)

    def issue(self, route_id: str) -> str:
        return self._sign(route_id)

    def verify(self, route_id: str, token: str | None) -> bool:
        if token is None:
            return False
        try:
            presented = token.encode()
        except UnicodeEncodeError:
            return False
        expected = self._sign(route_id).encode()
        return hmac.compare_digest(presented, expected)

    def _sign(self, route_id: str) -> str:
        digest = hmac.new(self._secret, route_id.encode(), hashlib.sha256).hexdigest()
        return f"{_PREFIX}_{digest}"
