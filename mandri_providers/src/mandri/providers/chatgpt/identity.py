"""ChatGPT request identity derived from the OAuth access token."""

import base64
import json
from dataclasses import dataclass

_AUTH_CLAIM = "https://api.openai.com/auth"
_MAX_CLAIM_BYTES = 65536


@dataclass(frozen=True)
class ChatGptIdentity:
    account_id: str | None = None
    plan_type: str | None = None
    residency: str | None = None


def claims(token: str) -> dict[str, object]:
    if not isinstance(token, str) or not token.strip():
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload)
        if len(raw) > _MAX_CLAIM_BYTES:
            return {}
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def identity(token: str) -> ChatGptIdentity:
    auth = claims(token).get(_AUTH_CLAIM)
    auth = auth if isinstance(auth, dict) else {}
    return ChatGptIdentity(
        account_id=_text(auth.get("chatgpt_account_id")),
        plan_type=_text(auth.get("chatgpt_plan_type")),
        residency=_text(auth.get("chatgpt_data_residency"))
        or _text(auth.get("chatgpt_compute_residency")),
    )


def expires_at_ms(token: str) -> int:
    value = claims(token).get("exp")
    return value * 1000 if type(value) is int else 0


def request_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    headers = {
        "Authorization": f"Bearer {token}",
        "originator": "mandri",
    }
    info = identity(token)
    if info.account_id:
        headers["ChatGPT-Account-ID"] = info.account_id
    if info.residency:
        headers["x-openai-internal-codex-residency"] = info.residency
    return headers


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
