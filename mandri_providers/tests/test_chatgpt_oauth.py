"""Tests for the ChatGPT OAuth flow, identity helpers, and token store."""

import asyncio
import base64
import json
import stat
import sys
from importlib.metadata import version
from pathlib import Path

import httpx
import pytest
from mandri.providers.chatgpt import oauth
from mandri.providers.chatgpt.codex_models import CLIENT_VERSION, catalog_entry, models_url
from mandri.providers.chatgpt.identity import expires_at_ms, identity, request_headers
from mandri.providers.chatgpt.store import ChatGptTokenStore, token_path
from mandri.providers.chatgpt.store import TokenSet as StoreTokenSet


def jwt(claims: dict[str, object]) -> str:
    def segment(payload: dict[str, object]) -> str:
        raw = json.dumps(payload).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'none'})}.{segment(claims)}.signature"


def test_pkce_challenge_is_s256_of_verifier() -> None:
    pair = oauth.pkce_pair()
    import hashlib

    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(pair.verifier.encode("ascii")).digest())
        .decode()
        .rstrip("=")
    )
    assert pair.challenge == expected
    assert "=" not in pair.challenge


def test_authorize_url_carries_pkce_and_client() -> None:
    url = oauth.authorize_url("http://localhost:1455/auth/callback", "state-1", "challenge-1")
    assert url.startswith(oauth.AUTHORIZE_URL + "?")
    assert "code_challenge=challenge-1" in url
    assert "code_challenge_method=S256" in url
    assert f"client_id={oauth.CLIENT_ID}" in url
    assert "state=state-1" in url


def test_callback_code_rejects_mismatched_state() -> None:
    with pytest.raises(oauth.ChatGptAuthError):
        oauth.callback_code("http://localhost/?code=abc&state=other", "expected")
    assert oauth.callback_code("http://localhost/?code=abc&state=expected", "expected") == "abc"


def test_callback_code_reports_provider_error() -> None:
    with pytest.raises(oauth.ChatGptAuthError, match="denied"):
        oauth.callback_code("http://localhost/?error=access_denied", "expected")


async def test_exchange_code_posts_form_and_returns_tokens() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type")
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"access_token": "access-1", "refresh_token": "refresh-1", "expires_in": 3600},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tokens = await oauth.exchange_code(
            client, "code-1", "verifier-1", "http://localhost:1455/auth/callback"
        )
    assert seen["url"] == oauth.TOKEN_URL
    assert seen["content_type"] == "application/x-www-form-urlencoded"
    assert "grant_type=authorization_code" in str(seen["body"])
    assert "code_verifier=verifier-1" in str(seen["body"])
    assert tokens.access_token == "access-1"
    assert tokens.refresh_token == "refresh-1"
    assert tokens.expires_at_ms > 0


async def test_refresh_keeps_old_refresh_token_when_not_rotated() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "access-2", "expires_in": 60})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tokens = await oauth.refresh_tokens(client, "refresh-old")
    assert tokens.access_token == "access-2"
    assert tokens.refresh_token == "refresh-old"


async def test_token_error_surfaces_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_grant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(oauth.ChatGptAuthError, match="invalid_grant"):
            await oauth.refresh_tokens(client, "stale")


def test_identity_and_headers_read_the_auth_claim() -> None:
    token = jwt(
        {
            "exp": 4102444800,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-1",
                "chatgpt_plan_type": "pro",
                "chatgpt_data_residency": "eu",
            },
        }
    )
    assert identity(token).account_id == "acct-1"
    assert identity(token).plan_type == "pro"
    assert expires_at_ms(token) == 4102444800000
    headers = request_headers(token)
    assert headers["Authorization"] == f"Bearer {token}"
    assert headers["ChatGPT-Account-ID"] == "acct-1"
    assert headers["x-openai-internal-codex-residency"] == "eu"


def test_headers_tolerate_malformed_tokens() -> None:
    assert request_headers("") == {}
    assert "ChatGPT-Account-ID" not in request_headers("not-a-jwt")
    assert "ChatGPT-Account-ID" not in request_headers("a.@@@.c")


def test_store_round_trips(tmp_path: Path) -> None:
    store = ChatGptTokenStore(tmp_path)
    assert store.load("work") is None
    store.save("work", StoreTokenSet(access_token="a", refresh_token="r", expires_at_ms=12345))
    loaded = store.load("work")
    assert loaded == StoreTokenSet(access_token="a", refresh_token="r", expires_at_ms=12345)
    store.clear("work")
    assert store.load("work") is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows permissions use ACLs")
def test_store_restricts_posix_permissions(tmp_path: Path) -> None:
    store = ChatGptTokenStore(tmp_path)
    store.save("work", StoreTokenSet(access_token="a", refresh_token="r", expires_at_ms=12345))
    path = token_path(tmp_path, "work")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_store_rejects_unsafe_names(tmp_path: Path) -> None:
    path = token_path(tmp_path, "../../etc/passwd")
    assert path.parent == tmp_path / "chatgpt"
    assert ".." not in path.name


def test_models_url_uses_the_configured_base() -> None:
    assert models_url(None) == "https://chatgpt.com/backend-api/codex/models"
    assert models_url("https://chatgpt.com/backend-api/codex") == (
        "https://chatgpt.com/backend-api/codex/models"
    )
    assert models_url("http://127.0.0.1:9/backend") == "http://127.0.0.1:9/backend/models"


def test_catalog_entry_keeps_slug_and_efforts() -> None:
    entry = catalog_entry(
        {
            "slug": "gpt-5.6-sol",
            "display_name": "GPT-5.6 Sol",
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
            "default_reasoning_level": "high",
            "input_modalities": ["text", "image"],
        }
    )
    assert entry == {
        "id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "reasoning_efforts": ["low", "high"],
        "default_effort": "high",
        "input_modalities": ["text", "image"],
    }
    assert catalog_entry({"display_name": "no slug"}) is None
    assert version("openai-codex-cli-bin") == CLIENT_VERSION


def test_catalog_entry_defaults_effort_to_medium_when_unlisted() -> None:
    entry = catalog_entry({"slug": "m", "supported_reasoning_levels": [{"effort": "medium"}]})
    assert entry is not None
    assert entry["default_effort"] == "medium"


async def test_loopback_server_captures_the_callback() -> None:
    server = oauth.LoopbackCallbackServer(ports=(0,))
    await server.start()
    try:
        url = f"{server.uri}?code=code-9&state=state-9"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        assert response.status_code == 200
        code, state = await asyncio.wait_for(server.wait(5.0), timeout=5.0)
        assert (code, state) == ("code-9", "state-9")
    finally:
        await server.aclose()


async def test_loopback_server_times_out() -> None:
    server = oauth.LoopbackCallbackServer(ports=(0,))
    await server.start()
    try:
        with pytest.raises(oauth.ChatGptLoginCancelled):
            await server.wait(0.05)
    finally:
        await server.aclose()


@pytest.mark.parametrize("fallback", [False, True])
async def test_loopback_server_handles_busy_ports(fallback: bool) -> None:
    occupied = oauth.LoopbackCallbackServer(ports=(0,))
    await occupied.start()
    assert occupied.port is not None
    ports = (occupied.port, 0) if fallback else (occupied.port,)
    server = oauth.LoopbackCallbackServer(ports=ports)
    try:
        if fallback:
            await server.start()
            assert server.port is not None and server.port != occupied.port
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server.uri}?code=code-9&state=state-9")
            assert response.status_code == 200
            assert await server.wait(5.0) == ("code-9", "state-9")
        else:
            with pytest.raises(oauth.ChatGptAuthError, match="loopback"):
                await server.start()
    finally:
        await server.aclose()
        await occupied.aclose()
