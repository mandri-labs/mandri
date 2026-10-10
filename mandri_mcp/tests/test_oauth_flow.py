import asyncio
import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import httpx2
from mandri.mcp.oauth import OAuthManager, OAuthStorage
from mandri.mcp.storage import SecretStore


async def test_sdk_discovers_registers_authorizes_with_pkce_and_refreshes(tmp_path):
    store = SecretStore(tmp_path)
    manager = OAuthManager(store, 9000)
    manager.begin("server")
    provider = manager.provider("server", "https://example.invalid/mcp", lambda identity: None)
    requests = []
    exchanges = []
    challenge = {}

    def handle(request):
        requests.append((request.method, str(request.url)))
        path = request.url.path
        if "oauth-protected-resource" in path:
            return httpx2.Response(
                200,
                json={
                    "resource": "https://example.invalid/mcp",
                    "authorization_servers": ["https://auth.example.invalid"],
                    "scopes_supported": ["read"],
                },
            )
        if ".well-known" in path:
            return httpx2.Response(
                200,
                json={
                    "issuer": "https://auth.example.invalid",
                    "authorization_endpoint": "https://auth.example.invalid/authorize",
                    "token_endpoint": "https://auth.example.invalid/token",
                    "registration_endpoint": "https://auth.example.invalid/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"],
                    "token_endpoint_auth_methods_supported": ["none"],
                },
            )
        if path == "/register":
            return httpx2.Response(
                201,
                json={
                    **json.loads(request.content),
                    "client_id": "registered-client",
                },
            )
        if path == "/token":
            data = parse_qs(request.content.decode())
            exchanges.append(data)
            if data["grant_type"] == ["authorization_code"]:
                verifier = data["code_verifier"][0]
                digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                assert digest.rstrip(b"=").decode() == challenge["code_challenge"][0]
                assert data["code"] == ["synthetic-code"]
                assert data["resource"] == ["https://example.invalid/mcp"]
            else:
                assert data["grant_type"] == ["refresh_token"]
                assert data["refresh_token"] == ["synthetic-refresh"]
            return httpx2.Response(
                200,
                json={
                    "access_token": "synthetic-access",
                    "token_type": "Bearer",
                    "refresh_token": "synthetic-refresh",
                    "expires_in": 3600,
                },
            )
        if request.headers.get("authorization") == "Bearer synthetic-access":
            return httpx2.Response(200, json={"ok": True})
        return httpx2.Response(
            401,
            headers={
                "WWW-Authenticate": 'Bearer resource_metadata="https://example.invalid/.well-known/oauth-protected-resource"',
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle), auth=provider) as client:
        pending = asyncio.create_task(client.get("https://example.invalid/mcp"))
        async with asyncio.timeout(3):
            while not manager.url("server"):
                await asyncio.sleep(0)
        challenge.update(parse_qs(urlsplit(manager.url("server")).query))
        assert challenge["code_challenge_method"] == ["S256"]
        assert challenge["redirect_uri"] == ["http://127.0.0.1:9000/v1/mcp/oauth/callback/server"]
        manager.complete("server", "synthetic-code", challenge["state"][0], None)
        assert (await pending).status_code == 200
    assert len(exchanges) == 1
    assert (await OAuthStorage(store, "server").get_client_info()).client_id == "registered-client"
    OAuthStorage(store, "server").save({"expires": 0})
    restarted = manager.provider("server", "https://example.invalid/mcp", lambda identity: None)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle), auth=restarted) as client:
        assert (await client.get("https://example.invalid/mcp")).status_code == 200
    assert len(exchanges) == 2
    assert not any(b"synthetic-access" in path.read_bytes() for path in tmp_path.glob("*"))
    manager.close()
