import asyncio
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from mandri.core.types.mcp import McpServerConfig
from mandri.mcp.oauth import AuthenticationRequired, OAuthManager, OAuthStorage
from mandri.mcp.registry import Registry
from mandri.mcp.service import McpService
from mandri.mcp.storage import SecretStore
from mcp.shared.auth import OAuthToken


async def test_oauth_stays_noninteractive_at_startup(tmp_path):
    service = McpService(tmp_path, 9000)
    service.start()
    row = await service.add(
        McpServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.invalid/mcp",
            auth="oauth",
        )
    )
    async with asyncio.timeout(1):
        while service.supervisors[row.id].state == "starting":
            await asyncio.sleep(0)
    assert service.supervisors[row.id].state == "auth_required"
    assert not service.oauth.pending
    await service.aclose()


async def test_callback_requires_explicit_login_and_exact_state(tmp_path):
    manager = OAuthManager(SecretStore(tmp_path), 9000)
    changed = []
    provider = manager.provider("server", "https://example.invalid/mcp", changed.append)
    url = "https://example.invalid/authorize?state=expected&code_challenge=challenge"
    with pytest.raises(AuthenticationRequired):
        await provider.context.redirect_handler(url)
    manager.begin("server")
    await provider.context.redirect_handler(url)
    assert changed == ["server"]
    assert parse_qs(urlsplit(manager.url("server")).query)["state"] == ["expected"]
    with pytest.raises(ValueError):
        manager.complete("server", "code", "wrong", None)
    manager.complete("server", "code", "expected", "https://example.invalid")
    result = await provider.context.callback_handler()
    assert result.code == "code" and result.state == "expected"
    assert result.iss == "https://example.invalid"
    with pytest.raises(ValueError):
        manager.complete("server", "again", "expected", None)
    manager.close()
    assert not manager.pending


async def test_token_expiry_survives_daemon_restart_and_preserves_server_secrets(
    tmp_path, monkeypatch
):
    store = SecretStore(tmp_path)
    store.save("server", {"env": {"KEY": "secret"}})
    storage = OAuthStorage(store, "server")
    now = time.time()
    monkeypatch.setattr("mandri.mcp.oauth.time.time", lambda: now)
    await storage.set_tokens(OAuthToken(access_token="private", token_type="Bearer", expires_in=60))
    monkeypatch.setattr("mandri.mcp.oauth.time.time", lambda: now + 70)
    restored = await OAuthStorage(SecretStore(tmp_path), "server").get_tokens()
    assert restored.expires_in == 0
    assert store.load("server")["env"] == {"KEY": "secret"}
    assert all(b"private" not in p.read_bytes() for p in tmp_path.glob("*"))
    storage.clear()
    assert store.load("server") == {"env": {"KEY": "secret"}}


def test_failed_update_restores_oauth_and_configuration(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    row = registry.add(
        McpServerConfig(name="remote", transport="streamable_http", url="https://example.invalid/a")
    )
    secret = registry.secrets.load(row.id) | {"oauth": {"tokens": {"access_token": "private"}}}
    registry.secrets.save(row.id, secret)

    def fail():
        raise OSError("Disk full")

    monkeypatch.setattr(registry, "save", fail)
    with pytest.raises(OSError):
        registry.update(row.id, row.revision, {"url": "https://example.invalid/b"})
    assert registry.get(row.id) == row
    assert registry.secrets.load(row.id) == secret
