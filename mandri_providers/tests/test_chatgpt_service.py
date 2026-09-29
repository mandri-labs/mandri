import asyncio
import json
from urllib.parse import urlencode

import httpx
import pytest
from mandri.core.ids import ProviderKind
from mandri.providers import chatgpt_login as login_module
from mandri.providers.chatgpt.oauth import LoopbackCallbackServer, TokenSet
from mandri.providers.chatgpt.resolver import ChatGptCredentialResolver
from mandri.providers.chatgpt.store import ChatGptTokenStore
from mandri.providers.chatgpt_login import ChatGptLoginService, ChatGptLoginStatus
from mandri.providers.errors import ProviderExistsError
from mandri.providers.service import ProvidersRegistry

from mandri_providers.tests.test_providers_service import FakeConfig, FakeRoutes


@pytest.mark.parametrize(
    "left,right", [("work a", "work_a"), ("a" * 65, "a" * 64 + "b"), ("é", "è")]
)
def test_accounts_never_share_credentials(tmp_path, left, right):
    store = ChatGptTokenStore(tmp_path)
    store.save(left, TokenSet("left-access", "left-refresh", 1))
    store.save(right, TokenSet("right-access", "right-refresh", 2))
    assert store.load(left).access_token == "left-access"
    assert store.load(right).access_token == "right-access"
    store.clear(right)
    assert store.load(left).access_token == "left-access"


def test_migration_preserves_unambiguous_account_and_skips_collisions(tmp_path):
    (tmp_path / "chatgpt").mkdir()
    payload = json.dumps(
        {"access_token": "synthetic", "refresh_token": "refresh", "expires_at_ms": 1}
    )
    (tmp_path / "chatgpt" / "work.json").write_text(payload)
    (tmp_path / "chatgpt" / "work_a.json").write_text(payload)
    store = ChatGptTokenStore(tmp_path)
    store.migrate(["work", "work a", "work_a"])
    assert store.load("work").access_token == "synthetic"
    assert not (tmp_path / "chatgpt" / "work.json").exists()
    assert store.load("work a") is None
    assert store.load("work_a") is None


@pytest.fixture
async def login(tmp_path, monkeypatch):
    monkeypatch.setattr(
        login_module, "LoopbackCallbackServer", lambda: LoopbackCallbackServer(ports=(0,))
    )
    store = ChatGptTokenStore(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                },
            )
        )
    ) as client:
        config = FakeConfig()
        registry = ProvidersRegistry(
            config, FakeRoutes(), token_resolver=ChatGptCredentialResolver(store, client)
        )
        service = ChatGptLoginService(registry, store, client)
        try:
            yield service, registry, store, config
        finally:
            await service.aclose()


async def test_conflicting_provider_never_loses_key(login):
    service, registry, store, _config = login
    await registry.add("work", ProviderKind.OPENAI, None, "original", verify=False)
    with pytest.raises(ProviderExistsError):
        await service.start("work")
    with pytest.raises(ProviderExistsError):
        await service._register("work", None)
    assert registry.get("work").api_key == "original"
    assert store.load("work") is None


async def test_pending_login_is_reused_for_same_account(login):
    service, _, _, _ = login
    left, right = await asyncio.gather(service.start("work"), service.start("work"))
    assert left.id == right.id


async def test_manual_callback_completes_and_reconnects_without_losing_base(login):
    service, registry, store, _ = login
    await registry.add("work", ProviderKind.CHATGPT, "https://synthetic.invalid", "", verify=False)
    store.save("work", TokenSet("old", "refresh", 1))
    session = await service.start("work")
    url = session.redirect + "?" + urlencode({"code": "test-code", "state": session.state})
    await service.submit_redirect(session.id, url)
    assert session.status is ChatGptLoginStatus.COMPLETED
    assert registry.get("work").api_key == "new-access"
    assert registry.get("work").api_base == "https://synthetic.invalid"
    assert session.provider.api_key == "new-access"


async def test_wrong_pasted_state_keeps_listener_available(login):
    service, _, _, _ = login
    session = await service.start("work")
    await service.submit_redirect(session.id, session.redirect + "?code=x&state=wrong")
    assert session.status is ChatGptLoginStatus.PENDING
    assert session.error_code == "chatgpt_callback_rejected"
    await service.submit_redirect(
        session.id, session.redirect + "?" + urlencode({"code": "x", "state": session.state})
    )
    assert session.status is ChatGptLoginStatus.COMPLETED


async def test_renamed_conflict_during_authorization_preserves_original(login):
    service, registry, store, _ = login
    session = await service.start("work")
    await registry.add("work", ProviderKind.OPENAI, None, "original", verify=False)
    await service.submit_redirect(
        session.id, session.redirect + "?" + urlencode({"code": "x", "state": session.state})
    )
    assert session.status is ChatGptLoginStatus.FAILED
    assert registry.get("work").api_key == "original"
    assert store.load("work") is None


async def test_refresh_single_flight_is_scoped_to_each_account(tmp_path):
    calls = []

    async def refresh(request):
        calls.append(request.content)
        await asyncio.sleep(0.01)
        return httpx.Response(
            200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600}
        )

    store = ChatGptTokenStore(tmp_path)
    for name in ("work", "personal"):
        store.save(name, TokenSet("expired", "refresh-" + name, 1))
    async with httpx.AsyncClient(transport=httpx.MockTransport(refresh)) as client:
        resolver = ChatGptCredentialResolver(store, client)
        await asyncio.gather(
            *(resolver.ensure_fresh(name) for name in ["work", "work", "personal", "personal"])
        )
    assert len(calls) == 2
    assert store.load("work").access_token == "fresh"
    assert store.load("personal").access_token == "fresh"


async def test_cancel_during_manual_exchange_does_not_save_credentials(login, monkeypatch):
    service, _, store, _ = login
    started = asyncio.Event()
    release = asyncio.Event()

    async def exchange(*args):
        started.set()
        await release.wait()
        return TokenSet("late-access", "late-refresh", 1)

    monkeypatch.setattr(login_module, "exchange_code", exchange)
    session = await service.start("work")
    callback = asyncio.create_task(
        service.submit_redirect(
            session.id, session.redirect + "?" + urlencode({"code": "x", "state": session.state})
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    await service.cancel(session.id)
    release.set()
    await asyncio.wait_for(callback, 1)
    assert session.status is ChatGptLoginStatus.CANCELLED
    assert store.load("work") is None
