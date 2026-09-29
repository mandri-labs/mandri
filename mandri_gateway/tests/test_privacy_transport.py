import asyncio
import json

import httpx
import pytest
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.privacy_call import guarded_call
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.privacy_transport import TransportScope, provider_transport
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState

_BASE = "https://provider.invalid/v1"
_EMAIL = "transport-canary-69314@example.invalid"


def scope(identifier: str = "scope-1") -> TransportScope:
    base = Url(_BASE)
    key = SecretRef("synthetic-test-key")
    kind = ProviderKind.CUSTOM
    provider = Provider(
        name="test", kind=kind, api_base=base, api_key=key, state=ProviderState.VERIFIED
    )
    route = ResolvedRoute(
        RouteId("route-test"),
        provider,
        Model(kind, ModelRef("openai/test-model"), base, key),
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id=identifier,
    )
    engine = SurrogateEngine(SurrogateScope(identifier))
    engine.protect_text(_EMAIL)
    return TransportScope(EgressGuard(route, engine))


def body(content: str) -> dict:
    return {"model": "test-model", "messages": [{"role": "user", "content": content}]}


async def test_alternate_sdk_client_accepts_known_provider_route():
    observed = []
    protected = scope()

    def receive(request):
        observed.append(request)
        return httpx.Response(200, json={})

    with provider_transport(protected):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            await client.post(
                _BASE + "/chat/completions", json=body(protected.guard.engine.protect_text(_EMAIL))
            )
    assert len(observed) == 1
    assert protected.guard.sends == 1


async def test_sdk_injected_known_header_is_blocked_before_transport():
    observed = []
    protected = scope()

    async def inject(request):
        request.headers["x-debug-owner"] = _EMAIL

    def receive(request):
        observed.append(request)
        return httpx.Response(200, json={})

    with provider_transport(protected):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(receive), event_hooks={"request": [inject]}
        ) as client:
            with pytest.raises(ProtectionError, match="Unmasked provider header"):
                await client.post(_BASE + "/chat/completions", json=body("Synthetic content"))
    assert observed == []
    assert protected.guard.sends == 0


async def test_concurrent_protected_and_standard_requests_keep_their_contexts():
    left, right = scope("scope-left"), scope("scope-right")
    observed = []

    def receive(request):
        observed.append(request.content)
        return httpx.Response(200, json={})

    async def send(protected):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            if protected is None:
                await asyncio.sleep(0)
                await client.post(_BASE + "/chat/completions", json=body(_EMAIL))
            else:
                with provider_transport(protected):
                    await asyncio.sleep(0)
                    alias = protected.guard.engine.protect_text(_EMAIL)
                    await client.post(_BASE + "/chat/completions", json=body(alias))

    await asyncio.gather(send(left), send(None), send(right))
    assert len(observed) == 3
    assert sum(_EMAIL.encode() in payload for payload in observed) == 1
    assert left.guard.sends == right.guard.sends == 1
    aliases = {
        json.loads(payload)["messages"][0]["content"]
        for payload in observed
        if _EMAIL.encode() not in payload
    }
    assert len(aliases) == 2


async def test_background_task_cannot_send_after_owning_call_ends():
    protected = scope()
    release = asyncio.Event()
    observed = []

    def receive(request):
        observed.append(request)
        return httpx.Response(200, json={})

    async def detached():
        await release.wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            await client.post(_BASE + "/chat/completions", json=body("Synthetic content"))

    with provider_transport(protected):
        task = asyncio.create_task(detached())
    protected.active = False
    release.set()
    with pytest.raises(ProtectionError, match="ended"):
        await task
    assert observed == []


def test_synchronous_fallback_cannot_bypass_protected_transport():
    observed = []

    def receive(request):
        observed.append(request)
        return httpx.Response(200, json={})

    with (
        provider_transport(scope()),
        httpx.Client(transport=httpx.MockTransport(receive)) as client,
        pytest.raises(ProtectionError, match="Synchronous"),
    ):
        client.post(_BASE + "/chat/completions", json=body(_EMAIL))
    assert observed == []


async def test_sdk_cannot_swallow_transport_denial_or_replace_its_typed_error():
    observed = []

    def receive(request):
        observed.append(request)
        return httpx.Response(200, json={})

    async def swallowing_sdk(**kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            try:
                await client.post(
                    "https://unapproved.invalid/v1/chat/completions", json=body(_EMAIL)
                )
            except ProtectionError:
                return {"choices": []}

    with pytest.raises(ProtectionError):
        await guarded_call(scope().guard, swallowing_sdk, {})
    assert observed == []


async def test_provider_response_has_no_surrogate_byte_limit():
    protected = scope()
    consumed = []
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(10):
                consumed.append(index)
                yield b"x" * 128

        async def aclose(self):
            closed.append(True)

    def receive(request):
        return httpx.Response(200, stream=Body())

    with provider_transport(protected):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            response = await client.post(
                _BASE + "/chat/completions", json=body("Synthetic content")
            )
            assert response.content == b"x" * 1280
    assert consumed == list(range(10))
    assert closed == [True]
    assert protected.failure is None


async def test_protected_transport_requests_identity_and_rejects_compression_before_reading():
    protected = scope()
    read = []
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            read.append(True)
            yield b"not read"

        async def aclose(self):
            closed.append(True)

    def receive(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Body())

    with provider_transport(protected):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            with pytest.raises(ProtectionError, match="uncompressed response requirement"):
                await client.post(_BASE + "/chat/completions", json=body("Synthetic content"))
    assert not read
    assert closed == [True]


@pytest.mark.parametrize("cancel", [False, True])
async def test_httpx_protected_response_closes_original_stream_once(cancel):
    protected = scope()
    consumed = asyncio.Event()
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            consumed.set()
            if cancel:
                await asyncio.Event().wait()
            yield b"second"

        async def aclose(self):
            closed.append(True)

    def receive(request):
        return httpx.Response(200, stream=Body())

    with provider_transport(protected):
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            pending = asyncio.create_task(
                client.post(_BASE + "/chat/completions", json=body("Synthetic content"))
            )
            await consumed.wait()
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                response = await pending
                assert response.content == b"firstsecond"
                await response.aclose()
    assert closed == [True]
