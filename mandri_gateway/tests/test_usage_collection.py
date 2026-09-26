import asyncio
import gzip
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.api.routers.gateway import _collect_usage, _UsageStreamingResponse
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.usage_pricing import bundled_prices
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.gateway.complete_response import complete_sse
from mandri.gateway.litellm_adapter import AnthropicHandler, GeminiHandler, OpenAIHandler
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_transport import install_transport_observers
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.gateway.usage import UsageCollector, collect_call
from mandri.gateway.usage_observation import to_observation
from mandri.gateway.usage_payload import observe_payload
from mandri.gateway.usage_transport import UsageDecoder
from mandri.providers.service import Provider, ProviderState
from starlette.requests import ClientDisconnect


@pytest.fixture
def route():
    return ResolvedRoute(
        RouteId("route"),
        Provider(
            "fixture", ProviderKind.OPENROUTER, None, SecretRef("secret"), ProviderState.VERIFIED
        ),
        Model(ProviderKind.OPENROUTER, ModelRef("openrouter/old"), None, SecretRef("secret")),
        conversation_id="arbitrary-first-child",
    )


def collection(route):
    records = []

    async def sink(record):
        records.append(record)

    return UsageCollector(route, "chat", sink), records


async def test_stalled_usage_sink_cannot_block_model_delivery(route, monkeypatch):
    monkeypatch.setattr("mandri.gateway.usage.PERSISTENCE_TIMEOUT_SECONDS", 0.01)
    calls = 0

    async def stalled_sink(record):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    async def complete():
        return {"choices": []}

    collector = UsageCollector(route, "chat", stalled_sink)
    result = await asyncio.wait_for(collect_call(collector, complete), timeout=0.2)
    assert result == {"choices": []}
    assert calls == 1
    assert collector.record.incomplete
    assert collector.persistence_disabled


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            if isinstance(part, BaseException):
                raise part
            yield part

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("compressed", [False, True])
async def test_http_json_before_lossy_conversion_and_synthetic_sse(route, compressed):
    collector, records = collection(route)
    install_transport_observers()
    raw = (
        b'{"id":"upstream-exact","model":"resolved","usage":{"prompt_tokens":9,'
        b'"completion_tokens":0,"cost":0.000000000000000123,"private":"secret"},"choices":[]}'
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                **({"content-encoding": "gzip"} if compressed else {}),
            },
            stream=ByteStream(gzip.compress(raw) if compressed else raw),
        )
    )

    async with httpx.AsyncClient(transport=transport) as client:

        async def call():
            response = await client.post("https://synthetic.invalid/chat")
            body = response.json()
            body["usage"].pop("cost")
            return body

        result = await collect_call(collector, call)
    complete_sse(result, GatewayProtocol.CHAT)
    record = records[-1]
    assert record.status == "completed"
    assert record.input_tokens == 9 and record.output_tokens == 0
    assert record.total_tokens is None and record.cache_read_tokens is None
    assert record.provider_cost == Decimal("0.000000000000000123")
    assert record.provider_cost_currency == "USD"
    assert record.upstream_request_id == "upstream-exact"
    assert record.observed_model == "resolved"
    assert "secret" not in record.raw_usage_json
    assert len({r.request_id for r in records}) == 1
    assert sum(r.status == "completed" for r in records) == 1
    assert record.session_id is None and record.project_path is None
    observation = to_observation(record)
    assert observation.reported_cost_usd == record.provider_cost
    assert observation.sequence == record.revision
    assert observation.source_key == record.request_id


async def test_usage_tail_after_finish_and_fragmented_frames(route):
    collector, records = collection(route)
    install_transport_observers()
    wire = (
        b'data: {"id":"request","choices":[{"finish_reason":"stop"}]}\r\n\r\n'
        b'data: {"id":"request","choices":[],"usage":{"prompt_tokens":12,'
        b'"completion_tokens":3,"total_tokens":15,"cost":0.12}}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    stream = ByteStream(*(wire[i : i + 1] for i in range(len(wire))))
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:

        async def call():
            response = await client.send(
                client.build_request("POST", "https://synthetic.invalid"), stream=True
            )
            return response.aiter_bytes()

        result = await collect_call(collector, call)
        assert b"".join([part async for part in result]) == wire
    assert records[-1].total_tokens == 15
    assert records[-1].provider_cost == Decimal("0.12")
    assert records[-1].status == "completed"
    assert not records[-1].incomplete


@pytest.mark.parametrize("cancel", [False, True])
async def test_partial_usage_survives_failure_or_cancellation(route, cancel):
    collector, records = collection(route)

    async def source():
        await observe_payload(
            collector,
            {
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                },
            },
        )
        yield b"first"
        raise asyncio.CancelledError() if cancel else RuntimeError("failure")

    async def call():
        return source()

    result = await collect_call(collector, call)
    assert await anext(result) == b"first"
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await anext(result)
    record = records[-1]
    assert record.status == ("cancelled" if cancel else "failed")
    assert record.input_tokens == 20 and record.output_tokens == 0
    assert record.incomplete
    await result.aclose()
    assert records[-1] is record


async def test_close_after_client_disconnect_marks_cancelled(route):
    collector, records = collection(route)

    async def source():
        yield b"part"
        yield b"unread"

    async def call():
        return source()

    result = await collect_call(collector, call)
    response = _UsageStreamingResponse(result, result)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("disconnected")

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert result.closed
    assert records[-1].status == "cancelled"


async def test_anthropic_snapshots_replace_and_merge_without_double_count(route):
    collector, records = collection(route)
    await observe_payload(
        collector,
        {
            "type": "message_start",
            "message": {
                "id": "msg",
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 1},
            },
        },
    )
    for output in [7, 7, 9]:
        await observe_payload(
            collector, {"type": "message_delta", "usage": {"output_tokens": output}}
        )
    record = records[-1]
    assert record.input_tokens == 100 and record.output_tokens == 9
    assert record.cache_read_tokens == 20 and record.total_tokens is None
    assert to_observation(record).input_includes_cache is False


async def test_dispatch_snapshot_is_frozen_and_request_ids_are_unique(route):
    collector, records = collection(route)
    new_route = replace(route, model=replace(route.model, model_ref=ModelRef("openrouter/new")))
    other, _ = collection(new_route)
    await collector.publish()
    assert records[0].selected_model == "openrouter/old"
    assert other.record.selected_model == "openrouter/new"
    assert other.record.request_id != records[0].request_id
    with pytest.raises(FrozenInstanceError):
        records[0].selected_model = "other"


@pytest.mark.parametrize(
    "payload",
    [{}, {"usage": None}, {"usage": 4}, {"usage": {"input_tokens": True, "output_tokens": -1}}],
)
async def test_unknown_counts_never_become_zero(route, payload):
    collector, _ = collection(route)
    await observe_payload(collector, payload)
    assert collector.record.input_tokens is None
    assert collector.record.output_tokens is None


async def test_responses_and_gemini_usage_semantics(route):
    collector, _ = collection(route)
    await observe_payload(
        collector,
        {
            "type": "response.completed",
            "response": {
                "id": "resp-exact",
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 12},
                    "output_tokens_details": {"reasoning_tokens": 4},
                },
            },
        },
    )
    assert collector.record.cache_read_tokens == 12 and collector.record.reasoning_tokens == 4
    assert collector.record.usage_protocol == "responses"
    gemini, _ = collection(route)
    await observe_payload(
        gemini,
        {
            "responseId": "gemini-exact",
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 6,
                "thoughtsTokenCount": 4,
            },
        },
    )
    assert gemini.record.output_tokens == 10 and gemini.record.reasoning_tokens == 4
    assert gemini.record.upstream_request_id == "gemini-exact"


async def test_sink_failure_does_not_fail_model_response(route):
    async def broken(record):
        raise RuntimeError("storage failed")

    async def call():
        return {"result": "ok"}

    assert await collect_call(UsageCollector(route, "chat", broken), call) == {"result": "ok"}


async def test_router_app_state_callback_fallback(route):
    records = []

    async def sink(record):
        records.append(record)

    async def call():
        return {"ok": True}

    app = FastAPI()
    app.state.gateway_usage_sink = sink
    request = Request({"type": "http", "app": app})
    await _collect_usage(SimpleNamespace(), request, route, GatewayProtocol.CHAT, call)
    assert [r.status for r in records] == ["pending", "completed"]
    assert records[-1].incomplete


async def test_decoder_recovers_after_oversized_content_event(route):
    collector, _ = collection(route)
    decoder = UsageDecoder(collector, "text/event-stream")
    await decoder.feed(b"data: " + b"x" * (1024 * 1024 + 1) + b"\n\n")
    await decoder.feed(b'data: {"usage":{"prompt_tokens":3}}\n\n', final=True)
    assert collector.record.input_tokens == 3
    assert collector.observation_incomplete
    assert json.loads(collector.record.raw_usage_json) == {"prompt_tokens": 3}


@pytest.mark.parametrize("protocol", ["chat", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_real_litellm_translation_captures_raw_cost(monkeypatch, route, protocol, stream):
    logging_tasks = []
    monkeypatch.setattr(
        GLOBAL_LOGGING_WORKER,
        "ensure_initialized_and_enqueue",
        lambda coroutine: logging_tasks.append(asyncio.create_task(coroutine)),
    )
    routed = replace(
        route,
        model=replace(
            route.model,
            provider=ProviderKind.CUSTOM,
            model_ref=ModelRef("custom_openai/fixture"),
            api_base=Url("https://synthetic.invalid/v1"),
        ),
    )
    collector, records = collection(routed)
    install_transport_observers()
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "cost": 0.123}
    payload = {
        "id": "upstream-id",
        "object": "chat.completion",
        "created": 1,
        "model": "fixture",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }

    async def send(client, request):
        assert request.url.host == "synthetic.invalid"
        body = json.loads(request.content)
        if body.get("stream"):
            frames = [
                {
                    **payload,
                    "object": "chat.completion.chunk",
                    "usage": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "OK"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **payload,
                    "object": "chat.completion.chunk",
                    "usage": None,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {**payload, "object": "chat.completion.chunk", "choices": []},
            ]
            raw = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=ByteStream(raw.encode(), b"data: [DONE]\n\n"),
            )
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr("mandri.gateway.privacy_transport._ASYNC_SEND", send)

    async def call():
        if protocol == "gemini":
            return await GeminiHandler().generate_content(
                routed, {"contents": [{"role": "user", "parts": [{"text": "fixture"}]}]}, stream
            )
        body = {
            "messages": [{"role": "user", "content": "fixture"}],
            "stream": stream,
            "max_tokens": 10,
        }
        if protocol == "anthropic":
            return await AnthropicHandler().messages(routed, body)
        return await OpenAIHandler().chat_completions(routed, body)

    result = await asyncio.wait_for(collect_call(collector, call), timeout=5)
    if stream:
        assert [part async for part in result]
    assert records[-1].input_tokens == 12
    assert records[-1].output_tokens == 3
    assert records[-1].provider_cost == Decimal("0.123")
    assert records[-1].upstream_request_id == "upstream-id"
    assert len({record.request_id for record in records}) == 1
    await asyncio.gather(*logging_tasks)


async def test_raw_gateway_snapshots_replace_one_priced_repository_fact(route, tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "usage.sqlite")
    await db.migrate()
    repository = UsageRepository(db)
    openai = replace(
        route,
        provider=replace(route.provider, kind=ProviderKind.OPENAI),
        model=replace(
            route.model, provider=ProviderKind.OPENAI, model_ref=ModelRef("openai/gpt-4.1-mini")
        ),
    )

    async def sink(record):
        await repository.record(to_observation(record))

    collector = UsageCollector(openai, "chat", sink)
    try:
        for price in bundled_prices():
            await repository.add_price(price)
        await collector.publish()
        assert to_observation(collector.record).model == "gpt-4.1-mini"
        collector.record = replace(collector.record, transport_attempts=1)
        payload = {
            "id": "actual-id",
            "model": "gpt-4.1-mini-2025-04-14",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        }
        await observe_payload(collector, payload)
        await observe_payload(collector, payload)
        await collector.finish("completed")
        result = await repository.overview()
        assert result["summary"]["fact_count"] == 1
        assert result["summary"]["request_count"] == 1
        assert result["summary"]["input_tokens"] == 1000
        assert result["summary"]["cache_write_tokens"] == 0
        assert result["breakdown"][0]["key"] == "gpt-4.1-mini"
        assert Decimal(result["summary"]["usd_equivalent"]) == Decimal("0.00054")
        assert to_observation(collector.record).observed_model == "gpt-4.1-mini-2025-04-14"
    finally:
        await db.close()


@pytest.mark.parametrize("kind", list(ProviderKind))
async def test_total_failure_removes_pending_usage_for_every_provider(route, kind):
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    repository = UsageRepository(db)

    async def sink(record):
        await repository.record(to_observation(record))

    selected = replace(route, provider=replace(route.provider, kind=kind))
    collector = UsageCollector(selected, "chat", sink)
    try:
        await collector.publish()
        assert (await repository.overview())["summary"]["fact_count"] == 1
        await observe_payload(collector, {"error": {"message": "Request rejected"}})
        await collector.finish("completed")
        result = await repository.overview()
        assert collector.record.status == "failed"
        assert result["summary"]["fact_count"] == 0
        assert result["summary"]["unpriced_fact_count"] == 0
        assert result["summary"]["incomplete_fact_count"] == 0
        assert result["breakdown"] == []
        assert result["timeseries"] == []
    finally:
        await db.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"delta": {"content": "partial"}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{"}}]}}]},
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": "content_block_delta", "delta": {"text": "partial"}},
        {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]},
        {"response": "partial"},
    ],
)
async def test_partial_stream_failure_keeps_usage_without_final_counters(route, payload):
    collector, records = collection(route)
    await observe_payload(collector, payload)
    await collector.finish("failed")
    item = to_observation(records[-1])
    assert item.authoritative
    assert item.output_tokens is None
    assert item.pricing_context["output_observed"] is True
    assert "partial" not in item.pricing_context["raw_usage"]
