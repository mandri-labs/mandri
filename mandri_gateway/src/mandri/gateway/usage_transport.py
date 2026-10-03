import json
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import httpx
from mandri.gateway.usage import CURRENT_USAGE, UsageCollector
from mandri.gateway.usage_payload import observe_payload, sum_usage

_MAX_BUFFER = 1024 * 1024


class UsageDecoder:
    def __init__(
        self,
        collector: UsageCollector,
        content_type: str,
        prior_usage: dict[str, Any] | None = None,
    ):
        self.collector = collector
        self.prior_usage = prior_usage
        self.sse = "text/event-stream" in content_type
        self.lines = self.sse or "ndjson" in content_type
        self.buffer = bytearray()
        self.event: list[bytes] = []
        self.event_size = 0
        self.skipping = False

    async def _json(self, data: bytes) -> None:
        if not data.strip() or data.strip() == b"[DONE]":
            return
        try:
            payload = json.loads(data, parse_float=Decimal)
        except (ValueError, UnicodeError, RecursionError):
            self.collector.observation_incomplete = True
            return
        if self.prior_usage is not None and isinstance(payload, dict):
            envelope = payload.get("response", payload)
            if isinstance(envelope, dict) and isinstance(envelope.get("usage"), dict):
                envelope["usage"] = sum_usage(self.prior_usage, envelope["usage"])
        await observe_payload(self.collector, payload)

    async def _line(self, line: bytes) -> None:
        if not self.sse:
            await self._json(line)
        elif not line:
            if self.event:
                await self._json(b"\n".join(self.event))
            self.event.clear()
            self.event_size = 0
        elif line.startswith(b"data:"):
            data = line[5:].lstrip(b" ")
            self.event_size += len(data)
            if self.event_size > _MAX_BUFFER:
                self.event.clear()
                self.collector.observation_incomplete = True
            else:
                self.event.append(data)

    async def feed(self, data: bytes, *, final: bool = False) -> None:
        if not self.lines:
            if len(self.buffer) + len(data) > _MAX_BUFFER:
                self.skipping = True
                self.buffer.clear()
                self.collector.observation_incomplete = True
            if not self.skipping:
                self.buffer.extend(data)
                if final:
                    await self._json(bytes(self.buffer))
                    self.buffer.clear()
            return
        for part in data.splitlines(keepends=True):
            self.buffer.extend(part)
            if len(self.buffer) > _MAX_BUFFER:
                self.skipping = True
                self.buffer.clear()
                self.collector.observation_incomplete = True
            if part.endswith(b"\n"):
                if not self.skipping:
                    await self._line(bytes(self.buffer).rstrip(b"\r\n"))
                self.buffer.clear()
                self.skipping = False
        if final:
            if self.buffer and not self.skipping:
                await self._line(bytes(self.buffer).rstrip(b"\r\n"))
            self.buffer.clear()
            if self.event:
                await self._json(b"\n".join(self.event))
                self.event.clear()


async def observe_response(response: httpx.Response, *, continuation: bool = False) -> None:
    collector = CURRENT_USAGE.get()
    if collector is None:
        return
    await _observe_request(collector, response)
    collector.upstream_failed = response.is_error
    await collector.publish(transport_attempts=collector.record.transport_attempts + 1)
    if collector.record.transport_attempts > 1 and not continuation:
        collector.observation_incomplete = True
    prior_usage = sum_usage(collector.raw_usage) if continuation else None
    decoder = UsageDecoder(collector, response.headers.get("content-type", ""), prior_usage)
    if response.is_stream_consumed:
        await decoder.feed(response.content, final=True)
        return
    original = response.aiter_bytes

    async def observed(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        async for chunk in original(chunk_size):
            await decoder.feed(chunk)
            yield chunk
        await decoder.feed(b"", final=True)

    observed_response: Any = response
    observed_response.aiter_bytes = observed


async def _observe_request(collector: UsageCollector, response: httpx.Response) -> None:
    try:
        request = response.request
        data = request.content
    except (RuntimeError, httpx.RequestNotRead):
        return
    if len(data) > _MAX_BUFFER:
        return
    try:
        body = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        return
    if not isinstance(body, dict):
        return
    changes: dict[str, Any] = {}
    model = body.get("model")
    if isinstance(model, str) and model.strip() and len(model) <= 512:
        changes["requested_model"] = model
    path = request.url.path.rstrip("/")
    if path.endswith("/responses"):
        changes["upstream_protocol"] = "responses"
    elif path.endswith("/messages"):
        changes["upstream_protocol"] = "anthropic"
    elif path.endswith("/chat/completions"):
        changes["upstream_protocol"] = "openai"
    if changes:
        await collector.publish(**changes)
