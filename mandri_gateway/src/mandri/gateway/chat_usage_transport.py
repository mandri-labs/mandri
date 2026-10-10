import json
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx
from mandri.gateway.privacy_response_sse import Frame

CURRENT_CHAT_USAGE: ContextVar[bool] = ContextVar("mandri_chat_usage", default=False)


@contextmanager
def chat_usage_transport() -> Iterator[None]:
    token = CURRENT_CHAT_USAGE.set(True)
    try:
        yield
    finally:
        CURRENT_CHAT_USAGE.reset(token)


def normalize_frame(data: bytes) -> bytes:
    try:
        lines = data.decode("utf-8").splitlines()
        parts = [
            line.partition(":")[2].removeprefix(" ") for line in lines if line.startswith("data:")
        ]
        payload = json.loads("\n".join(parts))
    except (UnicodeError, ValueError, RecursionError):
        return data
    if (
        not isinstance(payload, dict)
        or not payload.get("choices")
        or not isinstance(payload.get("usage"), dict)
    ):
        return data
    return b"".join(frame.encode() for frame in usage_frames(Frame(lines, payload, None)))


def usage_frames(frame: Frame) -> list[Frame]:
    payload = frame.payload
    if (
        not isinstance(payload, dict)
        or not payload.get("choices")
        or not isinstance(payload.get("usage"), dict)
    ):
        return [frame]
    content = {key: value for key, value in payload.items() if key != "usage"}
    usage = {**payload, "choices": []}
    return [Frame(frame.lines, content, None), Frame(frame.lines, usage, None)]


def normalize_chat_usage(response: httpx.Response) -> None:
    if response.is_error or response.is_stream_consumed:
        return
    if not response.request.url.path.rstrip("/").endswith("/chat/completions"):
        return
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return
    original = response.aiter_bytes

    async def normalized(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        pending = b""
        async for chunk in original(chunk_size):
            pending += chunk
            while boundary := re.search(rb"\r\n\r\n|\n\n|\r\r", pending):
                yield normalize_frame(pending[: boundary.end()])
                pending = pending[boundary.end() :]
        if pending:
            yield normalize_frame(pending)

    streaming_response: Any = response
    streaming_response.aiter_bytes = normalized
