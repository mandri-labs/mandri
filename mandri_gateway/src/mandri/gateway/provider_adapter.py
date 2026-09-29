from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

import httpx


class ResponseAdapter(Protocol):
    def request(self, request: httpx.Request) -> httpx.Request: ...

    async def response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response: ...


CURRENT_ADAPTER: ContextVar[ResponseAdapter | None] = ContextVar(
    "mandri_provider_adapter", default=None
)


@contextmanager
def use_adapter(adapter: ResponseAdapter) -> Iterator[None]:
    token = CURRENT_ADAPTER.set(adapter)
    try:
        yield
    finally:
        CURRENT_ADAPTER.reset(token)
