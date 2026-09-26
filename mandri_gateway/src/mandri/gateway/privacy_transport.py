import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import httpx
from mandri.core.ids import ProviderKind
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.usage_transport import observe_response


@dataclass
class TransportScope:
    guard: EgressGuard
    active: bool = True
    failure: ProtectionError | None = None


_CURRENT: ContextVar[TransportScope | None] = ContextVar("mandri_privacy_transport", default=None)
_ASYNC_SEND = httpx.AsyncClient._send_single_request
_SYNC_SEND = httpx.Client._send_single_request
_INSTALLED = False


async def _async_send(client: httpx.AsyncClient, request: httpx.Request) -> httpx.Response:
    scope = _CURRENT.get()
    if scope is not None:
        _require_active(scope)
        try:
            request.headers["accept-encoding"] = "identity"
            await scope.guard.check(request)
        except ProtectionError as error:
            scope.failure = error
            raise
    response = await _ASYNC_SEND(client, request)
    if scope is not None:
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            await response.aclose()
            encoding_error = ProtectionError(
                "privacy_transport_unsupported",
                "Provider ignored the uncompressed response requirement",
            )
            scope.failure = encoding_error
            raise encoding_error
    await observe_response(response)
    return response


def _sync_send(client: httpx.Client, request: httpx.Request) -> httpx.Response:
    scope = _CURRENT.get()
    if scope is not None:
        if _optional_model_discovery(scope, request):
            raise ProtectionError(
                "privacy_discovery_blocked", "Optional provider discovery is disabled"
            )
        scope.failure = ProtectionError(
            "privacy_transport_unsupported", "Synchronous provider transport is unsupported"
        )
        raise scope.failure
    return _SYNC_SEND(client, request)


def _optional_model_discovery(scope: TransportScope, request: httpx.Request) -> bool:
    if (
        scope.guard.route.model.provider is not ProviderKind.OLLAMA
        or request.method != "POST"
        or request.url.path != "/api/show"
        or request.url.query
    ):
        return False
    try:
        payload = json.loads(request.content)
    except (ValueError, UnicodeError, httpx.RequestNotRead):
        return False
    native_model = str(scope.guard.route.model.model_ref).partition("/")[2]
    return isinstance(payload, dict) and payload == {"name": native_model}


def _require_active(scope: TransportScope) -> None:
    if scope.failure is not None:
        raise scope.failure
    if not scope.active:
        raise ProtectionError("privacy_egress_blocked", "The provider request has ended")


def install_transport_observers() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if version("httpx") != "0.28.1":
        raise ProtectionError(
            "privacy_transport_unsupported", "The provider transport version is not qualified"
        )
    async_client: Any = httpx.AsyncClient
    sync_client: Any = httpx.Client
    async_client._send_single_request = _async_send
    sync_client._send_single_request = _sync_send
    _INSTALLED = True


@contextmanager
def provider_transport(scope: TransportScope) -> Iterator[None]:
    install_transport_observers()
    _require_active(scope)
    token = _CURRENT.set(scope)
    try:
        yield
    finally:
        _CURRENT.reset(token)
