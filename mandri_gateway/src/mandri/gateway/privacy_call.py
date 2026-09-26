import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from mandri.core.ids import ProviderKind
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_egress import EgressGuard, provider_base
from mandri.gateway.privacy_transport import TransportScope, provider_transport
from openai import AsyncOpenAI

_SDK_PROVIDERS = frozenset(
    {
        ProviderKind.OPENAI,
        ProviderKind.CUSTOM,
        ProviderKind.OPENCODE,
        ProviderKind.OPENCODE_GO,
        ProviderKind.LM_STUDIO,
    }
)


class GuardedStream(AsyncIterator[Any]):
    def __init__(
        self, stream: AsyncIterator[Any], handler: AsyncHTTPHandler, scope: TransportScope
    ):
        self.stream = stream
        self.handler = handler
        self.scope = scope
        self.guard = scope.guard
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    async def __anext__(self) -> Any:
        try:
            with provider_transport(self.scope):
                result = await anext(self.stream)
            if self.scope.failure is not None:
                raise self.scope.failure
        except BaseException as error:
            await self.aclose()
            if isinstance(error, Exception) and self.scope.failure is not None:
                raise self.scope.failure from None
            raise
        if not self.guard.sends:
            await self.aclose()
            raise ProtectionError(
                "privacy_egress_unobserved", "Provider transport was not validated"
            )
        return result

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.scope.active = False
        try:
            close = getattr(self.stream, "aclose", None)
            if close is not None:
                await close()
        finally:
            await self.handler.client.aclose()


async def guarded_call(
    guard: EgressGuard,
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    *,
    sdk: bool | None = None,
) -> Any:
    if os.environ.get("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "").lower() == "true":
        raise ProtectionError(
            "privacy_transport_unsupported", "Experimental provider transport is unsupported"
        )
    handler = AsyncHTTPHandler(timeout=kwargs.get("timeout"))
    handler.client.follow_redirects = False
    use_sdk = guard.route.model.provider in _SDK_PROVIDERS if sdk is None else sdk
    client: Any = handler
    if use_sdk:
        client = AsyncOpenAI(
            api_key=str(guard.route.model.api_key),
            base_url=provider_base(guard.route),
            http_client=handler.client,
            max_retries=0,
        )
    scope = TransportScope(guard)
    try:
        with provider_transport(scope):
            result = await call(**{**kwargs, "client": client})
        if scope.failure is not None:
            raise scope.failure
        if isinstance(result, AsyncIterator):
            return GuardedStream(result, handler, scope)
        if not guard.sends:
            raise ProtectionError(
                "privacy_egress_unobserved", "Provider transport was not validated"
            )
    except BaseException as error:
        scope.active = False
        await handler.client.aclose()
        if isinstance(error, Exception) and scope.failure is not None:
            raise scope.failure from None
        raise
    scope.active = False
    await handler.client.aclose()
    return result
