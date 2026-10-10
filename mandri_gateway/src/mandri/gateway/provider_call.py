import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from mandri.core.ids import ProviderKind
from mandri.gateway.chat_usage_transport import chat_usage_transport
from mandri.gateway.chatgpt_adapter import ChatGptAdapter
from mandri.gateway.output_budget import retry_kwargs
from mandri.gateway.privacy_call import guarded_call
from mandri.gateway.privacy_egress import EgressGuard, provider_base
from mandri.gateway.privacy_transport import install_transport_observers
from mandri.gateway.provider_adapter import use_adapter
from mandri.gateway.route_registry import ResolvedRoute


async def provider_call(
    route: ResolvedRoute,
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    guard: EgressGuard | None,
) -> Any:
    install_transport_observers()
    client = kwargs.get("client")
    if client is None:
        with chat_usage_transport():
            return await _provider_call(route, call, kwargs, guard)
    try:
        with chat_usage_transport():
            result = await _provider_call(route, call, kwargs, guard)
    except BaseException:
        await client.close()
        raise
    if isinstance(result, AsyncIterator):
        return _client_stream(result, client)
    await client.close()
    return result


async def _client_stream(stream: AsyncIterator[Any], client: Any) -> AsyncIterator[Any]:
    try:
        async for event in stream:
            yield event
    finally:
        close = getattr(stream, "aclose", None)
        try:
            if close is not None:
                await close()
        finally:
            await client.close()


async def _provider_call(
    route: ResolvedRoute,
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    guard: EgressGuard | None,
) -> Any:
    try:
        return await _attempt_provider_call(route, call, kwargs, guard)
    except Exception as error:
        fallback = retry_kwargs(kwargs, error)
        if fallback is None:
            raise
        return await _attempt_provider_call(route, call, fallback, guard)


async def _attempt_provider_call(
    route: ResolvedRoute,
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    guard: EgressGuard | None,
) -> Any:
    if route.model.provider is ProviderKind.CHATGPT:
        install_transport_observers()
        kwargs = {**kwargs, "stream": False}
        kwargs.pop("stream_options", None)
        async with asyncio.timeout(600):
            with use_adapter(ChatGptAdapter(provider_base(route))):
                if guard is not None:
                    return await guarded_call(guard, call, kwargs)
                return await call(**kwargs)
    if guard is not None:
        return await guarded_call(guard, call, kwargs)
    return await call(**kwargs)
