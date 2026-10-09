import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from mandri.core.ids import ProviderKind
from mandri.gateway.chatgpt_adapter import ChatGptAdapter
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
    client = kwargs.get("client")
    if client is None:
        return await _provider_call(route, call, kwargs, guard)
    try:
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
