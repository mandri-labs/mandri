import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.privacy_transport import TransportScope, provider_transport


async def guarded_call(
    guard: EgressGuard,
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
) -> Any:
    if os.environ.get("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "").lower() == "true":
        raise ProtectionError(
            "privacy_transport_unsupported", "Experimental provider transport is unsupported"
        )
    scope = TransportScope(guard)
    try:
        with provider_transport(scope):
            result = await call(**kwargs)
        if scope.failure is not None:
            raise scope.failure
        if isinstance(result, AsyncIterator):
            close = getattr(result, "aclose", None)
            if close is not None:
                await close()
            raise ProtectionError(
                "privacy_response_invalid", "Protected provider responses must be complete"
            )
        if not guard.sends:
            raise ProtectionError(
                "privacy_egress_unobserved", "Provider transport was not validated"
            )
        return result
    except Exception:
        if scope.failure is not None:
            raise scope.failure from None
        raise
    finally:
        scope.active = False
