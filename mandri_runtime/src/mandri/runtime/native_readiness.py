import asyncio

from mandri.core.ids import HarnessSessionId
from mandri.core.ports.control import HarnessControl
from mandri.core.types.execution import ProtectionError
from mandri.runtime.control.errors import ControlError


async def require_native_identity(
    control: HarnessControl | None, *, timeout_seconds: float = 30.0
) -> HarnessSessionId:
    if control is None:
        raise ProtectionError(
            "native_initialization_failed", "The native control channel is unavailable"
        )
    try:
        identity = await asyncio.wait_for(control.capture_identity(), timeout_seconds)
    except TimeoutError:
        raise ProtectionError(
            "native_initialization_timeout", "The native control channel did not become ready"
        ) from None
    except ControlError:
        raise ProtectionError(
            "native_initialization_failed", "The native control channel failed to initialize"
        ) from None
    if identity is None:
        raise ProtectionError(
            "native_initialization_failed", "The native conversation identity is unavailable"
        )
    return identity
