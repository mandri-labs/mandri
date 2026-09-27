import asyncio
import logging

from mandri.core.ids import HarnessSessionId
from mandri.core.ports.control import HarnessControl
from mandri.core.types.execution import ProtectionError
from mandri.runtime.control.errors import ControlError

logger = logging.getLogger(__name__)


async def require_native_identity(
    control: HarnessControl | None, *, timeout_seconds: float | None = None
) -> HarnessSessionId:
    if control is None:
        raise ProtectionError(
            "native_initialization_failed", "The native control channel is unavailable"
        )
    try:
        identity = await asyncio.wait_for(control.capture_identity(), timeout_seconds)
    except TimeoutError:
        logger.warning("native initialization timed out: %s", type(control).__name__)
        raise ProtectionError(
            "native_initialization_timeout", "The native control channel did not become ready"
        ) from None
    except ControlError:
        logger.warning("native initialization failed: %s", type(control).__name__, exc_info=True)
        raise ProtectionError(
            "native_initialization_failed", "The native control channel failed to initialize"
        ) from None
    if identity is None:
        raise ProtectionError(
            "native_initialization_failed", "The native conversation identity is unavailable"
        )
    return identity
