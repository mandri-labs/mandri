"""Routing and upstream gateway errors."""

from collections.abc import Mapping

from mandri.core.ids import ProviderKind
from mandri.gateway.errors.base import GatewayError


class RouteNotFoundError(GatewayError):
    """No gateway route exists under the requested id."""


class ProviderUnavailableError(GatewayError):
    """Configured provider cannot be reached or is disabled."""


class UpstreamError(GatewayError):
    def __init__(
        self,
        status: int,
        message: str,
        llm_provider: ProviderKind,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"upstream {llm_provider.value} returned {status}: {message}")
        self.status = status
        self.message = message
        self.llm_provider = llm_provider
        self.headers: dict[str, str] = dict(headers) if headers else {}
