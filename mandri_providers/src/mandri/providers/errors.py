"""Provider credential errors."""

from mandri.core.errors import MandriError
from mandri.core.ids import ProviderKind


class ProviderError(MandriError):
    """Base class for all provider errors."""


class ProviderNotFoundError(ProviderError):
    """No provider exists under the requested name."""


class ProviderExistsError(ProviderError):
    """A provider with this name is already registered."""


class ProviderInvalidError(ProviderError):
    """Provider configuration is invalid (bad kind, missing api_base, malformed model arg)."""


class ProviderInUseError(ProviderError):
    """Provider is still referenced by gateway routes."""

    def __init__(self, name: str, route_ids: list[str]) -> None:
        super().__init__(f"provider {name} is referenced by routes")
        self.name = name
        self.route_ids = route_ids


class ProviderVerificationError(ProviderError):
    def __init__(self, kind: ProviderKind, reason: str | None) -> None:
        super().__init__(f"provider verification failed for {kind.value}: {reason}")
        self.kind = kind
        self.reason = reason
