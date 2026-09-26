"""Route lookup port for services that must reason about route bindings."""

from typing import Protocol

from mandri.core.ids import RouteId


class RouteLookupPort(Protocol):
    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        raise NotImplementedError
