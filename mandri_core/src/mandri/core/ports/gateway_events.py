"""Gateway event sink port."""

from mandri.core.ids import RouteId
from mandri.core.types.gateway import GatewayEventKind


class GatewayEventSink:
    def publish_event(
        self,
        event: GatewayEventKind,
        route_id: RouteId | None,
        provider_name: str | None,
    ) -> None:
        raise NotImplementedError
