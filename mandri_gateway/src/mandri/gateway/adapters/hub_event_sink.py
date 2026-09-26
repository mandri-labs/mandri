"""Hub-backed gateway event sink publishing on the gateway.events topic."""

from mandri.core.hub import Hub, Topic
from mandri.core.ids import RouteId
from mandri.core.ports.gateway_events import GatewayEventSink
from mandri.core.types.gateway import GatewayEventKind

GATEWAY_EVENTS_TOPIC = Topic("gateway.events")


class HubGatewayEventSink(GatewayEventSink):
    def __init__(self, hub: Hub) -> None:
        self._hub = hub

    def publish_event(
        self,
        event: GatewayEventKind,
        route_id: RouteId | None,
        provider_name: str | None,
    ) -> None:
        payload: dict[str, str | None] = {
            "event": event.value,
            "route_id": None if route_id is None else str(route_id),
            "provider_name": provider_name,
        }
        self._hub.publish(GATEWAY_EVENTS_TOPIC, payload)
