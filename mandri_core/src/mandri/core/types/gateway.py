"""Gateway event kinds carried on the gateway.events feed topic."""

import enum


class GatewayEventKind(enum.StrEnum):
    ROUTE_CREATED = "route_created"
    ROUTE_UPDATED = "route_updated"
    ROUTE_DELETED = "route_deleted"
