"""Smoke tests for the gateway package surface."""

from mandri.gateway import route_registry
from mandri.gateway.errors.upstream import RouteNotFoundError


def test_gateway_surface_imports() -> None:
    assert route_registry.RouteRegistry is not None
    assert issubclass(RouteNotFoundError, Exception)
