"""Reasoning effort normalization and validation against the governed catalog."""

from mandri.api.deps import GatewayWiring
from mandri.api.errors import ApiError
from mandri.gateway.reasoning_catalog import ReasoningInfo
from mandri.providers.errors import ProviderInvalidError
from mandri.providers.service import parse_model_arg


def normalize_effort(effort: str | None) -> str | None:
    """Treat an empty effort as not provided."""
    return effort or None


def lookup_efforts(wiring: GatewayWiring, model_arg: str) -> ReasoningInfo | None:
    catalog = wiring.reasoning_catalog
    if catalog is None:
        return None
    try:
        provider_name, _ = parse_model_arg(model_arg)
    except ProviderInvalidError:
        return None
    return catalog.lookup(provider_name, model_arg)


def validate_effort(wiring: GatewayWiring, model_arg: str, effort: str | None) -> None:
    normalized = normalize_effort(effort)
    if normalized is None:
        return
    info = lookup_efforts(wiring, model_arg)
    if info is None or normalized in info.efforts:
        return
    raise ApiError(
        code="invalid_effort",
        message=(
            f"Unknown reasoning effort {normalized!r} for {model_arg}, "
            f"allowed efforts: {', '.join(info.efforts)}"
        ),
        status=400,
        detail={"effort": normalized, "allowed": info.efforts},
    )
