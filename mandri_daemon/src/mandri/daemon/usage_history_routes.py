from dataclasses import replace
from typing import Any

from mandri.core.ids import ProviderKind
from mandri.core.model_refs import MODEL_REF_PREFIXES
from mandri.core.types.usage import UsageObservation


def attribute_route(
    value: UsageObservation, routes: list[dict[str, Any]]
) -> UsageObservation | None:
    when = value.interval_start if value.interval_start is not None else value.occurred_at
    if when is None:
        return None
    applicable = [route for route in routes if route["effective_from"] <= when]
    if not applicable:
        return None
    route = max(applicable, key=lambda item: (item["effective_from"], item["id"]))
    kind = ProviderKind(route["provider_kind"])
    model = str(route["model_ref"]).removeprefix(MODEL_REF_PREFIXES[kind])
    if value.model not in {model, route["model_ref"]}:
        return None
    return replace(
        value,
        provider=route["provider_name"],
        model=model,
        billing_mode="subscription" if kind is ProviderKind.OPENCODE_GO else "api",
        pricing_context={
            **value.pricing_context,
            "provider_kind": kind.value,
            "route_id": route["route_id"],
            "selected_model": route["model_ref"],
            "attributed_source": "gateway",
            "model_basis": "historical_gateway_route",
        },
    )
