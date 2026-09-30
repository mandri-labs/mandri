from dataclasses import replace

from mandri.core.types.usage import UsageObservation
from mandri.database.usage_coverage import same_request
from mandri.database.usage_serialization import COUNTERS


def recover_gateway(gateway: UsageObservation, native: UsageObservation) -> UsageObservation:
    if (
        gateway.input_tokens is not None
        or gateway.output_tokens is not None
        or native.kind != "delta"
        or native.pricing_context.get("evidence") != "history_request"
        or native.input_tokens is None
        or native.output_tokens is None
        or not same_request(native, gateway)
    ):
        return gateway
    return replace(
        gateway,
        **{key: getattr(native, key) for key in COUNTERS},
        input_includes_cache=native.input_includes_cache,
        output_includes_reasoning=native.output_includes_reasoning,
        complete=native.complete,
        pricing_context={
            **gateway.pricing_context,
            "usage_evidence": "native_request",
            **(
                {"context_tokens": native.pricing_context["context_tokens"]}
                if "context_tokens" in native.pricing_context
                else {}
            ),
        },
    )
