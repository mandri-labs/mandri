from dataclasses import dataclass
from typing import Any

from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.privacy_known import KnownValues
from mandri.gateway.privacy_protocol import (
    GatewayProtocol,
    transform_content,
    validate_request,
    visit_content,
)
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine


@dataclass(frozen=True)
class PreparedRequest:
    body: dict[str, Any]
    guard: EgressGuard | None = None
    client_stream: bool = False


class GatewayPrivacy:
    def __init__(self, scopes: PrivacyScopes) -> None:
        self.scopes = scopes

    async def prepare(
        self, route: ResolvedRoute, protocol: GatewayProtocol, body: dict[str, Any]
    ) -> PreparedRequest:
        protected = route.privacy_mode is PrivacyMode.SURROGATE
        validate_request(body)
        if not protected:
            return PreparedRequest(body, client_stream=bool(body.get("stream")))
        if not route.privacy_scope_id:
            raise ProtectionError(
                "privacy_state_unavailable", "Protected route has no privacy scope"
            )

        def transform(engine: SurrogateEngine) -> dict[str, Any]:
            credential = str(route.model.api_key)
            if credential:
                engine.register(credential, kind="secret")
            if route.execution_backend is ExecutionBackend.DOCKER:
                engine.reserve_root("/workspace")
                engine.reserve_root("/home/worker")
            result = transform_content(body, engine)
            result = visit_content(result, KnownValues(engine).replace)
            if protocol in {
                GatewayProtocol.CHAT,
                GatewayProtocol.RESPONSES,
                GatewayProtocol.ANTHROPIC,
            }:
                result["stream"] = False
                result.pop("stream_options", None)
            return result

        payload, engine = await self.scopes.prepare(route.privacy_scope_id, transform)
        return PreparedRequest(payload, EgressGuard(route, engine), bool(body.get("stream")))


async def prepare_request(
    privacy: GatewayPrivacy | None,
    route: ResolvedRoute,
    protocol: GatewayProtocol,
    body: dict[str, Any],
) -> PreparedRequest:
    if privacy is not None:
        return await privacy.prepare(route, protocol, body)
    if route.privacy_mode is PrivacyMode.SURROGATE:
        raise ProtectionError("privacy_unavailable", "Privacy service is unavailable")
    validate_request(body)
    return PreparedRequest(body, client_stream=bool(body.get("stream")))
