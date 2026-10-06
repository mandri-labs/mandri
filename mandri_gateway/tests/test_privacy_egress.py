import json

import httpx
import pytest
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState


@pytest.fixture
def guard():
    kind = ProviderKind.OPENAI
    base = Url("https://provider.invalid/private-deployment/v1")
    key = SecretRef("sk-proj-syntheticcredential000000000")
    provider = Provider(
        name="test", kind=kind, api_base=base, api_key=key, state=ProviderState.VERIFIED
    )
    route = ResolvedRoute(
        route_id=RouteId("route"),
        provider=provider,
        model=Model(kind, ModelRef("openai/approved-model"), base, key),
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="scope",
    )
    engine = SurrogateEngine(SurrogateScope("scope"))
    engine.register("private-customer")
    engine.register(str(key), "secret")
    engine.register_root("/srv/private-customer/Project")
    return EgressGuard(route, engine)


def request(guard, *, path="/chat/completions", body=None, headers=None):
    return httpx.Request(
        "POST",
        str(guard.route.model.api_base) + path,
        json=body
        or {"model": "approved-model", "messages": [{"role": "user", "content": "Hello"}]},
        headers=headers or {"authorization": "Bearer " + str(guard.route.model.api_key)},
    )


async def test_exact_approved_routing_and_credentials_remain_authoritative(guard):
    await guard.check(request(guard))
    assert guard.sends == 1


@pytest.mark.parametrize(
    "path",
    [
        "/private-customer/chat/completions",
        "/chat/completions/private-customer",
        "/chat/completions?email=private-customer",
        "/chat/completions?type=private-customer",
        "/chat/completions?key=private-customer",
        "/chat/completions#private-customer",
        "/chat/completions/../../uploads",
        "/chat/completions?alt=sse",
    ],
)
async def test_added_sdk_path_query_or_fragment_is_rejected(guard, path):
    with pytest.raises(ProtectionError):
        await guard.check(request(guard, path=path))
    assert guard.sends == 0


@pytest.mark.parametrize(
    "body",
    [
        {"model": "private-customer", "messages": []},
        {"model": "different-model", "messages": []},
        {"model": "approved-model", "temperature": "private-customer", "messages": []},
        {"model": "approved-model", "messages": [], "sdk_extension": {"role": "private-customer"}},
        {"model": "approved-model", "messages": [], "metadata": {"role": "private-customer"}},
        {
            "model": "approved-model",
            "messages": [{"role": "tool", "content": {"type": "private-customer"}}],
        },
    ],
)
async def test_final_wire_validation_rejects_wrong_model_and_known_personal_data(guard, body):
    with pytest.raises(ProtectionError):
        await guard.check(request(guard, body=body))
    assert guard.sends == 0


@pytest.mark.parametrize(
    "headers",
    [
        {"x-sdk-user": "private-customer"},
        {"x-sdk-root": "/srv/private-customer/Project/file.py"},
        {"authorization": "Bearer wrong-credential"},
        {"x-sdk-token": "unprepared-credential"},
        {"x-sdk-debug": "sk-proj-syntheticcredential000000000"},
    ],
)
async def test_sdk_headers_reject_known_values_and_wrong_credentials(guard, headers):
    if "x-sdk-token" not in headers:
        with pytest.raises(ProtectionError):
            await guard.check(request(guard, headers=headers))
        assert guard.sends == 0
    else:
        await guard.check(request(guard, headers=headers))
        assert guard.sends == 1


async def test_duplicate_json_keys_are_rejected_before_transport(guard):
    wire = request(guard)
    payload = (
        b'{"model":"approved-model","metadata":{"role":"private-customer",'
        b'"role":"safe"},"messages":[]}'
    )
    duplicate = httpx.Request("POST", wire.url, content=payload)
    with pytest.raises(ProtectionError):
        await guard.check(duplicate)
    assert guard.sends == 0


async def test_prepared_nested_schema_and_application_data_pass_wire_guard(guard):
    body = {
        "model": "approved-model",
        "messages": [{"role": "tool", "content": {"type": "private-customer"}}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"private-customer": {"type": "string"}},
                        "required": ["private-customer"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    protected = guard.engine.protect(body)
    await guard.check(request(guard, body=protected))
    assert guard.sends == 1
    assert "private-customer" not in json.dumps(protected)


async def test_adapter_cannot_reenable_protected_provider_streaming(guard):
    with pytest.raises(ProtectionError, match="streaming is disabled"):
        await guard.check(
            request(guard, body={"model": "approved-model", "messages": [], "stream": True})
        )
    assert guard.sends == 0
