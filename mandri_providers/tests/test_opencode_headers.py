import httpx
import pytest
from mandri.core.ids import ProviderKind, Url
from mandri.providers.verify import ProviderVerifier, models_headers


@pytest.mark.parametrize("kind", [ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO])
async def test_metadata_verification_transmits_stable_technical_context(kind):
    captured = []

    def upstream(request):
        captured.append(request.headers["x-opencode-session"])
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["user-agent"].startswith("mandri/")
        return httpx.Response(200, json={"data": []})

    verifier = ProviderVerifier(httpx.MockTransport(upstream))
    for _ in range(2):
        result = await verifier.verify_async(kind, Url("http://provider.test/v1"), "test-key")
        assert result.ok
    assert captured[0] == captured[1]
    assert captured[0] == models_headers(kind, "test-key")["x-opencode-session"]


async def test_go_uses_hosted_endpoint_and_preserves_auth_failures():
    def upstream(request):
        assert str(request.url) == "https://opencode.ai/zen/go/v1/models"
        assert request.headers["user-agent"].startswith("mandri/")
        return httpx.Response(401)

    verifier = ProviderVerifier(httpx.MockTransport(upstream))
    result = await verifier.verify_async(ProviderKind.OPENCODE_GO, None, "invalid")
    assert not result.ok
    assert result.reason == "auth failed (status 401)"
