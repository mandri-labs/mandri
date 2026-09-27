import httpx
import pytest
from mandri.core.ids import ProviderKind, Url
from mandri.gateway import catalog_enrichment, model_metadata
from mandri.providers.service import provider_model_ref


@pytest.mark.parametrize("kind", [ProviderKind.OPENCODE_GO, ProviderKind.OPENAI])
async def test_sparse_catalog_enrichment_is_cached_and_keeps_live_exclusions(monkeypatch, kind):
    monkeypatch.setattr(catalog_enrichment, "_expires", 0)
    monkeypatch.setattr(catalog_enrichment, "_cached", {})
    calls = []

    def upstream(request):
        calls.append(request)
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "opencode-go": {
                    "models": {
                        "target": {
                            "reasoning": True,
                            "tool_call": True,
                            "modalities": {"input": ["text", "image"]},
                        }
                    }
                },
                "openai": {
                    "models": {
                        "target": {
                            "reasoning": True,
                            "tool_call": True,
                            "modalities": {"input": ["text", "image"]},
                        }
                    }
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        for _ in range(2):
            entries = await catalog_enrichment.enrich_entries(
                kind, [{"id": "target", "tool_call": False}], client
            )
            metadata = model_metadata._parse(kind, {"data": entries}, "target")
            assert metadata.image_input is True
            assert metadata.tool_call is False
            assert metadata.reasoning_supported is True
    assert len(calls) == 1


async def test_lm_studio_fetch_uses_native_metadata(monkeypatch):
    def upstream(request):
        assert str(request.url) == "http://localhost:1234/api/v1/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "org/target",
                        "max_context_length": 32768,
                        "capabilities": {
                            "vision": True,
                            "trained_for_tool_use": False,
                            "reasoning": {"allowed_options": ["on"]},
                        },
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(model_metadata.httpx, "AsyncClient", lambda **kwargs: client)
    metadata = await model_metadata.fetch(
        ProviderKind.LM_STUDIO,
        provider_model_ref(ProviderKind.LM_STUDIO, "org/target"),
        Url("http://localhost:1234/v1"),
        "",
    )
    assert metadata is not None
    assert metadata.context_window == 32768
    assert metadata.image_input is True
    assert metadata.tool_call is False
    assert metadata.reasoning_supported is True


async def test_openrouter_keeps_the_model_vendor_prefix(monkeypatch):
    def upstream(request):
        return httpx.Response(
            200, json={"data": [{"id": "openai/gpt-model", "context_length": 64000}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(model_metadata.httpx, "AsyncClient", lambda **kwargs: client)
    metadata = await model_metadata.fetch(
        ProviderKind.OPENROUTER, "openrouter/openai/gpt-model", None, "key"
    )
    assert metadata is not None
    assert metadata.context_window == 64000
