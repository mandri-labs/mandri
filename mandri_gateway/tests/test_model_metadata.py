from collections import OrderedDict

import httpx
import pytest
from mandri.core.ids import ProviderKind, Url
from mandri.core.model_metadata import ModelMetadata
from mandri.gateway import catalog_enrichment, model_metadata
from mandri.gateway.metadata_entry import consensus, parse_entry


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.setattr(model_metadata, "_previous", OrderedDict())
    monkeypatch.setattr(catalog_enrichment, "_cached", {})
    monkeypatch.setattr(catalog_enrichment, "_expires", 0)


@pytest.mark.parametrize(
    "entry,context,output",
    [
        ({"meta": {"n_ctx": 262144}}, 262144, None),
        ({"limit": {"context": 65536, "output": 4096}}, 65536, 4096),
        ({"context_window": 32768, "max_output_tokens": 8192}, 32768, 8192),
        ({"context_length": None, "limit": {"context": 65536}}, 65536, None),
        ({"context_length": False, "meta": {"n_ctx": 32768}}, 32768, None),
        ({"context_length": -1, "max_context_length": 65536}, 65536, None),
    ],
)
def test_limits_are_discovered_without_substituting_numbers(entry, context, output):
    metadata = parse_entry(entry)
    assert metadata.context_window == context
    assert metadata.output_tokens == output
    assert metadata.auto_compact_token_limit is None
    assert ModelMetadata.from_payload(metadata.to_payload()) == metadata


def test_loaded_context_takes_precedence_over_the_model_maximum():
    metadata = parse_entry(
        {
            "max_context_length": 262144,
            "loaded_instances": [{"config": {"context_length": 32768}}],
        }
    )
    assert metadata.context_window == 32768
    assert metadata.max_context_window == 262144


def test_input_and_output_limits_remain_distinct():
    metadata = parse_entry({"inputTokenLimit": 131072, "outputTokenLimit": 65536})
    assert metadata.context_window is None
    assert metadata.input_tokens == metadata.available_context == 131072
    assert metadata.output_tokens == 65536
    assert metadata.output_budget is None


def test_partial_payload_does_not_discard_capabilities_without_limits():
    metadata = ModelMetadata.from_payload({"reasoning_efforts": ["low", "high"], "tool_call": True})
    assert metadata is not None
    assert metadata.context_window is None
    assert metadata.reasoning_efforts == ("low", "high")
    assert metadata.tool_call is True


def test_explicit_exclusions_survive_fallbacks():
    metadata = parse_entry({"capabilities": {"reasoning": False, "vision": False}})
    metadata = metadata.with_fallback(
        parse_entry(
            {
                "reasoning_efforts": ["low", "high"],
                "capabilities": {"vision": True},
                "context_length": 32768,
            },
            "catalog",
        )
    )
    assert metadata.reasoning_supported is False
    assert metadata.reasoning_efforts == ()
    assert metadata.image_input is False
    assert metadata.context_window == 32768
    assert metadata.sources["context_window"] == "catalog"


def test_current_reasoning_support_does_not_inherit_a_previous_exclusion():
    previous = ModelMetadata(reasoning_supported=False, reasoning_efforts=()).with_source("server")
    current = ModelMetadata(reasoning_supported=True).with_source("server").with_fallback(previous)
    assert current.reasoning_supported is True
    assert current.reasoning_efforts is None
    assert "reasoning_efforts" not in current.sources


def test_ambiguous_catalogs_only_supply_agreed_fields():
    metadata = consensus(
        [
            parse_entry({"context_length": 32768, "tool_call": True}),
            parse_entry({"context_length": 131072, "tool_call": True}),
        ]
    )
    assert metadata.context_window is None
    assert metadata.tool_call is True


async def test_discovery_continues_after_catalog_failure_and_retains_previous_data(monkeypatch):
    calls = []
    available = True

    def upstream(request):
        calls.append(request.url.path)
        if request.url.host == "models.dev":
            return httpx.Response(200, json={})
        if request.url.path == "/v1/models":
            return httpx.Response(503)
        if request.url.path == "/props" and available:
            return httpx.Response(
                200,
                json={
                    "model_alias": "fixture",
                    "default_generation_settings": {"n_ctx": 65536},
                    "chat_template_caps": {"supports_tool_calls": True},
                    "reasoning_efforts": ["low", "high"],
                },
            )
        return httpx.Response(404)

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        model_metadata.httpx,
        "AsyncClient",
        lambda **kwargs: factory(
            transport=httpx.MockTransport(upstream),
            **kwargs,
        ),
    )
    first = await model_metadata.fetch(
        ProviderKind.CUSTOM, "fixture", Url("http://model.invalid/v1"), ""
    )
    available = False
    second = await model_metadata.fetch(
        ProviderKind.CUSTOM, "fixture", Url("http://model.invalid/v1"), ""
    )
    assert first == second
    assert second.context_window == 65536
    assert second.tool_call is True
    assert second.reasoning_efforts == ("low", "high")
    assert "/props" in calls and "/api.json" in calls


async def test_catalog_fallback_works_without_live_model_entries(monkeypatch):
    def upstream(request):
        if request.url.host == "models.dev":
            return httpx.Response(
                200,
                json={
                    "vendor": {
                        "models": {
                            "fixture": {
                                "limit": {"context": 65536, "output": 8192},
                                "reasoning_options": [
                                    {"type": "effort", "values": ["low", "high"]}
                                ],
                            }
                        }
                    }
                },
            )
        return httpx.Response(404)

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        model_metadata.httpx,
        "AsyncClient",
        lambda **kwargs: factory(
            transport=httpx.MockTransport(upstream),
            **kwargs,
        ),
    )
    metadata = await model_metadata.fetch(
        ProviderKind.CUSTOM, "fixture-Q4_K_M.gguf", Url("http://model.invalid/v1"), ""
    )
    assert metadata.context_window == 65536
    assert metadata.output_tokens == 8192
    assert metadata.reasoning_efforts == ("low", "high")
    assert metadata.sources["context_window"] == "catalog"


@pytest.mark.parametrize("running,context", [(True, 32768), (False, 8192)])
async def test_ollama_uses_the_loaded_window_before_parameters_and_training_limits(
    monkeypatch, running, context
):
    def upstream(request):
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "parameters": 'num_ctx 8192\nnum_predict -1\nstop "word"',
                    "model_info": {"fixture.context_length": 262144},
                    "capabilities": ["completion", "tools"],
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"model": "fixture", "context_length": 32768},
                        {"model": "unrelated", "context_length": 1024},
                    ]
                    if running
                    else []
                },
            )
        return httpx.Response(404)

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        model_metadata.httpx,
        "AsyncClient",
        lambda **kwargs: factory(transport=httpx.MockTransport(upstream), **kwargs),
    )
    metadata = await model_metadata.fetch(
        ProviderKind.OLLAMA, "ollama/fixture", Url("http://model.invalid"), ""
    )
    assert metadata.context_window == context
    assert metadata.max_context_window == 262144
    assert metadata.output_budget is None
    assert metadata.tool_call is True
