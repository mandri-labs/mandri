import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.codex_catalog import CATALOG_ENV
from mandri.core.ids import HarnessKind, ProviderKind
from mandri.core.launch import ModelMetadata as LaunchMetadata
from mandri.core.launch import build_harness_launch
from mandri.gateway.model_metadata import ModelMetadata, _parse
from mandri.runtime.service import RuntimeService
from mandri.runtime.wiring import build_harness_launch as runtime_launch


@pytest.mark.parametrize("supported", [True, False])
def test_explicit_catalog_hosted_search_capability_round_trips(supported):
    metadata = _parse(
        ProviderKind.OPENROUTER,
        {
            "data": [
                {
                    "id": "fixture",
                    "context_length": 128000,
                    "top_provider": {"max_completion_tokens": 4096},
                    "capabilities": {"hosted_web_search": supported},
                }
            ]
        },
        "fixture",
    )
    assert metadata is not None
    assert metadata.hosted_web_search is supported
    assert ModelMetadata.from_payload(metadata.to_payload()) == metadata
    launch = runtime_launch(HarnessKind.CODEX, 8000, "route", "fixture", "token", metadata)
    assert ('web_search="disabled"' in launch.args) is (not supported)


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (["tools", "tool_choice"], False),
        (["tools", "web_search_options"], True),
        (None, False),
    ],
)
def test_function_tools_do_not_imply_hosted_search(parameters, expected):
    metadata = _parse(
        ProviderKind.OPENROUTER,
        {
            "data": [
                {
                    "id": "fixture",
                    "context_length": 128000,
                    "top_provider": {"max_completion_tokens": 4096},
                    "supported_parameters": parameters,
                }
            ]
        },
        "fixture",
    )
    assert metadata is not None and metadata.hosted_web_search is expected


def test_unadvertised_search_does_not_disable_any_coding_tool_or_permission():
    unknown = build_harness_launch(HarnessKind.CODEX, 8000, "route", "fixture", "token")
    supported = build_harness_launch(
        HarnessKind.CODEX,
        8000,
        "route",
        "fixture",
        "token",
        LaunchMetadata(128000, 4096, hosted_web_search=True),
    )
    assert CATALOG_ENV not in unknown.env
    assert unknown.env == {key: value for key, value in supported.env.items() if key != CATALOG_ENV}
    catalog = json.loads(supported.env[CATALOG_ENV])
    assert catalog["models"][0]["supports_search_tool"] is True
    assert unknown.args[-2:] == ("-c", 'web_search="disabled"')
    assert supported.args[-2:] == ("-c", "model_context_window=128000")
    assert unknown.args[:-2] == supported.args[:-2]
    assert not any("sandbox" in value or "approval" in value for value in unknown.args)


async def test_named_provider_resolves_catalog_by_model_id(monkeypatch):
    provider = SimpleNamespace(kind=ProviderKind.OPENROUTER, api_base=None, api_key="scoped")
    runtime = SimpleNamespace(_routes=SimpleNamespace(providers={"custom-provider": provider}))
    fetch = AsyncMock(return_value=ModelMetadata(128000, 4096, hosted_web_search=True))
    monkeypatch.setattr("mandri.runtime.service.fetch_model_metadata", fetch)
    metadata = await RuntimeService._resolve_metadata(runtime, "custom-provider/vendor/model")
    fetch.assert_awaited_once_with(ProviderKind.OPENROUTER, "vendor/model", None, "scoped")
    assert metadata is not None and metadata.hosted_web_search
