"""Tests for reasoning effort catalog parsing and lookup."""

import mandri.gateway.reasoning_catalog as catalog_module
import pytest
from mandri.core.ids import ProviderKind, SecretRef, Url
from mandri.gateway.reasoning_catalog import (
    ReasoningCatalog,
    ReasoningInfo,
    parse_lm_studio_entry,
    parse_models_dev_entry,
    parse_ollama_entry,
)
from mandri.providers.service import Provider, ProviderState


def _provider(
    name: str = "prov",
    kind: ProviderKind = ProviderKind.OPENROUTER,
    api_base: str | None = None,
    api_key: str = "",
) -> Provider:
    return Provider(
        name=name,
        kind=kind,
        api_base=Url(api_base) if api_base else None,
        api_key=SecretRef(api_key),
        state=ProviderState.VERIFIED,
    )


def test_models_dev_entry_extracts_effort_values() -> None:
    entry = {
        "reasoning_options": [
            {"type": "effort", "values": ["low", "medium", "high"]},
            {"type": "toggle"},
            {"type": "budget_tokens", "min": 1024, "max": 32000},
        ]
    }
    assert parse_models_dev_entry(entry) == ReasoningInfo(
        efforts=["low", "medium", "high"], default_effort=None
    )


def test_models_dev_entry_ignores_non_effort_options() -> None:
    assert parse_models_dev_entry({"reasoning_options": [{"type": "toggle"}]}) is None


def test_models_dev_entry_rejects_malformed_payload() -> None:
    assert parse_models_dev_entry(None) is None
    assert parse_models_dev_entry({"reasoning_options": "low"}) is None


def test_normalize_dedupes_preserving_order() -> None:
    entry = {"reasoning_options": [{"type": "effort", "values": ["high", "low", "high", "medium"]}]}
    assert parse_models_dev_entry(entry) == ReasoningInfo(
        efforts=["high", "low", "medium"], default_effort=None
    )


def test_lm_studio_entry_parses_allowed_options_and_default() -> None:
    entry = {
        "capabilities": {"reasoning": {"allowed_options": ["off", "on", "medium"], "default": "on"}}
    }
    assert parse_lm_studio_entry(entry) == ReasoningInfo(
        efforts=["off", "on", "medium"], default_effort="on"
    )


def test_lm_studio_entry_drops_default_not_in_options() -> None:
    entry = {"capabilities": {"reasoning": {"allowed_options": ["low", "high"], "default": "on"}}}
    assert parse_lm_studio_entry(entry) == ReasoningInfo(
        efforts=["low", "high"], default_effort=None
    )


def test_lm_studio_entry_rejects_missing_reasoning() -> None:
    assert parse_lm_studio_entry({"capabilities": {}}) is None
    assert parse_lm_studio_entry("model") is None


def test_lookup_strips_model_ref_prefix() -> None:
    catalog = ReasoningCatalog({("prov", "z-ai/glm-5.2"): ReasoningInfo(["high"])})
    assert catalog.lookup("prov", "openrouter/z-ai/glm-5.2") == ReasoningInfo(["high"])


def test_lookup_falls_back_to_last_path_segment() -> None:
    catalog = ReasoningCatalog({("prov", "glm-5.2"): ReasoningInfo(["high"])})
    assert catalog.lookup("prov", "openrouter/z-ai/glm-5.2") == ReasoningInfo(["high"])


def test_lookup_unknown_returns_none() -> None:
    catalog = ReasoningCatalog({("prov", "m1"): ReasoningInfo(["low"])})
    assert catalog.lookup("prov", "openrouter/m2") is None
    assert catalog.lookup("other", "openrouter/m1") is None


@pytest.fixture()
def fetch_recorder(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, str]]] = []

    def install(payload_by_url: dict[str, object]) -> list[tuple[str, dict[str, str]]]:
        async def fake_fetch(url: str, headers: dict[str, str], body=None) -> object:
            calls.append((url, dict(headers)))
            key = url + "/" + body["model"] if body else url
            return payload_by_url.get(key)

        monkeypatch.setattr(catalog_module, "_fetch_json", fake_fetch)
        return calls

    return install


async def test_build_parses_lm_studio_probe(fetch_recorder) -> None:
    url = "http://127.0.0.1:1234/api/v1/models"
    calls = fetch_recorder(
        {
            url: {
                "models": [
                    {
                        "id": "qwen3-32b",
                        "capabilities": {
                            "reasoning": {
                                "allowed_options": ["off", "on"],
                                "default": "off",
                            }
                        },
                    },
                    {"id": "plain", "capabilities": {}},
                ]
            }
        }
    )
    catalog = await ReasoningCatalog.build(
        lambda: [_provider("local", ProviderKind.LM_STUDIO, "http://127.0.0.1:1234")]
    )
    assert [call[0] for call in calls if call[0].endswith("/api/v1/models")] == [url]
    probe = next(call for call in calls if call[0] == url)
    assert probe[1] == {}
    assert catalog.lookup("local", "lm_studio/qwen3-32b") == ReasoningInfo(["off", "on"], "off")
    assert catalog.lookup("local", "lm_studio/plain") is None


async def test_build_probe_sends_bearer_header(fetch_recorder) -> None:
    calls = fetch_recorder({})
    await ReasoningCatalog.build(
        lambda: [
            _provider(
                "local",
                ProviderKind.LM_STUDIO,
                "http://localhost:1234",
                api_key="secret",
            )
        ]
    )
    probe = next(call for call in calls if call[0].endswith("/api/v1/models"))
    assert probe[1] == {"Authorization": "Bearer secret"}


async def test_build_skips_probe_for_remote_base(fetch_recorder) -> None:
    calls = fetch_recorder({})
    await ReasoningCatalog.build(
        lambda: [_provider("remote", ProviderKind.CUSTOM, "https://example.com/v1")]
    )
    assert all(not call[0].endswith("/api/v1/models") for call in calls)


async def test_build_resolves_provider_by_name_fallback(fetch_recorder) -> None:
    models_dev_url = "https://models.dev/api.json"
    fetch_recorder(
        {
            models_dev_url: {
                "zai": {
                    "models": {
                        "glm-5.2": {"reasoning_options": {"type": "effort", "values": ["high"]}}
                    }
                }
            }
        }
    )
    catalog = await ReasoningCatalog.build(
        lambda: [_provider("zai", ProviderKind.CUSTOM, "https://example.com/v1")]
    )
    assert catalog.lookup("zai", "openai/glm-5.2") == ReasoningInfo(["high"])


async def test_build_prefers_live_catalog_over_models_dev(fetch_recorder) -> None:
    models_dev_url = "https://models.dev/api.json"
    live_url = "https://openrouter.ai/api/v1/models"
    fetch_recorder(
        {
            models_dev_url: {
                "openrouter": {
                    "models": {
                        "z-ai/glm-5.2": {"reasoning_options": {"type": "effort", "values": ["low"]}}
                    }
                }
            },
            live_url: {
                "data": [
                    {
                        "id": "z-ai/glm-5.2",
                        "reasoning": {"supported_efforts": ["low", "medium", "high"]},
                    }
                ]
            },
        }
    )
    catalog = await ReasoningCatalog.build(lambda: [_provider("prov", ProviderKind.OPENROUTER)])
    assert catalog.lookup("prov", "openrouter/z-ai/glm-5.2") == ReasoningInfo(
        ["low", "medium", "high"], None
    )


async def test_named_opencode_go_provider_preserves_namespaced_models(fetch_recorder) -> None:
    fetch_recorder(
        {
            "https://models.dev/api.json": {
                "opencode-go": {
                    "models": {
                        "vendor/model": {
                            "reasoning_options": {"type": "effort", "values": ["low", "high"]}
                        }
                    }
                }
            }
        }
    )
    catalog = await ReasoningCatalog.build(lambda: [_provider("my-go", ProviderKind.OPENCODE_GO)])
    assert catalog.lookup("my-go", "my-go/vendor/model") == ReasoningInfo(["low", "high"])


@pytest.mark.parametrize(
    "thinking,expected",
    [
        ({"values": [False, True], "default": True}, ReasoningInfo(["off", "on"], "on")),
        ({"values": ["low", "high"], "default": "high"}, ReasoningInfo(["low", "high"], "high")),
        ({"values": [False], "default": False}, ReasoningInfo([])),
        ({"values": [1, None]}, None),
        (None, None),
    ],
)
def test_ollama_native_thinking_options(thinking, expected):
    assert parse_ollama_entry({"thinking": thinking}) == expected


async def test_live_capabilities_are_provider_scoped_and_override_static_catalog(fetch_recorder):
    base = "http://localhost:9999"
    calls = fetch_recorder(
        {
            "https://models.dev/api.json": {
                "lmstudio": {
                    "models": {
                        "same-model": {"reasoning_options": {"type": "effort", "values": ["high"]}}
                    }
                },
                "ollama-cloud": {
                    "models": {
                        "same-model": {"reasoning_options": {"type": "effort", "values": ["high"]}}
                    }
                },
            },
            base + "/api/v1/models": {
                "models": [
                    {
                        "key": "same-model",
                        "capabilities": {
                            "reasoning": {"allowed_options": ["off", "on"], "default": "off"}
                        },
                    }
                ]
            },
            base + "/api/tags": {"models": [{"name": "same-model"}, {"name": "plain"}]},
            base + "/api/show/same-model": {
                "thinking": {"values": ["low", "high"], "default": "low"}
            },
            base + "/api/show/plain": {"thinking": {"values": [False]}},
        }
    )
    catalog = await ReasoningCatalog.build(
        lambda: [
            _provider("studio", ProviderKind.LM_STUDIO, base),
            _provider("ollama", ProviderKind.OLLAMA, base + "/v1"),
        ]
    )
    assert catalog.lookup("studio", "same-model") == ReasoningInfo(["off", "on"], "off")
    assert catalog.lookup("ollama", "same-model") == ReasoningInfo(["low", "high"], "low")
    assert catalog.lookup("ollama", "plain") == ReasoningInfo([])
    assert sum(url.endswith("/api/v1/models") for url, _ in calls) == 1
    assert sum(url.endswith("/api/show") for url, _ in calls) == 2
