"""Tests for the provider registry service."""

from pathlib import Path

import pytest
from mandri.core.ids import ProviderKind, RouteId, Url
from mandri.core.ports.provider_verifier import ProviderVerifierPort, VerificationResult
from mandri.core.types.config import DaemonConfig, ProviderConfig
from mandri.providers.errors import (
    ProviderExistsError,
    ProviderInUseError,
    ProviderInvalidError,
    ProviderNotFoundError,
    ProviderVerificationError,
)
from mandri.providers.service import (
    ProvidersRegistry,
    ProviderState,
    parse_model_arg,
    provider_model_ref,
    split_model_ref,
)


class FakeConfig:
    def __init__(self, config: DaemonConfig | None = None) -> None:
        self.config = config if config is not None else DaemonConfig()
        self.save_calls = 0

    @property
    def config_path(self) -> Path:
        return Path("config.toml")

    def load(self) -> DaemonConfig:
        return self.config

    def save(self, config: DaemonConfig) -> None:
        self.config = config
        self.save_calls += 1


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/", "/api/v0", "/api/v1/"])
async def test_lm_studio_normalizes_existing_and_new_provider_bases(suffix: str) -> None:
    base = "http://localhost:1234" + suffix
    config = FakeConfig(DaemonConfig(providers=[ProviderConfig("local", "lm_studio", base)]))
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier())
    assert registry.get("local").api_base == "http://localhost:1234/v1"
    assert config.save_calls == 0
    added = await registry.add("new", ProviderKind.LM_STUDIO, base, "", verify=False)
    assert added.api_base == "http://localhost:1234/v1"


class FakeRoutes:
    def __init__(self, route_ids: list[RouteId] | None = None) -> None:
        self.route_ids = route_ids if route_ids is not None else []
        self.queries: list[str] = []

    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        self.queries.append(provider_name)
        return self.route_ids


class FakeVerifier(ProviderVerifierPort):
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[ProviderKind, Url | None, str]] = []

    def verify(self, kind: ProviderKind, api_base: Url | None, api_key: str) -> VerificationResult:
        self.calls.append((kind, api_base, api_key))
        reason = None if self.ok else "denied"
        return VerificationResult(ok=self.ok, provider=kind, reason=reason)

    async def verify_async(
        self, kind: ProviderKind, api_base: Url | None, api_key: str
    ) -> VerificationResult:
        return self.verify(kind, api_base, api_key)


def test_parse_model_arg_splits_provider_and_model() -> None:
    assert parse_model_arg("openrouter/claude-x") == ("openrouter", "claude-x")


@pytest.mark.parametrize("arg", ["/model", "provider/", "provider", "  /model"])
def test_parse_model_arg_rejects_malformed(arg: str) -> None:
    with pytest.raises(ProviderInvalidError):
        parse_model_arg(arg)


def test_model_ref_format_and_parse_round_trip() -> None:
    ref = provider_model_ref(ProviderKind.OLLAMA, "llama3")
    assert str(ref) == "ollama_chat/llama3"
    assert split_model_ref(ProviderKind.OLLAMA, str(ref)) == "llama3"


def test_split_model_ref_passthrough_without_prefix() -> None:
    assert split_model_ref(ProviderKind.OPENAI, "gpt-4") == "gpt-4"


async def test_add_verified_provider_persists_entry() -> None:
    config = FakeConfig()
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier())
    provider = await registry.add("openrouter", ProviderKind.OPENROUTER, None, "sk-test")
    assert provider.state is ProviderState.VERIFIED
    assert [entry.name for entry in config.config.providers] == ["openrouter"]


async def test_add_with_failed_verification_raises() -> None:
    config = FakeConfig()
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier(ok=False))
    with pytest.raises(ProviderVerificationError):
        await registry.add("openrouter", ProviderKind.OPENROUTER, None, "sk-test")
    assert config.config.providers == []


async def test_add_duplicate_name_rejected() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier())
    await registry.add("openrouter", ProviderKind.OPENROUTER, None, "sk-test")
    with pytest.raises(ProviderExistsError):
        await registry.add("openrouter", ProviderKind.OPENROUTER, None, "sk-test")


async def test_add_blank_name_rejected() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier())
    with pytest.raises(ProviderInvalidError):
        await registry.add("   ", ProviderKind.OPENROUTER, None, "sk-test")


async def test_add_local_kind_requires_api_base() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier())
    with pytest.raises(ProviderInvalidError):
        await registry.add("ollama", ProviderKind.OLLAMA, None, "", verify=False)


async def test_get_unknown_provider_raises() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier())
    with pytest.raises(ProviderNotFoundError):
        registry.get("nope")


def test_load_maps_existing_config_entries() -> None:
    config = FakeConfig(
        DaemonConfig(providers=[ProviderConfig(name="p1", kind="openai", api_key="k1")])
    )
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier())
    provider = registry.get("p1")
    assert provider.kind is ProviderKind.OPENAI
    assert provider.state is ProviderState.VERIFIED


async def test_update_persists_replaced_entry() -> None:
    config = FakeConfig()
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier())
    await registry.add("openai", ProviderKind.OPENAI, None, "old", verify=False)
    await registry.update("openai", api_key="new", verify=False)
    assert config.config.providers[0].api_key == "new"


async def test_remove_deletes_entry() -> None:
    config = FakeConfig()
    registry = ProvidersRegistry(config, FakeRoutes(), FakeVerifier())
    await registry.add("openrouter", ProviderKind.OPENROUTER, None, "sk-test")
    await registry.remove("openrouter")
    with pytest.raises(ProviderNotFoundError):
        registry.get("openrouter")
    assert config.config.providers == []


async def test_remove_blocked_while_routes_reference_provider() -> None:
    registry = ProvidersRegistry(
        FakeConfig(),
        FakeRoutes([RouteId("route-1"), RouteId("route-2")]),
        FakeVerifier(),
    )
    await registry.add("openai", ProviderKind.OPENAI, None, "k", verify=False)
    with pytest.raises(ProviderInUseError) as error:
        await registry.remove("openai")
    assert error.value.route_ids == ["route-1", "route-2"]


async def test_remove_queries_routes_for_the_provider_name() -> None:
    routes = FakeRoutes()
    registry = ProvidersRegistry(FakeConfig(), routes, FakeVerifier())
    await registry.add("openai", ProviderKind.OPENAI, None, "k", verify=False)
    await registry.remove("openai")
    assert routes.queries == ["openai"]


async def test_verify_marks_provider_verified() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier(ok=True))
    await registry.add("openai", ProviderKind.OPENAI, None, "k", verify=False)
    provider = await registry.verify("openai")
    assert provider.state is ProviderState.VERIFIED


async def test_verify_marks_provider_degraded_on_failure() -> None:
    registry = ProvidersRegistry(FakeConfig(), FakeRoutes(), FakeVerifier(ok=False))
    await registry.add("openai", ProviderKind.OPENAI, None, "k", verify=False)
    provider = await registry.verify("openai")
    assert provider.state is ProviderState.DEGRADED
