"""Read-through coherence for the provider registry against external config writes."""

from typing import Any

import pytest
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ids import ProviderKind, RouteId
from mandri.providers.errors import ProviderNotFoundError
from mandri.providers.service import ProvidersRegistry


class _Routes:
    async def route_ids_for_provider(self, provider_name: str) -> list[RouteId]:
        return []


def _registry(tmp_path: object) -> ProvidersRegistry:
    return ProvidersRegistry(TomlConfigAdapter(tmp_path), _Routes())


def _external(tmp_path: object) -> ProvidersRegistry:
    return ProvidersRegistry(TomlConfigAdapter(tmp_path), _Routes())


async def test_provider_added_externally_visible_on_next_list(tmp_path: object) -> None:
    registry = _registry(tmp_path)
    assert registry.list() == []
    external = _external(tmp_path)
    await external.add("openrouter", ProviderKind.OPENROUTER, None, "sk", verify=False)
    assert [provider.name for provider in registry.list()] == ["openrouter"]


async def test_provider_removed_externally_invisible_on_next_get(tmp_path: object) -> None:
    registry = _registry(tmp_path)
    external = _external(tmp_path)
    await external.add("openrouter", ProviderKind.OPENROUTER, None, "sk", verify=False)
    assert [provider.name for provider in registry.list()] == ["openrouter"]
    await external.remove("openrouter")
    with pytest.raises(ProviderNotFoundError):
        registry.get("openrouter")


async def test_provider_updated_externally_reflected_on_next_get(tmp_path: object) -> None:
    registry = _registry(tmp_path)
    external = _external(tmp_path)
    await external.add("openai", ProviderKind.OPENAI, None, "old", verify=False)
    await external.update("openai", api_key="new", verify=False)
    provider: Any = registry.get("openai")
    assert str(provider.api_key) == "new"
