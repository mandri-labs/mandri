"""Read-through coherence for externally written provider entries."""

import dataclasses

from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.types.config import ProviderConfig


def test_provider_entry_written_externally_visible_on_next_read(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    assert adapter.load().providers == []
    external = TomlConfigAdapter(tmp_path)
    config = external.load()
    external.save(
        dataclasses.replace(
            config,
            providers=[ProviderConfig(name="external", kind="openai", api_key="k")],
        )
    )
    assert [entry.name for entry in adapter.load().providers] == ["external"]


def test_provider_entry_removed_externally_invisible_on_next_read(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    external = TomlConfigAdapter(tmp_path)
    external.save(
        dataclasses.replace(
            external.load(),
            providers=[ProviderConfig(name="external", kind="openai", api_key="k")],
        )
    )
    assert [entry.name for entry in adapter.load().providers] == ["external"]
    external.save(dataclasses.replace(external.load(), providers=[]))
    assert adapter.load().providers == []
