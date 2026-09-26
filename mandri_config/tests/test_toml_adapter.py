"""Tests for the TOML config adapter contract."""

import pytest
import tomlkit
from mandri.config.errors import ConfigError, ConfigParseError
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ports.config import ConfigPort
from mandri.core.types.config import (
    DEFAULT_CORS_ORIGINS,
    ApprovalsConfig,
    ApprovalTimeoutSeconds,
    ClaudeMode,
    CorsOrigin,
    DaemonConfig,
    DefaultsConfig,
    ProviderConfig,
    ServerConfig,
    SessionModeConfig,
    SessionsConfig,
    SyncConfig,
)


def _full_config() -> DaemonConfig:
    return DaemonConfig(
        server=ServerConfig(host="0.0.0.0", port=9000),
        sync=SyncConfig(
            ttl_seconds=45,
            mtime_poll_seconds=20,
            backoff_max_seconds=60,
            activity_quiet_seconds=120,
        ),
        providers=[
            ProviderConfig(name="openrouter", kind="openrouter", api_key="sk-test"),
            ProviderConfig(name="ollama", kind="ollama", api_base="http://127.0.0.1:11434"),
        ],
        defaults=DefaultsConfig(harness="claude", model="openrouter/claude-x"),
        sessions=SessionsConfig(
            opencode_db_path="C:/tmp/opencode.db",
            launch_args={"codex": ["--flag"]},
            mode=SessionModeConfig(claude=ClaudeMode("plan")),
        ),
        approvals=ApprovalsConfig(timeout_seconds=ApprovalTimeoutSeconds(30)),
    )


def _raise_serialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(doc: object) -> str:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(tomlkit, "dumps", _boom)


def test_conforms_to_config_port(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    assert isinstance(adapter, ConfigPort)


def test_missing_file_loads_defaults(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    assert adapter.load() == DaemonConfig()


def test_save_load_round_trip(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    config = _full_config()
    adapter.save(config)
    assert adapter.load() == config


def test_save_is_atomic_on_serialization_failure(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    original = _full_config()
    adapter.save(original)
    before = adapter.config_path.read_text(encoding="utf-8")
    _raise_serialization_failure(monkeypatch)
    with pytest.raises(RuntimeError):
        adapter.save(DaemonConfig(server=ServerConfig(port=1234)))
    assert adapter.config_path.read_text(encoding="utf-8") == before


def test_failed_first_save_leaves_no_file(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    _raise_serialization_failure(monkeypatch)
    with pytest.raises(RuntimeError):
        adapter.save(_full_config())
    assert not adapter.config_path.exists()


def test_read_through_sees_external_changes(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.save(DaemonConfig())
    writer = TomlConfigAdapter(tmp_path / "other")
    writer.save(DaemonConfig(server=ServerConfig(port=9999)))
    adapter.config_path.write_text(writer.config_path.read_text(encoding="utf-8"), encoding="utf-8")
    assert adapter.load().server.port == 9999


def test_malformed_toml_raises_parse_error(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text("not [valid", encoding="utf-8")
    with pytest.raises(ConfigParseError):
        adapter.load()


def test_parse_error_is_encapsulated_config_error(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text("not [valid", encoding="utf-8")
    with pytest.raises(ConfigError):
        adapter.load()


def test_unknown_top_level_key_rejected(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text("bogus = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        adapter.load()


def test_duplicate_provider_name_rejected(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text(
        '[[providers]]\nname = "dup"\nkind = "openai"\n'
        '[[providers]]\nname = "dup"\nkind = "openai"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        adapter.load()


def test_server_cors_origins_default_includes_dev_loopback(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    assert adapter.load().server.cors_origins == [
        CorsOrigin("http://localhost:1420"),
        CorsOrigin("http://127.0.0.1:1420"),
        CorsOrigin("http://tauri.localhost"),
        CorsOrigin("https://tauri.localhost"),
        CorsOrigin("tauri://localhost"),
        CorsOrigin("http://localhost:5173"),
        CorsOrigin("http://127.0.0.1:5173"),
        CorsOrigin("http://localhost:5174"),
        CorsOrigin("http://127.0.0.1:5174"),
    ]


def test_saved_default_cors_origins_include_the_alternate_dev_port(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    previous = [origin for origin in DEFAULT_CORS_ORIGINS if not origin.endswith(":5174")]
    adapter.save(DaemonConfig(server=ServerConfig(cors_origins=previous)))
    assert adapter.load().server.cors_origins == list(DEFAULT_CORS_ORIGINS)


@pytest.mark.parametrize("origins", [[], [CorsOrigin("http://localhost:5173")]])
def test_explicit_cors_restrictions_are_preserved(
    tmp_path: object, origins: list[CorsOrigin]
) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text(f"[server]\ncors_origins = {origins!r}\n", encoding="utf-8")
    assert adapter.load().server.cors_origins == origins


def test_server_cors_origins_round_trip(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    config = DaemonConfig(server=ServerConfig(cors_origins=[CorsOrigin("https://ops.example")]))
    adapter.save(config)
    assert adapter.load() == config


def test_server_cors_origins_missing_in_file_loads_defaults(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text("[server]\nport = 9000\n", encoding="utf-8")
    assert adapter.load().server.cors_origins == list(DEFAULT_CORS_ORIGINS)


def test_server_cors_origins_non_list_rejected(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text(
        '[server]\ncors_origins = "http://localhost:1420"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        adapter.load()


def test_server_cors_origins_non_string_entry_rejected(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text("[server]\ncors_origins = [42]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        adapter.load()


def test_local_provider_without_api_base_rejected(tmp_path: object) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    adapter.config_path.write_text(
        '[[providers]]\nname = "local"\nkind = "ollama"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        adapter.load()
