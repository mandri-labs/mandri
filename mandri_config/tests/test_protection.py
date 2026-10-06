from dataclasses import replace
from pathlib import Path

import pytest
from mandri.config.errors import ConfigError
from mandri.config.protection import parse_docker, parse_privacy
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.types.config import DaemonConfig, DefaultsConfig, DockerSettings, PrivacySettings
from mandri.core.types.execution import (
    ExecutionBackend,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.model_selection import ModelSource


@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_protected_native_selection_is_always_rejected(backend: ExecutionBackend) -> None:
    with pytest.raises(ProtectionError):
        SessionPolicy(backend, PrivacyMode.SURROGATE).validate(ModelSource.NATIVE)
    SessionPolicy(backend, PrivacyMode.NONE).validate(ModelSource.NATIVE)


def test_settings_roundtrip_and_existing_defaults(tmp_path: Path) -> None:
    adapter = TomlConfigAdapter(tmp_path)
    assert adapter.load().defaults.execution_backend is ExecutionBackend.HOST
    assert adapter.load().defaults.privacy_mode is PrivacyMode.NONE
    configured = replace(
        DaemonConfig(),
        defaults=DefaultsConfig(
            execution_backend=ExecutionBackend.DOCKER, privacy_mode=PrivacyMode.SURROGATE
        ),
        docker=DockerSettings(image="registry.invalid/worker@sha256:" + "a" * 64, cpus=2.5),
        privacy=PrivacySettings(key_file="/private-keys/privacy.key"),
    )
    adapter.save(configured)
    assert adapter.load() == configured


@pytest.mark.parametrize("value", [0, -1, True, "4", float("inf"), float("nan")])
def test_invalid_docker_cpu_limit_is_rejected(value: object) -> None:
    with pytest.raises(ConfigError):
        parse_docker({"cpus": value})


@pytest.mark.parametrize(
    "value",
    [
        {"max_depth": 0},
        {"max_entities": True},
        {"unknown": "x"},
        {"exact_values": ["private@example.invalid"]},
    ],
)
def test_removed_and_unknown_privacy_settings_are_not_loaded(value: dict) -> None:
    assert parse_privacy(value) == PrivacySettings()


def test_privacy_rules_file_is_a_local_path_setting_without_inline_values():
    assert (
        parse_privacy({"rules_file": "/private-config/rules.json"}).rules_file
        == "/private-config/rules.json"
    )
    with pytest.raises(ConfigError):
        parse_privacy({"rules_file": {"email": "private@example.invalid"}})


def test_image_pull_defaults_and_explicit_digest_roundtrip(tmp_path):
    assert parse_docker({}).pull_policy == "always"
    digest = "registry.invalid/worker@sha256:" + "b" * 64
    docker = parse_docker(
        {"image": digest, "pull_policy": "if-missing", "pull_timeout_seconds": 120}
    )
    adapter = TomlConfigAdapter(tmp_path)
    configured = replace(DaemonConfig(), docker=docker)
    adapter.save(configured)
    assert adapter.load() == configured


@pytest.mark.parametrize(
    "value",
    [
        {"pull_policy": "unknown"},
        {"pull_policy": []},
        {"pull_policy": "if-missing"},
        {"pull_policy": "if-missing", "image": "worker:latest"},
        {"image": "user:secret@registry.invalid/worker"},
        {"pull_timeout_seconds": 0},
        {"pull_timeout_seconds": True},
        {"pull_timeout_seconds": "600"},
    ],
)
def test_unsafe_image_acquisition_configuration_is_rejected(value):
    with pytest.raises(ConfigError):
        parse_docker(value)


def test_saved_privacy_settings_drop_obsolete_options(tmp_path):
    adapter = TomlConfigAdapter(tmp_path)
    adapter.save(DaemonConfig())
    path = tmp_path / "config.toml"
    content = path.read_text().replace(
        "[privacy]", "[privacy]\nmax_request_bytes = 10\nmax_entities = 1\nmax_depth = 2"
    )
    path.write_text(content)
    config = adapter.load()
    assert config.privacy == PrivacySettings()
    adapter.save(config)
    assert "max_request_bytes" not in path.read_text()
    assert "max_entities" not in path.read_text()
    assert "max_depth" not in path.read_text()
