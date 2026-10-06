import math
from dataclasses import asdict
from typing import Any

from mandri.config.errors import ConfigError
from mandri.core.image_reference import valid_image_reference
from mandri.core.types.config import DockerSettings, PrivacySettings


def _values(name: str, data: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a table")
    unknown = set(data) - defaults.keys()
    if unknown:
        raise ConfigError(f"unknown keys in {name}: {sorted(unknown)}")
    return defaults | data


def _positive(name: str, value: Any, *, integer: bool = True) -> None:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types):
        raise ConfigError(f"{name} must be a positive {'integer' if integer else 'number'}")
    if value <= 0 or not math.isfinite(value):
        raise ConfigError(f"{name} must be positive and finite")


def _string(name: str, value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{name} must be a non-empty string")


def parse_docker(data: Any) -> DockerSettings:
    values = _values("docker", data, asdict(DockerSettings()))
    for name in ("image", "state_root"):
        _string(f"docker.{name}", values[name], optional=True)
    if not isinstance(values["pull_policy"], str) or values["pull_policy"] not in {
        "never",
        "if-missing",
        "always",
    }:
        raise ConfigError("docker.pull_policy must be never, if-missing or always")
    _positive("docker.pull_timeout_seconds", values["pull_timeout_seconds"])
    if values["image"] is not None and not valid_image_reference(values["image"]):
        raise ConfigError("docker.image must be an image reference without credentials")
    if values["pull_policy"] == "if-missing" and not valid_image_reference(
        values["image"], digest_required=True
    ):
        raise ConfigError("docker.pull_policy if-missing requires an image repository and digest")
    for name in ("ingress_bind", "ingress_host", "docker_binary"):
        _string(f"docker.{name}", values[name])
    _positive("docker.cpus", values["cpus"], integer=False)
    for name in ("memory_mb", "pids_limit", "tmpfs_mb"):
        _positive(f"docker.{name}", values[name])
    exceptions = values["network_exceptions"]
    if not isinstance(exceptions, (list, tuple)):
        raise ConfigError("docker.network_exceptions must be an array of strings")
    for value in exceptions:
        _string("docker.network_exceptions", value)
    values["network_exceptions"] = tuple(exceptions)
    values["cpus"] = float(values["cpus"])
    return DockerSettings(**values)


def parse_privacy(data: Any) -> PrivacySettings:
    if not isinstance(data, dict):
        raise ConfigError("privacy must be a table")
    defaults = asdict(PrivacySettings())
    values = defaults | {key: value for key, value in data.items() if key in defaults}
    _string("privacy.key_file", values["key_file"], optional=True)
    _string("privacy.key_service", values["key_service"])
    _string("privacy.rules_file", values["rules_file"], optional=True)
    return PrivacySettings(**values)
