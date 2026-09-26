"""Config persistence port."""

from pathlib import Path

from mandri.core.types.config import DaemonConfig


class ConfigPort:
    """Abstract base for config persistence backends."""

    @property
    def config_path(self) -> Path:
        raise NotImplementedError

    def load(self) -> DaemonConfig:
        raise NotImplementedError

    def save(self, config: DaemonConfig) -> None:
        raise NotImplementedError
