"""Config errors."""


class ConfigError(Exception):
    """Base class for configuration failures."""


class ConfigParseError(ConfigError):
    """Raised when the config file cannot be parsed."""


class SecretRefError(ConfigError):
    """Raised when a secret reference is invalid."""
