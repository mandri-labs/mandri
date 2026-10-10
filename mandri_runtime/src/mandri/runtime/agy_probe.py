from collections.abc import Mapping


def agy_probe_env(env: Mapping[str, str]) -> dict[str, str]:
    environment = dict(env)
    environment.pop("AGY_CLI_INTERACTIVE_HEADLESS", None)
    environment["AGY_CLI_NONINTERACTIVE_HEADLESS"] = "true"
    environment["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
    return environment
