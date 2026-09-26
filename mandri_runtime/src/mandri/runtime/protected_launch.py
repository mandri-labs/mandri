import dataclasses
import json

from mandri.core.ids import HarnessKind
from mandri.core.launch import HarnessLaunch, route_base_url
from mandri.core.opencode import GATEWAY_MODEL_ID, GATEWAY_MODEL_REF
from mandri.core.types.execution import ProtectionError

_CLAUDE_MODELS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


def protected_launch(
    launch: HarnessLaunch,
    harness: HarnessKind,
    port: int,
    route_id: str,
    token: str,
    model: str,
) -> HarnessLaunch:
    base = route_base_url(port, route_id, HarnessKind.CLAUDE)
    env = dict(launch.env)
    strip = set(launch.strip)
    if harness is HarnessKind.CLAUDE:
        env.update(dict.fromkeys(_CLAUDE_MODELS, model))
        strip.update(("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"))
    if harness is HarnessKind.OPENCODE:
        env["OPENCODE_CONFIG_CONTENT"] = _opencode_config(env, base, token)
    return dataclasses.replace(launch, env=env, strip=tuple(sorted(strip)))


def _opencode_config(env: dict[str, str], base: str, token: str) -> str:
    try:
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["mandri"]
        options = provider["options"]
        models = provider["models"]
        if (
            not isinstance(config, dict)
            or not isinstance(models, dict)
            or GATEWAY_MODEL_ID not in models
            or provider["npm"] != "@ai-sdk/openai-compatible"
            or options["baseURL"] != f"{base}/v1"
            or options["apiKey"] != token
        ):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise ProtectionError(
            "privacy_state_unavailable", "The protected OpenCode provider configuration is invalid"
        ) from None
    for key in ("agent", "mode"):
        if key not in config:
            continue
        definitions = config[key]
        if not isinstance(definitions, dict):
            raise ProtectionError(
                "privacy_state_unavailable", "Protected agent definitions are invalid"
            )
        for definition in definitions.values():
            if not isinstance(definition, dict):
                raise ProtectionError(
                    "privacy_state_unavailable", "Protected agent configuration is invalid"
                )
            if "model" in definition:
                if not isinstance(definition["model"], str) or not definition["model"].strip():
                    raise ProtectionError(
                        "privacy_state_unavailable", "Protected agent model is invalid"
                    )
                definition["model"] = GATEWAY_MODEL_REF
    config.update(model=GATEWAY_MODEL_REF, small_model=GATEWAY_MODEL_REF)
    config["enabled_providers"] = ["mandri"]
    config["provider"] = {"mandri": provider}
    return json.dumps(config)
