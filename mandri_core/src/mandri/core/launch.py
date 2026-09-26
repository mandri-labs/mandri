"""Per-harness gateway env wiring and launch plans for harness processes."""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from mandri.core.codex_catalog import CATALOG_ENV, catalog
from mandri.core.ids import HarnessKind
from mandri.core.model_metadata import ModelMetadata as ModelMetadata
from mandri.core.opencode import GATEWAY_MODEL_ID, GATEWAY_MODEL_REF, model_entry
from mandri.core.pi import gateway_env, gateway_extension_path

_CLAUDE_BASE_SUFFIX = ""
_CODEX_BASE_SUFFIX = "/v1"
_OPENCODE_BASE_SUFFIX = "/v1"

_CLAUDE_STRIP_VARS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
)


def route_base_url(port: int, route_id: str, harness: HarnessKind) -> str:
    root = f"http://127.0.0.1:{port}/v1/gateway/llm/{route_id}"
    if harness in (HarnessKind.CLAUDE, HarnessKind.AGY):
        return root + _CLAUDE_BASE_SUFFIX
    if harness is HarnessKind.CODEX:
        return root + _CODEX_BASE_SUFFIX
    return root + _OPENCODE_BASE_SUFFIX


@dataclasses.dataclass(frozen=True)
class HarnessLaunch:
    """Gateway context to apply to a harness process."""

    env: dict[str, str]
    strip: tuple[str, ...] = ()
    args: tuple[str, ...] = ()


def merged_env(
    parent_env: Mapping[str, str],
    launch: HarnessLaunch,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(parent_env)
    if extra is not None:
        env.update(extra)
    env.update(launch.env)
    for name in launch.strip:
        env.pop(name, None)
    return env


def build_harness_launch(
    harness: HarnessKind,
    gateway_port: int,
    route_id: str,
    model: str,
    token: str,
    metadata: ModelMetadata | None = None,
    opencode_config: Mapping[str, Any] | None = None,
    *,
    effort: str | None = None,
) -> HarnessLaunch:
    route_base = route_base_url(gateway_port, route_id, harness)
    if harness is HarnessKind.CLAUDE:
        return HarnessLaunch(claude_env(route_base, token, model, metadata), _CLAUDE_STRIP_VARS)
    if harness is HarnessKind.CODEX:
        return HarnessLaunch(
            {
                "MANDRI_API_KEY": token,
                **({CATALOG_ENV: catalog(model, metadata)} if metadata else {}),
            },
            args=tuple(codex_args(route_base, model, metadata)),
        )
    if harness is HarnessKind.AGY:
        return HarnessLaunch(
            {
                "GEMINI_API_KEY": token,
                "GOOGLE_GEMINI_BASE_URL": route_base,
                "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
            },
            strip=("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"),
            args=("--model", "mandri"),
        )
    if harness is HarnessKind.PI:
        return HarnessLaunch(
            gateway_env(route_base, token, model, metadata),
            args=(
                "--extension",
                gateway_extension_path(),
                "--provider",
                "mandri",
                "--model",
                "gateway",
            ) + (("--thinking", effort) if effort else ()),
        )
    return HarnessLaunch(opencode_env(route_base, token, opencode_config, metadata))


def claude_env(
    route_base: str, auth_token: str, model: str, metadata: ModelMetadata | None = None
) -> dict[str, str]:
    env = {
        "ANTHROPIC_BASE_URL": route_base,
        "ANTHROPIC_AUTH_TOKEN": auth_token,
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if metadata is not None:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(metadata.context_window)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(metadata.output_tokens)
    return env


def codex_args(route_base: str, model: str, metadata: ModelMetadata | None = None) -> list[str]:
    args = [
        "-c",
        'model_provider="mandri"',
        "-c",
        f'model="{model}"',
        "-c",
        'model_providers.mandri.name="Mandri Gateway"',
        "-c",
        f'model_providers.mandri.base_url="{route_base}"',
        "-c",
        'model_providers.mandri.env_key="MANDRI_API_KEY"',
        "-c",
        'model_providers.mandri.wire_api="responses"',
    ]
    if metadata is not None:
        args.extend(("-c", f"model_context_window={metadata.context_window}"))
    if metadata is None or not metadata.hosted_web_search:
        args.extend(("-c", 'web_search="disabled"'))
    return args


def opencode_env(
    route_base: str,
    api_key: str,
    extra_config: Mapping[str, Any] | None = None,
    metadata: ModelMetadata | None = None,
) -> dict[str, str]:
    config: dict[str, Any] = {
        "model": GATEWAY_MODEL_REF,
        "small_model": GATEWAY_MODEL_REF,
        "provider": {
            "mandri": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": route_base, "apiKey": api_key},
                "models": {GATEWAY_MODEL_ID: model_entry(metadata)},
            }
        },
        "autoupdate": False,
        "share": "disabled",
    }
    if extra_config:
        config.update(extra_config)
    config["model"] = config["small_model"] = GATEWAY_MODEL_REF
    return {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}
