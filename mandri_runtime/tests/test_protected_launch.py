import json
from dataclasses import replace
from typing import Any

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.launch import build_harness_launch
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.runtime.control.modes import LaunchMode
from mandri.runtime.launch_preparation import LaunchPreparation
from mandri.runtime.protected_launch import protected_launch

_BASE = "http://127.0.0.1:8175/v1/gateway/llm/synthetic-route"


@pytest.mark.parametrize("kind", list(HarnessKind))
def test_protected_launch_preserves_independent_plugin_credentials(kind: HarnessKind) -> None:
    parent = {
        "OPENROUTER_API_KEY": "synthetic-plugin-key",
        "OPENCODE_API_KEY": "synthetic-plugin-key",
        "AZURE_OPENAI_API_KEY": "synthetic-plugin-key",
        "HTTPS_PROXY": "http://proxy.example.invalid:8080",
    }
    prepared = LaunchPreparation(8175, lambda _: "token", parent).prepare(
        [kind.value],
        kind,
        "chosen",
        None,
        False,
        "synthetic-route",
        None,
        None,
        privacy_mode=PrivacyMode.SURROGATE,
    )
    for key, value in parent.items():
        assert prepared.env[key] == value
    assert "OPENAI_BASE_URL" not in prepared.env
    assert "OPENAI_API_BASE" not in prepared.env
    if kind is not HarnessKind.CLAUDE:
        assert "ANTHROPIC_BASE_URL" not in prepared.env


def test_protected_codex_preserves_plugins_and_sdk_config_with_gateway_model_routing() -> None:
    parent = {
        "OPENAI_API_KEY": "synthetic-plugin-key",
        "OPENAI_BASE_URL": "https://plugin.example.invalid/v1",
        "ANTHROPIC_API_KEY": "synthetic-plugin-key",
    }
    command = ["codex", "app-server", "-c", "features.plugins=true"]
    prepared = LaunchPreparation(8175, lambda _: "token", parent).prepare(
        command,
        HarnessKind.CODEX,
        "chosen",
        None,
        False,
        "synthetic-route",
        None,
        None,
        privacy_mode=PrivacyMode.SURROGATE,
    )
    for key, value in parent.items():
        assert prepared.env[key] == value
    assert prepared.env["MANDRI_API_KEY"] == "token"
    assert 'model_provider="mandri"' in prepared.argv
    assert f'model_providers.mandri.base_url="{_BASE}/v1"' in prepared.argv
    assert prepared.argv[: len(command)] == command
    assert not any("plugins=false" in arg or "remote_plugin=false" in arg for arg in prepared.argv)


def test_claude_auxiliary_models_share_the_session_route() -> None:
    plan = build_harness_launch(HarnessKind.CLAUDE, 8175, "synthetic-route", "chosen", "token")
    protected = protected_launch(
        plan, HarnessKind.CLAUDE, 8175, "synthetic-route", "token", "chosen"
    )
    for key in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert protected.env[key] == "chosen"
    assert (
        plan.env
        == build_harness_launch(HarnessKind.CLAUDE, 8175, "synthetic-route", "chosen", "token").env
    )


def test_opencode_helpers_keep_permissions_and_only_use_the_scoped_provider() -> None:
    extra: dict[str, Any] = {
        "permission": {"bash": "ask", "edit": "deny"},
        "small_model": "external/fast",
        "enabled_providers": ["external"],
        "agent": {"audit": {"model": "external/large", "permission": {"edit": "deny"}}},
    }
    prepared = LaunchPreparation(8175, lambda _: "token", {}).prepare(
        ["opencode"],
        HarnessKind.OPENCODE,
        "chosen",
        None,
        False,
        "synthetic-route",
        None,
        LaunchMode(opencode_config=extra),
        privacy_mode=PrivacyMode.SURROGATE,
    )
    config = json.loads(prepared.env["OPENCODE_CONFIG_CONTENT"])
    assert config["enabled_providers"] == ["mandri"]
    assert list(config["provider"]) == ["mandri"]
    assert config["model"] == config["small_model"] == "mandri/mandri_gateway"
    assert config["permission"] == extra["permission"]
    assert config["agent"] == {
        "audit": {"model": "mandri/mandri_gateway", "permission": {"edit": "deny"}}
    }
    assert extra["agent"]["audit"]["model"] == "external/large"
    assert extra["small_model"] == "external/fast"


@pytest.mark.parametrize("field,value", [("npm", "unqualified-sdk"), ("options", {})])
def test_opencode_provider_adapter_override_fails_before_launch(field: str, value: Any) -> None:
    launch = build_harness_launch(HarnessKind.OPENCODE, 8175, "synthetic-route", "chosen", "token")
    config = json.loads(launch.env["OPENCODE_CONFIG_CONTENT"])
    config["provider"]["mandri"][field] = value
    altered = replace(launch, env={"OPENCODE_CONFIG_CONTENT": json.dumps(config)})
    with pytest.raises(ProtectionError, match="provider configuration"):
        protected_launch(altered, HarnessKind.OPENCODE, 8175, "synthetic-route", "token", "chosen")


@pytest.mark.parametrize(
    "native,kind,route",
    [(True, HarnessKind.CODEX, "r"), (False, None, "r"), (False, HarnessKind.CODEX, None)],
)
def test_unbound_protected_helper_launch_is_rejected(
    native: bool, kind: HarnessKind | None, route: str | None
) -> None:
    with pytest.raises(ProtectionError, match="scoped gateway"):
        LaunchPreparation(8175, lambda _: "token", {}).prepare(
            ["synthetic"],
            kind,
            "chosen",
            None,
            native,
            route,
            None,
            None,
            privacy_mode=PrivacyMode.SURROGATE,
        )


def test_standard_launch_keeps_existing_sdk_environment() -> None:
    prepared = LaunchPreparation(
        8175,
        lambda _: "token",
        {
            "OPENAI_API_KEY": "synthetic-existing-key",
            "OPENAI_BASE_URL": "https://example.invalid",
        },
    ).prepare(["codex"], HarnessKind.CODEX, "chosen", None, False, "synthetic-route", None, None)
    assert prepared.env["OPENAI_API_KEY"] == "synthetic-existing-key"
    assert prepared.env["OPENAI_BASE_URL"] == "https://example.invalid"
    assert "ANTHROPIC_API_KEY" not in prepared.env


@pytest.mark.parametrize("field", ["agent", "mode"])
def test_protected_named_agents_preserve_tools_and_permissions_when_routed(field):
    definition = {
        "audit": {
            "model": "other/independent",
            "tools": {"bash": False},
            "permission": {"edit": "deny"},
            "description": "Inspect only",
            "prompt": "existing",
        },
        "default": {"tools": {"read": True}},
    }
    prepared = LaunchPreparation(8175, lambda _: "token", {}).prepare(
        ["opencode"],
        HarnessKind.OPENCODE,
        "chosen",
        None,
        False,
        "synthetic-route",
        None,
        LaunchMode(opencode_config={field: definition}),
        privacy_mode=PrivacyMode.SURROGATE,
    )
    selected = json.loads(prepared.env["OPENCODE_CONFIG_CONTENT"])[field]
    assert selected["audit"] == {**definition["audit"], "model": "mandri/mandri_gateway"}
    assert selected["default"] == definition["default"]
    assert definition["audit"]["model"] == "other/independent"


@pytest.mark.parametrize(
    "definition",
    [
        [],
        "plugin",
        {"audit": None},
        {"audit": {"model": None}},
        {"audit": {"model": ""}},
        {"audit": {"model": {"provider": "external"}}},
    ],
)
def test_unknown_agent_model_shapes_fail_before_process_launch(definition):
    with pytest.raises(ProtectionError) as raised:
        LaunchPreparation(8175, lambda _: "token", {}).prepare(
            ["opencode"],
            HarnessKind.OPENCODE,
            "chosen",
            None,
            False,
            "synthetic-route",
            None,
            LaunchMode(opencode_config={"agent": definition}),
            privacy_mode=PrivacyMode.SURROGATE,
        )
    assert raised.value.code == "privacy_state_unavailable"
