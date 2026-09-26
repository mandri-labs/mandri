"""Tests for the Codex bound-model listing payload."""

from mandri.gateway.listings import codex_listing
from mandri.gateway.reasoning_catalog import ReasoningInfo

_REQUIRED_MODEL_KEYS = (
    "slug",
    "display_name",
    "supported_in_api",
    "supported_reasoning_levels",
    "shell_type",
    "visibility",
    "priority",
    "support_verbosity",
    "truncation_policy",
    "experimental_supported_tools",
    "input_modalities",
)


def test_codex_listing_has_full_model_info_shape() -> None:
    payload = codex_listing("z-ai/glm-5.2")
    assert payload["base_instructions"] == ""
    entry = payload["models"][0]
    assert entry["slug"] == "z-ai/glm-5.2"
    assert entry["display_name"] == "z-ai/glm-5.2"
    assert entry["supported_in_api"] is True
    for key in _REQUIRED_MODEL_KEYS:
        assert key in entry
    assert entry["shell_type"] == "unified_exec"
    assert entry["visibility"] == "list"
    assert isinstance(entry["priority"], int)
    assert entry["support_verbosity"] is False
    assert entry["truncation_policy"]["mode"] == "bytes"
    assert isinstance(entry["truncation_policy"]["limit"], int)
    assert entry["experimental_supported_tools"] == []
    assert entry["input_modalities"] == ["text", "image"]


def test_codex_listing_default_levels_without_catalog() -> None:
    entry = codex_listing("m1")["models"][0]
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == [
        "none",
        "low",
        "medium",
        "high",
    ]
    assert all(level["description"] for level in entry["supported_reasoning_levels"])
    assert entry["default_reasoning_level"] == "none"


def test_codex_listing_uses_catalog_efforts_and_default() -> None:
    reasoning = ReasoningInfo(efforts=["off", "on", "medium"], default_effort="medium")
    entry = codex_listing("m1", reasoning)["models"][0]
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == [
        "off",
        "on",
        "medium",
    ]
    assert entry["default_reasoning_level"] == "medium"


def test_codex_listing_is_backward_compatible_without_reasoning() -> None:
    payload = codex_listing("m1")
    assert set(payload) == {"models", "base_instructions"}
