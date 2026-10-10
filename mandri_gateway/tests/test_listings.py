import json

from mandri.core.codex_catalog import catalog
from mandri.core.model_metadata import ModelMetadata
from mandri.gateway.listings import codex_listing, gemini_listing
from mandri.gateway.reasoning_catalog import ReasoningInfo


def test_listing_preserves_the_native_prompt_and_capabilities():
    native = json.loads(catalog("fixture", ModelMetadata()))
    assert codex_listing("fixture") == native
    entry = native["models"][0]
    assert entry["model_messages"]["instructions_template"]
    assert entry["shell_type"] == "shell_command"
    assert entry["supported_in_api"] is True


def test_known_efforts_are_filtered_to_levels_the_harness_understands():
    reasoning = ReasoningInfo(efforts=["off", "on", "medium"], default_effort="medium")
    entry = codex_listing("fixture", reasoning)["models"][0]
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == ["medium"]
    assert entry["default_reasoning_level"] == "medium"


def test_explicit_metadata_takes_precedence_over_reasoning_fallback():
    entry = codex_listing(
        "fixture",
        ReasoningInfo(efforts=["high"], default_effort="high"),
        ModelMetadata(context_window=65536, reasoning_supported=False, reasoning_efforts=()),
    )["models"][0]
    assert entry["context_window"] == 65536
    assert entry["supported_reasoning_levels"] == []


def test_gemini_listing_supplies_only_known_limits():
    assert gemini_listing("fixture", ModelMetadata(input_tokens=32768)) == {
        "name": "models/fixture",
        "displayName": "fixture",
        "inputTokenLimit": 32768,
    }
    assert "outputTokenLimit" not in gemini_listing("fixture", ModelMetadata())
