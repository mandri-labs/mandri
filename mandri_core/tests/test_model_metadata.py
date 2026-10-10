import json
from importlib.metadata import version
from importlib.resources import files

from mandri.core.codex_catalog import catalog
from mandri.core.ids import HarnessKind
from mandri.core.launch import build_harness_launch
from mandri.core.model_metadata import ModelMetadata
from mandri.core.opencode import GATEWAY_MODEL_ID
from mandri.core.pi import model_entry as pi_model_entry


def test_unknown_metadata_preserves_native_instructions_tools_and_reasoning():
    native = json.loads(
        files("mandri.core").joinpath("resources/codex_model.json").read_text(encoding="utf-8")
    )
    assert native["version"] == version("openai-codex-cli-bin")
    entry = json.loads(catalog("fixture", ModelMetadata()))["models"][0]
    for key in (
        "model_messages",
        "experimental_supported_tools",
        "supported_reasoning_levels",
        "include_apps_usage_instructions",
        "include_skills_usage_instructions",
    ):
        assert entry[key] == native["model"][key]


def test_maximum_output_does_not_force_compaction_at_half_the_context():
    metadata = ModelMetadata(context_window=131072, output_tokens=65536)
    entry = json.loads(catalog("fixture", metadata))["models"][0]
    assert entry["context_window"] == entry["max_context_window"] == 131072
    assert entry["auto_compact_token_limit"] is None
    launch = build_harness_launch(HarnessKind.CODEX, 8000, "route", "fixture", "token", metadata)
    assert "model_context_window=131072" in launch.args
    assert not any("model_auto_compact_token_limit" in arg for arg in launch.args)


def test_explicit_compaction_policy_is_passed_to_codex_and_claude():
    metadata = ModelMetadata(65536, 8192, auto_compact_token_limit=48000)
    codex = build_harness_launch(HarnessKind.CODEX, 8000, "route", "fixture", "token", metadata)
    claude = build_harness_launch(HarnessKind.CLAUDE, 8000, "route", "fixture", "token", metadata)
    assert "model_auto_compact_token_limit=48000" in codex.args
    assert claude.env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "48000"


def test_claude_partial_metadata_keeps_verified_capabilities_without_fake_limits():
    launch = build_harness_launch(
        HarnessKind.CLAUDE, 8000, "route", "fixture", "token", ModelMetadata(tool_call=True)
    )
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in launch.env
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in launch.env
    assert launch.env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


def test_explicit_text_only_and_no_reasoning_are_respected_by_codex():
    entry = json.loads(
        catalog(
            "fixture",
            ModelMetadata(image_input=False, reasoning_supported=False, reasoning_efforts=()),
        )
    )["models"][0]
    assert entry["input_modalities"] == ["text"]
    assert entry["supported_reasoning_levels"] == []


def test_partial_exclusion_is_respected_without_an_effort_list():
    entry = json.loads(catalog("fixture", ModelMetadata(reasoning_supported=False)))["models"][0]
    assert entry["supported_reasoning_levels"] == []
    assert pi_model_entry(
        "fixture", ModelMetadata(image_input=False, input_modalities=("text", "image"))
    )["input"] == ["text"]


def test_compaction_and_catalog_are_stable_for_identical_metadata():
    metadata = ModelMetadata(65536, 8192, reasoning_efforts=("low", "high"))
    assert catalog("fixture", metadata) == catalog(
        "fixture", ModelMetadata.from_payload(metadata.to_payload())
    )


def test_opencode_native_config_receives_partial_metadata_and_gateway_transport():
    metadata = ModelMetadata(
        context_window=65536, reasoning_efforts=("low", "high"), tool_call=True
    )
    launch = build_harness_launch(HarnessKind.OPENCODE, 8000, "route", "fixture", "token", metadata)
    provider = json.loads(launch.env["OPENCODE_CONFIG_CONTENT"])["providers"]["mandri"]
    assert provider["package"] == "@opencode/ai/providers/openai-compatible"
    assert provider["settings"] == {
        "baseURL": "http://127.0.0.1:8000/v1/gateway/llm/route/v1",
        "apiKey": "token",
    }
    model = provider["models"][GATEWAY_MODEL_ID]
    assert model["limit"] == {"context": 65536}
    assert model["capabilities"]["tools"] is True
    assert model["variants"] == [
        {"id": "low", "settings": {"reasoningEffort": "low"}},
        {"id": "high", "settings": {"reasoningEffort": "high"}},
    ]


def test_agy_launch_only_supplies_verified_limits():
    unknown = build_harness_launch(
        HarnessKind.AGY, 8000, "route", "fixture", "token", ModelMetadata()
    )
    assert "MANDRI_AGY_MODEL" not in unknown.env
    known = build_harness_launch(
        HarnessKind.AGY, 8000, "route", "fixture", "token", ModelMetadata(context_window=65536)
    )
    assert json.loads(known.env["MANDRI_AGY_MODEL"]) == {"maxTokens": 65536}
