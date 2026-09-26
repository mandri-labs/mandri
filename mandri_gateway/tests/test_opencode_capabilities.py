import json

import pytest
from mandri.core.ids import HarnessKind, ProviderKind
from mandri.core.model_metadata import ModelMetadata
from mandri.core.opencode import model_entry
from mandri.gateway.model_metadata import _parse
from mandri.runtime.wiring import build_harness_launch


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        ModelMetadata(),
        ModelMetadata.from_payload(
            {
                "context_window": 64000,
                "output_tokens": 4096,
            }
        ),
    ],
)
def test_unknown_capabilities_stay_enabled(metadata):
    entry = model_entry(metadata)
    assert all(entry[key] for key in ("reasoning", "attachment", "tool_call", "temperature"))
    assert entry["interleaved"] == {"field": "reasoning_content"}
    assert set(entry["modalities"]["input"]) == {"text", "image", "audio", "video", "pdf"}
    assert entry["limit"]["context"] > entry["limit"]["output"] > 0


@pytest.mark.parametrize(
    "kind", [ProviderKind.OPENROUTER, ProviderKind.OPENCODE_GO, ProviderKind.OPENAI]
)
def test_partial_catalog_retains_explicit_exclusions_without_limits(kind):
    metadata = _parse(
        kind,
        {
            "data": [
                {
                    "id": "target",
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "capabilities": {"reasoning": False, "tool_call": False, "temperature": False},
                }
            ]
        },
        "target",
    )
    assert metadata is not None
    entry = model_entry(metadata)
    assert entry["reasoning"] is False and entry["interleaved"] is False
    assert entry["tool_call"] is False and entry["temperature"] is False
    assert entry["attachment"] is True
    assert entry["modalities"] == {"input": ["text", "image"], "output": ["text"]}
    assert ModelMetadata.from_payload(metadata.to_payload()) == metadata


def test_openrouter_authoritative_capabilities_and_reasoning_efforts():
    metadata = _parse(
        ProviderKind.OPENROUTER,
        {
            "data": [
                {
                    "id": "target",
                    "context_length": 200000,
                    "top_provider": {"max_completion_tokens": 16384},
                    "architecture": {
                        "input_modalities": ["text", "image", "file"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools", "reasoning"],
                    "reasoning": {"supported_efforts": ["low", "high"]},
                }
            ]
        },
        "target",
    )
    entry = model_entry(metadata)
    assert entry["reasoning"] is True and entry["tool_call"] is True
    assert entry["temperature"] is False
    assert entry["modalities"]["input"] == ["text", "image", "pdf"]
    assert entry["variants"] == {
        "low": {"reasoningEffort": "low"},
        "high": {"reasoningEffort": "high"},
    }
    assert entry["limit"] == {"context": 200000, "output": 16384}


def test_explicit_vision_exclusion_does_not_disable_other_unknown_modalities():
    entry = model_entry(ModelMetadata(image_input=False))
    assert "image" not in entry["modalities"]["input"]
    assert "pdf" in entry["modalities"]["input"]
    assert entry["attachment"] is True


def test_sparse_compatible_catalog_does_not_disable_vision():
    metadata = _parse(ProviderKind.OPENCODE_GO, {"data": [{"id": "target"}]}, "target")
    assert metadata is not None and metadata.image_input is None
    assert "image" in model_entry(metadata)["modalities"]["input"]


def test_text_only_catalog_disables_attachments():
    metadata = _parse(
        ProviderKind.OPENROUTER,
        {
            "data": [
                {
                    "id": "target",
                    "architecture": {"input_modalities": ["text"]},
                }
            ]
        },
        "target",
    )
    entry = model_entry(metadata)
    assert entry["modalities"]["input"] == ["text"]
    assert entry["attachment"] is False
    assert entry["reasoning"] is True


@pytest.mark.parametrize("target", ["provider/first", "provider/second"])
def test_launch_only_exposes_stable_gateway_alias(target):
    plan = build_harness_launch(
        HarnessKind.OPENCODE,
        8175,
        "route",
        target,
        "synthetic",
        ModelMetadata(64000, 4096, image_input=False),
    )
    serialized = plan.env["OPENCODE_CONFIG_CONTENT"]
    assert target not in serialized
    config = json.loads(serialized)
    assert config["model"] == config["small_model"] == "mandri/mandri_gateway"
    assert list(config["provider"]["mandri"]["models"]) == ["mandri_gateway"]
    assert (
        "image"
        not in config["provider"]["mandri"]["models"]["mandri_gateway"]["modalities"]["input"]
    )
