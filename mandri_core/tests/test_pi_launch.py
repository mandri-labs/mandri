from pathlib import Path

from mandri.core.ids import HarnessKind
from mandri.core.launch import build_harness_launch
from mandri.core.model_metadata import ModelMetadata
from mandri.core.pi import model_entry


def test_gateway_preserves_native_resources_and_exposes_verified_metadata():
    launch = build_harness_launch(HarnessKind.PI, 8000, "route", "provider/model", "token")
    assert launch.env["MANDRI_PI_BASE_URL"] == "http://127.0.0.1:8000/v1/gateway/llm/route/v1"
    assert launch.env["MANDRI_API_KEY"] == "token"
    assert Path(launch.args[1]).is_file()
    assert launch.args[0] == "--extension"
    assert launch.args[2:] == ("--provider", "mandri", "--model", "gateway")
    assert "PI_CODING_AGENT_DIR" not in launch.env
    assert not any("no-" in value for value in launch.args)


def test_gateway_thinking_metadata_includes_explicit_extended_levels():
    entry = model_entry(
        "fixture", ModelMetadata(reasoning_efforts=("low", "high", "max"), image_input=True)
    )
    assert entry["input"] == ["text", "image"]
    assert entry["reasoning"] is True
    assert entry["thinkingLevelMap"] == {
        "off": "none",
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }
