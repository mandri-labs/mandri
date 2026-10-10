import json
import os
import shutil
import subprocess
from importlib.resources import files

import pytest


@pytest.mark.parametrize(
    "supplied",
    [
        {"id": "gateway", "name": "fixture"},
        {
            "id": "gateway",
            "name": "fixture",
            "contextWindow": 8192,
            "reasoning": False,
            "input": ["text"],
        },
    ],
)
def test_native_fallback_preserves_limits_without_inheriting_an_external_transport(supplied):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    native = {
        "id": "native",
        "api": "openai-completions",
        "contextWindow": 32768,
        "maxTokens": 16384,
        "input": ["text", "image"],
        "reasoning": True,
        "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
        "baseUrl": "https://external.invalid",
        "headers": {"authorization": "external"},
        "compat": {"external": True},
    }
    models = [{**native, "api": "bedrock-converse-stream"}, native]
    source = files("mandri.core").joinpath("resources/pi_gateway.ts").read_text()
    source = source.replace(
        'import { getModels, getProviders } from "@earendil-works/pi-ai/compat";',
        f"const getProviders = () => ['fixture']; const getModels = () => {json.dumps(models)};",
    )
    source = source.replace("export default function (pi: any)", "function register(pi)")
    source += (
        "\nregister({registerProvider: (name, config) => "
        "process.stdout.write(JSON.stringify(config))});"
    )
    result = subprocess.run(
        [node, "--input-type=module"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "MANDRI_PI_BASE_URL": "http://gateway.invalid/v1",
            "MANDRI_API_KEY": "synthetic",
            "MANDRI_PI_MODEL": json.dumps(supplied),
        },
    )
    provider = json.loads(result.stdout)
    model = provider["models"][0]
    assert model["api"] == "openai-completions"
    assert model["baseUrl"] == "http://gateway.invalid/v1"
    assert "headers" not in model and "compat" not in model
    assert model["contextWindow"] == supplied.get("contextWindow", native["contextWindow"])
    assert model["maxTokens"] == min(native["maxTokens"], model["contextWindow"])
    assert model["reasoning"] is supplied.get("reasoning", native["reasoning"])
    assert model["input"] == supplied.get("input", native["input"])
