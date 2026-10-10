import json

import tomlkit
from mandri.core.ids import HarnessKind
from mandri.core.mcp_launch import mcp_launch
from mandri.core.types.mcp import McpBinding, McpEndpoint


def binding() -> McpBinding:
    return McpBinding(
        id="binding",
        token="private",
        servers=[
            McpEndpoint(id="server", name="github", url="http://127.0.0.1:9000/mcp"),
        ],
    )


def test_codex_inline_overrides_are_valid_toml():
    launch = mcp_launch(HarnessKind.CODEX, binding(), {})
    assert "private" not in " ".join(launch.args)
    parsed = tomlkit.loads(launch.args[1])
    assert parsed["mcp_servers"]["github"]["required"] is False
    assert parsed["mcp_servers"]["github"]["bearer_token_env_var"] == "MANDRI_MCP_TOKEN"


def test_opencode_deep_merge_preserves_native_entries_and_provider():
    env = {
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            {
                "provider": {"custom": {"options": {"baseURL": "https://example.invalid"}}},
                "mcp": {"native": {"type": "local", "command": ["fixture"]}},
            }
        )
    }
    launch = mcp_launch(HarnessKind.OPENCODE, binding(), env)
    parsed = json.loads(launch.env["OPENCODE_CONFIG_CONTENT"])
    assert parsed["provider"]["custom"]["options"]["baseURL"] == "https://example.invalid"
    assert set(parsed["mcp"]) == {"github", "native"}
    assert parsed["mcp"]["github"]["oauth"] is False


def test_claude_and_pi_tokens_stay_in_environment():
    for harness in (HarnessKind.CLAUDE, HarnessKind.PI):
        launch = mcp_launch(harness, binding(), {})
        assert "private" not in " ".join(launch.args)
        assert launch.env["MANDRI_MCP_TOKEN"] == "private"
    assert mcp_launch(HarnessKind.AGY, binding(), {}).args == ()
