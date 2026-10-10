import hashlib

from mandri.core.ids import HarnessKind
from mandri.core.mcp_names import avoid_collisions
from mandri.core.types.mcp import McpBinding, McpEndpoint


def test_native_config_is_unchanged_and_alias_is_per_server(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[mcp_servers.github]\ncommand = "native"\n')
    original = hashlib.sha256(config.read_bytes()).digest()
    binding = McpBinding(
        id="binding",
        token="private",
        servers=[
            McpEndpoint(id="12345678-abcd", name="github", url="http://127.0.0.1/mcp"),
            McpEndpoint(id="other", name="filesystem", url="http://127.0.0.1/other"),
        ],
    )
    renamed, diagnostics = avoid_collisions(
        HarnessKind.CODEX, binding, {"CODEX_HOME": str(tmp_path)}, tmp_path
    )
    assert [row.name for row in renamed.servers] == ["github_12345678", "filesystem"]
    assert diagnostics == ["github: github_12345678"]
    assert hashlib.sha256(config.read_bytes()).digest() == original
    assert binding.servers[0].name == "github"


def test_unreadable_config_omits_injection(tmp_path):
    (tmp_path / "config.toml").write_text("not valid toml")
    binding = McpBinding(
        id="binding",
        token="private",
        servers=[
            McpEndpoint(id="server", name="github", url="http://127.0.0.1/mcp"),
        ],
    )
    result, diagnostics = avoid_collisions(
        HarnessKind.CODEX, binding, {"CODEX_HOME": str(tmp_path)}, tmp_path
    )
    assert result.servers == []
    assert diagnostics


def test_jsonc_ancestor_configuration_is_preserved(tmp_path):
    root = tmp_path / "project"
    cwd = root / "nested"
    cwd.mkdir(parents=True)
    config = root / "opencode.jsonc"
    content = (
        '{"mcp": {"github": {"type": "remote", "url": "https://host/mcp//path",},}, /* note */}'
    )
    config.write_text(content)
    binding = McpBinding(
        id="binding",
        token="private",
        servers=[
            McpEndpoint(id="12345678-abcd", name="github", url="http://127.0.0.1/mcp"),
        ],
    )
    renamed, diagnostics = avoid_collisions(
        HarnessKind.OPENCODE, binding, {"HOME": str(tmp_path)}, cwd
    )
    assert renamed.servers[0].name == "github_12345678"
    assert diagnostics
    assert config.read_text() == content
