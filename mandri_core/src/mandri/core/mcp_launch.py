import json
from collections.abc import Mapping
from importlib.resources import files

import tomlkit
from mandri.core.ids import HarnessKind
from mandri.core.launch import HarnessLaunch
from mandri.core.types.mcp import McpBinding


def mcp_launch(harness: HarnessKind, binding: McpBinding, env: Mapping[str, str]) -> HarnessLaunch:
    if harness is HarnessKind.AGY or not binding.servers:
        return HarnessLaunch({})
    token_env = {"MANDRI_MCP_TOKEN": binding.token}
    if harness is HarnessKind.CODEX:
        args: list[str] = []
        for server in binding.servers:
            config = tomlkit.inline_table()
            config.update(
                url=server.url,
                bearer_token_env_var="MANDRI_MCP_TOKEN",
                required=False,
                startup_timeout_sec=1,
            )
            key = tomlkit.key(["mcp_servers", server.name])
            document = tomlkit.document()
            document.add(key, config)
            args.extend(("-c", tomlkit.dumps(document).strip()))
        return HarnessLaunch(token_env, args=tuple(args))
    if harness is HarnessKind.CLAUDE:
        servers = {
            server.name: {
                "type": "http",
                "url": server.url,
                "headers": {"Authorization": "Bearer ${MANDRI_MCP_TOKEN}"},
            }
            for server in binding.servers
        }
        return HarnessLaunch(token_env, args=("--mcp-config", json.dumps({"mcpServers": servers})))
    if harness is HarnessKind.OPENCODE:
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}"))
        if not isinstance(config, dict):
            raise ValueError("Invalid inline OpenCode configuration")
        native = config.get("mcp", {})
        if not isinstance(native, dict):
            raise ValueError("Invalid inline OpenCode MCP configuration")
        added = {
            server.name: {
                "type": "remote",
                "url": server.url,
                "headers": {"Authorization": "Bearer {env:MANDRI_MCP_TOKEN}"},
                "oauth": False,
                "enabled": True,
                "timeout": 1000,
            }
            for server in binding.servers
            if server.name not in native
        }
        config["mcp"] = native | added
        return HarnessLaunch(token_env | {"OPENCODE_CONFIG_CONTENT": json.dumps(config)})
    manifest = [{"name": server.name, "url": server.url} for server in binding.servers]
    return HarnessLaunch(
        token_env | {"MANDRI_MCP_SERVERS": json.dumps(manifest)},
        args=("--extension", str(files("mandri.core").joinpath("resources/pi_mcp.ts"))),
    )
