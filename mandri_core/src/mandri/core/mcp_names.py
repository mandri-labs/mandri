import json
import tomllib
from collections.abc import Mapping
from pathlib import Path

from mandri.core.ids import HarnessKind
from mandri.core.jsonc import loads as load_jsonc
from mandri.core.types.mcp import McpBinding


def native_paths(harness: HarnessKind, env: Mapping[str, str], cwd: Path) -> list[Path]:
    home = Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())
    if harness is HarnessKind.CODEX:
        return [Path(env.get("CODEX_HOME") or home / ".codex") / "config.toml"]
    projects = [cwd, *cwd.parents]
    if harness is HarnessKind.CLAUDE:
        return [
            home / ".claude.json",
            Path(env.get("CLAUDE_CONFIG_DIR") or home / ".claude") / ".claude.json",
            Path(env.get("CLAUDE_CONFIG_DIR") or home / ".claude") / "settings.json",
            *(directory / ".mcp.json" for directory in projects),
        ]
    if harness is HarnessKind.PI:
        return [
            Path(env.get("PI_CODING_AGENT_DIR") or home / ".pi" / "agent") / "mcp.json",
            cwd / ".pi" / "mcp.json",
        ]
    config_home = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
    return [
        config_home / "opencode" / "opencode.json",
        config_home / "opencode" / "opencode.jsonc",
        *(
            directory / name
            for directory in projects
            for name in (
                "opencode.json",
                "opencode.jsonc",
                ".opencode/opencode.json",
                ".opencode/opencode.jsonc",
            )
        ),
        *([Path(env["OPENCODE_CONFIG"])] if env.get("OPENCODE_CONFIG") else []),
    ]


def native_names(harness: HarnessKind, env: Mapping[str, str], cwd: Path) -> set[str]:
    names: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, dict):
            return
        for key in ("mcp_servers", "mcpServers", "mcp"):
            entries = value.get(key)
            if isinstance(entries, dict):
                names.update(str(name).lower() for name in entries)
        projects = value.get("projects")
        if isinstance(projects, dict):
            for project in projects.values():
                add(project)

    for path in native_paths(harness, env, cwd):
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".toml":
            add(tomllib.loads(content))
        else:
            add(load_jsonc(content) if path.suffix == ".jsonc" else json.loads(content))
    if harness is HarnessKind.OPENCODE:
        add(json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}")))
    return names


def avoid_collisions(
    harness: HarnessKind, binding: McpBinding, env: Mapping[str, str], cwd: Path
) -> tuple[McpBinding, list[str]]:
    try:
        used = native_names(harness, env, cwd)
    except (OSError, ValueError):
        return binding.model_copy(update={"servers": []}), ["Native MCP configuration unavailable"]
    names = used | {server.name.lower() for server in binding.servers}
    diagnostics: list[str] = []
    servers = []
    for server in binding.servers:
        if server.name.lower() not in used:
            servers.append(server)
            continue
        name = f"{server.name[:32]}_{server.id.replace('-', '')[:8]}"
        if name.lower() in names:
            diagnostics.append(f"{server.name}: name collision")
            continue
        names.add(name.lower())
        diagnostics.append(f"{server.name}: {name}")
        servers.append(server.model_copy(update={"name": name}))
    return binding.model_copy(update={"servers": servers}), diagnostics
