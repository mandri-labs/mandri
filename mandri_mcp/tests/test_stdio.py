import asyncio
import sys

import psutil
from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import McpServerConfig
from mandri.mcp.service import McpService

SERVER = """
import asyncio
import os
from pathlib import Path
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

Path(os.environ["PID_FILE"]).write_text(str(os.getpid()))

async def tools(ctx, params):
    return types.ListToolsResult(tools=[
        types.Tool(name="pid", input_schema={"type": "object"})
    ])

async def call(ctx, params):
    return types.CallToolResult(content=[
        types.TextContent(type="text", text=str(os.getpid()))
    ])

server = Server("fixture", on_list_tools=tools, on_call_tool=call)

async def main():
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

asyncio.run(main())
"""


async def test_stdio_process_survives_sessions_and_dies_with_daemon(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    pid_file = tmp_path / "pid"
    service = McpService(tmp_path, 9000)
    service.start()
    await service.add(
        McpServerConfig(
            name="fixture",
            command=sys.executable,
            args=["-u", str(script)],
            env={"PID_FILE": str(pid_file)},
        )
    )
    try:
        async with asyncio.timeout(10):
            while service.snapshot().servers[0].state != "ready":
                await asyncio.sleep(0.01)
        pid = int(pid_file.read_text())
        for index in range(3):
            binding = service.bind(str(index), HarnessKind.CODEX)
            assert binding.servers[0].name == "fixture"
            supervisor = next(iter(service.supervisors.values()))
            result = await supervisor.call(lambda client: client.call_tool("pid"))
            assert result.content[0].text == str(pid)
            service.release(str(index))
            assert psutil.pid_exists(pid)
        assert int(pid_file.read_text()) == pid
    finally:
        await service.aclose()
    assert not psutil.pid_exists(pid)
