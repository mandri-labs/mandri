import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
from mandri.core.types.mcp import McpServerConfig

from mcp import Client, StdioServerParameters
from mcp.client.session import MessageHandlerFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@asynccontextmanager
async def connect(
    config: McpServerConfig,
    base_dir: Path,
    auth: httpx2.Auth | None = None,
    message_handler: MessageHandlerFnT | None = None,
) -> AsyncIterator[Client]:
    if config.transport == "stdio":
        env = dict(config.env)
        for key, reference in config.env_refs.items():
            if reference not in os.environ:
                raise ValueError("Missing environment reference")
            env[key] = os.environ[reference]
        cwd = Path(config.cwd).expanduser() if config.cwd else base_dir
        if not cwd.is_absolute():
            cwd = base_dir / cwd
        params = StdioServerParameters(
            command=config.command or "", args=config.args, env=env, cwd=str(cwd)
        )
        with open(os.devnull, "w") as errors:
            async with Client(
                stdio_client(params, errlog=errors),
                read_timeout_seconds=config.call_timeout,
                message_handler=message_handler,
            ) as client:
                yield client
    elif config.transport == "sse":
        async with Client(
            sse_client(config.url or "", headers=config.headers, auth=auth),
            read_timeout_seconds=config.call_timeout,
            message_handler=message_handler,
        ) as client:
            yield client
    else:
        async with (
            httpx2.AsyncClient(
                headers=config.headers, timeout=config.call_timeout, trust_env=False, auth=auth
            ) as http,
            Client(
                streamable_http_client(config.url or "", http_client=http),
                read_timeout_seconds=config.call_timeout,
                message_handler=message_handler,
            ) as client,
        ):
            yield client
