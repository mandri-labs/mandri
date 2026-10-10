from typing import Any

from mandri.mcp.catalog import Catalog
from mandri.mcp.supervisor import Supervisor
from starlette.applications import Starlette

from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server


class Proxy:
    def __init__(self, supervisor: Supervisor) -> None:
        self.upstream = supervisor
        self.server: Server[Any] = Server(
            supervisor.definition.config.name,
            on_list_tools=self.list_tools,
            on_call_tool=self.call_tool,
            on_list_resources=self.list_resources,
            on_list_resource_templates=self.list_templates,
            on_read_resource=self.read_resource,
            on_list_prompts=self.list_prompts,
            on_get_prompt=self.get_prompt,
            get_tool_input_schema=self.tool_schema,
        )
        self.app: Starlette = self.server.streamable_http_app(
            streamable_http_path="/", json_response=False, stateless_http=True
        )

    def catalog(self) -> Catalog:
        return self.upstream.catalog or Catalog([], [], [], [])

    def tool_schema(self, name: str) -> dict[str, Any] | None:
        return next((tool.input_schema for tool in self.catalog().tools if tool.name == name), None)

    async def list_tools(
        self, ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.catalog().tools)

    async def list_resources(
        self, ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=self.catalog().resources)

    async def list_templates(
        self, ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resource_templates=self.catalog().templates)

    async def list_prompts(
        self, ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=self.catalog().prompts)

    async def call_tool(
        self, ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        async def report(progress: float, total: float | None, message: str | None) -> None:
            await ctx.session.report_progress(progress, total, message)

        return await self.upstream.call(
            lambda client: client.session.call_tool(
                params.name,
                params.arguments,
                read_timeout_seconds=self.upstream.definition.config.call_timeout,
                progress_callback=report,
                meta=ctx.meta,
            )
        )

    async def read_resource(
        self, ctx: ServerRequestContext[Any], params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        return await self.upstream.call(
            lambda client: client.session.read_resource(params.uri, meta=ctx.meta)
        )

    async def get_prompt(
        self, ctx: ServerRequestContext[Any], params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        return await self.upstream.call(
            lambda client: client.session.get_prompt(params.name, params.arguments, meta=ctx.meta)
        )
