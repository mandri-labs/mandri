from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mcp import Client, types

_MAX_ITEMS = 10000
_MAX_PAGES = 100
_MAX_BYTES = 8 * 1024 * 1024


async def collect[T](
    fetch: Callable[[str | None], Awaitable[tuple[list[T], str | None]]],
) -> list[T]:
    items: list[T] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(_MAX_PAGES):
        page, cursor = await fetch(cursor)
        items.extend(page)
        if len(items) > _MAX_ITEMS:
            raise ValueError("Catalog is too large")
        if cursor is None:
            return items
        if cursor in seen:
            raise ValueError("Repeated catalog cursor")
        seen.add(cursor)
    raise ValueError("Too many catalog pages")


@dataclass(frozen=True)
class Catalog:
    tools: list[types.Tool]
    resources: list[types.Resource]
    templates: list[types.ResourceTemplate]
    prompts: list[types.Prompt]

    @classmethod
    async def fetch(cls, client: Client) -> "Catalog":
        async def tools(cursor: str | None) -> tuple[list[types.Tool], str | None]:
            result = await client.list_tools(cursor=cursor, cache_mode="refresh")
            return result.tools, result.next_cursor

        async def resources(cursor: str | None) -> tuple[list[types.Resource], str | None]:
            result = await client.list_resources(cursor=cursor, cache_mode="refresh")
            return result.resources, result.next_cursor

        async def templates(cursor: str | None) -> tuple[list[types.ResourceTemplate], str | None]:
            result = await client.list_resource_templates(cursor=cursor, cache_mode="refresh")
            return result.resource_templates, result.next_cursor

        async def prompts(cursor: str | None) -> tuple[list[types.Prompt], str | None]:
            result = await client.list_prompts(cursor=cursor, cache_mode="refresh")
            return result.prompts, result.next_cursor

        capabilities = client.server_capabilities
        catalog = cls(
            await collect(tools) if capabilities.tools else [],
            await collect(resources) if capabilities.resources else [],
            await collect(templates) if capabilities.resources else [],
            await collect(prompts) if capabilities.prompts else [],
        )
        size = sum(
            len(item.model_dump_json().encode())
            for group in (catalog.tools, catalog.resources, catalog.templates, catalog.prompts)
            for item in group
        )
        if size > _MAX_BYTES:
            raise ValueError("Catalog is too large")
        names = [tool.name for tool in catalog.tools]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate tool names")
        return catalog
